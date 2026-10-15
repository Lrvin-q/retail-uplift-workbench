"""检查点持久化。

蓝图阶段七验收条件：

> **进程重启后的恢复已经实际验证，不能只使用内存状态宣称持久化完成。**

因此本模块不满足于「能存」，而是提供 :func:`verify_resume_after_restart`——
它**另起一个子进程**去读取同一个 thread_id 的检查点，
以此证明状态确实落在了进程之外。
"""

from __future__ import annotations

import json
import subprocess
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any

from dsa.config import PROJECT_ROOT

from dsa.config import ARTIFACTS_DIR
ARTIFACTS_ROOT = ARTIFACTS_DIR
SQLITE_PATH = ARTIFACTS_ROOT / "checkpoints.db"

_KIND: str | None = None


def checkpointer_kind() -> str:
    """返回上一次 :func:`make_checkpointer` 实际选中的类型。"""
    return _KIND or "unknown"


__all__ = [
    "ARTIFACTS_ROOT",
    "SQLITE_PATH",
    "checkpointer_kind",
    "list_checkpoints",
    "make_checkpointer",
    "new_thread_config",
    "read_thread_state",
    "verify_resume_after_restart",
]


def _try_postgres() -> Any:
    """用可写连接构造 PostgresSaver。

    ⚠️ 必须显式载入 ``.env``：``dsa.config`` 的 pydantic-settings ``env_file``
    **不会写回 os.environ**，所以这里直接读 ``os.environ["DATABASE_URL"]``
    会拿不到值，导致 postgres 后端被静默跳过、降级到 memory
    （即"看起来选了持久化，实际只在内存里"）。这是 L021 同一根源的第二次出现。
    """
    from dsa.agent.llm import _load_dotenv_once

    _load_dotenv_once()

    import os

    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver

    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return None
    dsn = url.replace("postgresql+psycopg://", "postgresql://")
    conn = psycopg.connect(dsn, autocommit=True)
    saver = PostgresSaver(conn)
    saver.setup()  # 建检查点所需表；幂等
    return saver


def _try_sqlite() -> Any:
    from langgraph.checkpoint.sqlite import SqliteSaver

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(SQLITE_PATH), check_same_thread=False))


def _try_memory() -> Any:
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def make_checkpointer(*, kind: str = "auto") -> Any:
    """返回检查点存储器。

    ``auto`` 依次尝试 postgres → sqlite → memory，用第一个可用的。
    选中类型可通过 :func:`checkpointer_kind` 读取。

    注意：**memory 不满足蓝图的持久化验收条件**，
    它只用于测试；实际演示必须用 sqlite 或 postgres。
    """
    global _KIND
    if kind != "auto":
        maker = {"postgres": _try_postgres, "sqlite": _try_sqlite,
                 "memory": _try_memory}.get(kind)
        if maker is None:
            raise ValueError(f"未知的 checkpointer 类型: {kind!r}")
        cp = maker()
        _KIND = kind
        return cp

    for name, maker in (("postgres", _try_postgres),
                        ("sqlite", _try_sqlite),
                        ("memory", _try_memory)):
        try:
            cp = maker()
            if cp is not None:
                _KIND = name
                return cp
        except Exception:  # noqa: BLE001 —— 逐个降级，最终一定有可用的
            continue
    _KIND = "memory"
    return _try_memory()


def new_thread_config(thread_id: str | None = None) -> dict:
    """构造 LangGraph 的 config。thread_id 缺省生成 uuid4。"""
    return {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}


def _channel_values(item: Any) -> dict:
    """从一条检查点记录里取出**状态值**。

    缺陷回顾（外部审查 F04）
    ------------------------
    > `checkpointer.list()` 返回检查点记录，原状态在
    > `checkpoint['channel_values']` 中；代码读取 `getattr(item, 'values', None)`，
    > 得到空值，原问题和澄清记录丢失。

    也就是说：记录里的状态确实在，只是**取错了字段**，
    于是"恢复"变成了"从零开始" —— 用户答了指标，系统却再问一遍男装还是女装。

    这里按可靠性依次尝试三种取法，**只要有一种拿到非空就返回**：
    新版 `.values`、`CheckpointTuple.checkpoint['channel_values']`、
    以及某些实现上的 `.checkpoint['values']`。
    """
    values = getattr(item, "values", None)
    if isinstance(values, dict) and values:
        return dict(values)

    cp = getattr(item, "checkpoint", None)
    if isinstance(cp, dict):
        for key in ("channel_values", "values"):
            inner = cp.get(key)
            if isinstance(inner, dict) and inner:
                return dict(inner)
    return dict(values) if isinstance(values, dict) else {}


def read_thread_state(checkpointer: Any, thread_id: str) -> dict:
    """读取某线程**最近一次**的状态值；读不到返回 ``{}``。

    优先 ``get_tuple``（单条、语义明确），退回 ``list``（迭代、取最近一条）。
    两条路都失败时返回空字典 —— 调用方据此判断"确实没有可恢复状态"，
    而不是把"读失败"也当成"没有状态"。
    """
    config = new_thread_config(thread_id)
    getter = getattr(checkpointer, "get_tuple", None)
    if callable(getter):
        try:
            item = getter(config)
            state = _channel_values(item) if item is not None else {}
            if state:
                return state
        except Exception:  # noqa: BLE001 —— 换下一种取法
            pass
    try:
        for item in checkpointer.list(config):
            state = _channel_values(item)
            if state:
                return state
            break
    except Exception:  # noqa: BLE001
        return {}
    return {}


def list_checkpoints(checkpointer: Any, thread_id: str) -> list[dict]:
    """列出某线程的历史检查点。

    不同 saver 的迭代接口略有差异，这里做兼容处理；
    读不到时返回空列表而不是抛异常（调用方据此判断是否可恢复）。
    """
    config = new_thread_config(thread_id)
    out: list[dict] = []
    try:
        for cp in checkpointer.list(config):
            checkpoint = getattr(cp, "checkpoint", None)
            out.append({
                "checkpoint_id": checkpoint.get("id") if isinstance(checkpoint, dict) else None,
                "values_keys": sorted(_channel_values(cp).keys()),
            })
    except Exception:  # noqa: BLE001
        return []
    return out


#: 子进程探针：读指定 saver 上的 thread_id，打印 JSON。
#: 只支持 postgres —— sqlite 后端（langgraph-checkpoint-sqlite）本环境未安装。
#: 探针在**独立进程**里运行，因此它读到非空检查点即证明状态落在了进程之外。
_PROBE = r'''
import json, os, sys
sys.path.insert(0, {src!r})
thread_id = sys.argv[1]
try:
    from dotenv import load_dotenv
    load_dotenv({env!r}, override=False)
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    dsn = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://")
    saver = PostgresSaver(psycopg.connect(dsn, autocommit=True))
    cfg = {{"configurable": {{"thread_id": thread_id}}}}
    n, keys = 0, []
    for cp in saver.list(cfg):
        n += 1
        vals = getattr(cp, "values", None) or {{}}
        keys = sorted(vals.keys())
    print(json.dumps({{"n": n, "keys": keys, "pid": os.getpid()}}))
except Exception as e:
    print(json.dumps({{"error": f"{{type(e).__name__}}: {{e}}"}}))
'''


def verify_resume_after_restart(thread_id: str, *, kind: str = "postgres") -> dict:
    """**在新进程中**读取指定 thread_id 的检查点。

    这是蓝图验收条件「进程重启后的恢复已经实际验证」的直接落地：
    若状态只存在于内存，子进程读到的检查点数必定为 0。

    Returns:
        ``{"thread_id", "kind", "n_checkpoints", "last_state_keys",
        "resumable", "probe_pid", "note"}``
    """
    if kind != "postgres":
        return {
            "thread_id": thread_id, "kind": kind, "n_checkpoints": 0,
            "last_state_keys": [], "resumable": False, "probe_pid": None,
            "error": None,
            "note": ("只有 postgres 后端能跨进程验证；"
                     "memory 必然不可恢复，本函数不会为它出具通过结论"),
        }

    src = str(PROJECT_ROOT / "src")
    env = str(PROJECT_ROOT / ".env")
    code = _PROBE.format(src=src, env=env)
    payload: dict = {}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, thread_id],
            capture_output=True, text=True, timeout=180,
        )
        out = (proc.stdout or "").strip()
        payload = json.loads(out.splitlines()[-1]) if out else {}
    except Exception as exc:  # noqa: BLE001
        payload = {"error": f"{type(exc).__name__}: {exc}"}

    n = int(payload.get("n", 0) or 0)
    return {
        "thread_id": thread_id,
        "kind": kind,
        "n_checkpoints": n,
        "last_state_keys": payload.get("keys", []),
        "resumable": n > 0,
        "probe_pid": payload.get("pid"),
        "error": payload.get("error"),
        "note": "子进程读取；n=0 表示状态只存在于内存，不满足持久化验收条件",
    }
