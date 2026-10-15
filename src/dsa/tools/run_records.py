"""运行记录读写：`analysis_runs` 与 `query_executions`。

依据：
- [docs/08-工具协议.md](../../../docs/08-工具协议.md) §6「运行记录」与 §6.2「重复执行检查」
- [docs/05-报告结构与异常分类.md](../../../docs/05-报告结构与异常分类.md) §2.3 引用完整性与 `run_id` 追溯
- `sql/001_schema.sql` 第 8、9 节（表结构是唯一权威，本模块不建表、不改表）

## 状态机

`pending` → `running` → `succeeded` / `blocked` / `failed`

- `blocked`：数据质量检查未通过，**不输出因果结论**，但保留描述性结果。
- `failed`：工具或数据库错误。
- 状态取值受 `analysis_runs_status_ck` 约束，本模块在**进入数据库之前**用
  `RUN_STATUSES` 校验，避免把非法值交给数据库再翻译错误信息。

## 幂等（铁律 3：重复执行检查）

LangGraph 的中断恢复会**重新执行节点中的前置代码**，没有幂等保证就会产生
无法区分的重复产物。本模块提供三层保证：

1. `create_run` 支持调用方传入显式 `run_id`；同 `run_id` 重复创建**不新增行**
   （`ON CONFLICT (run_id) DO NOTHING`）。若已存在行与本次请求的实验/比较不一致，
   直接抛 `ValueError`——静默复用会让两次不同含义的运行共用一个 `run_id`，
   这正是"无法区分的重复产物"。
2. `record_query` 依赖 `UNIQUE(run_id, seq)`，冲突时 `DO UPDATE`，不插入新行。
3. `deterministic_seq(tool_name, params)` 由**工具名 + 参数哈希**生成确定性 `seq`，
   重放得到同一个 `seq`；调用方也可以自己传 `seq`（例如节点内的递增计数器）。

## 敏感信息

本模块**不记录连接串、密码或 API key**：写入的只有业务参数、SQL 文本与错误码。
异常信息由调用方（`dsa.api.main`）在返回前脱敏。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import date, datetime, time
from decimal import Decimal
from functools import lru_cache
from typing import Any, Mapping, Sequence

from sqlalchemy import Engine, text

from dsa.config import PROJECT_ROOT

__all__ = [
    "DEFAULT_LIST_LIMIT",
    "DEFAULT_QUERY_LIMIT",
    "MAX_LIST_LIMIT",
    "QUERY_STATUSES",
    "RUN_STATUSES",
    "TERMINAL_RUN_STATUSES",
    "code_version",
    "create_run",
    "deterministic_seq",
    "get_run",
    "list_queries",
    "list_runs",
    "record_query",
    "to_jsonable",
    "update_run_status",
]

#: `analysis_runs.status` 的全部合法取值（与 `analysis_runs_status_ck` 一致）
RUN_STATUSES: tuple[str, ...] = ("pending", "running", "succeeded", "blocked", "failed")

#: 终态：进入终态时写 `finished_at`；回到 `pending`/`running` 时清空
TERMINAL_RUN_STATUSES: tuple[str, ...] = ("succeeded", "blocked", "failed")

#: `query_executions.status` 的全部合法取值（与 `query_executions_status_ck` 一致）
QUERY_STATUSES: tuple[str, ...] = ("ok", "error", "timeout", "rejected")

DEFAULT_LIST_LIMIT = 20
MAX_LIST_LIMIT = 200
DEFAULT_QUERY_LIMIT = 200

_UUID_NS = "run_id"

_RUN_COLUMNS = (
    "run_id", "parent_run_id",
    "experiment_id",
    "dataset_id",
    "comparison_key",
    "question",
    "plan",
    "status",
    "result_refs",
    "report_ref",
    "code_version",
    "stats_config",
    "error_code",
    "error_detail",
    "started_at",
    "finished_at",
)

_QUERY_COLUMNS = (
    "query_exec_id",
    "run_id",
    "seq",
    "tool_name",
    "sql_text",
    "params",
    "row_count",
    "duration_ms",
    "status",
    "error_code",
    "error_detail",
    "executed_at",
)


# ============================================================================
#  内部工具
# ============================================================================
def to_jsonable(value: Any) -> Any:
    """把数据库返回值转成可 `json.dumps` 的结构。

    psycopg 会把 `uuid` / `timestamptz` / `numeric` 还原成 Python 对象，
    直接交给 FastAPI 会被 `jsonable_encoder` 处理，但**先转一遍**可让
    `get_run()` 的结果在任何调用方（脚本、CLI、测试）手里都能直接序列化。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        # numeric 在本项目中用于统计量与金额；float 足够表达，且 JSON 无 Decimal
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _json_dumps(value: Any) -> str:
    """序列化为 jsonb 参数（`ensure_ascii=False` 保留中文原样）。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _require_non_empty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串，收到: {value!r}")
    return value


def as_run_id(run_id: Any) -> str:
    """校验并规范化 `run_id`（数据库列类型为 `uuid`）。

    提前校验而不是交给数据库：非法 UUID 走到数据库会变成一条难读的
    `DataError`，调用方无法区分"参数写错"和"数据库故障"。
    """
    if isinstance(run_id, uuid.UUID):
        return str(run_id)
    _require_non_empty_str(run_id, _UUID_NS)
    try:
        return str(uuid.UUID(str(run_id)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"run_id 必须是合法 UUID，收到: {run_id!r}") from exc


def _validate_seq(seq: Any) -> int:
    if isinstance(seq, bool) or not isinstance(seq, int):
        raise ValueError(f"seq 必须是整数，收到: {seq!r}")
    if not 1 <= seq <= 2_147_483_647:
        raise ValueError(f"seq 必须在 1..2147483647（integer 列范围）内，收到: {seq!r}")
    return seq


def _validate_optional_count(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} 必须是整数或 None，收到: {value!r}")
    if value < 0:
        raise ValueError(f"{field} 不能为负，收到: {value!r}")
    return value


# ============================================================================
#  代码版本
# ============================================================================
@lru_cache(maxsize=1)
def code_version() -> str:
    """返回可复现信息用的代码版本字符串，例如 `a1b2c3d` 或 `a1b2c3d-dirty`。

    实现已收敛到 :func:`dsa.version.code_version`（**一处实现，两处引用**）：
    本模块与 `dsa.stats.experiment` 原本各有一份，返回口径不同，
    于是"这次用的哪个版本"取决于谁写的记录。

    保留这个同名包装是为了不打断既有调用点，
    但它不再包含任何自己的逻辑 —— 这样两份实现不可能再漂移。
    """
    from dsa.version import code_version as _impl

    return _impl()


# ============================================================================
#  确定性 seq
# ============================================================================
def deterministic_seq(tool_name: str, params: Mapping[str, Any] | None = None, *,
                      salt: str | None = None) -> int:
    """由「工具名 + 参数」生成确定性 `seq`，用于重放安全的查询记录。

    为什么不用自增计数器：节点被中断恢复后重新执行，计数器会从当前内存状态
    继续（或从头开始），同一个工具调用会拿到不同的 `seq`，于是产生两条
    无法区分的记录。哈希 `seq` 让"同一个调用"永远落在同一行上，
    `ON CONFLICT (run_id, seq) DO UPDATE` 因此是幂等的。

    取值范围 1..2147483647（`query_executions.seq` 是 `integer`）。
    31 位空间下，几十条记录的碰撞概率可忽略；真碰撞时会覆盖同一行，
    所以**要求严格计数的场景请由调用方传入自己的 `seq`**。
    """
    _require_non_empty_str(tool_name, "tool_name")
    payload = _json_dumps({"tool": tool_name, "params": params or {}, "salt": salt})
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:4], "big") % 2_147_483_647 + 1


# ============================================================================
#  写：analysis_runs
# ============================================================================
_INSERT_RUN_SQL = text(
    """
    insert into analysis_runs (
        run_id, experiment_id, dataset_id, comparison_key, question,
        plan, status, code_version, stats_config, parent_run_id
    )
    values (
        :run_id, :experiment_id, :dataset_id, :comparison_key, :question,
        cast(:plan as jsonb), :status, :code_version, cast(:stats_config as jsonb), :parent_run_id
    )
    on conflict (run_id) do nothing
    returning run_id
    """
)

_SELECT_EXPERIMENT_SQL = text(
    "select experiment_id, dataset_id from experiments where experiment_key = :experiment_key"
)

_SELECT_COMPARISON_SQL = text(
    """
    select comparison_id from experiment_comparisons
    where experiment_id = :experiment_id and comparison_key = :comparison_key
    """
)

_SELECT_RUN_SQL = text(
    "select run_id, experiment_id, dataset_id, comparison_key, status, parent_run_id from analysis_runs "
    "where run_id = :run_id"
)


def create_run(
    engine_writable: Engine,
    *,
    experiment_key: str,
    comparison_key: str,
    question: str | None = None,
    plan: Mapping[str, Any] | None = None,
    stats_config: Mapping[str, Any] | None = None,
    run_id: str | uuid.UUID | None = None,
    parent_run_id: str | uuid.UUID | None = None,
) -> str:
    """登记一次分析运行，返回 `run_id`（写入时 `status='pending'`）。

    Args:
        engine_writable: **可写**引擎（`dsa_app`）。分析查询一律用只读引擎，
            但运行记录本身是写入操作。
        experiment_key: `experiments.experiment_key`，必须已登记。
        comparison_key: `experiment_comparisons.comparison_key`，必须已在
            该实验下预注册（如 `primary` / `secondary_womens`）。
        question: 用户原始问题，原样保存不改写（报告第 1 章要求）。
        plan: 结构化分析计划（JSON）。
        stats_config: 本次运行的统计参数（α、区间方法、种子……），复现所需。
        run_id: 调用方显式指定的 `run_id`（幂等重放用）。缺省生成新 UUID。

    Returns:
        `run_id` 字符串。

    Raises:
        ValueError: `run_id` 非法、实验或比较未登记、或同一 `run_id`
            已存在但与本请求的实验/比较不一致。
    """
    rid = as_run_id(run_id) if run_id is not None else str(uuid.uuid4())
    if question is not None and not isinstance(question, str):
        raise ValueError(f"question 必须是字符串或 None，收到: {type(question).__name__}")
    experiment_key = _require_non_empty_str(experiment_key, "experiment_key")
    comparison_key = _require_non_empty_str(comparison_key, "comparison_key")

    params = {
        "run_id": rid,
        "parent_run_id": as_run_id(parent_run_id) if parent_run_id else None,
        "experiment_id": None,
        "dataset_id": None,
        "comparison_key": comparison_key,
        "question": question,
        "plan": None if plan is None else _json_dumps(plan),
        "status": "pending",
        "code_version": code_version(),
        "stats_config": None if stats_config is None else _json_dumps(stats_config),
    }

    with engine_writable.begin() as conn:
        experiment = (
            conn.execute(_SELECT_EXPERIMENT_SQL, {"experiment_key": experiment_key})
            .mappings()
            .first()
        )
        if experiment is None:
            raise ValueError(
                f"experiment_key 未登记: {experiment_key!r}；"
                "实验配置缺失属 E_CONFIG_001，需先补齐 experiments 表"
            )
        comparison = (
            conn.execute(
                _SELECT_COMPARISON_SQL,
                {"experiment_id": experiment["experiment_id"], "comparison_key": comparison_key},
            )
            .mappings()
            .first()
        )
        if comparison is None:
            raise ValueError(
                f"comparison_key 未在该实验下预注册: {comparison_key!r}"
                f"（experiment_key={experiment_key!r}）"
            )

        params["experiment_id"] = experiment["experiment_id"]
        params["dataset_id"] = experiment["dataset_id"]

        inserted = conn.execute(_INSERT_RUN_SQL, params).scalar_one_or_none()
        if inserted is not None:
            return rid

        # 冲突：同 run_id 已存在。内容一致才算幂等重放，否则必须报错。
        existing = conn.execute(_SELECT_RUN_SQL, {"run_id": rid}).mappings().first()
        if existing is None:
            # 极端并发下另一事务刚回滚；重试由调用方决定，这里明确报错。
            raise RuntimeError(f"run_id 冲突但读不到已存在的行: {rid}")
        if (
            existing["experiment_id"] != params["experiment_id"]
            or existing["comparison_key"] != comparison_key
            or (str(existing["parent_run_id"]) if existing["parent_run_id"] else None) != params["parent_run_id"]
        ):
            raise ValueError(
                f"run_id 已存在且内容不同: {rid}（已存在 experiment_id="
                f"{existing['experiment_id']}, comparison_key={existing['comparison_key']!r}；"
                f"本次 experiment_id={params['experiment_id']}, comparison_key={comparison_key!r}）。"
                "同一个 run_id 不得代表两次不同含义的运行"
            )
        return rid


_UPDATE_RUN_SQL = text(
    """
    update analysis_runs set
        status       = :status,
        error_code   = coalesce(:error_code, error_code),
        error_detail = coalesce(:error_detail, error_detail),
        report_ref   = coalesce(:report_ref, report_ref),
        result_refs  = coalesce(cast(:result_refs as jsonb), result_refs),
        finished_at  = case
                         when :status in ('succeeded', 'blocked', 'failed')
                           then coalesce(finished_at, now())
                         else null
                       end
    where run_id = :run_id
    returning run_id
    """
)


def update_run_status(
    engine_writable: Engine,
    run_id: str | uuid.UUID,
    *,
    status: str,
    error_code: str | None = None,
    error_detail: str | None = None,
    report_ref: str | None = None,
    result_refs: Sequence[Any] | None = None,
) -> None:
    """更新运行状态与终态字段。

    行为约定：

    - 进入 `succeeded` / `blocked` / `failed` 时写 `finished_at`；
      回到 `pending` / `running` 时**清空** `finished_at`（支持失败后重跑）。
    - 同一终态重复提交时 `finished_at` 保持首次值（`coalesce(finished_at, now())`），
      重放不改变追溯记录。
    - `error_code` / `error_detail` / `report_ref` / `result_refs` 传 `None` 表示
      "本次不改动该字段"，不表示清空——避免局部更新把已有证据抹掉。

    Raises:
        ValueError: `status` 不在 `RUN_STATUSES` 内，或 `result_refs` 不是序列。
        LookupError: `run_id` 不存在（不静默成功）。
    """
    if status not in RUN_STATUSES:
        raise ValueError(
            f"status 取值不合法: {status!r}；允许值: {', '.join(RUN_STATUSES)}"
        )
    rid = as_run_id(run_id)
    if result_refs is not None:
        if isinstance(result_refs, (str, bytes)) or not isinstance(result_refs, Sequence):
            raise ValueError(
                f"result_refs 必须是序列（列表），收到: {type(result_refs).__name__}"
            )
        result_refs_json: str | None = _json_dumps(list(result_refs))
    else:
        result_refs_json = None

    params = {
        "run_id": rid,
        "status": status,
        "error_code": error_code,
        "error_detail": error_detail,
        "report_ref": report_ref,
        "result_refs": result_refs_json,
    }
    with engine_writable.begin() as conn:
        updated = conn.execute(_UPDATE_RUN_SQL, params).scalar_one_or_none()
    if updated is None:
        raise LookupError(f"run_id 不存在，无法更新状态: {rid}")


# ============================================================================
#  写：query_executions
# ============================================================================
_UPSERT_QUERY_SQL = text(
    """
    insert into query_executions (
        run_id, seq, tool_name, sql_text, params,
        row_count, duration_ms, status, error_code, error_detail
    )
    values (
        :run_id, :seq, :tool_name, :sql_text, cast(:params as jsonb),
        :row_count, :duration_ms, :status, :error_code, :error_detail
    )
    on conflict (run_id, seq) do update set
        tool_name    = excluded.tool_name,
        sql_text     = excluded.sql_text,
        params       = excluded.params,
        row_count    = excluded.row_count,
        duration_ms  = excluded.duration_ms,
        status       = excluded.status,
        error_code   = excluded.error_code,
        error_detail = excluded.error_detail,
        executed_at  = now()
    """
)


def record_query(
    engine_writable: Engine,
    *,
    run_id: str | uuid.UUID,
    seq: int,
    tool_name: str,
    sql_text: str,
    params: Mapping[str, Any] | None = None,
    row_count: int | None = None,
    duration_ms: int | None = None,
    status: str = "ok",
    error_code: str | None = None,
    error_detail: str | None = None,
) -> None:
    """记录一次工具调用/SQL 执行；同 `(run_id, seq)` 重复调用**更新同一行**。

    Args:
        seq: 确定性序号，建议由 `deterministic_seq()` 生成，或由调用方传入
            节点内的稳定计数。`UNIQUE(run_id, seq)` 是幂等的落点。
        status: `ok` / `error` / `timeout` / `rejected`。
        sql_text: 实际执行的 SQL 文本（已参数化形态），用于报告第 4 章追溯。

    Raises:
        ValueError: 参数类型或取值非法。
        LookupError: `run_id` 未登记时由外键约束抛出，转换为 `LookupError`，
            避免调用方把"运行记录没建"误判成数据库故障。
    """
    if status not in QUERY_STATUSES:
        raise ValueError(
            f"status 取值不合法: {status!r}；允许值: {', '.join(QUERY_STATUSES)}"
        )
    rid = as_run_id(run_id)
    seq = _validate_seq(seq)
    tool_name = _require_non_empty_str(tool_name, "tool_name")
    sql_text = _require_non_empty_str(sql_text, "sql_text")
    row_count = _validate_optional_count(row_count, "row_count")
    duration_ms = _validate_optional_count(duration_ms, "duration_ms")

    sql_params = {
        "run_id": rid,
        "seq": seq,
        "tool_name": tool_name,
        "sql_text": sql_text,
        "params": None if params is None else _json_dumps(params),
        "row_count": row_count,
        "duration_ms": duration_ms,
        "status": status,
        "error_code": error_code,
        "error_detail": error_detail,
    }
    try:
        with engine_writable.begin() as conn:
            conn.execute(_UPSERT_QUERY_SQL, sql_params)
    except Exception as exc:  # noqa: BLE001 - 仅用于识别外键冲突，其余原样抛出
        if "foreign key" in str(exc).lower() or "query_executions_run_id_fkey" in str(exc):
            raise LookupError(
                f"run_id 未登记，无法写入查询记录: {rid}（请先 create_run）"
            ) from exc
        raise


# ============================================================================
#  读
# ============================================================================
def list_queries(
    engine_readonly: Engine,
    run_id: str | uuid.UUID,
    *,
    limit: int = DEFAULT_QUERY_LIMIT,
) -> list[dict[str, Any]]:
    """按 `seq` 升序返回某次运行的查询执行记录。"""
    rid = as_run_id(run_id)
    _validate_limit(limit, MAX_LIST_LIMIT, "limit")
    stmt = text(
        f"select {', '.join(_QUERY_COLUMNS)} from query_executions "
        "where run_id = :run_id order by seq asc limit :limit"
    )
    with engine_readonly.connect() as conn:
        rows = conn.execute(stmt, {"run_id": rid, "limit": limit}).mappings().all()
    return [to_jsonable(dict(row)) for row in rows]


def get_run(engine_readonly: Engine, run_id: str | uuid.UUID) -> dict[str, Any] | None:
    """返回运行详情；不存在时返回 `None`。

    返回结构 = `analysis_runs` 的全部列 + `queries`（该运行的 `query_executions`，
    按 `seq` 升序）。`queries` 随详情一起返回，是为了让报告第 4 章的
    "数字出自哪次执行"能在一次调用里取齐。
    """
    rid = as_run_id(run_id)
    stmt = text(
        f"select {', '.join(_RUN_COLUMNS)} from analysis_runs where run_id = :run_id"
    )
    with engine_readonly.connect() as conn:
        row = conn.execute(stmt, {"run_id": rid}).mappings().first()
    if row is None:
        return None
    record = to_jsonable(dict(row))
    record["queries"] = list_queries(engine_readonly, rid)
    return record


def _validate_limit(limit: Any, maximum: int, field: str = "limit") -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError(f"{field} 必须是整数，收到: {limit!r}")
    if not 1 <= limit <= maximum:
        raise ValueError(f"{field} 必须在 1..{maximum} 内，收到: {limit!r}")
    return limit


def list_runs(
    engine_readonly: Engine,
    *,
    limit: int = DEFAULT_LIST_LIMIT,
    offset: int = 0,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """按 `started_at` 倒序列出运行记录（分页）。

    Args:
        limit: 1..`MAX_LIST_LIMIT`。
        offset: 起始偏移，>= 0。
        status: 可选状态过滤；非法取值直接 `ValueError`，不做静默忽略。

    Note:
        每条记录附带 `query_count`（该运行的查询执行条数），便于列表页展示
        "这次运行到底跑了多少查询"。
    """
    _validate_limit(limit, MAX_LIST_LIMIT)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError(f"offset 必须是 >= 0 的整数，收到: {offset!r}")
    if status is not None and status not in RUN_STATUSES:
        raise ValueError(
            f"status 取值不合法: {status!r}；允许值: {', '.join(RUN_STATUSES)}"
        )

    columns = ", ".join(f"r.{col}" for col in _RUN_COLUMNS)
    stmt = (
        f"select {columns}, "
        "(select count(*) from query_executions q where q.run_id = r.run_id) as query_count "
        "from analysis_runs r "
    )
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if status is not None:
        stmt += "where r.status = :status "
        params["status"] = status
    stmt += "order by r.started_at desc, r.run_id desc limit :limit offset :offset"

    with engine_readonly.connect() as conn:
        rows = conn.execute(text(stmt), params).mappings().all()
    return [to_jsonable(dict(row)) for row in rows]
