"""运行导出、产物引用与运行重放（**包内实现**）。

依据《实现蓝图.md》阶段九：「实现报告导出、产物引用与运行重放。」

三个能力：

| 函数 | 作用 |
|---|---|
| :func:`cmd_export` | 把该运行的全部产物 + 清单打包成 zip |
| :func:`cmd_list` | 列出可重放的运行 |
| :func:`cmd_replay` | 用**登记的配置**重跑，并逐字段比对结果 |

**为什么要「重放」而不只是「导出」**：
导出证明「产物还在」，重放证明「产物**可被重新产生**」。
后者才是可复现性 —— 一个只导出不重放的系统，
无法区分「结果稳定」与「结果只出现过一次」。

⚠️ **这份实现为什么在 `src/dsa/` 而不在 `scripts/`**（外部审查 F11 的连带发现）：

它原先只存在于 `scripts/export_run.py`，而工作台为了提供导出/重放按钮，
写了 `from scripts.export_run import cmd_export` ——
**包内的代码去 import 包外的脚本目录**。这要求"项目根目录在 `sys.path` 上"，
而这个前提在安装成包之后并不成立：
`pip install .` 装的是 `dsa`，不会把 `scripts/` 一起装进去，
于是工作台会在导入期失败。

正确的分层是：**能力在包里，入口在脚本里**。
`scripts/export_run.py` 现在只剩一个薄的命令行包装。

用法：
    python scripts/export_run.py list
    python scripts/export_run.py export <run_id>
    python scripts/export_run.py replay <run_id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

from dsa.config import ARTIFACTS_DIR, get_engine


def _run_row(engine, run_id: str) -> dict | None:
    with engine.connect() as c:
        row = c.execute(text("""
            select r.run_id, r.comparison_key, r.status, r.code_version,
                   r.stats_config, r.report_ref, r.result_refs, r.plan, r.question, r.parent_run_id,
                   r.started_at, r.finished_at, e.experiment_key
            from analysis_runs r
            left join experiments e on e.experiment_id = r.experiment_id
            where r.run_id = :rid
        """), {"rid": run_id}).mappings().first()
    return dict(row) if row else None


def _queries(engine, run_id: str) -> list[dict]:
    with engine.connect() as c:
        rows = c.execute(text("""
            select seq, tool_name, sql_text, params, row_count, duration_ms, status, error_code
            from query_executions where run_id = :rid order by seq
        """), {"rid": run_id}).mappings().all()
    return [dict(r) for r in rows]


def cmd_list() -> int:
    engine = get_engine(readonly=True)
    with engine.connect() as c:
        rows = c.execute(text("""
            select run_id, comparison_key, status, code_version, finished_at
            from analysis_runs where finished_at is not null
            order by finished_at desc limit 20
        """)).mappings().all()
    if not rows:
        print("没有已完成的运行记录。")
        return 0
    print(f"{'run_id':38} {'比较':10} {'状态':12} {'代码版本':10} 完成时间")
    for r in rows:
        # ⚠️ run_id 列是 UUID 类型，psycopg 返回 uuid.UUID 对象。
        # 直接写 `f"{x:38}"` 会抛 TypeError: unsupported format string
        # passed to UUID.__format__ —— 必须先 str()。
        print(f"{str(r['run_id']):38} {str(r['comparison_key'] or ''):10} "
              f"{str(r['status'] or ''):12} {str(r['code_version'] or '')[:8]:10} "
              f"{r['finished_at']}")
    return 0


def _resolve_ref(ref: str) -> Path:
    """把产物引用解析成绝对路径。

    ⚠️ **引用有两种约定，必须都支持**：数据库里的 ``report_ref``
    实测存的是 ``artifacts/test_report.md``（**带** ``artifacts/`` 前缀），
    而 ``result_refs`` 与工具层返回的 ``refs`` 是**相对** ``artifacts/`` 的
    （见 docs/08-工具协议.md §5）。

    早期版本一律按后者拼接，于是 ``artifacts/artifacts/xxx`` 永远不存在 ——
    **导出会报"产物缺失"，而文件其实好好的**。
    这是一个"报警方向正确、原因错误"的缺陷：它确实报了缺失，
    但缺失的原因是路径拼错，不是文件真的没了。
    """
    p = Path(ref)
    if p.is_absolute():
        return p
    # 去掉可能存在的 artifacts/ 前缀（两种写法都接受）
    parts = p.parts
    if parts and parts[0] == ARTIFACTS_DIR.name:
        p = Path(*parts[1:])
    return ARTIFACTS_DIR / p


#: 对外名称：`resolve_artifact_ref`（包外调用方用这个名字，
#: 不带下划线 —— 它是 `__all__` 的一部分）。
resolve_artifact_ref = _resolve_ref


def cmd_export(run_id: str) -> int:
    engine = get_engine(readonly=True)
    rec = _run_row(engine, run_id)
    if rec is None:
        print(f"未找到运行 {run_id}")
        return 1

    qs = _queries(engine, run_id)
    if not rec.get("report_ref") or not rec.get("plan") or not rec.get("question"):
        print("导出不完整：缺少最终报告、问题或分析计划。旧运行不能当作完整Agent归档。")
        return 2
    refs: list[str] = []
    for key in ("report_ref", "result_refs"):
        v = rec.get(key)
        if isinstance(v, str):
            refs.append(v)
        elif isinstance(v, list):
            refs.extend(x for x in v if isinstance(x, str))

    state_ref=next((x for x in refs if x.endswith("/state.json")),None)
    if state_ref is None or not _resolve_ref(state_ref).is_file():
        print("导出不完整：缺少完整终态证据。")
        return 2
    saved_state=json.loads(_resolve_ref(state_ref).read_text(encoding="utf-8"))
    hashes={}
    for ref in refs:
        path=_resolve_ref(ref)
        if path.is_file() and ARTIFACTS_DIR.resolve() in path.resolve().parents:
            hashes[ref]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version":"v3",
        "run_id": run_id,
        "parent_run_id":rec.get("parent_run_id"),
        "question":rec.get("question"),
        "original_question":saved_state.get("original_question") or rec.get("question"),
        "answers":saved_state.get("answers") or [],
        "plan":rec.get("plan"),
        "report_generation":saved_state.get("explanation_guard"),
        "artifact_hashes":hashes,
        "experiment_key": rec.get("experiment_key"),
        "comparison_key": rec.get("comparison_key"),
        "status": rec.get("status"),
        "code_version": rec.get("code_version"),
        "stats_config": rec.get("stats_config"),
        "started_at": str(rec.get("started_at")),
        "finished_at": str(rec.get("finished_at")),
        "queries": qs,
        "artifact_refs": refs,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "note": ("清单记录该运行的全部可追溯信息：模型/数据/代码版本、"
                 "工具参数与耗时、产物引用。"),
    }

    out = ARTIFACTS_DIR / f"export_{run_id[:8]}.zip"
    included, missing = [], []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False,
                                               indent=1, default=str))
        for ref in refs:
            p = _resolve_ref(ref)
            if p.exists() and p.is_file():
                # ⚠️ 只打包解析后仍在 ARTIFACTS_DIR 内的文件（防目录穿越）
                if ARTIFACTS_DIR.resolve() in p.resolve().parents:
                    z.write(p, arcname=ref)
                    included.append(ref)
                else:
                    missing.append(f"{ref}（越界，已拒绝）")
            else:
                missing.append(ref)

    print(f"已导出 {out}")
    print(f"  清单: manifest.json（{len(qs)} 条查询记录）")
    print(f"  产物: {len(included)} 个已打包" + (f"，{len(missing)} 个缺失" if missing else ""))
    for m in missing:
        print(f"    ⚠️ {m}")
    if missing:
        # 缺失必须显式暴露：静默跳过会让导出包看起来"完整"
        print("  **导出不完整**：上列产物未找到，不要当作完整归档使用。")
        return 2
    return 0


#: 重放时逐字段比对的数值字段。**必须逐个列出**，不能整包 JSON 比较 ——
#: 整包比较会因为时间戳、内部 id 等无关字段而永远"不一致"，
#: 于是"重放比对"退化成"永远报差异"，等于没比。
_REPLAY_NUMERIC_FIELDS: tuple[str, ...] = (
    "treatment_n", "control_n", "treatment_k", "control_k",
    "treatment_value", "control_value", "difference",
    "ci_low", "ci_high", "p_value", "ratio", "relative_lift",
)


def _effect_artifact_ref(rec: dict) -> str | None:
    """从运行登记的产物引用里找出效果载荷 JSON。"""
    refs = rec.get("result_refs")
    if isinstance(refs, str):
        try:
            refs = json.loads(refs)
        except Exception:  # noqa: BLE001
            refs = []
    for ref in refs or []:
        if isinstance(ref, str) and ref.startswith("effect_") or (
            isinstance(ref, str) and "/effect_" in ref
        ):
            return ref
    return None


def _compare_payloads(old: dict, new: dict) -> dict:
    """逐字段比对两次效果载荷，返回结构化差异。

    按 ``metric`` 对齐；某个指标只在一边出现也算差异 ——
    "少算了一个指标"和"数值变了"同样是重放不一致。
    """
    old_by = {e.get("metric"): e for e in old.get("effects") or []}
    new_by = {e.get("metric"): e for e in new.get("effects") or []}
    diffs: list[dict] = []

    for metric in sorted(set(old_by) | set(new_by)):
        a, b = old_by.get(metric), new_by.get(metric)
        if a is None or b is None:
            diffs.append({
                "metric": metric,
                "field": "(整体)",
                "recorded": "存在" if a else "缺失",
                "replayed": "存在" if b else "缺失",
            })
            continue
        for field_name in _REPLAY_NUMERIC_FIELDS:
            av, bv = a.get(field_name), b.get(field_name)
            if av is None and bv is None:
                continue
            if av is None or bv is None:
                diffs.append({"metric": metric, "field": field_name,
                              "recorded": av, "replayed": bv})
                continue
            # 逐位相等才算一致：bootstrap 用固定种子，统计函数是纯函数，
            # 因此重放**应当**给出完全相同的浮点数。容差只会掩盖非确定性。
            if float(av) != float(bv):
                diffs.append({"metric": metric, "field": field_name,
                              "recorded": float(av), "replayed": float(bv)})

    meta_fields = ("treatment_arm", "control_arm", "dataset_version",
                   "window_days", "alpha", "blocked")
    for field_name in meta_fields:
        if old.get(field_name) != new.get(field_name):
            diffs.append({"metric": "(元信息)", "field": field_name,
                          "recorded": old.get(field_name),
                          "replayed": new.get(field_name)})
    return {"n_differences": len(diffs), "differences": diffs}


def cmd_replay(run_id: str) -> int:
    """用登记的配置重跑，并**逐字段比对**关键结果。

    ⚠️ 本项目的重放是**幂等重跑**而非事件回放：
    它验证「同样的配置能否再产生同样的结果」，而不是「逐字节复现历史」。
    后者需要记录每次工具调用的完整返回，本项目只记录效果载荷。

    缺陷回顾（外部审查 F06）
    ------------------------
    > `cmd_replay()` 对阶段四仅打印手动重跑建议，对其他类型打印不支持，
    > 然后返回 0；**没有实际重跑和结果比对**。
    > 它目前不能支持"已实现运行重放"的表述。

    返回 0 的"什么都不做"是最糟的一种：调用方看到成功退出码，
    会以为重放通过了。

    现在的行为：

    | 情形 | 行为 | 退出码 |
    |---|---|---|
    | 有登记配置且能读到历史效果产物 | 真重跑 + 逐字段比对 | 0 一致 / 3 有差异 |
    | 没有历史效果产物（旧运行） | **明确说明未执行**，不回退到"打印建议" | 2 |
    | 运行不存在 | 明确报错 | 1 |
    """
    engine = get_engine(readonly=True)
    rec = _run_row(engine, run_id)
    if rec is None:
        print(f"未找到运行 {run_id}")
        return 1

    cfg = rec.get("stats_config") or {}
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    plan = rec.get("plan") or {}
    if isinstance(plan, str):
        plan = json.loads(plan)

    comparison = str(rec.get("comparison_key") or plan.get("comparison_key") or "primary")
    experiment = str(rec.get("experiment_key") or plan.get("experiment_key")
                     or "hillstrom_email_v1")
    print(f"重放 {run_id}")
    print(f"  实验/比较: {experiment} / {comparison}")

    ref = _effect_artifact_ref(rec)
    if ref is None:
        print("  ⛔ **未执行重放**：该运行没有登记可比对的效果产物。")
        print("     原因：效果载荷是修复审查 F06 之后才开始随运行落盘的；")
        print("     更早的运行只登记了参数，没有可读的历史结果，")
        print("     因此**无法**判断重跑结果是否一致。")
        print("     请在当前版本上重新跑一次分析，再对该运行重放。")
        return 2

    old_path = _resolve_ref(ref)
    if not old_path.exists():
        print(f"  ⛔ **未执行重放**：登记的效果产物不存在：{old_path}")
        return 2
    old = json.loads(old_path.read_text(encoding="utf-8"))

    # ---- 真重跑：**用登记的统计参数**，不是用当前默认值 ----
    historical={**(old.get("stats_config") or {}),**cfg}
    if not historical.get("n_boot") or historical.get("bootstrap_seed") is None or not old.get("dataset_version"):
        print("未执行效果重算：历史统计参数或数据版本不完整")
        return 2
    n_boot=int(historical["n_boot"])
    seed=int(historical["bootstrap_seed"])
    print(f"  重跑参数: comparison={comparison} n_boot={n_boot} seed={seed}")

    from dsa.tools.registry import call_tool

    from dsa.tools.run_records import create_run, update_run_status
    from dsa.agent.nodes import call_tool_captured, _record_query, node_report
    from dsa.agent.graph import _finish_run
    from dsa.agent.state import initial_state
    from dsa.tools.run_records import to_jsonable
    replay_run_id = create_run(get_engine(readonly=False),experiment_key=experiment,
                               comparison_key=comparison,question=rec.get("question") or "效果重算",
                               plan=plan,stats_config=cfg,parent_run_id=run_id)
    update_run_status(get_engine(readonly=False),replay_run_id,status="running")
    # 数据版本必须仍与历史载荷一致，不把新数据结果冒充历史重现。
    config_result=call_tool("get_experiment_config",engine,experiment_key=experiment)
    current_dataset=((config_result.data or {}).get("dataset") or {}).get("version_label") if config_result.ok else None
    if not current_dataset or current_dataset != old.get("dataset_version"):
        update_run_status(get_engine(readonly=False),replay_run_id,status="failed",
                          error_code="E_CAP_001",error_detail="数据版本无法匹配；未执行效果重算")
        print(f"未执行效果重算：数据版本无法匹配，运行 {replay_run_id}")
        return 2
    kwargs={"experiment_key":experiment,"comparison_key":comparison,
            "metrics":old.get("requested_metrics") or None,"n_boot":n_boot,"bootstrap_seed":seed}
    result, cap = call_tool_captured("analyze_experiment_effect", engine,
                       run_id=replay_run_id, **kwargs)
    _record_query(replay_run_id,"analyze_experiment_effect",kwargs,result,cap)
    child_state=initial_state(rec.get("question") or "效果重算",run_id=replay_run_id,
                              thread_id=str(uuid.uuid4()))
    child_state.update(category="executable",plan=plan,tool_results=[result.to_dict()],
                       last_execute_ok=result.ok,blocked=bool((result.data or {}).get("blocked")),
                       explanation_guard={"mode":"deterministic","free_explanation_enabled":False})
    if result.ok:
        child_state.update(node_report(child_state))
    else:
        child_state.update(report="效果重算失败："+result.error.message,
                           errors=[{"node":"execute","code":result.error.code.value,"message":result.error.message}])
    _finish_run(replay_run_id,child_state)
    if not result.ok:
        print(f"效果重算失败：{replay_run_id}")
        return 1
    new = result.data

    # ---- 结构化比对 ----
    diff = _compare_payloads(old, new)
    consistent = diff["n_differences"] == 0

    report = {
        "scope":"effect_recompute",
        "replayed_from": run_id,
        "replay_run_id": replay_run_id,
        "experiment_key": experiment,
        "comparison_key": comparison,
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "recorded_code_version": rec.get("code_version"),
        "current_code_version": _current_code_version(),
        "consistent": consistent,
        "diff": diff,
        "replayed_at": datetime.now(timezone.utc).isoformat(),
        "note": ("重放 = 用登记配置幂等重跑 + 逐字段比对，不是事件回放。"
                 "bootstrap 使用固定种子，统计函数是纯函数，"
                 "因此一致时应逐位相等。"),
    }
    out = ARTIFACTS_DIR / f"replay_{run_id[:8]}_{replay_run_id[:8]}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")

    print(f"  新执行的证据: {out}")
    print(f"  重放运行标识: {replay_run_id}（其查询已记入 query_executions）")
    if consistent:
        print("  ✅ **重放一致**：全部字段逐位相同。")
        return 0
    print(f"  ❌ **重放不一致**：{diff['n_differences']} 处差异")
    for d in diff["differences"][:20]:
        print(f"     {d['metric']}.{d['field']}: 记录={d['recorded']} 重放={d['replayed']}")
    return 3


def _current_code_version() -> str:
    from dsa.version import code_version

    return code_version()


__all__ = [
    "cmd_export",
    "cmd_list",
    "cmd_replay",
    "resolve_artifact_ref",
]
