"""受限分析工具：实现 docs/08-工具协议.md §4 的工具清单。

设计约定（协议 §1、§4.1）：
- 每个工具第一行**校验参数**，不合法立刻返回 ``INVALID_PARAM`` 且**不碰数据库**。
- 分析查询一律走**只读**引擎；SQL 全部参数化。
- ``data`` 中的列表不超过 ``MAX_INLINE_ROWS``；超出部分截断并在 ``summary`` 说明。
- 错误分类：参数错误 / 业务条件不足 / 能力范围 / 数据库异常 / 超时 / 未知，六类可分辨。

注意：本模块只实现"读"和"算"，不写数据库；运行记录由 ``run_records`` 负责。
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError

from dsa.config import PROJECT_ROOT
from dsa.tools.schemas import (
    MAX_INLINE_ROWS,
    PERMISSION_SQLSTATES,
    TIMEOUT_SQLSTATES,
    ToolErrorCode,
    ToolResult,
    error_result,
)

from importlib.resources import files
METRICS_YAML = files("dsa.resources").joinpath("metrics.yaml")

#: 允许的实验（白名单）。未登记的 experiment_key 一律判为 CAPABILITY。
KNOWN_EXPERIMENTS: tuple[str, ...] = ("hillstrom_email_v1",)
#: 允许的数据集（白名单）。未登记的一律判为 CAPABILITY。
#: ⚠️ 新增数据集时**必须同步在 docs/metrics.yaml 的 datasets 段登记**，
#: 否则 describe_capability 会返回 key 存在但 capabilities 为空 —— 那比直接
#: 报「未登记」更糟，因为调用方会以为「这个数据集没有任何能力」而不是「没登记」。
KNOWN_DATASETS: tuple[str, ...] = ("hillstrom", "retailhero")
KNOWN_COMPARISONS: tuple[str, ...] = ("primary", "secondary_womens")


# ============================================================ 内部辅助


def _classify(exc: Exception) -> ToolErrorCode:
    """把数据库异常映射到协议的错误码。``PERMISSION_SQLSTATES`` / ``TIMEOUT_SQLSTATES``
    由 schemas 提供，保证与 docs/05 的 E_DB_001..004 分类一致。"""
    sqlstate = getattr(getattr(exc, "orig", None), "sqlstate", None)
    if sqlstate and sqlstate in TIMEOUT_SQLSTATES:
        return ToolErrorCode.TIMEOUT
    if sqlstate and sqlstate in PERMISSION_SQLSTATES:
        return ToolErrorCode.DB_ERROR
    if isinstance(exc, (OperationalError, DBAPIError, SQLAlchemyError)):
        return ToolErrorCode.DB_ERROR
    return ToolErrorCode.UNKNOWN


def _run(tool: str, engine: Any, fn: Callable[[], dict]) -> ToolResult:
    """统一执行壳：计时、异常分类、结果封装。"""
    started = time.perf_counter()
    try:
        data = fn()
    except Exception as exc:  # noqa: BLE001 —— 必须分类后返回，不向上抛
        code = _classify(exc)
        return error_result(
            tool,
            code,
            f"{type(exc).__name__}: {exc}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    rows = data.get("rows")
    truncated = False
    if isinstance(rows, list) and len(rows) > MAX_INLINE_ROWS:
        data = {**data, "rows": rows[:MAX_INLINE_ROWS]}
        truncated = True

    summary = data.pop("summary", "")
    if truncated:
        summary += f"（结果超过内联上限 {MAX_INLINE_ROWS} 行，已截断；完整结果请用引用获取）"

    return ToolResult(
        ok=True,
        tool=tool,
        data=data,
        summary=summary,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


def _load_yaml() -> dict:
    import yaml

    return yaml.safe_load(METRICS_YAML.read_text(encoding="utf-8"))


def _bad(tool: str, msg: str, detail: str = "") -> ToolResult:
    return error_result(tool, ToolErrorCode.INVALID_PARAM, msg, detail)


def _no_capability(tool: str, msg: str, detail: str = "") -> ToolResult:
    return error_result(tool, ToolErrorCode.CAPABILITY, msg, detail)


def _is_nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


# ============================================================ 工具实现


def tool_get_experiment_config(engine: Any, *, experiment_key: str) -> ToolResult:
    """读取实验配置：组别、窗口、预期比例来源、预注册比较。"""
    tool = "get_experiment_config"
    if not _is_nonempty_str(experiment_key):
        return _bad(tool, "experiment_key 必须是非空字符串", f"收到 {experiment_key!r}")
    if experiment_key not in KNOWN_EXPERIMENTS:
        return _no_capability(tool, f"未登记的实验：{experiment_key}",
                              f"已登记：{list(KNOWN_EXPERIMENTS)}")

    def _q() -> dict:
        with engine.connect() as conn:
            exp = conn.execute(
                text(
                    "select experiment_key, name, randomization_unit, analysis_unit, "
                    "       outcome_window_days, outcome_window_fixed, "
                    "       expected_ratio_source, expected_ratio_note "
                    "from experiments where experiment_key = :k"
                ),
                {"k": experiment_key},
            ).mappings().one()
            dataset = conn.execute(
                text("select d.version_label from datasets d join experiments e "
                     "on e.dataset_id=d.dataset_id where e.experiment_key=:k"),
                {"k":experiment_key}).mappings().one()
            arms = conn.execute(
                text(
                    "select ar.arm_key, ar.segment_value, ar.role, ar.expected_ratio "
                    "from experiment_arms ar join experiments e on e.experiment_id = ar.experiment_id "
                    "where e.experiment_key = :k order by ar.arm_id"
                ),
                {"k": experiment_key},
            ).mappings().all()
            cmps = conn.execute(
                text(
                    "select ec.comparison_key, ec.kind, ec.alpha, ec.sided, "
                    "       tr.arm_key as treatment_arm, ct.arm_key as control_arm "
                    "from experiment_comparisons ec "
                    "join experiments e on e.experiment_id = ec.experiment_id "
                    "join experiment_arms tr on tr.arm_id = ec.treatment_arm_id "
                    "join experiment_arms ct on ct.arm_id = ec.control_arm_id "
                    "where e.experiment_key = :k order by ec.comparison_id"
                ),
                {"k": experiment_key},
            ).mappings().all()
        return {
            "experiment": dict(exp),
            "dataset": dict(dataset),
            "arms": [dict(a) for a in arms],
            "comparisons": [dict(c) for c in cmps],
            "summary": f"实验 {experiment_key}：{len(arms)} 个组，{len(cmps)} 个预注册比较",
        }

    return _run(tool, engine, _q)


def tool_get_metric_definition(engine: Any, *, metric_id: str) -> ToolResult:
    """读取指标定义（来自 docs/metrics.yaml，文件而非数据库）。"""
    tool = "get_metric_definition"
    if not _is_nonempty_str(metric_id):
        return _bad(tool, "metric_id 必须是非空字符串", f"收到 {metric_id!r}")

    # ⚠️ 能力判定必须在 _run 之外完成：_run 会把所有异常分类为 DB/UNKNOWN，
    # 若把 KeyError 留给它处理，"指标不存在"就会被误报成未知错误。
    try:
        doc = _load_yaml()
        metrics = doc.get("metrics") or {}
    except Exception as exc:  # noqa: BLE001
        return error_result(tool, ToolErrorCode.UNKNOWN, f"读取指标字典失败: {exc}")

    if metric_id not in metrics:
        return _no_capability(
            tool, f"未登记的指标：{metric_id}",
            f"已登记指标：{sorted(metrics)}；定义见 docs/metrics.yaml",
        )

    def _q() -> dict:
        m = metrics[metric_id]
        return {
            "metric": m,
            "unsupported": (doc.get("unsupported_metrics") or []),
            "summary": f"指标 {metric_id}（{m.get('name_zh', '')}）：角色 {m.get('role')}",
        }

    return _run(tool, engine, _q)


def tool_query_arm_aggregates(engine: Any, *, experiment_key: str) -> ToolResult:
    """各组聚合：n / 购买人数 / 购买率 / 访问率 / 人均消费。

    **分母恒为 assignments 行数**（全部被随机分配到该组的客户），
    不受任何干预后条件影响（docs/02-分析协议.md §8.1）。
    """
    tool = "query_arm_aggregates"
    if not _is_nonempty_str(experiment_key):
        return _bad(tool, "experiment_key 必须是非空字符串", f"收到 {experiment_key!r}")
    if experiment_key not in KNOWN_EXPERIMENTS:
        return _no_capability(tool, f"未登记的实验：{experiment_key}", "")

    def _q() -> dict:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    select ar.arm_key, ar.role, ar.segment_value,
                           count(*)                                   as n_customers,
                           count(*) filter (where o.outcome_id is null) as n_outcome_missing,
                           count(*) filter (where o.converted)        as n_converted,
                           count(*) filter (where o.visited)          as n_visited,
                           coalesce(sum(o.spend), 0)                  as total_spend
                    from assignments a
                    join experiments e on e.experiment_id = a.experiment_id
                    join experiment_arms ar on ar.arm_id = a.arm_id
                    left join outcomes o on o.experiment_id = a.experiment_id
                                        and o.customer_id = a.customer_id
                    where e.experiment_key = :k
                    group by ar.arm_key, ar.role, ar.segment_value, ar.arm_id
                    order by ar.arm_id
                    """
                ),
                {"k": experiment_key},
            ).mappings().all()

        out = []
        for r in rows:
            n = int(r["n_customers"])
            out.append({
                **{k: r[k] for k in ("arm_key", "role", "segment_value")},
                "n_customers": n,
                "n_outcome_missing": int(r["n_outcome_missing"]),
                "n_converted": int(r["n_converted"]),
                "n_visited": int(r["n_visited"]),
                "total_spend": float(r["total_spend"]),
                "conversion_rate": (int(r["n_converted"]) / n) if n else None,
                "visit_rate": (int(r["n_visited"]) / n) if n else None,
                "spend_per_customer": (float(r["total_spend"]) / n) if n else None,
            })
        return {
            "experiment_key": experiment_key,
            "arms": out,
            "denominator_note": "分母 = 全部被随机分配到该组的客户数，不受干预后行为影响",
            "summary": f"{experiment_key}：{len(out)} 个组，" + ", ".join(
                f"{a['arm_key']} n={a['n_customers']}" for a in out),
        }

    return _run(tool, engine, _q)


def tool_run_quality_checks(
    engine: Any, *, experiment_key: str, dataset_version: str
) -> ToolResult:
    """运行数据质量检查（包装 dsa.stats.checks）。"""
    tool = "run_quality_checks"
    if not _is_nonempty_str(experiment_key):
        return _bad(tool, "experiment_key 必须是非空字符串", f"收到 {experiment_key!r}")
    if not _is_nonempty_str(dataset_version):
        return _bad(tool, "dataset_version 必须是非空字符串", f"收到 {dataset_version!r}")
    if experiment_key not in KNOWN_EXPERIMENTS:
        return _no_capability(tool, f"未登记的实验：{experiment_key}", "")

    def _q() -> dict:
        from dsa.stats.checks import run_quality_checks

        checks = run_quality_checks(engine, experiment_key=experiment_key,
                                   dataset_version=dataset_version)
        items = [{
            "code": c.code, "name": c.name, "severity": c.severity,
            "passed": bool(c.passed), "detail": c.detail, "value": c.value,
        } for c in checks]
        blocking = [c["code"] for c in items
                    if c["severity"] == "blocking" and not c["passed"]]
        return {
            "checks": items,
            "blocking_codes": blocking,
            "causal_conclusion_allowed": not blocking,
            "summary": (f"共 {len(items)} 项检查，"
                        + (f"阻断项 {blocking}" if blocking else "无阻断项")),
        }

    return _run(tool, engine, _q)


def tool_describe_capability(engine: Any, *, dataset_key: str) -> ToolResult:
    """说明该数据适配支持/不支持哪些分析。"""
    tool = "describe_capability"
    if not _is_nonempty_str(dataset_key):
        return _bad(tool, "dataset_key 必须是非空字符串", f"收到 {dataset_key!r}")
    if dataset_key not in KNOWN_DATASETS:
        return _no_capability(tool, f"未登记的数据集：{dataset_key}",
                              f"已登记：{list(KNOWN_DATASETS)}")

    def _q() -> dict:
        doc = _load_yaml()
        ds = (doc.get("datasets") or {}).get(dataset_key) or {}
        return {
            "dataset_key": dataset_key,
            "version": ds.get("version"),
            "grain": ds.get("grain"),
            "capabilities": ds.get("capabilities") or {},
            "unsupported_metrics": doc.get("unsupported_metrics") or [],
            "summary": f"数据集 {dataset_key} 的能力范围已返回（含不支持项）",
        }

    return _run(tool, engine, _q)


#: policy_kind 白名单 → 对应的排序列。**列名来自 `dsa.models.policy` 的权威定义**，
#: 不在这里另写一份。
#:
#: 缺陷回顾（外部审查 5.2）
#: ------------------------
#: > 训练脚本用 `mu0_test` 评估名为 `propensity` 的策略，但全量分数表中的
#: > `propensity_score` 保存的是合并训练的购买预测模型输出，名单工具按该字段排序。
#: > **两条排序可能不同。**
#:
#: 于是"名为 propensity 的策略"在**离线评估**与**名单查询**里代表两个不同的模型输出。
#: 修法不是"把两处都改成一样"，而是**让它们不可能不一样**：
#: 列名只在 `POLICY_DEFINITIONS` 里写一次，两处都从它派生。
#: 这里再导出一份，是为了保持工具层的白名单语义（只允许已登记的列，
#: 杜绝任意列名注入）—— 但**内容不再是本地常量**。
def _load_policy_order_columns() -> dict[str, str]:
    """从权威定义取「策略 → 排序列」。

    ⚠️ **返回的是权威定义里的那个字典本身，不是副本。**

    为什么不做拷贝（哪怕拷贝看起来更"安全"）：
    拷贝是一个**新的字面量**，它会独立演化 —— 而这正是缺陷 5.2 的形态：
    两处各自维护一份"看起来一样"的映射，直到有一天不一样，
    而且**没有任何测试会失败**，因为两边各自都自洽。
    共享同一个对象，使"两边不一致"在结构上不可能发生。
    """
    try:
        from dsa.models.policy import POLICY_ORDER_COLUMNS as _official

        return _official
    except Exception:  # noqa: BLE001 —— 导入失败即无可用策略
        # 失败时返回空字典而不是内置兜底：兜底值会和权威定义漂移。
        # 宁可让名单工具明确报"策略未登记"。
        return {}


POLICY_ORDER_COLUMNS: dict[str, str] = _load_policy_order_columns()

#: 已废弃的策略名 → 拒绝理由。**必须显式拒绝，不能静默接受。**
#:
#: 静默接受旧名正是本条缺陷的成因：调用方以为自己在要"购买概率"，
#: 实际拿到的是另一个模型的排序。所以这里不是"兼容旧名"，
#: 而是**把旧名变成一个带解释的错误**。
DEPRECATED_POLICY_KINDS: dict[str, str] = {
    "propensity": (
        "策略名 `propensity` 已废弃：它曾在「离线评估按 mu0」与"
        "「名单按 propensity_score」两处代表不同的模型输出（缺陷 5.2）。"
        "请改用 `purchase_probability`（合并购买预测分数）或 "
        "`control_probability`（无干预购买概率 mu0）—— **两者含义不同**"
    ),
}


def tool_get_model_registry(engine: Any, *, dataset_version: str | None = None) -> ToolResult:
    """读取模型登记（模型元信息：特征列、参数、划分、代码版本）。"""
    tool = "get_model_registry"
    if dataset_version is not None and not _is_nonempty_str(dataset_version):
        return _bad(tool, "dataset_version 若提供必须是非空字符串", f"收到 {dataset_version!r}")

    def _q() -> dict:
        from dataclasses import asdict, is_dataclass

        from dsa.models.registry import list_models

        models = list_models(engine, dataset_version=dataset_version)
        rows = []
        for m in models:
            # ModelRecord 是 frozen+slots 的 dataclass —— **没有 __dict__**，
            # 必须用 dataclasses.asdict（frozen/slots 下这是唯一可靠方式）。
            if is_dataclass(m) and not isinstance(m, type):
                rows.append(asdict(m))
            elif isinstance(m, dict):
                rows.append(dict(m))
            else:
                rows.append({k: getattr(m, k) for k in dir(m) if not k.startswith("_")})
        from dsa.tools.effect_tools import tool_get_scoring_population
        for row in rows:
            scoring = tool_get_scoring_population(engine, model_key=row["model_key"])
            row["scoring"] = scoring.data if scoring.ok else {}
        return {
            "models": rows,
            "summary": f"共 {len(rows)} 个已登记模型"
                       + (f"（数据版本 {dataset_version}）" if dataset_version else ""),
        }

    return _run(tool, engine, _q)


def tool_get_policy_evaluations(engine: Any, *, model_key: str) -> ToolResult:
    """读取已验证的策略评估结果（各策略各比例的增量估计与区间）。"""
    tool = "get_policy_evaluations"
    if not _is_nonempty_str(model_key):
        return _bad(tool, "model_key 必须是非空字符串", f"收到 {model_key!r}")

    def _q() -> dict:
        from dsa.models.registry import load_model_record, load_policy_evaluations

        if load_model_record(engine, model_key) is None:
            return {"__missing__": model_key}
        evals = load_policy_evaluations(engine, model_key)
        rows = [dict(e) if not isinstance(e, dict) else e for e in evals]
        return {
            "model_key": model_key,
            "evaluations": rows,
            "note": "区间由留出集 bootstrap 给出；all_treat/none 资源不同，仅作补充参照",
            "summary": f"模型 {model_key}：{len(rows)} 条策略评估结果",
        }

    result = _run(tool, engine, _q)
    if result.ok and isinstance(result.data, dict) and "__missing__" in result.data:
        return _no_capability(tool, f"未登记的模型：{model_key}", "")
    return result


def tool_get_customer_scores(
    engine: Any,
    *,
    model_key: str,
    policy_kind: str = "uplift",
    limit_n: int = 100,
    run_id: str | None = None,
) -> ToolResult:
    """按策略取客户名单前 N 名。

    **大名单不内联返回**：完整结果写入 ``artifacts/`` 并把路径放进 ``refs``，
    ``data`` 只保留前 ``MAX_INLINE_ROWS`` 行（协议 §2、§5）。
    """
    tool = "get_customer_scores"
    if not _is_nonempty_str(model_key):
        return _bad(tool, "model_key 必须是非空字符串", f"收到 {model_key!r}")
    if policy_kind in DEPRECATED_POLICY_KINDS:
        return _bad(tool, f"策略名已废弃：{policy_kind!r}",
                    DEPRECATED_POLICY_KINDS[policy_kind])
    if policy_kind not in POLICY_ORDER_COLUMNS:
        return _bad(tool, f"policy_kind 不在白名单：{policy_kind!r}",
                    f"允许值：{sorted(POLICY_ORDER_COLUMNS)}")
    if not isinstance(limit_n, int) or isinstance(limit_n, bool) or limit_n <= 0:
        return _bad(tool, "limit_n 必须是正整数", f"收到 {limit_n!r}")
    if limit_n > 200_000:
        return _bad(tool, "limit_n 超出上限 200000", f"收到 {limit_n}")

    order_col = POLICY_ORDER_COLUMNS[policy_kind]  # 来自白名单，非用户输入

    def _q() -> dict:
        from dsa.models.registry import load_model_record

        rec = load_model_record(engine, model_key)
        if rec is None:
            return {"__missing__": model_key}

        sql = text(
            f"""
            select cu.customer_key, ar.arm_key,
                   cs.mu1, cs.mu0, cs.uplift, cs.propensity_score,
                   cu.recency, cu.history_band, cu.history,
                   cu.mens, cu.womens, cu.zip_code, cu.newbie, cu.channel
            from customer_scores cs
            join models m      on m.model_id = cs.model_id
            join customers cu  on cu.customer_id = cs.customer_id
            left join assignments a on a.customer_id = cs.customer_id and a.experiment_id = m.experiment_id
            left join experiment_arms ar on ar.arm_id = a.arm_id
            where m.model_key = :mk
            order by {order_col} desc nulls last, cu.customer_key
            limit :lim
            """
        )
        with engine.connect() as conn:
            rows = [dict(r) for r in conn.execute(
                sql, {"mk": model_key, "lim": limit_n}).mappings().all()]

        refs: list[str] = []
        if len(rows) > MAX_INLINE_ROWS:
            try:
                from dsa.tools.artifacts import write_table_artifact

                ref = write_table_artifact(
                    rows, run_id=run_id or "adhoc",
                    name=f"customer_scores_{model_key}_{policy_kind}_top{limit_n}",
                    fmt="csv",
                )
                refs.append(str(ref))
            except Exception:  # noqa: BLE001 —— 写产物失败不应让查询整体失败
                refs = []

        return {
            "model_key": model_key,
            "policy_kind": policy_kind,
            "limit_n": limit_n,
            "rows": rows,
            "row_count": len(rows),
            "refs": refs,
            "note": "名单为回顾性评估输出，不构成对未来的效果承诺",
            "summary": f"{model_key} 按 {policy_kind} 排序取前 {len(rows)} 名"
                       + (f"，完整结果见 {refs[0]}" if refs else ""),
        }

    result = _run(tool, engine, _q)
    if result.ok and isinstance(result.data, dict) and "__missing__" in result.data:
        return _no_capability(tool, f"未登记的模型：{model_key}", "")
    if result.ok and isinstance(result.data, dict):
        # _run 只认识 data["rows"]，把 refs 从 data 提到顶层
        extra_refs = result.data.pop("refs", []) or []
        if extra_refs:
            data = dict(result.data)
            data.pop("refs", None)
            return ToolResult(
                ok=True, tool=tool, data=data, refs=list(extra_refs),
                summary=result.summary, duration_ms=result.duration_ms,
            )
    return result


#: 名称 → 函数
TOOL_FUNCS: dict[str, Callable[..., ToolResult]] = {
    "get_experiment_config": tool_get_experiment_config,
    "get_metric_definition": tool_get_metric_definition,
    "query_arm_aggregates": tool_query_arm_aggregates,
    "run_quality_checks": tool_run_quality_checks,
    "describe_capability": tool_describe_capability,
    "get_model_registry": tool_get_model_registry,
    "get_policy_evaluations": tool_get_policy_evaluations,
    "get_customer_scores": tool_get_customer_scores,
}
