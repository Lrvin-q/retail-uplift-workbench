"""实验效果的结构化载荷（阶段三 → 阶段六/七的桥）。

**这个模块存在的唯一理由**：把「已经验证过的统计计算」（`dsa.stats.experiment`）
变成**一个可以被工具、API、工作台和报告共同消费的结构**。

为什么需要它（对应一次真实缺陷）
--------------------------------
项目早期，统计计算（`analyze_experiment`）只在命令行脚本里被调用过。
Agent 的报告节点拿不到它，于是自己从 `query_arm_aggregates` 的描述性聚合里
拼了一份「三组购买率」的表格 —— 用户问「男装邮件有没有提高购买率」，
得到的报告里**没有效果差、没有区间、没有检验**。

问题不在报告写得不好，而在**结构上就没有一条路把统计结果送到报告**。
所以修法不是"在报告里多写几行"，而是：

    analyze_experiment  →  effect_payload  →  {工具 / API / 工作台 / 报告}

**同一份计算产物**。报告层不允许再自己算任何效果量。

口径不可动摇（AGENTS.md §2.2）
-----------------------------
载荷里的每一个 `n` 都来自 `assignments` 的行数，即**全部被随机分配到该组的客户**。
本模块不做任何筛选，也不可能引入干预后字段作为分母。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Sequence

#: 允许做**因果比较**的指标。顺序即报告的默认顺序。
#:
#: ⚠️ `spend_per_buyer` **不在其中**：它的分母是「购买者」，
#: 而"是否购买"是干预后行为。把它当因果指标会直接违反 AGENTS.md §2.2。
#: 它只登记为 `descriptive_only`（docs/metrics.yaml）。
CAUSAL_METRICS: tuple[str, ...] = (
    "conversion_rate",
    "visit_rate",
    "spend_per_customer",
)

#: 被**显式拒绝**为因果指标的登记指标 → 拒绝理由。
REFUSED_AS_CAUSAL: dict[str, str] = {
    "spend_per_buyer": (
        "该指标的分母是「购买者」，而是否购买属于**干预后**行为；"
        "按它计算组间差异等于按干预后行为筛选人群，违反分母口径红线"
        "（AGENTS.md §2.2）。它只作为描述性指标提供。"
    ),
}

#: 指标中文名兜底（读不到 docs/metrics.yaml 时使用，保证报告不出现裸 id）
_FALLBACK_NAMES: dict[str, str] = {
    "conversion_rate": "购买率",
    "visit_rate": "访问率",
    "spend_per_customer": "人均消费",
}

#: 指标单位。比例型按「百分点」表述，金额型按货币表述。
#: **单位必须随数值一起传递**，否则报告很容易把"个百分点"写成"%"。
_FALLBACK_UNITS: dict[str, str] = {
    "conversion_rate": "proportion",
    "visit_rate": "proportion",
    "spend_per_customer": "currency",
}


@lru_cache(maxsize=1)
def metric_meta() -> dict[str, dict[str, Any]]:
    """从 `docs/metrics.yaml` 读指标元信息（名称、角色、单位）。

    读不到时退回内置兜底值：**报告里绝不出现裸 metric id**。
    """
    meta: dict[str, dict[str, Any]] = {}
    try:
        from dsa.tools.analysis_tools import _load_yaml  # 复用同一份 YAML 读取

        doc = _load_yaml()
        for key, val in (doc.get("metrics") or {}).items():
            role = (val or {}).get("role")
            meta[key] = {
                "name_zh": (val or {}).get("name_zh") or _FALLBACK_NAMES.get(key, key),
                "role": role,
                # `unit` 在 YAML 里是 "customer" / "currency"，不是数值单位；
                # 这里按指标类型判定展示单位，避免把 "customer" 当成单位印出来。
                "unit": _unit_for(key, role),
                "denominator": (val or {}).get("denominator", ""),
                "constraints": (val or {}).get("constraints") or [],
            }
    except Exception:  # noqa: BLE001 —— 元信息缺失不能让分析失败
        meta = {}

    for key in CAUSAL_METRICS:
        meta.setdefault(key, {
            "name_zh": _FALLBACK_NAMES.get(key, key),
            "role": "unknown",
            "unit": _FALLBACK_UNITS.get(key, "unknown"),
            "denominator": "",
            "constraints": [],
        })
    return meta


def _unit_for(metric: str, role: Any) -> str:
    if metric in _FALLBACK_UNITS:
        return _FALLBACK_UNITS[metric]
    return "currency" if "spend" in metric else "proportion"


def metric_name(metric: str) -> str:
    return metric_meta().get(metric, {}).get("name_zh") or metric


def metric_unit(metric: str) -> str:
    return metric_meta().get(metric, {}).get("unit") or _FALLBACK_UNITS.get(metric, "unknown")


def causal_metrics() -> tuple[str, ...]:
    """可以做因果比较的指标（登记在 metrics.yaml 且分母不受干预影响）。"""
    return CAUSAL_METRICS


def refuse_reason(metric: str) -> str | None:
    """若该指标不能做因果比较，返回**理由**；可以做则返回 None。"""
    if metric in REFUSED_AS_CAUSAL:
        return REFUSED_AS_CAUSAL[metric]
    if metric not in CAUSAL_METRICS:
        return (
            f"未登记为可做因果比较的指标：{metric}；"
            f"可用指标：{list(CAUSAL_METRICS)}"
        )
    return None


# ============================================================================
#  载荷
# ============================================================================


def _interval(ci: Any) -> dict[str, Any]:
    """把 `Interval` 拍平。`defined=False` 必须**保留**为缺失，不能填 0。"""
    return {
        "low": None if not getattr(ci, "defined", False) else float(ci.low),
        "high": None if not getattr(ci, "defined", False) else float(ci.high),
        "level": float(getattr(ci, "level", float("nan"))),
        "method": str(getattr(ci, "method", "")),
        "defined": bool(getattr(ci, "defined", False)),
    }


def _number(value: Any) -> float | None:
    """float；NaN / inf → None（JSON 无此二者，"未定义"就是缺失）。"""
    import math

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def proportion_effect(stats: Any) -> dict[str, Any]:
    """一个比例型指标的对比 → 结构化效果。

    `difference` 采用**原尺度**（如 0.00680501）；
    `difference_pp` 是同一数值的百分点表示（0.680501）。
    两者都给出，是为了让报告**不必自己做单位换算** ——
    换算一旦散落在报告里，就迟早会出现「把 0.68 写成 68%」这类错误。
    """
    diff = _number(stats.risk_difference)
    return {
        "metric": stats.metric,
        "metric_name_zh": metric_name(stats.metric),
        "kind": "proportion",
        "unit": "proportion",
        "treatment_n": int(stats.treatment_n),
        "treatment_k": int(stats.treatment_k),
        "control_n": int(stats.control_n),
        "control_k": int(stats.control_k),
        "treatment_value": _number(stats.treatment_rate),
        "control_value": _number(stats.control_rate),
        "difference": diff,
        "difference_pp": None if diff is None else diff * 100.0,
        "ci_low": _interval(stats.risk_difference_ci)["low"],
        "ci_high": _interval(stats.risk_difference_ci)["high"],
        "ci_low_pp": None if _interval(stats.risk_difference_ci)["low"] is None
        else _interval(stats.risk_difference_ci)["low"] * 100.0,
        "ci_high_pp": None if _interval(stats.risk_difference_ci)["high"] is None
        else _interval(stats.risk_difference_ci)["high"] * 100.0,
        "ci_level": _interval(stats.risk_difference_ci)["level"],
        "ci_method": _interval(stats.risk_difference_ci)["method"],
        "ci_defined": _interval(stats.risk_difference_ci)["defined"],
        "ratio": _number(stats.risk_ratio),
        "ratio_ci_low": _interval(stats.risk_ratio_ci)["low"],
        "ratio_ci_high": _interval(stats.risk_ratio_ci)["high"],
        "ratio_method": _interval(stats.risk_ratio_ci)["method"],
        "ratio_defined": _interval(stats.risk_ratio_ci)["defined"],
        "relative_lift": _number(stats.relative_lift),
        "p_value": _number(stats.test.p_value),
        "test_method": str(stats.test.method),
        "min_expected": _number(getattr(stats.test, "min_expected", None)),
        "used_fisher": bool(getattr(stats.test, "used_fisher", False)),
        "alpha": float(stats.alpha),
        "significant": bool(stats.significant),
    }


def continuous_effect(stats: Any) -> dict[str, Any]:
    """一个连续型指标的对比 → 结构化效果。"""
    return {
        "metric": stats.metric,
        "metric_name_zh": metric_name(stats.metric),
        "kind": "continuous",
        "unit": "currency",
        "treatment_n": int(stats.treatment_n),
        "control_n": int(stats.control_n),
        "treatment_value": _number(stats.treatment_mean),
        "control_value": _number(stats.control_mean),
        "difference": _number(stats.difference),
        "ci_low": _number(stats.ci_low),
        "ci_high": _number(stats.ci_high),
        "ci_level": 1.0 - float(stats.alpha),
        "ci_method": "bootstrap_percentile",
        "ci_defined": _number(stats.ci_low) is not None,
        "p_value": _number(stats.p_value),
        "test_method": str(stats.method),
        "treatment_zero_share": _number(stats.treatment_zero_share),
        "control_zero_share": _number(stats.control_zero_share),
        "bootstrap_seed": int(stats.bootstrap_seed),
        "alpha": float(stats.alpha),
        "significant": bool(stats.significant),
    }


def effect_payload(analysis: Any, *, metrics: Sequence[str] | None = None) -> dict[str, Any]:
    """`ExperimentAnalysis` → 可 JSON 序列化的效果载荷。

    Args:
        analysis: `dsa.stats.experiment.analyze_experiment` 的返回值。
        metrics: 只要这些指标；None 表示全部（按 :data:`CAUSAL_METRICS` 顺序）。
            传入未登记/不可做因果的指标会被**忽略**，其拒绝理由见
            :func:`refuse_reason`（调用方负责把理由告诉用户）。

    Returns:
        含 ``effects`` 列表的结构化载荷。数值全部来自 `analysis`，
        **本函数不计算任何新的统计量**。
    """
    wanted = list(metrics) if metrics else list(CAUSAL_METRICS)
    effects: list[dict[str, Any]] = []
    missing: list[str] = []

    for metric in wanted:
        if metric in analysis.metric_comparisons:
            effects.append(proportion_effect(analysis.metric_comparisons[metric]))
        elif metric in analysis.continuous_comparisons:
            effects.append(continuous_effect(analysis.continuous_comparisons[metric]))
        else:
            missing.append(metric)

    return {
        "analysis_run_id": analysis.run_id,
        "experiment_key": analysis.experiment_key,
        "comparison_key": analysis.comparison_key,
        "comparison_kind": analysis.comparison_kind,
        "treatment_arm": analysis.treatment_arm,
        "control_arm": analysis.control_arm,
        "dataset_version": analysis.dataset_version,
        "window_days": int(analysis.window_days),
        "alpha": float(analysis.alpha),
        "blocked": bool(analysis.blocked),
        "blocking_codes": list(analysis.blocking_codes),
        "causal_conclusion_allowed": bool(analysis.causal_conclusion_allowed),
        "stats_config": dict(analysis.stats_config),
        "code_version": analysis.code_version,
        "generated_at": analysis.generated_at,
        "requested_metrics": list(wanted),
        "effects": effects,
        "metrics_missing": missing,
        "denominator_note": (
            "每个 n 均为**全部被随机分配到该组的客户数**（assignments 行数），"
            "不受访问/购买/消费等干预后行为影响"
        ),
    }


def select_effects(payload: dict[str, Any], metrics: Iterable[str]) -> list[dict[str, Any]]:
    """从载荷里挑出指定指标的效果，**保持 `metrics` 的顺序**。"""
    index = {e["metric"]: e for e in (payload.get("effects") or [])}
    return [index[m] for m in metrics if m in index]


__all__ = [
    "CAUSAL_METRICS",
    "REFUSED_AS_CAUSAL",
    "causal_metrics",
    "continuous_effect",
    "effect_payload",
    "metric_meta",
    "metric_name",
    "metric_unit",
    "proportion_effect",
    "refuse_reason",
    "select_effects",
]
