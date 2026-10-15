"""实验分析的整合层。

职责：把「数据库里的数据」变成「结构化统计结果」。
这是**唯一**允许把 SQL 输出与统计函数接起来的地方——
统计函数本身（`proportions`）不认识数据库，数据库查询也不认识统计。

依据：docs/02-分析协议.md；实现蓝图.md 阶段三。

关键设计
--------
- **分母不可动摇**：所有指标的 n 都来自 `assignments` 的行数（= 全部被随机分配的客户），
  由 SQL 保证，Python 侧不做任何筛选。
- **结果缺失单独计数**，绝不并入未购买。
- 运行配置（统计方法、α、数据版本、代码版本）随结果一起返回，写进 `analysis_runs`，
  这是复现的凭据。
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import Engine, text

from dsa.config import PROJECT_ROOT
from dsa.stats.proportions import ComparisonStats, analyze_comparison

#: 每个实验的指标配置：SQL 列名 → (中文名, 是否比例型)
#: `metric` 标识必须与 docs/03-指标字典.md 一致
PROPORTION_METRICS: dict[str, tuple[str, str]] = {
    "conversion_rate": ("购买率", "converted"),
    "visit_rate": ("访问率", "visited"),
}

#: 连续型指标：SQL 列名 → (中文名, 金额列)
CONTINUOUS_METRICS: dict[str, tuple[str, str]] = {
    "spend_per_customer": ("人均消费", "spend"),
}


# ============================================================================
#  结构化结果
# ============================================================================


@dataclass(frozen=True, slots=True)
class ContinuousStats:
    """连续型指标的两组比较（Welch t 检验 + bootstrap 区间）。"""

    metric: str
    treatment_n: int
    control_n: int
    treatment_mean: float
    control_mean: float
    difference: float
    ci_low: float
    ci_high: float
    alpha: float
    method: str
    p_value: float
    bootstrap_seed: int
    treatment_zero_share: float
    control_zero_share: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def summary(self) -> str:
        return (
            f"指标            : {self.metric}\n"
            f"干预组 均值     : {self.treatment_mean:.4f}  (n={self.treatment_n:,}, 零值占比 {self.treatment_zero_share:.4%})\n"
            f"对照组 均值     : {self.control_mean:.4f}  (n={self.control_n:,}, 零值占比 {self.control_zero_share:.4%})\n"
            f"绝对差异        : {self.difference:+.4f}   {self.alpha:.0%} bootstrap 区间 [{self.ci_low:+.4f}, {self.ci_high:+.4f}]\n"
            f"检验            : {self.method}  p = {self.p_value:.6g}  "
            f"（bootstrap 种子 {self.bootstrap_seed}）"
        )


@dataclass
class ExperimentAnalysis:
    """一次实验分析的完整结构化结果。**报告只能从这个对象取数。**"""

    run_id: str
    experiment_key: str
    dataset_version: str
    comparison_key: str
    comparison_kind: str
    alpha: float
    window_days: int
    metric_comparisons: dict[str, ComparisonStats] = field(default_factory=dict)
    continuous_comparisons: dict[str, ContinuousStats] = field(default_factory=dict)
    checks: list[Any] = field(default_factory=list)
    blocked: bool = False
    blocking_codes: list[str] = field(default_factory=list)
    stats_config: dict[str, Any] = field(default_factory=dict)
    code_version: str = ""
    generated_at: str = ""

    #: 干预组 / 对照组的 `arm_key`。
    #:
    #: **为什么必须随结果一起返回**：报告要写「男装邮件 vs 无邮件」，
    #: 而 `comparison_key`（`primary`）只是登记表里的标识，不是给人看的组名。
    #: 早期版本只返回 `comparison_key`，导致消费/访问率这类问题在报告里
    #: 无法说清「比的到底是哪两组」—— 而"比的谁"正是结论的一部分。
    treatment_arm: str = ""
    control_arm: str = ""

    @property
    def causal_conclusion_allowed(self) -> bool:
        """存在阻断级异常时，禁止输出因果结论。"""
        return not self.blocked

    def summary(self) -> str:
        parts = [
            f"run_id          : {self.run_id}",
            f"实验            : {self.experiment_key} / 比较 {self.comparison_key} ({self.comparison_kind})",
            f"比较对象        : {self.treatment_arm or '?'} (干预) vs {self.control_arm or '?'} (对照)",
            f"数据版本        : {self.dataset_version}",
            f"窗口            : 实验后 {self.window_days} 天",
            f"α               : {self.alpha}",
            f"阻断            : {'是 ' + ','.join(self.blocking_codes) if self.blocked else '否'}",
            "",
        ]
        for m in self.metric_comparisons.values():
            parts.append(m.summary())
            parts.append("")
        for c in self.continuous_comparisons.values():
            parts.append(c.summary())
            parts.append("")
        return "\n".join(parts)


# ============================================================================
#  辅助
# ============================================================================


def code_version() -> str:
    """当前代码版本（git 短哈希，工作区脏时带 `-dirty`）。

    实现已收敛到 :func:`dsa.version.code_version`：
    本模块与 `dsa.tools.run_records` 原本各有一份、口径不同，
    于是"这次用的哪个版本"取决于谁写的记录，而不是事实。

    取不到时返回 `unknown`，不抛异常。
    """
    from dsa.version import code_version as _impl

    return _impl()


# ============================================================================
#  取数
# ============================================================================


_ARM_SQL = """
select
    ar.arm_key,
    ar.role,
    count(*)                                        as n_total,
    count(*) filter (where o.outcome_id is null)    as n_missing,
    count(*) filter (where o.converted)             as k_converted,
    count(*) filter (where o.visited)               as k_visited,
    coalesce(sum(o.spend), 0)                       as total_spend,
    coalesce(array_agg(o.spend order by cu.source_row_number)
             filter (where o.spend is not null), '{}') as spend_values
from assignments a
join experiments     e  on e.experiment_id = a.experiment_id
join experiment_arms ar on ar.arm_id       = a.arm_id
join customers       cu on cu.customer_id  = a.customer_id
left join outcomes   o  on o.experiment_id = a.experiment_id
                       and o.customer_id   = a.customer_id
where e.experiment_key = :experiment_key
group by ar.arm_key, ar.role, ar.arm_id
order by ar.arm_id
"""

_COMPARISON_SQL = """
select ec.comparison_key, ec.kind, ec.alpha, ec.sided,
       tr.arm_key as treatment_arm, ct.arm_key as control_arm,
       e.outcome_window_days, d.version_label
from experiment_comparisons ec
join experiment_arms tr on tr.arm_id = ec.treatment_arm_id
join experiment_arms ct on ct.arm_id = ec.control_arm_id
join experiments e on e.experiment_id = ec.experiment_id
join datasets d on d.dataset_id = e.dataset_id
where e.experiment_key = :experiment_key and ec.comparison_key = :comparison_key
"""


def fetch_arm_data(engine: Engine, experiment_key: str) -> dict[str, dict[str, Any]]:
    """取出每组的分母、分子与消费明细。

    **分母 `n_total` 是 assignments 的行数**，即全部被随机分配到该组的客户，
    不受任何干预后条件影响。
    """
    with engine.connect() as conn:
        rows = conn.execute(text(_ARM_SQL), {"experiment_key": experiment_key}).mappings().all()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[r["arm_key"]] = {
            "role": r["role"],
            "n_total": int(r["n_total"]),
            "n_missing": int(r["n_missing"]),
            "k_converted": int(r["k_converted"]),
            "k_visited": int(r["k_visited"]),
            "total_spend": float(r["total_spend"]),
            "spend_values": np.asarray(r["spend_values"], dtype=float),
        }
    return out


def fetch_comparison(engine: Engine, experiment_key: str, comparison_key: str) -> dict[str, Any]:
    with engine.connect() as conn:
        row = conn.execute(
            text(_COMPARISON_SQL),
            {"experiment_key": experiment_key, "comparison_key": comparison_key},
        ).mappings().one()
    return dict(row)


# ============================================================================
#  连续型指标比较
# ============================================================================


def compare_continuous(
    treatment_values: np.ndarray,
    control_values: np.ndarray,
    *,
    metric: str,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 20261011,
) -> ContinuousStats:
    """两组连续指标的比较：Welch t 检验 + bootstrap 百分位区间。

    选择理由（协议 §12.2）：消费金额高度零膨胀（中位数 0），
    方差不齐，因此用**不假设方差齐性**的 Welch t 检验，
    并用 bootstrap 给出不依赖分布假设的区间。

    **必须报告零值占比**——只报均值会掩盖「多数客户消费为 0」这一事实。

    Args:
        treatment_values: 干预组全部分母上的观测值（含 0）。
        control_values: 对照组全部分母上的观测值（含 0）。
        n_boot: bootstrap 重抽样次数。
        seed: 随机种子，**固定以保证可复现**。
    """
    from scipy import stats as sstats

    if treatment_values.size == 0 or control_values.size == 0:
        raise ValueError("两组都必须有观测值")

    t_mean = float(treatment_values.mean())
    c_mean = float(control_values.mean())
    diff = t_mean - c_mean

    t_stat, p_value = sstats.ttest_ind(treatment_values, control_values, equal_var=False)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        a = rng.choice(treatment_values, size=treatment_values.size, replace=True)
        b = rng.choice(control_values, size=control_values.size, replace=True)
        boots[i] = a.mean() - b.mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return ContinuousStats(
        metric=metric,
        treatment_n=int(treatment_values.size),
        control_n=int(control_values.size),
        treatment_mean=t_mean,
        control_mean=c_mean,
        difference=diff,
        ci_low=float(lo),
        ci_high=float(hi),
        alpha=alpha,
        method="welch_t_test + bootstrap_percentile",
        p_value=float(p_value) if np.isfinite(p_value) else float("nan"),
        bootstrap_seed=seed,
        treatment_zero_share=float((treatment_values == 0).mean()),
        control_zero_share=float((control_values == 0).mean()),
    )


# ============================================================================
#  主流程
# ============================================================================


def analyze_experiment(
    engine: Engine,
    *,
    experiment_key: str = "hillstrom_email_v1",
    comparison_key: str = "primary",
    run_quality_checks: bool = True,
    bootstrap_seed: int = 20261011,
    n_boot: int = 10_000,
) -> ExperimentAnalysis:
    """执行一次完整的实验分析。

    Args:
        engine: **只读**引擎。
        experiment_key: 实验标识。
        comparison_key: 比较标识（'primary' 或 'secondary_womens'）。
        run_quality_checks: 是否运行数据质量检查（阻断判定的依据）。
        bootstrap_seed: bootstrap 随机种子，固定以保证可复现。
        n_boot: bootstrap 次数。

    Returns:
        ExperimentAnalysis —— 报告层的唯一数据来源。
    """
    cmp_cfg = fetch_comparison(engine, experiment_key, comparison_key)
    arm_data = fetch_arm_data(engine, experiment_key)

    t_arm = cmp_cfg["treatment_arm"]
    c_arm = cmp_cfg["control_arm"]
    if t_arm not in arm_data or c_arm not in arm_data:
        raise ValueError(f"比较 {comparison_key} 的组在数据中不存在: {t_arm} / {c_arm}")

    alpha = float(cmp_cfg["alpha"])
    t_data, c_data = arm_data[t_arm], arm_data[c_arm]

    # ---- 比例型指标 ----
    metric_comparisons: dict[str, ComparisonStats] = {}
    for metric_id, (zh, col) in PROPORTION_METRICS.items():
        k_t = t_data["k_converted"] if col == "converted" else t_data["k_visited"]
        k_c = c_data["k_converted"] if col == "converted" else c_data["k_visited"]
        metric_comparisons[metric_id] = analyze_comparison(
            treatment_k=k_t,
            treatment_n=t_data["n_total"],
            control_k=k_c,
            control_n=c_data["n_total"],
            alpha=alpha,
            metric=metric_id,
        )

    # ---- 连续型指标 ----
    continuous_comparisons: dict[str, ContinuousStats] = {}
    for metric_id, (_zh, _col) in CONTINUOUS_METRICS.items():
        continuous_comparisons[metric_id] = compare_continuous(
            t_data["spend_values"],
            c_data["spend_values"],
            metric=metric_id,
            alpha=alpha,
            n_boot=n_boot,
            seed=bootstrap_seed,
        )

    # ---- 质量检查 ----
    checks: list[Any] = []
    if run_quality_checks:
        from dsa.stats.checks import run_quality_checks as _rqc

        checks = _rqc(
            engine,
            experiment_key=experiment_key,
            dataset_version=cmp_cfg["version_label"],
        )

    blocking = [c for c in checks if c.severity == "blocking" and not c.passed]

    return ExperimentAnalysis(
        run_id=str(uuid.uuid4()),
        experiment_key=experiment_key,
        dataset_version=cmp_cfg["version_label"],
        comparison_key=comparison_key,
        comparison_kind=cmp_cfg["kind"],
        alpha=alpha,
        window_days=int(cmp_cfg["outcome_window_days"]),
        metric_comparisons=metric_comparisons,
        continuous_comparisons=continuous_comparisons,
        checks=checks,
        blocked=bool(blocking),
        blocking_codes=[c.code for c in blocking],
        stats_config={
            "proportion_interval": "wilson",
            "difference_interval": "newcombe",
            "ratio_interval": "katz",
            "hypothesis_test": "chi2_no_correction (fisher if min_expected < 5)",
            "continuous_test": "welch_t_test + bootstrap_percentile",
            "bootstrap_seed": bootstrap_seed,
            "n_boot": n_boot,
            "alpha": alpha,
            "sided": int(cmp_cfg["sided"]),
        },
        code_version=code_version(),
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        treatment_arm=t_arm,
        control_arm=c_arm,
    )


def record_run(engine_writable: Engine, analysis: ExperimentAnalysis, report_ref: str | None = None) -> str:
    """把一次分析运行登记到 `analysis_runs`（可写引擎）。

    这是报告追溯的入口：报告里的 `run_id` 能在这里找回
    数据版本、统计参数、代码版本与产物引用。
    """
    import json

    with engine_writable.begin() as conn:
        conn.execute(
            text(
                """
                insert into analysis_runs (
                    run_id, experiment_id, dataset_id, comparison_key, question, status,
                    report_ref, code_version, stats_config, finished_at
                )
                select
                    :run_id, e.experiment_id, e.dataset_id, :cmp_key, :question,
                    :status, :report_ref, :code_version, cast(:stats_config as jsonb), now()
                from experiments e where e.experiment_key = :exp_key
                on conflict (run_id) do nothing
                """
            ),
            {
                "run_id": analysis.run_id,
                "cmp_key": analysis.comparison_key,
                "question": f"{analysis.comparison_key} 比较的统计分析",
                "status": "blocked" if analysis.blocked else "succeeded",
                "report_ref": report_ref,
                "code_version": analysis.code_version,
                "stats_config": json.dumps(analysis.stats_config, ensure_ascii=False),
                "exp_key": analysis.experiment_key,
            },
        )
    return analysis.run_id
