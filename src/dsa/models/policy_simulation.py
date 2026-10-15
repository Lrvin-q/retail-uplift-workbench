"""已知机制的模拟验证：核对策略评估（SNIPS + bootstrap）估计量与区间。

口径依据
--------
- `docs/06-建模与策略评估协议.md` §7（模拟验证必须完成）：已知真实 CATE 的模拟、
  排序正确性、区间覆盖、并写明模拟假设与实际数据的差异。
- 同文件 §5.2："具体公式与区间方法必须在实现阶段用模拟数据核对后固定下来"。

为什么必须先做这一步
--------------------
真实留出集只有**一次实现**，``τ(x)`` 的真值未知：估出 +0.68pp 既可能是对的，
也可能是估计量本身有系统偏差。把生成机制设成已知以后，偏差与区间覆盖都变成可核对的量：

====================  ==========================================================
指标                    含义
====================  ==========================================================
``bias``               点估计在重复间的均值 - 已知真值。抓系统偏差。
``ci_coverage``        真值落进报告区间的比例。对账名义 1 - alpha。
``t_statistic``        排序比较里"配对差的蒙特卡洛均值 / 其标准误"。抓"排序有没有用"。
====================  ==========================================================

**被检验的是真实代码路径**：每次重复都调用
:func:`dsa.models.policy.evaluate_policies`（= ``snips_estimate`` + bootstrap），
而不是在模拟里重写一遍公式。模拟若绕过真实调用链，就只是验证了另一份代码。

已知机制
--------
``X ~ Uniform(0, 1)``；``W ~ Bernoulli(0.5)`` 且与 X 独立；``Y ~ Bernoulli(μ0(X) + W·τ(X))``。

* 基线购买率 ``μ0(x) = 0.006 + 0.006·(1 - x)``，取值 0.6%~1.2%，
  与 Hillstrom 对照组的量级一致。**它随 x 递减**：x 越大的人本来购买率越低，
  这正是"本来就会买的人"与"需要推一把的人"错开的那种情形。
* ``homogeneous``：``τ(x) ≡ 0.0068``（同质），量级取自本项目主比较。
* ``heterogeneous``：``τ(x) = 0.0068·x³``（异质，严格递增且连续）。
  选它而不是"只有 X 高端 30% 有正效果"的阶跃形式，理由见
  :data:`SCENARIO_PARAMS` 的说明。

真值
----
目标策略是"按**真实** τ(x) 降序取前 q 比例"，其总体增量
``V = E[π(X)·τ(X)] = ∫ τ(x)·π(x)dx`` 有闭式解，写在
:meth:`ScenarioSpec.uplift_value` 里，并由 ``tests/test_stage4_policy.py``
用大样本蒙特卡洛独立核对。目标策略实际使用的打分见 :meth:`ScenarioSpec.policy_score`
（``cubic`` 场景下它与 τ 同序；``constant`` 场景下 τ 并列，"按 τ 排序"不唯一，
用 x 作并列键，真值不变）。

模拟假设与实际数据的差异（协议 §7.4，必须如实记录）
----------------------------------------------------
见 :data:`ASSUMPTIONS`。**模拟只验证"给定生成机制时程序算得对不对"**，
它不能证明真实数据满足这些假设，也不能替代真实留出集上的评估。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dsa.models.policy import (
    DEFAULT_ALPHA,
    DEFAULT_SEED,
    PROPENSITY,
    TREATMENT_FRACTIONS,
    _selected_mask,
    _snips_core,
    evaluate_policies,
    evaluate_policies_with_contrasts,
    snips_estimate,
)

#: 基线购买率：μ0(x) = MU0_BASE + MU0_SPAN·(1 - x)。
MU0_BASE: float = 0.0060
MU0_SPAN: float = 0.0060

#: 干预效果量级，取自本项目主比较的绝对效果（+0.68pp）。
TAU_SCALE: float = 0.0068

#: 模拟内部 bootstrap 重复次数。写死在模块里，保证 ``run_policy_simulation`` 可复现。
#: 注意它与真实评估的 ``DEFAULT_N_BOOTSTRAP`` 不同：模拟要跑 n_sims 次，
#: 每次都要一套区间，次数取小一些才能在可接受时间内跑完。
SIM_CI_REPLICATES: int = 200

#: 模拟使用的显著性水平（与协议 §12.3 一致）。``run_policy_simulation`` 的签名里
#: 没有 alpha 参数，因此由模块常量固定。
SIM_ALPHA: float = DEFAULT_ALPHA


def baseline_rate(x: np.ndarray) -> np.ndarray:
    """对照组（``No E-Mail``）的真实购买概率 μ0(x)，随 x 递减。"""
    return MU0_BASE + MU0_SPAN * (1.0 - np.asarray(x, dtype=float))


def tau_constant(x: np.ndarray) -> np.ndarray:
    """同质效果：每名客户的真实增量都一样。"""
    return np.full_like(np.asarray(x, dtype=float), TAU_SCALE)


def tau_cubic(x: np.ndarray) -> np.ndarray:
    """异质效果：真实增量随 x 递增（τ(x) = TAU_SCALE · x³）。"""
    x = np.asarray(x, dtype=float)
    return TAU_SCALE * x**3


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """一个预设场景：已知的生成机制与可解析的真值。"""

    name: str
    tau_kind: str  # "constant" | "cubic"
    fraction: float  # 目标策略使用的触达比例，必须来自 TREATMENT_FRACTIONS
    description: str

    def tau(self, x: np.ndarray) -> np.ndarray:
        """真实条件平均增量 τ(x)。"""
        if self.tau_kind == "constant":
            return tau_constant(x)
        if self.tau_kind == "cubic":
            return tau_cubic(x)
        raise ValueError(f"未登记的 tau_kind={self.tau_kind!r}")

    def policy_score(self, x: np.ndarray) -> np.ndarray:
        """目标策略实际使用的打分（越大越优先触达）。

        * ``cubic``：打分 = x³ = τ(x)/TAU_SCALE，与 τ 同序，即"按真实增量排序"。
        * ``constant``：τ 是常数，**"按 τ 排序"在并列下不唯一**，任何 q 子集的真值都一样
          （都是 q·TAU）。这里用 x 本身当并列键。

        为什么不用常数打分（例如直接传 ``tau(x)``）：常数打分下
        ``np.argsort`` 只能按位置取前 k 个，bootstrap 里"按打分重排"就退化成
        "随机抽 k 行（有放回）"，重抽样的方差会相对无放回真值被系统性放大，
        覆盖率虚高（实测 1.00）。改成对 x 取值后，两个场景的选取都是"对 x 取阈值"，
        与真实打分（连续、单调）的结构一致，覆盖率回到名义水平附近（实测约 0.93~0.95）。
        """
        x = np.asarray(x, dtype=float)
        if self.tau_kind == "constant":
            return x
        if self.tau_kind == "cubic":
            return x**3
        raise ValueError(f"未登记的 tau_kind={self.tau_kind!r}")

    def uplift_value(self, fraction: float) -> float:
        """按**真实** τ(x) 降序取前 ``fraction`` 比例的总体增量（解析真值）。

        X 均匀分布、τ 单调时，总体规则就是 ``π(x) = 1[x ≥ 1 - q]``：
        ``V = ∫_{1-q}^{1} τ(x)dx``。阶跃/常数 τ 需按各自闭式处理：

        * ``constant``：``V = q · TAU_SCALE``；
        * ``cubic``：``V = TAU_SCALE · (1 - (1-q)⁴) / 4``。

        注：样本里选的是"最大的 k = round(q·n) 个"，其期望与上面的总体积分相差
        ``O(1/n)``，在 n ≥ 3000 时可忽略（模拟里 n 默认 20000）。
        """
        q = float(fraction)
        if not 0.0 < q <= 1.0:
            raise ValueError(f"fraction 必须落在 (0, 1] 内，实际为 {fraction!r}")
        if self.tau_kind == "constant":
            return q * TAU_SCALE
        if self.tau_kind == "cubic":
            return TAU_SCALE * (1.0 - (1.0 - q) ** 4) / 4.0
        raise ValueError(f"未登记的 tau_kind={self.tau_kind!r}")

    def propensity_value(self, fraction: float) -> float:
        """按预测购买概率 μ0(x) 降序取前 ``fraction`` 比例的总体增量（解析真值）。

        μ0(x) 随 x 递减，因此"按购买概率降序"等价于取 ``π(x) = 1[x ≤ q]``：
        ``V = ∫_0^q τ(x)dx``。

        * ``constant``：``V = q · TAU_SCALE``——与 :meth:`uplift_value` **完全相同**。
          同质效果下排序不可能带来差别，这是模拟的自洽性检查（反例对照）：
          若 :func:`run_ranking_comparison` 在同质场景里也报出"增量排序更好"，
          那一定是代码有问题，而不是策略有效。
        * ``cubic``：``V = TAU_SCALE · q⁴ / 4``。
        """
        q = float(fraction)
        if not 0.0 < q <= 1.0:
            raise ValueError(f"fraction 必须落在 (0, 1] 内，实际为 {fraction!r}")
        if self.tau_kind == "constant":
            return q * TAU_SCALE
        if self.tau_kind == "cubic":
            return TAU_SCALE * q**4 / 4.0
        raise ValueError(f"未登记的 tau_kind={self.tau_kind!r}")

    @property
    def true_policy_value(self) -> float:
        """目标策略（按真实 τ 排序）在 :attr:`fraction` 下的解析真值。"""
        return self.uplift_value(self.fraction)


#: 所有预设场景。顺序固定，:func:`run_all_policy_scenarios` 严格按该顺序执行。
SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        name="homogeneous",
        tau_kind="constant",
        fraction=0.30,
        description=(
            "同质效果：τ(x) ≡ +0.68pp。真值 = q·τ，与选谁无关；"
            "用于核对估计量的无偏性与区间覆盖。"
        ),
    ),
    ScenarioSpec(
        name="heterogeneous",
        tau_kind="cubic",
        fraction=0.30,
        description=(
            "异质效果：τ(x) = 0.68pp · x³，严格递增且连续；"
            "μ0(x) 随 x 递减，于是'按购买概率排序'恰好选到增量最小的人。"
        ),
    ),
)

#: 场景名 -> 场景配置。
SCENARIO_PARAMS: dict[str, ScenarioSpec] = {s.name: s for s in SCENARIOS}

#: ``assumptions`` 字段的固定正文：机制、估计量、以及与真实数据的差异。
#: **如实写明**，不含未做的假设。
ASSUMPTIONS = (
    "模拟机制：(1) X ~ Uniform(0,1)，一维；"
    "(2) 干预分配 W ~ Bernoulli(0.5)，iid 且与 X 独立，等同于完全随机化；"
    "(3) 结果 Y ~ Bernoulli(clip(μ0(X) + W·τ(X), 0, 1))，每人只观测一个结果，"
    "潜在结果 Y(0)、Y(1) 不同时可见；"
    f"(4) 基线 μ0(x) = {MU0_BASE} + {MU0_SPAN}·(1-x)，随 x 递减；"
    f"(5) 效果量级 TAU_SCALE = {TAU_SCALE}；"
    "(6) 条件化倾向 e(x) = 0.5 且**已知**（与项目一致，协议 §2.2），不是估计出来的；"
    "(7) 目标策略 = 按真实 τ(x) 降序取前 q 比例，q 取自预先固定的触达比例集合；"
    "(8) 真值由 E[π(X)·τ(X)] 的闭式积分给出，不依赖任何估计；"
    f"(9) 每次重复都走真实调用链 evaluate_policies（SNIPS + {SIM_CI_REPLICATES} 次"
    "百分位 bootstrap），不在模拟里重写公式；"
    "(10) 随机数由 numpy.random.default_rng(seed) 产生，同种子逐位复现；"
    "(11) 无干扰、无流失、无多版本干预（SUTVA 成立），这是假设而非可验证的事实。"
    " 与真实数据的差异：真实 τ(x) 未知，模拟里已知，所以只有模拟能核对偏差与覆盖率；"
    "真实数据是 9 个决策前特征（recency/history/mens/…）而不是一维均匀 X；"
    "真实留出集约 8.5k 人，模拟默认 20k；"
    "真实结果窗口固定为实验后两周，模拟里没有时间维度；"
    "真实购买率约 0.6%~1.3%，模拟的 μ0 在 0.6%~1.2%，量级相近但不相等；"
    "真实数据只有一次实现，模拟可以重复，因此模拟结论**不能**替代真实留出集上的评估，"
    "也不能证明真实数据满足上述假设。"
)


@dataclass(frozen=True, slots=True)
class PolicySimResult:
    """一个模拟场景的汇总结果。"""

    scenario: str
    n_sims: int
    n: int
    true_policy_value: float  # 已知真值
    mean_estimate: float
    bias: float
    ci_coverage: float
    alpha: float
    seed: int
    assumptions: str


@dataclass(frozen=True, slots=True)
class RankingComparison:
    """同质/异质场景下"按真实增量排序"与"按购买概率排序"的配对比较。"""

    scenario: str
    fraction: float
    n_sims: int
    n: int
    true_uplift_value: float
    true_propensity_value: float
    mean_uplift_estimate: float
    mean_propensity_estimate: float
    mean_difference: float
    true_difference: float
    se_difference: float
    t_statistic: float
    uplift_win_rate: float
    seed: int
    assumptions: str


@dataclass(frozen=True, slots=True)
class BootstrapDesignDiagnostic:
    """两种 bootstrap 做法的覆盖率对照：重抽样时"名单重选"还是"名单固定"。"""

    scenario: str
    n_sims: int
    n: int
    n_bootstrap: int
    true_policy_value: float
    coverage_reselect: float
    coverage_fixed_list: float
    alpha: float
    seed: int
    note: str


def _resolve(scenario: str) -> ScenarioSpec:
    """按场景名取配置；未知场景名报错，**不静默退化成默认参数**。"""
    if scenario not in SCENARIO_PARAMS:
        raise ValueError(
            f"未知场景 {scenario!r}；可选：{sorted(SCENARIO_PARAMS)}。"
            "未知场景不会被静默当作默认参数处理，请先在 SCENARIOS 中登记。"
        )
    return SCENARIO_PARAMS[scenario]


def _check_sizes(*, n_sims: int, n: int) -> None:
    if n_sims <= 0:
        raise ValueError(f"n_sims 必须为正整数，实际为 {n_sims!r}")
    if n <= 0:
        raise ValueError(f"n 必须为正整数，实际为 {n!r}")


def _draw(spec: ScenarioSpec, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """抽一份模拟数据，返回 ``(x, w, y, tau)``。"""
    x = rng.random(n)
    w = rng.random(n) < PROPENSITY
    tau = spec.tau(x)
    p = np.clip(baseline_rate(x) + w * tau, 0.0, 1.0)
    y = rng.random(n) < p
    return x, w, y, tau


def simulate_estimates(
    *,
    scenario: str = "heterogeneous",
    n_sims: int = 200,
    n: int = 20000,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """只算点估计（不算区间）的重复模拟，返回每次重复的 SNIPS 估计。

    用于核对无偏性：``bias = mean(估计) - 真值``，而"经验标准误"就是本数组的标准差。
    走的是 :func:`dsa.models.policy.snips_estimate` 这条真实路径（不经过 bootstrap，
    因此比 :func:`run_policy_simulation` 快一到两个数量级）。

    Args:
        scenario: 场景名，必须是 :data:`SCENARIO_PARAMS` 的键。
        n_sims: 重复次数。
        n: 每次重复的样本量。
        seed: 随机种子。

    Returns:
        长度 n_sims 的估计数组。
    """
    spec = _resolve(scenario)
    _check_sizes(n_sims=n_sims, n=n)
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        x, w, y, _tau = _draw(spec, n, rng)
        mask = _selected_mask(spec.policy_score(x), spec.fraction, name=scenario)
        out[i] = snips_estimate(w, y, mask, PROPENSITY)
    return out


def run_policy_simulation(
    *,
    scenario: str = "heterogeneous",
    n_sims: int = 200,
    n: int = 20000,
    seed: int = DEFAULT_SEED,
) -> PolicySimResult:
    """已知机制的模拟，用于验证 IPS/SNIPS 估计量无偏且区间覆盖达标。

    每次重复：抽一份数据 -> 按**真实** τ(x) 排序取前 30% -> 调用
    :func:`~dsa.models.policy.evaluate_policies` 得到点估计与 bootstrap 区间 ->
    看真值是否落进区间。汇总出 ``bias`` 与 ``ci_coverage``。

    区间种子：第 i 次重复用 ``seed + i`` 作为 bootstrap 种子。
    重复之间不能共用同一批重抽样索引——共用会让重抽样误差在重复之间完全相关，
    ``ci_coverage`` 就不再是覆盖率。真实评估里种子固定为 ``20261011``（协议 §6.5），
    两者用途不同。

    Args:
        scenario: 场景名，必须是 :data:`SCENARIO_PARAMS` 的键。
        n_sims: 重复次数（蒙特卡洛样本量）。
        n: 每次重复的样本量。
        seed: 随机种子，同种子逐位复现。

    Returns:
        :class:`PolicySimResult`。``bias`` 是"估计均值 - 真值"，
        ``ci_coverage`` 是名义 ``1 - alpha`` 区间的实际覆盖率。

    Raises:
        ValueError: 场景名未知；n_sims / n 非正。
    """
    spec = _resolve(scenario)
    _check_sizes(n_sims=n_sims, n=n)
    true_value = spec.true_policy_value
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_sims, dtype=float)
    covered = 0
    undefined_intervals = 0
    for i in range(n_sims):
        x, w, y, _tau = _draw(spec, n, rng)
        rows = evaluate_policies(
            y=y,
            treatment=w,
            scores={"uplift": spec.policy_score(x)},
            fractions=(spec.fraction,),
            alpha=SIM_ALPHA,
            n_bootstrap=SIM_CI_REPLICATES,
            bootstrap_seed=seed + i,
        )
        row = next(r for r in rows if r.policy == "uplift")
        estimates[i] = row.ips_estimate
        if not row.ci_defined:
            undefined_intervals += 1
            continue
        if row.ci_low <= true_value <= row.ci_high:
            covered += 1

    assumptions = ASSUMPTIONS
    if undefined_intervals:
        assumptions += (
            f" 本次运行有 {undefined_intervals} 次区间无定义（已排除在覆盖率分子分母之外）。"
        )
    mean_estimate = float(estimates.mean())
    return PolicySimResult(
        scenario=scenario,
        n_sims=n_sims,
        n=n,
        true_policy_value=true_value,
        mean_estimate=mean_estimate,
        bias=mean_estimate - true_value,
        ci_coverage=covered / n_sims,
        alpha=SIM_ALPHA,
        seed=seed,
        assumptions=assumptions,
    )


def run_ranking_comparison(
    *,
    scenario: str = "heterogeneous",
    fraction: float | None = None,
    n_sims: int = 200,
    n: int = 20000,
    seed: int = DEFAULT_SEED,
) -> RankingComparison:
    """配对比较"按真实增量排序"与"按购买概率排序"两个策略的估计值。

    同一次重复里两份数据完全相同，两条排序规则各自取前 q 比例，得到一对**配对**估计。
    配对差的蒙特卡洛标准误 ``se_difference = sd(差)/sqrt(n_sims)``，
    ``t_statistic = mean_difference / se_difference`` 就是"排序是否真的更有用"的判据。

    **反例对照**：``homogeneous`` 场景下 τ 是常数，任何排序的总体增量都相同
    （``true_difference == 0``）。若那里也报出"增量排序更好"，就是代码有问题。
    这条对照是防止"结论怎么都能成立"的关键。

    Args:
        scenario: 场景名。
        fraction: 触达比例，``None`` 表示取场景预设（默认 0.30）。
        n_sims: 重复次数。
        n: 每次重复的样本量。
        seed: 随机种子。

    Returns:
        :class:`RankingComparison`。``true_uplift_value`` / ``true_propensity_value``
        都是解析真值，不由模拟估计。
    """
    spec = _resolve(scenario)
    _check_sizes(n_sims=n_sims, n=n)
    q = spec.fraction if fraction is None else float(fraction)
    if q not in TREATMENT_FRACTIONS:
        raise ValueError(
            f"触达比例 {q!r} 不在预先固定的集合 {TREATMENT_FRACTIONS} 中（协议 §6.1）"
        )
    true_uplift = spec.uplift_value(q)
    true_propensity = spec.propensity_value(q)
    rng = np.random.default_rng(seed)
    uplift_est = np.empty(n_sims, dtype=float)
    propensity_est = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        x, w, y, _tau = _draw(spec, n, rng)
        uplift_mask = _selected_mask(spec.policy_score(x), q, name="uplift")
        propensity_mask = _selected_mask(baseline_rate(x), q, name="propensity")
        uplift_est[i] = snips_estimate(w, y, uplift_mask, PROPENSITY)
        propensity_est[i] = snips_estimate(w, y, propensity_mask, PROPENSITY)
    diffs = uplift_est - propensity_est
    mean_difference = float(diffs.mean())
    se_difference = float(diffs.std(ddof=1) / math.sqrt(n_sims)) if n_sims > 1 else math.nan
    return RankingComparison(
        scenario=scenario,
        fraction=q,
        n_sims=n_sims,
        n=n,
        true_uplift_value=true_uplift,
        true_propensity_value=true_propensity,
        mean_uplift_estimate=float(uplift_est.mean()),
        mean_propensity_estimate=float(propensity_est.mean()),
        mean_difference=mean_difference,
        true_difference=true_uplift - true_propensity,
        se_difference=se_difference,
        t_statistic=(mean_difference / se_difference if se_difference else math.nan),
        uplift_win_rate=float(np.mean(diffs > 0)),
        seed=seed,
        assumptions=ASSUMPTIONS,
    )


def _fixed_list_coverage(
    *,
    spec: ScenarioSpec,
    n_sims: int,
    n: int,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> float:
    """对照做法：重抽样时**固定住原样本选出的名单**，只重抽 (W, Y)。

    实现放在模拟模块里（而不是给 ``evaluate_policies`` 加开关），
    是为了让主模块的接口只保留一种、也就是已经过核对的做法。
    """
    true_value = spec.true_policy_value
    rng = np.random.default_rng(seed)
    covered = 0
    defined = 0
    for i in range(n_sims):
        x, w, y, _tau = _draw(spec, n, rng)
        mask = _selected_mask(spec.policy_score(x), spec.fraction, name="fixed")
        share = float(mask.mean())
        boot_rng = np.random.default_rng(seed + i)
        values = np.empty(n_bootstrap, dtype=float)
        for b in range(n_bootstrap):
            idx = boot_rng.integers(0, n, n)
            m = mask[idx]
            t_b = w[idx]
            values[b] = _snips_core(m & t_b, m & ~t_b, y[idx].astype(float), float(m.mean()))
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
        defined += 1
        if low <= true_value <= high:
            covered += 1
    return covered / defined if defined else math.nan


def diagnose_bootstrap_design(
    *,
    scenario: str = "heterogeneous",
    n_sims: int = 200,
    n: int = 20000,
    n_bootstrap: int = SIM_CI_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> BootstrapDesignDiagnostic:
    """核对两种 bootstrap 做法的区间覆盖率，为"固定用哪一种"提供证据。

    * ``reselect``（主模块采用）：每次重抽样**重新执行选取规则**——按打分重排、再取前 q。
    * ``fixed_list``（对照）：固定住原样本选出的名单，只重抽结果与分配。

    为什么默认选 reselect：留出集是从总体抽出的**一份样本**，策略是"排序取前 q"这条
    **规则**；固定名单等于把"名单本身会变"这一层不确定性人为去掉，区间会偏窄。
    两种做法的实际覆盖率由本函数给出，测试断言主模块采用的那种达标，
    数字同时写进模拟验证记录。名单固定这种做法的代码只存在于本函数内，
    不进 :func:`~dsa.models.policy.evaluate_policies` 的接口。
    """
    spec = _resolve(scenario)
    _check_sizes(n_sims=n_sims, n=n)
    result = run_policy_simulation(scenario=scenario, n_sims=n_sims, n=n, seed=seed)
    fixed = _fixed_list_coverage(
        spec=spec, n_sims=n_sims, n=n, n_bootstrap=n_bootstrap, alpha=SIM_ALPHA, seed=seed
    )
    return BootstrapDesignDiagnostic(
        scenario=scenario,
        n_sims=n_sims,
        n=n,
        n_bootstrap=n_bootstrap,
        true_policy_value=spec.true_policy_value,
        coverage_reselect=result.ci_coverage,
        coverage_fixed_list=fixed,
        alpha=SIM_ALPHA,
        seed=seed,
        note=(
            "覆盖率为真值落入 95% 百分位 bootstrap 区间的比例，名义水平 0.95；"
            "n_sims 次重复下覆盖率本身的蒙特卡洛标准误约 sqrt(0.95·0.05/n_sims)。"
            "'名单固定'一列只作对照，不用于报告。"
            + ASSUMPTIONS
        ),
    )


def run_all_policy_scenarios(
    *, n_sims: int = 200, n: int = 20000, seed: int = DEFAULT_SEED
) -> list[PolicySimResult]:
    """跑完所有预设场景，参数由 :func:`run_policy_simulation` 自己按场景名解析。"""
    return [run_policy_simulation(scenario=s.name, n_sims=n_sims, n=n, seed=seed) for s in SCENARIOS]


# ============================================================================
#  差值区间（策略之间的**配对**比较）的模拟表现
# ============================================================================


@dataclass(frozen=True, slots=True)
class DifferenceCoverage:
    """两个策略差值区间的覆盖率与检验力（已知真值）。"""

    scenario: str
    fraction: float
    n_sims: int
    n: int
    n_bootstrap: int
    true_difference: float
    mean_difference: float
    bias: float
    se_difference: float
    ci_coverage: float
    excludes_zero_rate: float
    alpha: float
    seed: int
    assumptions: str


def run_difference_coverage(
    *,
    scenario: str = "heterogeneous",
    fraction: float = 0.30,
    n_sims: int = 40,
    n: int = 3000,
    n_bootstrap: int = SIM_CI_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> DifferenceCoverage:
    """已知真值下核对**差值区间**：覆盖率是否接近名义水平、偏差是否可忽略。

    每次重复：抽一份数据 -> 用**真实** τ(x) 排序当 ``uplift`` 策略、
    用真实基线购买率 μ0(x) 排序当 ``purchase_probability`` 策略 ->
    调用 :func:`~dsa.models.policy.evaluate_policies_with_contrasts`（真实代码路径）
    -> 看解析真值差 ``uplift_value(q) - propensity_value(q)`` 是否落进差值区间。

    与边际区间覆盖率不同，这里**必须走配对抽样**：两个策略跑在同一份数据上，
    它们的估计相关，差值区间才可能比"两条边际区间相减"更紧、也更贴近名义覆盖。

    Raises:
        ValueError: 场景名未知；n_sims / n 非正；fraction 不在固定集合内。
    """
    spec = _resolve(scenario)
    _check_sizes(n_sims=n_sims, n=n)
    q = float(fraction)
    if q not in TREATMENT_FRACTIONS:
        raise ValueError(
            f"触达比例 {q!r} 不在预先固定的集合 {TREATMENT_FRACTIONS} 中（协议 §6.1）"
        )
    true_difference = spec.uplift_value(q) - spec.propensity_value(q)
    rng = np.random.default_rng(seed)
    differences = np.empty(n_sims)
    covered = 0
    defined = 0
    excludes_zero = 0
    for i in range(n_sims):
        x, w, y, _tau = _draw(spec, n, rng)
        evaluation = evaluate_policies_with_contrasts(
            y=y,
            treatment=w,
            scores={
                "uplift": spec.policy_score(x),
                # 模拟里"合并购买预测"用真实基线购买率 μ0(x) 排序（真值口径见 propensity_value）
                "purchase_probability": baseline_rate(x),
            },
            fractions=(q,),
            alpha=SIM_ALPHA,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=seed + i,
        )
        contrast = next(
            c for c in evaluation.contrasts
            if c.policy == "uplift" and c.reference_policy == "purchase_probability"
        )
        differences[i] = contrast.difference
        if not contrast.ci_defined:
            continue
        defined += 1
        if contrast.ci_low <= true_difference <= contrast.ci_high:
            covered += 1
        if contrast.excludes_zero:
            excludes_zero += 1

    se = float(differences.std(ddof=1) / math.sqrt(n_sims)) if n_sims > 1 else math.nan
    mean_difference = float(differences.mean())
    return DifferenceCoverage(
        scenario=scenario,
        fraction=q,
        n_sims=n_sims,
        n=n,
        n_bootstrap=n_bootstrap,
        true_difference=true_difference,
        mean_difference=mean_difference,
        bias=mean_difference - true_difference,
        se_difference=se,
        ci_coverage=(covered / defined if defined else math.nan),
        excludes_zero_rate=(excludes_zero / defined if defined else math.nan),
        alpha=SIM_ALPHA,
        seed=seed,
        assumptions=ASSUMPTIONS,
    )


#: 「区间重叠 vs 配对差值」诊断的生成机制（见 :func:`diagnose_overlap_vs_paired`）。
OVERLAP_TAU_THRESHOLD: float = 0.70
OVERLAP_TAU_STEP: float = 0.05
OVERLAP_CONTAMINATION_BUMP: float = 0.10
OVERLAP_MU0: float = 0.01


@dataclass(frozen=True, slots=True)
class OverlapDiagnostic:
    """同一批重复里，"边际区间是否重叠"与"配对差值区间是否排除 0"两个判据的对照。"""

    fraction: float
    n_sims: int
    n: int
    n_bootstrap: int
    bump: float
    true_difference: float
    mean_difference: float
    se_difference: float
    marginal_overlap_rate: float
    paired_excludes_zero_rate: float
    paired_coverage: float
    alpha: float
    seed: int
    assumptions: str
    note: str


def _overlap_design_truths(
    fraction: float,
    *,
    threshold: float = OVERLAP_TAU_THRESHOLD,
    step: float = OVERLAP_TAU_STEP,
    bump: float = OVERLAP_CONTAMINATION_BUMP,
) -> tuple[float, float]:
    """两个策略的**解析真值** ``(V_oracle, V_proxy)``。

    机制：``τ(x) = step·1[x ≥ threshold]``，``μ0(x) ≡ OVERLAP_MU0``（与 x 无关，
    这样"按购买概率排序"没有意义，两个策略的差别只来自排序规则本身）。

    * ``oracle``：打分 = x，取前 q 比例的 ``x`` → 恰好选中 ``x ≥ 1-q`` 的人。
      当 ``1-q > threshold``（本项目 q=0.3、threshold=0.7 时两者相等）时名单里
      只有有效果的人：``V_oracle = q·step``。
    * ``proxy``：打分 = ``x + bump·1[x < threshold]``。抬升让一部分**没有效果**的人
      （``x ∈ [threshold - bump/2, threshold)``）挤进名单，同时把同等数量的有效果的人挤出去。
      阈值满足 ``F(s*) = 1 - q``，其中 ``F(s) = (s - bump) + (s - threshold)``
      （两个区间各贡献一段），解得 ``s* = (1 + threshold - q + bump)/2``。
      名单里有比例 ``(1 - s*) / q`` 的人有效果：``V_proxy = q·step·(1 - s*)/q``。

    要求 ``s*`` 落在 ``[threshold, threshold + bump)`` 内，否则两段分布在阈值处的
    接法不成立——不成立就报错，**不悄悄给一个错的真值**。
    """
    q = float(fraction)
    s_star = (1.0 + threshold - q + bump) / 2.0
    if not threshold <= s_star < threshold + bump:
        raise ValueError(
            f"该参数下名单阈值 s*={s_star:.4f} 不在 [{threshold}, {threshold + bump}) 内，"
            "解析真值不成立；请调整 fraction / bump / threshold。"
        )
    v_oracle = q * step
    v_proxy = q * step * ((1.0 - s_star) / q)
    return v_oracle, v_proxy


def diagnose_overlap_vs_paired(
    *,
    fraction: float = 0.30,
    n_sims: int = 40,
    n: int = 10000,
    n_bootstrap: int = SIM_CI_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> OverlapDiagnostic:
    """把"区间重叠"这条判据与"配对差值区间"这条判据放在一起量。

    每次重复生成一份数据（机制与解析真值见 :func:`_overlap_design_truths`），
    两个策略高度重叠（名单只差一小截），因此两个估计**高度相关**：
    这正是审查者说的"同一批留出客户上估计相关"的情形。

    同时记录三件事：

    1. ``marginal_overlap_rate``：两条**边际** 95% 区间重叠的比例——
       旧报告逻辑会把它读成"未检出显著差异"；
    2. ``paired_excludes_zero_rate``：**配对差值** 95% 区间排除 0 的比例（检验力）；
    3. ``paired_coverage``：配对差值区间覆盖解析真值的比例（名义 0.95）。

    判据的读法：真值差与配对标准误之比决定检验力，覆盖率才对得上名义水平。
    这是**合成数据**上的判据性质演示，不是真实数据结论。
    """
    _check_sizes(n_sims=n_sims, n=n)
    q = float(fraction)
    if q not in TREATMENT_FRACTIONS:
        raise ValueError(
            f"触达比例 {q!r} 不在预先固定的集合 {TREATMENT_FRACTIONS} 中（协议 §6.1）"
        )
    v_oracle, v_proxy = _overlap_design_truths(q)
    true_difference = v_oracle - v_proxy

    rng = np.random.default_rng(seed)
    differences = np.empty(n_sims)
    overlap_hits = 0
    excludes_hits = 0
    covered = 0
    defined = 0
    for i in range(n_sims):
        x = rng.random(n)
        w = rng.random(n) < PROPENSITY
        tau = np.where(x >= OVERLAP_TAU_THRESHOLD, OVERLAP_TAU_STEP, 0.0)
        p = np.clip(OVERLAP_MU0 + w * tau, 0.0, 1.0)
        y = rng.random(n) < p
        evaluation = evaluate_policies_with_contrasts(
            y=y,
            treatment=w,
            scores={
                "uplift": x,  # oracle：按 x 排序 = 按真实 τ 排序
                "control_probability": x + OVERLAP_CONTAMINATION_BUMP * (x < OVERLAP_TAU_THRESHOLD),
            },
            fractions=(q,),
            alpha=SIM_ALPHA,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=seed + i,
        )
        by_policy = {s.policy: s for s in evaluation.scores}
        oracle = by_policy["uplift"]
        proxy = by_policy["control_probability"]
        # 判据一：两条边际区间是否重叠（旧报告的逻辑）
        if not (
            oracle.ci_high < proxy.ci_low
            or proxy.ci_high < oracle.ci_low
            or not oracle.ci_defined
            or not proxy.ci_defined
        ):
            overlap_hits += 1
        contrast = next(
            c for c in evaluation.contrasts
            if c.policy == "uplift" and c.reference_policy == "control_probability"
        )
        differences[i] = contrast.difference
        if not contrast.ci_defined:
            continue
        defined += 1
        if contrast.excludes_zero:
            excludes_hits += 1
        if contrast.ci_low <= true_difference <= contrast.ci_high:
            covered += 1

    se = float(differences.std(ddof=1) / math.sqrt(n_sims)) if n_sims > 1 else math.nan
    mean_difference = float(differences.mean())
    return OverlapDiagnostic(
        fraction=q,
        n_sims=n_sims,
        n=n,
        n_bootstrap=n_bootstrap,
        bump=OVERLAP_CONTAMINATION_BUMP,
        true_difference=true_difference,
        mean_difference=mean_difference,
        se_difference=se,
        marginal_overlap_rate=(overlap_hits / n_sims),
        paired_excludes_zero_rate=(excludes_hits / defined if defined else math.nan),
        paired_coverage=(covered / defined if defined else math.nan),
        alpha=SIM_ALPHA,
        seed=seed,
        assumptions=ASSUMPTIONS,
        note=(
            "两个策略的名单高度重叠（只差被 bump 抬升的那一小截），因此两个估计高度相关；"
            "边际区间重叠不等于没有差异，差值区间才是差异的判据。"
            "本诊断是合成数据上的判据性质演示，不是真实数据结论。" + ASSUMPTIONS
        ),
    )


def illustrative_sketch_estimate(
    treatment: np.ndarray,
    y: np.ndarray,
    selected_treat: np.ndarray,
    propensity: float = PROPENSITY,
) -> float:
    """协议 §5.2 **示意式**的字面实现——**只作反例，不用于任何报告**。

    字面执行 :math:`\\frac{\\sum_i \\mathbb{1}[W_i = \\pi(X_i)] Y_i / e}{\\sum_i
    \\mathbb{1}[W_i = \\pi(X_i)] / e} \\times |\\pi| / n`：先按"实际动作是否等于策略动作"
    筛客户，求他们的购买率，再乘触达比例。

    在 ``e ≡ 0.5`` 下 ``1/e`` 在比值里整体约掉，于是它等于"匹配上的客户的购买率 ×
    触达比例"。这种做法**漏掉了对照那一侧的反事实**，估出来的是**水平而不是增量**：
    ``π ≡ 1`` 时它直接给出干预组购买率（≈ 1.25%），而真实增量是两组之差（≈ 0.68pp）。
    差距就是基线购买率，量级上与效果本身相同甚至更大。

    保留它的唯一用途：在测试与模拟记录里**量化这个偏差**，说明主模块为什么不用它。
    """
    treat = np.asarray(treatment).astype(bool)
    outcome = np.asarray(y).astype(float)
    selected = np.asarray(selected_treat).astype(bool)
    e = float(propensity)
    if not 0.0 < e < 1.0:
        raise ValueError(f"propensity 必须落在 (0, 1) 内，实际为 {propensity!r}")
    if selected.shape[0] == 0:
        return math.nan
    matched = treat == selected
    # 示意式里的权重是 1/e(X_i)，而 e 在本项目里对所有人都等于 0.5：
    # 权重在分子分母里完全一样，整体约掉——所以它实际就是"匹配人群的购买率"。
    weight = 1.0 / e
    den = float(matched.sum()) * weight
    if den <= 0.0:
        return math.nan
    num = float(outcome[matched].sum()) * weight
    return float(selected.mean()) * (num / den)


def summarize(results: list[PolicySimResult]) -> str:
    """把结果列表格式化成对齐的文本表，供人工核对与日志粘贴。"""
    header = (
        f"{'场景':<15}{'样本量':>9}{'重复':>6}{'真值':>12}{'估计均值':>13}"
        f"{'bias':>12}{'覆盖率':>9}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.scenario:<15}{r.n:>9}{r.n_sims:>6}{r.true_policy_value:>+12.6f}"
            f"{r.mean_estimate:>+13.6f}{r.bias:>+12.2e}{r.ci_coverage:>9.4f}"
        )
    return "\n".join(lines)


def summarize_ranking(comparisons: list[RankingComparison]) -> str:
    """把排序比较格式化成文本表。"""
    header = (
        f"{'场景':<15}{'比例':>6}{'真值(增量排序)':>16}{'真值(购买概率排序)':>19}"
        f"{'估计均值差':>13}{'SE':>11}{'t':>8}{'单次胜率':>10}"
    )
    lines = [header, "-" * len(header)]
    for c in comparisons:
        lines.append(
            f"{c.scenario:<15}{c.fraction:>6.2f}{c.true_uplift_value:>+16.6f}"
            f"{c.true_propensity_value:>+19.6f}{c.mean_difference:>+13.6f}"
            f"{c.se_difference:>11.2e}{c.t_statistic:>8.1f}{c.uplift_win_rate:>10.4f}"
        )
    return "\n".join(lines)


def _main() -> None:  # pragma: no cover - 手工核对用的入口
    """命令行入口：打印两个场景的真实数字，粘贴进模拟验证记录。"""
    print("=" * 100)
    print("策略评估模拟验证（已知真值）")
    print("=" * 100)
    results = run_all_policy_scenarios()
    print(summarize(results))
    print()
    for r in results:
        est = simulate_estimates(scenario=r.scenario, n_sims=r.n_sims, n=r.n)
        se = float(est.std(ddof=1))
        mc_se = se / math.sqrt(r.n_sims)
        print(
            f"{r.scenario:<15} 经验标准误(单次) = {se:.3e}；"
            f"均值的蒙特卡洛标准误 = {mc_se:.3e}；bias = {r.bias:+.3e} "
            f"({abs(r.bias) / mc_se:.2f} 个 MC 标准误)"
        )
    print()
    print("排序比较：按真实增量排序 vs 按购买概率排序")
    comparisons = [
        run_ranking_comparison(scenario=s.name, fraction=s.fraction) for s in SCENARIOS
    ]
    print(summarize_ranking(comparisons))
    print()
    print("bootstrap 做法对照（区间覆盖率，名义 0.95）")
    for s in SCENARIOS:
        d = diagnose_bootstrap_design(scenario=s.name, n_sims=100)
        print(
            f"{s.name:<15} 名单重选 = {d.coverage_reselect:.4f}；"
            f"名单固定 = {d.coverage_fixed_list:.4f}"
        )
    print()
    print("协议 §5.2 示意式（只作反例）：同质场景下的偏差")
    spec = SCENARIO_PARAMS["homogeneous"]
    rng = np.random.default_rng(DEFAULT_SEED)
    sketch = np.empty(200)
    truth = spec.true_policy_value
    for i in range(200):
        x, w, y, _tau = _draw(spec, 20000, rng)
        sketch[i] = illustrative_sketch_estimate(
            w, y, _selected_mask(spec.policy_score(x), spec.fraction, name="sketch")
        )
    print(
        f"{spec.fraction:.0%} 触达：真值 = {truth:+.6f}；示意式均值 = {float(sketch.mean()):+.6f}；"
        f"偏差 = {float(sketch.mean()) - truth:+.6f}"
    )
    rng = np.random.default_rng(DEFAULT_SEED)
    sketch = np.empty(200)
    for i in range(200):
        _x, w, y, _tau = _draw(spec, 20000, rng)
        sketch[i] = illustrative_sketch_estimate(w, y, np.ones(20000, dtype=bool))
    print(
        f"全部触达：真值 = {spec.uplift_value(1.0):+.6f}；示意式均值 = {float(sketch.mean()):+.6f}；"
        f"偏差 = {float(sketch.mean()) - spec.uplift_value(1.0):+.6f}"
        "（示意式给的是干预组购买率这个水平，缺了对照那一侧）"
    )


if __name__ == "__main__":  # pragma: no cover
    _main()
