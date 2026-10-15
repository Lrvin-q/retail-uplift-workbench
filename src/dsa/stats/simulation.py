"""模拟验证：用已知机制的合成数据检查统计程序的行为。

口径依据：docs/02-分析协议.md §12；蓝图阶段三「用模拟数据验证统计程序」。

为什么要做模拟验证
------------------
真实数据（Hillstrom 邮件实验）只提供**一次实现**。在那一次实现里：

- 我们不知道真实效果是多少，因此无法判断点估计有没有偏；
- 我们不知道 Newcombe 区间是否真的覆盖了真值，只能「相信」名义水平；
- 一次「p > 0.05」既可能是真的没效果，也可能只是没检验出来。

把真值设为已知、把数据生成机制设为已知，上面三个问题就都变成可计算、可核对的量：

============================  ==========================================
指标                            含义
============================  ==========================================
``bias``                        点估计的均值 - 真实效果。判断有没有系统偏差。
``empirical_se``                点估计的经验标准误（模拟得到的实际波动）。
``mean_reported_se``            程序自己报告的区间半宽 / z。与上者对账。
``ci_coverage``                 真实效果落入报告区间的比例。对账名义 95%。
``false_positive_rate``         真实无效果时 p < alpha 的比例。对账名义 α。
============================  ==========================================

**被检验的代码路径是真实路径**：每一次重复都逐次调用
:func:`dsa.stats.proportions.analyze_comparison`，而不是在模拟里重写一遍
Wilson / Newcombe / 卡方的公式。模拟若绕过真实调用链，就只是验证了另一份代码。

**场景名驱动参数**：:func:`run_scenario` 的默认参数来自 :data:`SCENARIO_PARAMS`
里以场景名为键的配置，而不是函数签名上的字面默认值。这样
``run_scenario("positive")`` **不可能**静默跑成零效果场景——那正是最容易
得出错误结论的一类事故。结果里的 ``params_source`` 字段说明本次参数
是「取自场景预设」还是「被显式覆盖」。

**模拟结论的适用边界**：这里验证的是「给定数据是两组独立二项抽样时，程序算得对不对」。
它**不能**验证真实数据是否满足这个假设（随机化是否完整、有没有干扰、结果窗口是否合适）。
见 `SimulationResult.assumptions`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dsa.stats.proportions import analyze_comparison, z_for

#: `params_source` 取值：参数全部取自场景预设。
PARAMS_FROM_SCENARIO = "scenario_default"
#: `params_source` 取值：调用方显式覆盖了至少一个参数。
PARAMS_OVERRIDDEN = "overridden"


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """一个预设场景：已知的生成机制。"""

    name: str
    n_per_arm: int
    control_rate: float
    true_effect: float
    description: str

    @property
    def treatment_rate(self) -> float:
        """干预组真实比例。"""
        return self.control_rate + self.true_effect

    @property
    def is_null(self) -> bool:
        """真实无效果——只有这种情况下假阳性率才有意义。"""
        return self.true_effect == 0.0


#: 所有预设场景。`run_all_scenarios` 严格按该顺序执行；
#: `run_scenario` 的默认参数取自这里，测试与文档也都引用这里，避免多处口径打架。
SCENARIOS: tuple[ScenarioSpec, ...] = (
    # 真实无效果：检验的误报率应接近名义 alpha
    ScenarioSpec(
        name="null",
        n_per_arm=20000,
        control_rate=0.006,
        true_effect=0.0,
        description="真实无效果，用于核对假阳性率（名义 α=0.05）",
    ),
    # 真实正效果：约等于实测主比较的效果量（+0.68pp）
    ScenarioSpec(
        name="positive",
        n_per_arm=20000,
        control_rate=0.006,
        true_effect=0.0068,
        description="真实正效果 +0.68pp，约等于实测主比较的效果量",
    ),
    # 真实负效果：干预把购买率压低了
    ScenarioSpec(
        name="negative",
        n_per_arm=20000,
        control_rate=0.006,
        true_effect=-0.003,
        description="真实负效果 -0.30pp，干预反而压低购买率",
    ),
    # 稀疏结果：基线 0.1%，期望频数逼近卡方→Fisher 的切换阈值
    ScenarioSpec(
        name="sparse",
        n_per_arm=20000,
        control_rate=0.001,
        true_effect=0.001,
        description="稀疏结果：基线 0.1%、真实效果 +0.10pp",
    ),
    # 小样本：每组 2000，检验的功效明显不足，用于观察区间宽度与功效
    ScenarioSpec(
        name="small_sample",
        n_per_arm=2000,
        control_rate=0.006,
        true_effect=0.0068,
        description="小样本：每组仅 2000，区间显著更宽、功效不足",
    ),
)

#: 场景名 -> 场景配置。
SCENARIO_PARAMS: dict[str, ScenarioSpec] = {s.name: s for s in SCENARIOS}

#: `assumptions` 字段的固定正文。**如实写明**，不含未做的假设。
ASSUMPTIONS = (
    "数据生成假设：(1) 干预组与对照组相互独立，各自服从二项分布 "
    "Bin(n, p_treat) 与 Bin(n, p_control)，即每个客户的购买是独立同分布的伯努利试验；"
    "(2) 两组样本量固定且相等（每条臂 n_per_arm），不是随机样本量，也不做序贯/中途查看；"
    "(3) 随机化完全、无混淆、无流失、无干扰（SUTVA 成立），"
    "即不存在「一个人影响另一个人购买」或「未收到邮件的人被已收到的人带动」；"
    "(4) 干预组真实比例固定为 control_rate + true_effect，不随重复次数变化，"
    "因此模拟衡量的是固定真值下的频率性质，而非对真值分布的平均；"
    "(5) 结果只有一个二元结局、一个固定时间点，不含延迟转化与重复购买；"
    "(6) 每一次重复都走真实调用链 analyze_comparison（Wilson 单比例区间、"
    "Newcombe 混合得分区间、卡方不加连续性校正、任一格期望频数 < 5 时改用 Fisher 精确检验）；"
    "(7) 随机数由 numpy.random.default_rng(seed) 产生，同一种子可逐位复现。"
    " 与真实数据的差异：真实数据只有一次实现，真值未知；"
    "真实的随机化细节（分层、区块、分配比例）在数据中不完整可见；"
    "结果窗口在本项目中被固定为两周，模拟里没有时间维度；"
    "模拟里真值已知，因此区间覆盖率可以核对，而真实数据只能用区间宽度与功效来间接判断。"
)


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """一个模拟场景的汇总结果。"""

    scenario: str
    n_sims: int
    n_per_arm: int
    true_effect: float
    true_control_rate: float
    mean_estimate: float
    bias: float
    empirical_se: float
    mean_reported_se: float
    ci_coverage: float
    false_positive_rate: float
    alpha: float
    seed: int
    assumptions: str
    #: 本次参数来源："scenario_default"（全部取自 SCENARIOS[scenario]）或
    #: "overridden"（调用方显式传了至少一个参数）。
    #: 用途：让调用方一眼看出这次跑的**不是**场景预设，避免场景名与实际机制静默错配。
    params_source: str

    @property
    def is_null(self) -> bool:
        """本次是否为「真实无效果」——决定 ``false_positive_rate`` 是否有意义。"""
        return self.true_effect == 0.0


def _check_rate(rate: float, *, name: str) -> None:
    """比例的取值检查：必须在 [0, 1] 内，且不能是 nan。"""
    if math.isnan(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} 必须落在 [0, 1] 内，实际为 {rate!r}")


def run_scenario(
    scenario: str,
    *,
    n_sims: int = 2000,
    n_per_arm: int | None = None,
    control_rate: float | None = None,
    true_effect: float | None = None,
    alpha: float = 0.05,
    seed: int = 20261011,
) -> SimulationResult:
    """跑一个模拟场景，返回汇总指标。

    **参数默认来自 :data:`SCENARIO_PARAMS`\\ [scenario]，显式传参优先。**
    也就是说 ``run_scenario("positive")`` 跑的是 ``true_effect=+0.0068``，
    而不是零效果——场景名与生成机制不会静默错配。
    只有 ``n_sims`` 与 ``seed`` 不参与场景定义（它们是模拟规模，不是数据生成机制）。

    机制：干预组真实比例 = ``control_rate + true_effect``，对照组 = ``control_rate``；
    每次重复从两个二项分布各抽一个计数，交给
    :func:`~dsa.stats.proportions.analyze_comparison` 分析，
    再看点估计、报告区间与 p 值是否与已知真值一致。

    Args:
        scenario: 场景名，必须是 :data:`SCENARIO_PARAMS` 里的键。
            未知场景名会抛 ``ValueError``，**不会**退化成默认参数。
        n_sims: 重复次数（蒙特卡洛样本量）。
        n_per_arm: 每组固定样本量。``None`` 表示取场景预设。
        control_rate: 对照组真实比例。``None`` 表示取场景预设。
        true_effect: 真实的**绝对**效果（风险差 RD）。``None`` 表示取场景预设。
        alpha: 名义显著性水平，同时决定区间置信水平 1 - alpha。
        seed: 随机种子，同种子逐位复现。

    Returns:
        SimulationResult。``false_positive_rate`` 仅在 ``true_effect == 0`` 时有意义，
        否则为 nan。``params_source`` 说明参数是场景预设还是被显式覆盖。

    Note:
        ``params_source`` 只看「调用方有没有显式传参」，不做「传的值是否等于预设」的
        比对：显式传 ``true_effect=<场景预设值>`` 也会记为 ``"overridden"``。
        这是一个保守的标注——宁可多标一次，也不要漏标一次。

    Raises:
        ValueError: 场景名未知；n_sims/n_per_arm 非正；比例越界；
            干预组真实比例 ``control_rate + true_effect`` 超出 [0, 1]；alpha 不在 (0, 1) 内。
    """
    if scenario not in SCENARIO_PARAMS:
        raise ValueError(
            f"未知场景 {scenario!r}；可选：{sorted(SCENARIO_PARAMS)}。"
            "未知场景不会被静默当作默认参数处理，请先在 SCENARIOS 中登记。"
        )
    spec = SCENARIO_PARAMS[scenario]

    # 显式传参优先，未传则取场景预设。
    overridden = [
        name
        for name, value in (
            ("n_per_arm", n_per_arm),
            ("control_rate", control_rate),
            ("true_effect", true_effect),
        )
        if value is not None
    ]
    resolved_n_per_arm = spec.n_per_arm if n_per_arm is None else n_per_arm
    resolved_control_rate = spec.control_rate if control_rate is None else control_rate
    resolved_true_effect = spec.true_effect if true_effect is None else true_effect
    params_source = PARAMS_OVERRIDDEN if overridden else PARAMS_FROM_SCENARIO

    if n_sims <= 0:
        raise ValueError(f"n_sims 必须为正整数，实际为 {n_sims!r}")
    if resolved_n_per_arm <= 0:
        raise ValueError(f"n_per_arm 必须为正整数，实际为 {resolved_n_per_arm!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha 必须落在 (0, 1) 内，实际为 {alpha!r}")

    _check_rate(resolved_control_rate, name="control_rate")
    treat_rate = resolved_control_rate + resolved_true_effect
    if math.isnan(treat_rate) or not 0.0 <= treat_rate <= 1.0:
        raise ValueError(
            f"干预组真实比例 control_rate + true_effect = {resolved_control_rate!r} + "
            f"{resolved_true_effect!r} = {treat_rate!r} 超出 [0, 1]，请检查设定"
        )

    rng = np.random.default_rng(seed)
    # 逐次调用 rng.binomial(n_per_arm, rate) 的向量化等价写法：一次抽完 n_sims 个计数，
    # 每个元素与单次标量调用**逐位相同**（已核对）。循环里仍然逐次调用
    # analyze_comparison，保证被检验的是真实调用路径，而不是在模拟里重写的公式。
    treat_counts = rng.binomial(resolved_n_per_arm, treat_rate, size=n_sims)
    control_counts = rng.binomial(resolved_n_per_arm, resolved_control_rate, size=n_sims)

    z = z_for(alpha, 2)
    estimates = np.empty(n_sims, dtype=float)
    half_widths = np.empty(n_sims, dtype=float)
    p_values = np.empty(n_sims, dtype=float)
    covered = 0
    undefined_intervals = 0

    for i in range(n_sims):
        stats = analyze_comparison(
            treatment_k=int(treat_counts[i]),
            treatment_n=resolved_n_per_arm,
            control_k=int(control_counts[i]),
            control_n=resolved_n_per_arm,
            alpha=alpha,
        )
        ci = stats.risk_difference_ci
        estimates[i] = stats.risk_difference
        p_values[i] = stats.test.p_value
        if not ci.defined:
            # 区间无定义（分母为 0）时不计数覆盖，也不计入半宽。
            undefined_intervals += 1
            half_widths[i] = math.nan
            continue
        half_widths[i] = (ci.high - ci.low) / 2.0
        if ci.low <= resolved_true_effect <= ci.high:
            covered += 1

    mean_estimate = float(estimates.mean())
    # 经验标准误：点估计在重复之间的实际波动（ddof=1，样本标准差）。
    empirical_se = float(estimates.std(ddof=1)) if n_sims > 1 else math.nan
    # 程序报告的平均区间半宽换算成一个 z 单位，便于与经验标准误直接对账。
    mean_reported_se = float(np.nanmean(half_widths) / z)
    ci_coverage = covered / n_sims
    if resolved_true_effect == 0.0:
        false_positive_rate = float(np.mean(p_values < alpha))
    else:
        false_positive_rate = math.nan

    assumptions = ASSUMPTIONS
    if undefined_intervals:
        assumptions += f" 本次运行有 {undefined_intervals} 次区间无定义（已排除在覆盖率分子分母之外）。"

    return SimulationResult(
        scenario=scenario,
        n_sims=n_sims,
        n_per_arm=resolved_n_per_arm,
        true_effect=resolved_true_effect,
        true_control_rate=resolved_control_rate,
        mean_estimate=mean_estimate,
        bias=mean_estimate - resolved_true_effect,
        empirical_se=empirical_se,
        mean_reported_se=mean_reported_se,
        ci_coverage=ci_coverage,
        false_positive_rate=false_positive_rate,
        alpha=alpha,
        seed=seed,
        assumptions=assumptions,
        params_source=params_source,
    )


def run_all_scenarios(*, n_sims: int = 2000, seed: int = 20261011) -> list[SimulationResult]:
    """跑完所有预设场景。

    参数由 :func:`run_scenario` 自己按场景名解析，这里**不再展开场景参数**——
    两处各写一份参数正是口径走偏的来源。

    各场景共用同一个 ``seed``，但各自从该种子重新初始化 RNG，
    因此场景之间相互独立、且任一场景可单独复现。

    Args:
        n_sims: 每个场景的重复次数。
        seed: 随机种子。

    Returns:
        与 :data:`SCENARIOS` 同序的结果列表，每项的 ``params_source`` 都是
        ``"scenario_default"``。
    """
    return [run_scenario(s.name, n_sims=n_sims, seed=seed) for s in SCENARIOS]


def summarize(results: list[SimulationResult]) -> str:
    """把结果列表格式化成对齐的文本表，供人工核对与日志粘贴。"""
    header = (
        f"{'场景':<13}{'真实RD':>11}{'估计均值':>13}{'bias':>12}"
        f"{'经验SE':>12}{'报告SE':>12}{'覆盖率':>9}{'假阳性率':>10}  参数来源"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        fpr = "  n/a" if math.isnan(r.false_positive_rate) else f"{r.false_positive_rate:>10.4f}"
        lines.append(
            f"{r.scenario:<13}{r.true_effect:>+11.6f}{r.mean_estimate:>+13.6f}"
            f"{r.bias:>+12.2e}{r.empirical_se:>12.2e}{r.mean_reported_se:>12.2e}"
            f"{r.ci_coverage:>9.4f}{fpr:>10}  {r.params_source}"
        )
    return "\n".join(lines)
