"""比例推断：区间估计与假设检验。

口径依据：docs/02-分析协议.md §12。

方法选择说明（协议 §12.1 已记录理由）
--------------------------------------
购买率约为 0.6%~1.3%，属**小比例**。此场景下：

- **不采用 Wald 区间**：小比例 + 有限样本时覆盖率不足，且区间可能越界（出现负数或 >1）。
- 单比例区间用 **Wilson score interval**（`wilson_interval`）。
- 比例之差用 **Newcombe hybrid score interval**（`newcombe_diff_interval`），
  由两个 Wilson 区间组合而成（Newcombe 1998 方法 10）。
- 风险比用 **Katz 对数法**（`katz_rr_interval`），在计数为 0 时返回无定义而非 inf。

假设检验用卡方（**不加连续性校正**）；当任一格**期望频数 < 5** 时改用 Fisher 精确检验。

所有函数均为纯函数，无随机性、无全局状态，因此结果可复现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

#: 标准正态分位数缓存（避免重复计算）
_Z_CACHE: dict[float, float] = {}


def z_for(alpha: float, sided: int = 2) -> float:
    """给定显著性水平返回临界 z 值。

    Args:
        alpha: 显著性水平（双侧总错误率）。
        sided: 1 或 2。

    Examples:
        >>> round(z_for(0.05, 2), 4)
        1.96
    """
    key = (alpha, sided)
    if key not in _Z_CACHE:
        _Z_CACHE[key] = float(stats.norm.ppf(1 - alpha / sided))
    return _Z_CACHE[key]


@dataclass(frozen=True, slots=True)
class Interval:
    """一个置信区间。`defined=False` 表示该方法在当前数据下无定义。"""

    low: float
    high: float
    level: float
    method: str
    defined: bool = True

    def __str__(self) -> str:
        if not self.defined:
            return f"{self.method}: 无定义"
        return f"{self.method}: [{self.low:.8f}, {self.high:.8f}]"


# ============================================================================
#  单比例
# ============================================================================


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval（单比例）。

    相比 Wald 区间，它在小比例、小样本下覆盖率更好，且**永不越界**。

    Args:
        k: 成功计数。
        n: 总计数（分母）。
        alpha: 显著性水平，双侧。

    Returns:
        区间。n = 0 时返回 `defined=False`。
    """
    level = 1 - alpha
    if n <= 0:
        return Interval(math.nan, math.nan, level, "wilson", defined=False)

    z = z_for(alpha, 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(max(0.0, centre - half), min(1.0, centre + half), level, "wilson")


def wald_interval(k: int, n: int, alpha: float = 0.05) -> Interval:
    """Wald 区间。**仅用于对照演示其缺陷，不作为报告方法。**

    协议 §12.1 明确不采用它。
    """
    level = 1 - alpha
    if n <= 0:
        return Interval(math.nan, math.nan, level, "wald", defined=False)
    z = z_for(alpha, 2)
    p = k / n
    half = z * math.sqrt(p * (1 - p) / n)
    return Interval(p - half, p + half, level, "wald")


# ============================================================================
#  两个比例之差
# ============================================================================


def newcombe_diff_interval(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05) -> Interval:
    """Newcombe hybrid score interval（比例之差 p1 - p2）。

    构造方式（Newcombe 1998, method 10）：

        diff  = p1 - p2
        lower = diff - sqrt((p1 - l1)^2 + (u2 - p2)^2)
        upper = diff + sqrt((u1 - p1)^2 + (p2 - l2)^2)

    其中 (l1, u1) 是组 1 的 Wilson 区间，(l2, u2) 是组 2 的 Wilson 区间。

    注意区间**不强制包含 0**——这正是判断差异是否显著的依据之一。

    Args:
        k1, n1: 组 1（干预组）的成功数与分母。
        k2, n2: 组 2（对照组）的成功数与分母。
        alpha: 显著性水平，双侧。
    """
    level = 1 - alpha
    if n1 <= 0 or n2 <= 0:
        return Interval(math.nan, math.nan, level, "newcombe", defined=False)

    p1, p2 = k1 / n1, k2 / n2
    w1 = wilson_interval(k1, n1, alpha)
    w2 = wilson_interval(k2, n2, alpha)
    diff = p1 - p2

    lower = diff - math.sqrt((p1 - w1.low) ** 2 + (w2.high - p2) ** 2)
    upper = diff + math.sqrt((w1.high - p1) ** 2 + (p2 - w2.low) ** 2)
    # 比例之差的理论取值范围
    return Interval(max(-1.0, lower), min(1.0, upper), level, "newcombe")


# ============================================================================
#  风险比
# ============================================================================


def katz_rr_interval(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05) -> Interval:
    """风险比 RR = p1 / p2 的 Katz 对数法区间。

    SE(log RR) = sqrt(1/k1 - 1/n1 + 1/k2 - 1/n2)

    **边界处理**（协议 §8.3）：当任一组的成功计数为 0 时，
    RR 的对数无定义，返回 `defined=False`，调用方应改为只报告绝对差异。
    """
    level = 1 - alpha
    if n1 <= 0 or n2 <= 0 or k1 <= 0 or k2 <= 0:
        return Interval(math.nan, math.nan, level, "katz", defined=False)

    p1, p2 = k1 / n1, k2 / n2
    rr = p1 / p2
    se = math.sqrt(1 / k1 - 1 / n1 + 1 / k2 - 1 / n2)
    z = z_for(alpha, 2)
    return Interval(rr * math.exp(-z * se), rr * math.exp(z * se), level, "katz")


# ============================================================================
#  假设检验
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProportionTest:
    """两个独立比例的检验结果。"""

    method: str
    statistic: float
    p_value: float
    #: 2x2 表中最小期望频数。用于判断卡方是否适用。
    min_expected: float
    #: 本次是否因期望频数不足而改用 Fisher
    used_fisher: bool

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def two_proportion_test(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    *,
    min_expected_threshold: float = 5.0,
) -> ProportionTest:
    """比较两个独立比例。卡方（无连续性校正），必要时自动改用 Fisher 精确检验。

    Args:
        k1, n1: 组 1 成功数与分母。
        k2, n2: 组 2 成功数与分母。
        min_expected_threshold: 期望频数阈值，低于它改用 Fisher。

    Returns:
        ProportionTest

    Note:
        协议 §12.1 规定卡方**不加连续性校正**。是否改用 Fisher 由
        **期望频数**（而非观测频数）决定，这是更稳妥的判据。
    """
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    total = n1 + n2
    if total == 0:
        return ProportionTest("undefined", math.nan, math.nan, math.nan, False)

    # 期望频数
    row_totals = [n1, n2]
    col_totals = [k1 + k2, (n1 - k1) + (n2 - k2)]
    expected = [[r * c / total for c in col_totals] for r in row_totals]
    min_expected = min(min(row) for row in expected)

    if min_expected < min_expected_threshold:
        odds, p = stats.fisher_exact(table)
        return ProportionTest("fisher_exact", float(odds), float(p), min_expected, True)

    chi2, p, _dof, _exp = stats.chi2_contingency(table, correction=False)
    return ProportionTest("chi2_no_correction", float(chi2), float(p), min_expected, False)


# ============================================================================
#  汇总：一次给出一个比较的全部统计量
# ============================================================================


@dataclass(frozen=True, slots=True)
class ComparisonStats:
    """一次两组比较的完整统计结果。"""

    # 输入
    treatment_n: int
    treatment_k: int
    control_n: int
    control_k: int
    alpha: float
    metric: str

    # 组内比例与区间
    treatment_rate: float
    control_rate: float
    treatment_ci: Interval
    control_ci: Interval

    # 效果量
    risk_difference: float
    risk_difference_ci: Interval
    risk_ratio: float
    risk_ratio_ci: Interval
    relative_lift: float | None

    # 检验
    test: ProportionTest

    @property
    def significant(self) -> bool:
        return self.test.significant(self.alpha)

    def summary(self) -> str:
        lines = [
            f"指标            : {self.metric}",
            f"干预组          : {self.treatment_k:,} / {self.treatment_n:,} = {self.treatment_rate:.6%}",
            f"  区间          : {self.treatment_ci}",
            f"对照组          : {self.control_k:,} / {self.control_n:,} = {self.control_rate:.6%}",
            f"  区间          : {self.control_ci}",
            f"绝对差异 (RD)   : {self.risk_difference:+.6%}   {self.risk_difference_ci}",
        ]
        if self.risk_ratio_ci.defined:
            lines.append(f"风险比 (RR)     : {self.risk_ratio:.4f}   {self.risk_ratio_ci}")
            if self.relative_lift is not None:
                lines.append(f"相对提升        : {self.relative_lift:+.2%}")
        else:
            lines.append("风险比 (RR)     : 无定义（对照组或干预组计数为 0），只报告绝对差异")
        lines.append(
            f"检验            : {self.test.method}  p = {self.test.p_value:.6g}  "
            f"最小期望频数 = {self.test.min_expected:.2f}"
            f"{'  (已改用 Fisher)' if self.test.used_fisher else ''}"
        )
        lines.append(
            f"结论(α={self.alpha:g})   : "
            + ("差异有统计显著性" if self.significant else "未检出统计显著差异")
        )
        return "\n".join(lines)


def analyze_comparison(
    *,
    treatment_k: int,
    treatment_n: int,
    control_k: int,
    control_n: int,
    alpha: float = 0.05,
    metric: str = "conversion_rate",
) -> ComparisonStats:
    """对一个两组比较做完整分析。

    **分母由调用方提供，且必须是「全部被随机分配到该组的客户数」。**
    本函数不做任何筛选，因此不可能把干预后行为引入分母。
    """
    t_rate = treatment_k / treatment_n if treatment_n else math.nan
    c_rate = control_k / control_n if control_n else math.nan
    rd = t_rate - c_rate

    rr = t_rate / c_rate if c_rate > 0 else math.nan
    lift = (t_rate - c_rate) / c_rate if c_rate > 0 else None

    return ComparisonStats(
        treatment_n=treatment_n,
        treatment_k=treatment_k,
        control_n=control_n,
        control_k=control_k,
        alpha=alpha,
        metric=metric,
        treatment_rate=t_rate,
        control_rate=c_rate,
        treatment_ci=wilson_interval(treatment_k, treatment_n, alpha),
        control_ci=wilson_interval(control_k, control_n, alpha),
        risk_difference=rd,
        risk_difference_ci=newcombe_diff_interval(treatment_k, treatment_n, control_k, control_n, alpha),
        risk_ratio=rr,
        risk_ratio_ci=katz_rr_interval(treatment_k, treatment_n, control_k, control_n, alpha),
        relative_lift=lift,
        test=two_proportion_test(treatment_k, treatment_n, control_k, control_n),
    )
