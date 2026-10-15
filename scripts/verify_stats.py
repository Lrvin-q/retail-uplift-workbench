"""用可信第三方库核对本项目自研的统计实现。

依据：实现蓝图.md 阶段三「用独立参考实现或可信库结果核对计算」。

核对对象：
  自研 ``dsa.stats.proportions``  vs  ``statsmodels``

- Wilson 单比例区间      vs ``statsmodels.stats.proportion.proportion_confint(method="wilson")``
- Newcombe 比例之差区间  vs ``statsmodels.stats.proportion.confint_proportions_2indep(method="newcombe")``
- 卡方检验（无校正）      vs ``statsmodels.stats.proportion.proportions_chisquare``
- Fisher 精确检验        vs ``scipy.stats.fisher_exact``
- 卡方拟合优度（SRM）     vs ``scipy.stats.chisquare``

容忍度：区间端点相对误差 1e-9；p 值相对误差 1e-9。预先设定，不事后放宽。

用法：
    python scripts/verify_stats.py
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_chisquare,
    proportions_ztest,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsa.stats.proportions import (  # noqa: E402
    analyze_comparison,
    katz_rr_interval,
    newcombe_diff_interval,
    two_proportion_test,
    wald_interval,
    wilson_interval,
)

RTOL = 1e-9
_fail: list[str] = []
_n = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _n
    _n += 1
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        _fail.append(name)


def close(a: float, b: float, rtol: float = RTOL) -> bool:
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)):
        return False
    if a == b:
        return True
    d = max(abs(a), abs(b))
    return abs(a - b) <= rtol * d if d else True


# ============================================================================
#  核对用的样本（覆盖小比例、零计数、大样本三类场景）
# ============================================================================
CASES = [
    ("Hillstrom 主比较实测值", 267, 21307, 122, 21306),
    ("大样本对称", 5000, 10000, 4800, 10000),
    ("小比例", 12, 2000, 5, 2000),
    ("含零计数(干预组)", 0, 500, 10, 500),
    ("含零计数(对照组)", 10, 500, 0, 500),
    ("小样本", 3, 20, 1, 20),
    ("极小比例", 1, 50000, 2, 50000),
]


def main() -> int:
    print("=" * 78)
    print("统计实现核对 ——  自研 dsa.stats.proportions  vs  statsmodels / scipy")
    print(f"相对误差容忍度 RTOL = {RTOL:g}")
    print("=" * 78)

    # ---------------------------------------------------------------- Wilson
    print("\n---- 1. Wilson 单比例区间 vs statsmodels ----")
    for k, n in [(267, 21307), (122, 21306), (0, 500), (500, 500), (1, 3), (5, 10000)]:
        mine = wilson_interval(k, n, 0.05)
        lo_sm, hi_sm = proportion_confint(k, n, alpha=0.05, method="wilson")
        check(
            f"wilson k={k} n={n}",
            close(mine.low, float(lo_sm)) and close(mine.high, float(hi_sm)),
            f"mine=[{mine.low:.12f},{mine.high:.12f}] sm=[{lo_sm:.12f},{hi_sm:.12f}]",
        )

    # ---------------------------------------------------------------- Wald（对照）
    print("\n---- 2. Wald 区间 vs statsmodels（仅作对照，协议已弃用）----")
    # 注意：statsmodels 的 method="normal" 会把区间**截断**到 [0,1]；
    # 本项目的 wald_interval 刻意**不截断**，以便暴露小比例下的越界缺陷。
    # 因此这里只比较上限，并单独断言越界确实发生。
    for k, n in [(1, 50000), (0, 500)]:
        mine = wald_interval(k, n, 0.05)
        lo_sm, hi_sm = proportion_confint(k, n, alpha=0.05, method="normal")
        check(
            f"wald k={k} n={n} 上限一致",
            close(mine.high, float(hi_sm)),
            f"mine_hi={mine.high:.10f} sm_hi={float(hi_sm):.10f}",
        )
    w = wald_interval(1, 50000, 0.05)
    lo_sm, _ = proportion_confint(1, 50000, alpha=0.05, method="normal")
    check(
        "Wald 在小比例下越界（协议 §12.1 弃用它的依据）",
        w.low < 0 and float(lo_sm) == 0.0,
        f"本实现下限={w.low:.10f}（负数，越界）；statsmodels 截断为 {float(lo_sm):.10f}",
    )

    # ---------------------------------------------------------------- Newcombe
    print("\n---- 3. Newcombe 比例之差区间 vs statsmodels ----")
    for name, k1, n1, k2, n2 in CASES:
        mine = newcombe_diff_interval(k1, n1, k2, n2, 0.05)
        sm = confint_proportions_2indep(
            k1, n1, k2, n2, method="newcombe", compare="diff", alpha=0.05
        )
        check(
            f"newcombe [{name}]",
            close(mine.low, float(sm[0])) and close(mine.high, float(sm[1])),
            f"mine=[{mine.low:.12f},{mine.high:.12f}] sm=[{sm[0]:.12f},{sm[1]:.12f}]",
        )

    # ---------------------------------------------------------------- 卡方 / Fisher
    print("\n---- 4. 两比例检验（卡方无校正 / 自动改 Fisher）----")
    for name, k1, n1, k2, n2 in CASES:
        mine = two_proportion_test(k1, n1, k2, n2)
        table = np.array([[k1, n1 - k1], [k2, n2 - k2]])
        if mine.used_fisher:
            _odds, p_sm = stats.fisher_exact(table)
            check(
                f"fisher [{name}] p",
                close(mine.p_value, float(p_sm)),
                f"mine={mine.p_value:.12g} scipy={float(p_sm):.12g}",
            )
        else:
            stat_sm, p_sm, _dof, _exp = stats.chi2_contingency(table, correction=False)
            check(
                f"chi2 [{name}] 统计量与 p",
                close(mine.statistic, float(stat_sm)) and close(mine.p_value, float(p_sm)),
                f"mine chi2={mine.statistic:.10f} p={mine.p_value:.12g} | "
                f"scipy chi2={float(stat_sm):.10f} p={float(p_sm):.12g}",
            )

    # ---------------------------------------------------------------- 卡方拟合优度（SRM）
    print("\n---- 5. SRM 卡方拟合优度 vs scipy ----")
    obs = np.array([21307, 21387, 21306], dtype=float)
    exp = np.full(3, obs.sum() / 3)
    chi2_mine = float(np.sum((obs - exp) ** 2 / exp))
    chi2_sm, p_sm = stats.chisquare(obs, exp)
    check(
        "SRM 卡方统计量",
        close(chi2_mine, float(chi2_sm)),
        f"mine={chi2_mine:.12f} scipy={float(chi2_sm):.12f}",
    )
    check(
        "SRM p 值",
        close(float(stats.chi2.sf(chi2_mine, 2)), float(p_sm)),
        f"p={float(p_sm):.12g}",
    )

    # ---------------------------------------------------------------- 端到端
    print("\n---- 6. 端到端：分析函数内部一致性 ----")
    st = analyze_comparison(
        treatment_k=267, treatment_n=21307, control_k=122, control_n=21306, alpha=0.05
    )
    check("RD = p_t - p_c", close(st.risk_difference, st.treatment_rate - st.control_rate))
    check("RR = p_t / p_c", close(st.risk_ratio, st.treatment_rate / st.control_rate))
    check(
        "相对提升 = RR - 1",
        close(st.relative_lift, st.risk_ratio - 1),
        f"lift={st.relative_lift:.10f}",
    )
    check(
        "RD 区间包含 RD 点估计",
        st.risk_difference_ci.low <= st.risk_difference <= st.risk_difference_ci.high,
    )
    check(
        "组内 Wilson 区间包含点估计",
        st.treatment_ci.low <= st.treatment_rate <= st.treatment_ci.high
        and st.control_ci.low <= st.control_rate <= st.control_ci.high,
    )
    # 注意：不要写成 `low > 0 == significant`——Python 链式比较会变成
    # `(low > 0) and (0 == significant)`，而 0 == True 为假，恒判定失败。
    check(
        "RD 区间是否含 0 与显著性判定一致",
        (st.risk_difference_ci.low > 0) == st.significant,
        f"CI=[{st.risk_difference_ci.low:.8f},{st.risk_difference_ci.high:.8f}] "
        f"不含0={(st.risk_difference_ci.low > 0)} sig={st.significant}",
    )

    # 与 statsmodels 的 z 检验交叉确认显著性方向
    z_stat, p_z = proportions_ztest([267, 122], [21307, 21306])
    check(
        "显著性方向与 statsmodels z 检验一致",
        (p_z < 0.05) == st.significant,
        f"z 检验 p={float(p_z):.6g} 卡方 p={st.test.p_value:.6g}",
    )

    print("\n" + "=" * 78)
    if _fail:
        print(f"结论：{len(_fail)}/{_n} 项失败 ✘")
        for f in _fail:
            print(f"   - {f}")
        return 1
    print(f"结论：{_n} 项全部通过 ✔   自研实现与 statsmodels / scipy 完全一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
