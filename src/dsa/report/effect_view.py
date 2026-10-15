"""把效果载荷渲染成报告的「效果估计」章节。

设计约束
--------
1. **本模块不做任何统计计算，也不做任何单位换算之外的算术。**
   所有数值都来自 `dsa.stats.effect.effect_payload` 的载荷。
   「报告里的数字必须能追到一个唯一的计算点」——所以渲染层只负责排版。

2. **百分点与百分比必须分开写。** 购买率从 0.5726% 升到 1.2531%，
   绝对差异是 **+0.680501 个百分点**，相对提升是 **+118.84%**。
   把前者写成 "0.68%" 或把后者写成 "118.84 个百分点" 都是事实错误，
   而这两种写法在中文营销报告里都极常见。因此：
   - 绝对差异一律用「个百分点」；
   - 相对提升一律用「%」并**显式标注它是相对量**。

3. **阻断时不写因果结论。** 载荷里的 `blocked` 为真时，
   只呈现描述性数字并明确声明不得据此判断有效性。
"""

from __future__ import annotations

from typing import Any

#: 数值小数位。取 6 位是为了让报告里的数字与结构化载荷**逐位可核对**：
#: 独立统计脚本打印的 `+0.680501` 与本报告文本一致，不需要读者猜四舍五入。
_ND = 6


def _pct(value: Any) -> str:
    """比例 → 百分数文本（1.253109%）。"""
    return "—" if value is None else f"{value * 100:.{_ND}f}%"


def _pp(value: Any, *, signed: bool = True) -> str:
    """原尺度差异 → **百分点**文本（+0.680501）。"""
    if value is None:
        return "—"
    return f"{value * 100:+.{_ND}f}" if signed else f"{value * 100:.{_ND}f}"


def _num(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.{_ND}f}" if signed else f"{value:.{_ND}f}"


def _ratio(value: Any) -> str:
    return "—" if value is None else f"{value:.4f}"


def _lift(value: Any) -> str:
    return "—" if value is None else f"{value * 100:+.2f}%"


def render_effect_section(payload: dict[str, Any]) -> str:
    """效果载荷 → Markdown 章节。

    Args:
        payload: `dsa.stats.effect.effect_payload` 的返回值，或工具
            `analyze_experiment_effect` 的 `data`（两者结构相同）。

    Returns:
        Markdown 文本。`effects` 为空时返回空串 —— 让调用方决定怎么降级，
        而不是在这里编一段没有数字的"结论"。
    """
    effects = payload.get("effects") or []
    if not effects:
        return ""

    treat = payload.get("treatment_arm") or payload.get("comparison_key") or "干预组"
    ctrl = payload.get("control_arm") or "对照组"
    alpha = payload.get("alpha")
    window = payload.get("window_days")
    blocked = bool(payload.get("blocked"))

    lines: list[str] = []
    lines.append("## 效果估计\n")
    lines.append(
        f"比较对象：**{treat}**（干预） vs **{ctrl}**（对照）"
        + (f"　｜　结果窗口：实验后 **{window}** 天" if window else "")
        + (f"　｜　显著性水平 α = {alpha:g}" if isinstance(alpha, (int, float)) else "")
        + "\n"
    )

    if blocked:
        codes = "、".join(payload.get("blocking_codes") or []) or "数据质量检查未通过"
        lines.append(
            f"> ⛔ **本次分析被阻断（{codes}），下列数字仅作描述，不给出因果结论。**\n"
        )

    for e in effects:
        kind = e.get("kind")
        name = e.get("metric_name_zh") or e.get("metric")
        lines.append(f"### {name}\n")
        from dsa.stats.effect import metric_meta
        role=metric_meta().get(e.get("metric"),{}).get("role")
        primary=payload.get("comparison_kind") in {"confirmatory","primary"} and role=="primary"
        lines.append("结果角色：**主要结果**。" if primary else
                     "结果角色：**探索性结果**；名义区间与p值不能作为多重比较后的确证证据。")
        if kind=="continuous":
            lines.append("单位：原数据货币单位／被分配客户；原资料未明确币种。")
        difference=e.get("difference")
        if not blocked and difference is not None:
            direction="高于" if difference>0 else "低于" if difference<0 else "等于"
            lines.append(f"点估计方向：{treat} {name}{direction} {ctrl}；是否显著以已登记检验为准。")


        if kind == "proportion":
            lines.append("| 项 | 数值 |")
            lines.append("|---|---|")
            lines.append(
                f"| {treat} {name} | **{_pct(e.get('treatment_value'))}** "
                f"（{e.get('treatment_k'):,} / {e.get('treatment_n'):,}） |")
            lines.append(
                f"| {ctrl} {name} | **{_pct(e.get('control_value'))}** "
                f"（{e.get('control_k'):,} / {e.get('control_n'):,}） |")
            lines.append(
                f"| **绝对差异（{treat} - {ctrl}）** | "
                f"**{_pp(e.get('difference'))} 个百分点** |")
            lo, hi = e.get("ci_low"), e.get("ci_high")
            if lo is None or hi is None:
                lines.append(
                    f"| {int(round((e.get('ci_level') or 0.95) * 100))}% 置信区间 | "
                    "**无定义**（该方法在当前数据下不可用） |")
            else:
                lines.append(
                    f"| {int(round((e.get('ci_level') or 0.95) * 100))}% 置信区间"
                    f"（{e.get('ci_method')}） | "
                    f"**[{_pp(lo)}, {_pp(hi)}] 个百分点** |")
            if e.get("ratio_defined"):
                lines.append(
                    f"| 风险比 RR（{e.get('ratio_method')}） | "
                    f"{_ratio(e.get('ratio'))} "
                    f"[{_ratio(e.get('ratio_ci_low'))}, {_ratio(e.get('ratio_ci_high'))}] |")
                lines.append(f"| 相对提升（相对量，非百分点） | {_lift(e.get('relative_lift'))} |")
            else:
                lines.append("| 风险比 RR | 无定义（某组计数为 0），只报告绝对差异 |")
            lines.append(
                f"| 检验 | {e.get('test_method')}，p = "
                f"{_pval(e.get('p_value'))}"
                + ("（已改用 Fisher 精确检验）" if e.get("used_fisher") else "")
                + " |")
            lines.append(
                f"| 结论（α = {e.get('alpha'):g}） | "
                + ("**差异有统计显著性**" if e.get("significant")
                   else "**未检出统计显著差异**")
                + " |")
        else:  # continuous
            lines.append("| 项 | 数值 |")
            lines.append("|---|---|")
            lines.append(
                f"| {treat} 人均 | **{_num(e.get('treatment_value'))}** "
                f"（n = {e.get('treatment_n'):,}，零值占比 "
                f"{_pct(e.get('treatment_zero_share'))}） |")
            lines.append(
                f"| {ctrl} 人均 | **{_num(e.get('control_value'))}** "
                f"（n = {e.get('control_n'):,}，零值占比 "
                f"{_pct(e.get('control_zero_share'))}） |")
            lines.append(
                f"| **绝对差异（{treat} - {ctrl}）** | "
                f"**{_num(e.get('difference'), signed=True)}** |")
            lines.append(
                f"| {int(round((e.get('ci_level') or 0.95) * 100))}% 置信区间"
                f"（{e.get('ci_method')}） | "
                f"**[{_num(e.get('ci_low'), signed=True)}, "
                f"{_num(e.get('ci_high'), signed=True)}]** |")
            lines.append(
                f"| 检验 | {e.get('test_method')}，p = {_pval(e.get('p_value'))}"
                f"（bootstrap 种子 {e.get('bootstrap_seed')}） |")
            lines.append(
                f"| 结论（α = {e.get('alpha'):g}） | "
                + ("**差异有统计显著性**" if e.get("significant")
                   else "**未检出统计显著差异**")
                + " |")
        lines.append("")

    lines.append(
        "> **口径**："
        + str(payload.get("denominator_note") or "分母为全部被随机分配的客户数。")
        + "\n"
    )
    if blocked:
        lines.append(
            "> 上表数字**不得**用于判断邮件是否有效 —— 阻断原因使「干预效果」"
            "与「人群差异」无法区分。\n"
        )
    lines.append("> 局限：历史随机实验的回顾性结果，不保证未来投放收益；没有成本数据。")
    return "\n".join(lines)


def _pval(value: Any) -> str:
    """p 值文本。极小值用科学计数法，不写成 0。"""
    if value is None:
        return "—"
    if value == 0:
        return "0（小于双精度下限）"
    if value < 1e-4:
        return f"{value:.4e}"
    return f"{value:.6g}"


__all__ = ["render_effect_section"]
