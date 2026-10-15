"""效果区间图：matplotlib 静态 PNG + plotly 交互 HTML。

一张图说清一次两组比较（docs/05 §1.6 要求「至少一张效果与区间图」）：

- 上panel：干预组 / 对照组的比例点位 + Wilson 区间（误差棒），另画一条对照组比例的参照线；
- 下panel：两组之差（风险差 RD）点位 + Newcombe 区间；
- 0 参考线：表示两组无差异（画在下 panel，并标注区间是否跨越它）；
- 每个点位旁标注数值（比例用百分数 4 位小数，差异用百分点）。

设计约束
--------
- 数值**全部取自传入的 ``ComparisonStats``**，本模块不做任何统计计算；
  标注文本复用 ``dsa.report.stats_report`` 的格式化层，保证图上数字与报告表格逐字一致。
- ``out_path`` 由调用方给出，**不硬编码项目根**，也不依赖当前工作目录。
- 中文字体优先 Microsoft YaHei / SimHei，找不到时可回退到 DejaVu Sans，
  **函数不会因此崩溃**；实际选用的字体名会写进图注，便于复核。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

# 本模块只把图写进文件，不需要交互后端；显式指定 Agg，
# 以免在无显示环境（CI / 服务）下因为默认后端不可用而失败。
matplotlib.use("Agg")

from dsa.report.stats_report import (  # noqa: E402  (必须在 matplotlib.use 之后导入 pyplot)
    fmt_interval,
    fmt_pp,
    fmt_proportion_pair,
    fmt_rate,
)

__all__ = ["effect_interval_chart"]

#: 中文字体候选（按优先级）。Windows 上第一个即可命中。
_CJK_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
    "DejaVu Sans",
)

#: 字体配置状态（幂等缓存），键为 'resolved'。
_FONT_STATE: dict[str, str] = {}

_TREATMENT_COLOR = "#1f77b4"
_CONTROL_COLOR = "#ff7f0e"
_DIFF_COLOR = "#d62728"
_GRID_COLOR = "#dddddd"


def _finite(value: Any) -> bool:
    """是否为有限实数（None / NaN / inf 一律 False）。"""
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _setup_cjk_font() -> str:
    """配置中文字体并返回选用的字体名。重复调用幂等。

    - 设置 ``font.sans-serif``（含 Microsoft YaHei / SimHei / DejaVu Sans 回退链）
      与 ``axes.unicode_minus = False``，否则负号会显示成方块；
    - 找不到任何中文字体时**不抛异常**，回退到 DejaVu Sans，
      调用方可从返回值或图注看出当前字体（中文会显示为方块，但图仍生成）。
    """
    from matplotlib import font_manager

    if "resolved" in _FONT_STATE:
        return _FONT_STATE["resolved"]

    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in _CJK_CANDIDATES if name in available), None)

    ordered: list[str] = []
    for name in ([chosen] if chosen else []) + list(_CJK_CANDIDATES):
        if name and name not in ordered:
            ordered.append(name)

    matplotlib.rcParams["font.sans-serif"] = ordered
    matplotlib.rcParams["axes.unicode_minus"] = False
    _FONT_STATE["resolved"] = chosen or "DejaVu Sans（未找到中文字体，中文可能显示为方块）"
    return _FONT_STATE["resolved"]


def _crosses_zero(low: Any, high: Any) -> bool | None:
    """区间是否跨越 0。端点缺失时返回 None（不猜）。"""
    if not (_finite(low) and _finite(high)):
        return None
    return float(low) <= 0.0 <= float(high)


def _rate_row(stats: Any) -> tuple[float, Any, float, Any]:
    """取出 (干预组比例, 干预组区间, 对照组比例, 对照组区间)。"""
    return (
        getattr(stats, "treatment_rate", None),
        getattr(stats, "treatment_ci", None),
        getattr(stats, "control_rate", None),
        getattr(stats, "control_ci", None),
    )


def _interval_bounds(interval: Any) -> tuple[float | None, float | None]:
    """区间上下界；无定义或缺失时返回 (None, None)。"""
    if not bool(getattr(interval, "defined", True)):
        return None, None
    low, high = getattr(interval, "low", None), getattr(interval, "high", None)
    if not (_finite(low) and _finite(high)):
        return None, None
    return float(low), float(high)


def _caption(stats: Any, font_name: str) -> str:
    """图注：方法、α 与阅读边界（全部由入参派生，无写死数值）。

    **手工分行**：matplotlib 的自动换行只在空格处断行，中文长句不会折行，
    单行过长会直接溢出画布被裁掉。因此每行都控制在画布宽度以内（≈78 字符）。
    """
    alpha = getattr(stats, "alpha", None)
    alpha_txt = f"{float(alpha):g}" if _finite(alpha) else "未提供"
    test = getattr(stats, "test", None)
    method = getattr(test, "method", None) or "未提供"
    p_value = getattr(test, "p_value", None)
    p_txt = f"{float(p_value):.4e}" if _finite(p_value) else "不适用"
    return "\n".join(
        (
            "0 参考线表示两组无差异；橙色点线为对照组比例，用作目视参照。",
            "区间是否跨越 0 与假设检验是两种方法，结论不一定完全一致，"
            "判定以报告第 4 章的检验结果为准。",
            f"α = {alpha_txt}；检验方法 {method}；p = {p_txt}。",
            "图中数值与报告表格同源，由程序回填；本图仅呈现统计结果，不构成投放建议。",
            f"字体：{font_name}",
        )
    )


def _draw_matplotlib(
    stats: Any,
    png_path: Path,
    *,
    title: str,
    metric_zh: str,
    dpi: int,
) -> Path:
    """用 matplotlib 画 PNG（无中文字体时降级但不报错）。"""
    import matplotlib.pyplot as plt

    font_name = _setup_cjk_font()
    treatment_rate, treatment_ci, control_rate, control_ci = _rate_row(stats)
    rd = getattr(stats, "risk_difference", None)
    rd_ci = getattr(stats, "risk_difference_ci", None)
    rd_low, rd_high = _interval_bounds(rd_ci)

    fig, (ax_rate, ax_diff) = plt.subplots(
        2,
        1,
        figsize=(9.6, 5.9),
        dpi=dpi,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.30},
        layout="constrained",
    )

    # ---------------- 上 panel：两组比例与 Wilson 区间 ----------------
    x_values: list[float] = [0.0]
    rows = (
        (1.0, "干预组", treatment_rate, treatment_ci, _TREATMENT_COLOR),
        (0.0, "对照组", control_rate, control_ci, _CONTROL_COLOR),
    )
    for y, name, rate, interval, color in rows:
        low, high = _interval_bounds(interval)
        if _finite(rate):
            rate_value = float(rate)
            x_values.append(rate_value * 100)
            if low is not None and high is not None:
                x_values.extend([low * 100, high * 100])
                ax_rate.errorbar(
                    [rate_value * 100],
                    [y],
                    xerr=[[rate_value * 100 - low * 100], [high * 100 - rate_value * 100]],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=1.8,
                    capsize=5,
                    markersize=9,
                    zorder=3,
                )
                label = (
                    f"{name}：{fmt_proportion_pair(rate_value)}  "
                    f"区间 [{fmt_rate(low)}, {fmt_rate(high)}]"
                )
            else:
                ax_rate.plot([rate_value * 100], [y], "o", color=color, markersize=9, zorder=3)
                label = f"{name}：{fmt_proportion_pair(rate_value)}  区间 无定义"
        else:
            label = f"{name}：比例未提供"
        # 数值标注放在**该行上方**并左对齐到坐标区左缘：
        # 既不压住误差棒与点位，也不会被右侧裁剪（x 用轴分数、y 用数据坐标）。
        ax_rate.text(
            0.012,
            y + 0.22,
            label,
            transform=ax_rate.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=9,
            color=color,
        )

    if _finite(control_rate):
        ax_rate.axvline(
            float(control_rate) * 100,
            color=_CONTROL_COLOR,
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            zorder=1,
        )
    ax_rate.set_yticks([1.0, 0.0])
    ax_rate.set_yticklabels(["干预组", "对照组"])
    ax_rate.set_ylim(-0.42, 1.72)
    ax_rate.set_title(
        f"各组比例与区间（{metric_zh}）", fontsize=11, loc="left", color="#333333"
    )
    ax_rate.set_xlabel("比例（百分数，%）", fontsize=9)
    ax_rate.grid(axis="x", color=_GRID_COLOR, linewidth=0.8, zorder=0)
    ax_rate.set_axisbelow(True)

    # ---------------- 下 panel：RD 与 Newcombe 区间 ----------------
    ax_diff.axvline(0.0, color="black", linestyle="--", linewidth=1.3, zorder=1)
    ax_diff.text(
        0.0,
        -0.42,
        "0 参考线（两组无差异）",
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
    )
    if _finite(rd):
        rd_value = float(rd)
        x_values.append(rd_value * 100)
        if rd_low is not None and rd_high is not None:
            x_values.extend([rd_low * 100, rd_high * 100])
            ax_diff.errorbar(
                [rd_value * 100],
                [0.0],
                xerr=[[rd_value * 100 - rd_low * 100], [rd_high * 100 - rd_value * 100]],
                fmt="D",
                color=_DIFF_COLOR,
                ecolor=_DIFF_COLOR,
                elinewidth=2.0,
                capsize=6,
                markersize=9,
                zorder=3,
            )
        else:
            ax_diff.plot([rd_value * 100], [0.0], "D", color=_DIFF_COLOR, markersize=9, zorder=3)
        crosses = _crosses_zero(rd_low, rd_high)
        crosses_txt = (
            "区间跨越 0 参考线"
            if crosses is True
            else ("区间不跨越 0 参考线" if crosses is False else "区间是否跨越 0 无法判定")
        )
        label = (
            f"绝对差异（风险差）{fmt_pp(rd_value)}  "
            f"区间 {fmt_interval(rd_ci, kind='pp')}  {crosses_txt}"
        )
    else:
        label = "绝对差异未提供"
    ax_diff.text(
        0.012,
        0.24,
        label,
        transform=ax_diff.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=9,
        color=_DIFF_COLOR,
    )
    ax_diff.set_yticks([0.0])
    ax_diff.set_yticklabels(["RD"])
    ax_diff.set_ylim(-0.55, 1.05)
    ax_diff.set_title("两组之差（风险差）与区间", fontsize=11, loc="left", color="#333333")
    ax_diff.set_xlabel("差异（百分点，pp）", fontsize=9)
    ax_diff.grid(axis="x", color=_GRID_COLOR, linewidth=0.8, zorder=0)
    ax_diff.set_axisbelow(True)

    # ---------------- 横轴范围：包含全部数值，右侧留少量余量 ----------------
    low_bound = min(x_values)
    high_bound = max(x_values)
    span = high_bound - low_bound if high_bound > low_bound else max(abs(high_bound), 1.0)
    left = low_bound - 0.06 * span
    right = high_bound + 0.12 * span
    ax_rate.set_xlim(left, right)
    ax_diff.set_xlim(left, right)

    fig.suptitle(title or f"{metric_zh}：效果与区间", fontsize=13, x=0.01, ha="left")
    # 图注用 supxlabel：constrained layout 会为它预留空间，避免与坐标轴重叠。
    fig.supxlabel(_caption(stats, font_name), fontsize=8, color="#555555", ha="left", x=0.01)
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return png_path


def _draw_plotly(
    stats: Any,
    html_path: Path,
    *,
    title: str,
    metric_zh: str,
) -> Path:
    """用 plotly 画交互版 HTML（与 PNG 同源、同数值）。"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - 环境未装 plotly 时给出可执行提示
        raise ImportError("生成交互版 HTML 需要 plotly（pip install plotly）") from exc

    treatment_rate, treatment_ci, control_rate, control_ci = _rate_row(stats)
    rd = getattr(stats, "risk_difference", None)
    rd_ci = getattr(stats, "risk_difference_ci", None)
    rd_low, rd_high = _interval_bounds(rd_ci)

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.22,
        subplot_titles=(f"各组比例与区间（{metric_zh}）", "两组之差（风险差）与区间"),
    )

    # ---------------- 上 panel ----------------
    for name, rate, interval, color in (
        ("干预组", treatment_rate, treatment_ci, _TREATMENT_COLOR),
        ("对照组", control_rate, control_ci, _CONTROL_COLOR),
    ):
        if not _finite(rate):
            continue
        rate_value = float(rate)
        low, high = _interval_bounds(interval)
        hover = f"{name}<br>比例 {fmt_proportion_pair(rate_value)}"
        error_x = None
        if low is not None and high is not None:
            error_x = {
                "type": "data",
                "symmetric": False,
                "array": [high * 100 - rate_value * 100],
                "arrayminus": [rate_value * 100 - low * 100],
                "color": color,
                "thickness": 1.8,
                "width": 6,
            }
            hover += f"<br>区间 [{fmt_rate(low)}, {fmt_rate(high)}]"
        else:
            hover += "<br>区间 无定义"
        fig.add_trace(
            go.Scatter(
                x=[rate_value * 100],
                y=[name],
                mode="markers+text",
                text=[f"{fmt_rate(rate_value)}"],
                textposition="top center",
                marker={"size": 13, "color": color},
                error_x=error_x,
                name=name,
                hovertemplate=hover + "<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # ---------------- 下 panel ----------------
    if _finite(rd):
        rd_value = float(rd)
        error_x = None
        hover = f"绝对差异（风险差）{fmt_pp(rd_value)}"
        if rd_low is not None and rd_high is not None:
            error_x = {
                "type": "data",
                "symmetric": False,
                "array": [rd_high * 100 - rd_value * 100],
                "arrayminus": [rd_value * 100 - rd_low * 100],
                "color": _DIFF_COLOR,
                "thickness": 2.0,
                "width": 8,
            }
            crosses = _crosses_zero(rd_low, rd_high)
            crosses_txt = (
                "区间跨越 0 参考线"
                if crosses is True
                else ("区间不跨越 0 参考线" if crosses is False else "区间是否跨越 0 无法判定")
            )
            hover += f"<br>区间 {fmt_interval(rd_ci, kind='pp')}<br>{crosses_txt}"
        fig.add_trace(
            go.Scatter(
                x=[rd_value * 100],
                y=["RD"],
                mode="markers+text",
                text=[f"{fmt_pp(rd_value)}"],
                textposition="top center",
                marker={"size": 13, "color": _DIFF_COLOR, "symbol": "diamond"},
                error_x=error_x,
                name="绝对差异（风险差）",
                hovertemplate=hover + "<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.add_vline(x=0.0, line_dash="dash", line_color="black", row=2, col=1)
    if _finite(control_rate):
        fig.add_vline(
            x=float(control_rate) * 100,
            line_dash="dot",
            line_color=_CONTROL_COLOR,
            row=1,
            col=1,
        )

    rate_values = [float(v) * 100 for v in (treatment_rate, control_rate) if _finite(v)]
    diff_values = [float(v) * 100 for v in (rd_low, rd_high, rd) if _finite(v)]
    candidates = [0.0] + rate_values + diff_values
    low_bound = min(candidates)
    high_bound = max(candidates)
    span = high_bound - low_bound if high_bound > low_bound else max(abs(high_bound), 1.0)
    x_range = [low_bound - 0.10 * span, high_bound + 0.55 * span]

    fig.update_xaxes(title_text="比例（百分数，%）", range=x_range, row=1, col=1)
    fig.update_xaxes(title_text="差异（百分点，pp）", range=x_range, row=2, col=1)
    fig.update_layout(
        title=title or f"{metric_zh}：效果与区间",
        template="plotly_white",
        showlegend=True,
        height=680,
        margin={"l": 90, "r": 40, "t": 90, "b": 80},
        font={
            "family": "Microsoft YaHei, SimHei, Noto Sans CJK SC, sans-serif",
            "size": 12,
        },
    )
    fig.add_annotation(
        text=_caption(stats, "浏览器字体").replace("\n", "<br>"),
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.22,
        showarrow=False,
        align="left",
        font={"size": 10, "color": "#555555"},
    )
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    return html_path


def effect_interval_chart(
    stats: Any,
    out_path: Path,
    *,
    title: str = "",
    metric_zh: str = "购买率",
    dpi: int = 150,
) -> Path:
    """画效果区间图并保存为 PNG，返回路径。同时生成同名的 .html（plotly 交互版）。

    Args:
        stats: ``ComparisonStats`` —— 图上的每个数值都取自该对象，本模块不重算。
        out_path: PNG 输出路径。父目录不存在时创建；**不硬编码项目根**。
            HTML 写在同目录、同主文件名的 ``.html``。
        title: 图标题；留空时用 ``f"{metric_zh}：效果与区间"``。
        metric_zh: 指标中文名，仅用于图标题与轴标题。
        dpi: PNG 分辨率。

    Returns:
        PNG 文件路径（``Path``）。

    Note:
        - 中文字体：设置 ``font.sans-serif`` 为
          ``["Microsoft YaHei", "SimHei", ..., "DejaVu Sans"]`` 且
          ``axes.unicode_minus = False``；找不到中文字体时不抛异常，
          回退 DejaVu Sans 并继续出图。
        - 本函数只写文件，不读数据库、不读配置文件、不修改入参。
    """
    png_path = Path(out_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = png_path.with_suffix(".html")

    _draw_matplotlib(stats, png_path, title=title, metric_zh=metric_zh, dpi=dpi)
    _draw_plotly(stats, html_path, title=title, metric_zh=metric_zh)
    return png_path
