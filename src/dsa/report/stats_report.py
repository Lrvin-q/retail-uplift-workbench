"""固定模板统计报告：数值回填、阻断规则与措辞约束。

依据
----
- ``docs/02-分析协议.md`` §13（效果解释规则与措辞禁令）、§15（报告必填信息）
- ``docs/05-报告结构与异常分类.md`` §1（七章结构）、§2（数值回填规则）、§4（阻断规则）

三条硬性约束（违反即返工）
--------------------------
1. **数值只来自传入的结构化结果对象。** 本模块不做任何统计计算，只做格式化与模板
   渲染。报告里出现的每个数字都能追溯到 ``ComparisonStats`` / ``CheckResult`` 的
   某个字段，或由入参直接派生的计数（指标数、检查项数、阻断项数、比较数）。
2. **措辞禁令**（协议 §13.2）。结论由**固定模板分支**产出，不做自由文本拼装；
   渲染结束前执行一次程序自检 ``find_forbidden_wording``，程序生成的文本命中禁用
   措辞即抛错、不落盘（对应 ``E_REPORT_001`` 的处置原则）。外部透传文本（检查项
   的 ``name`` / ``detail``）按追溯要求原样保留，若命中则在报告末尾显式标注。
3. **结论分四段**（协议 §13.1）：统计证据 / 效果大小 / 业务价值 / 数据可信度。
   没有成本数据时**不得**声称净收益或 ROI。

阻断规则（docs/05 §4）
----------------------
``checks`` 中存在 ``severity == "blocking"`` 且 ``passed == False`` 的检查项时：

- 报告开头输出「⛔ 已阻断」区块并列出阻断原因；
- 第 6 章不输出任何因果结论，改为固定句
  「因数据质量问题，本报告不提供因果结论；以下为描述性结果供排查」；
- 已算出的效果估计仍保留在第 4 章供排查，但逐处标注「前提不成立，不作为结论」。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "FORBIDDEN_PHRASES",
    "METRIC_LABELS_ZH",
    "METRIC_NOTES_ZH",
    "METRIC_ROLES_ZH",
    "SEVERITY_ZH",
    "find_forbidden_wording",
    "fmt_count",
    "fmt_interval",
    "fmt_number",
    "fmt_p",
    "fmt_pp",
    "fmt_proportion_pair",
    "fmt_rate",
    "fmt_rr",
    "fmt_signed",
    "metric_label_zh",
    "render_stats_report",
    "severity_zh",
]

# ============================================================================
#  0. 登记表与常量
# ============================================================================

#: 指标标识 → 中文名。镜像 ``docs/metrics.yaml`` 的 ``name_zh``
#: （``tests/test_stage3_report.py`` 有一致性断言，防止与指标字典漂移）。
#: 未登记的标识不猜名字，回退为标识本身并在报告中显式说明。
METRIC_LABELS_ZH: dict[str, str] = {
    "conversion_rate": "购买率",
    "visit_rate": "访问率",
    "spend_per_customer": "人均消费",
    "spend_per_buyer": "购买者人均消费",
}

#: 指标标识 → 角色（取值来自 docs/metrics.yaml 的 ``role``）。
METRIC_ROLES_ZH: dict[str, str] = {
    "conversion_rate": "主要指标（primary）",
    "visit_rate": "次要指标（secondary）",
    "spend_per_customer": "次要指标（secondary）",
    "spend_per_buyer": "仅描述（descriptive_only）",
}

#: 指标标识 → 就地标注（口径边界，逐字取自 docs/metrics.yaml 的 constraints）。
METRIC_NOTES_ZH: dict[str, str] = {
    "conversion_rate": "分母为全部被随机分配到该组的客户数，不得按 visit 或任何干预后行为过滤。",
    "visit_rate": "分母为全部被随机分配到该组的客户数。",
    "spend_per_customer": (
        "分母包含未购买客户（spend = 0），否则会高估；"
        "消费分布高度零膨胀，必须与零值占比同时报告（协议 §12.2）。"
    ),
    "spend_per_buyer": "条件于干预后结果，仅作描述，不得用于因果结论。",
}

#: 连续型指标的零膨胀解读提示（协议 §12.2：不得只报均值）。
_ZERO_INFLATION_NOTE = (
    "**零值占比必须与均值同时阅读**：本指标分母为全部被随机分配到该组的客户"
    "（含消费为 0 的客户）。只报均值会让人误以为该均值是「普遍客户的消费水平」，"
    "而实际分布高度零膨胀——均值差异主要由少数购买者驱动。"
)

#: 严重级别 → 中文。兼容英文（``blocking`` / ``warning`` / ``info``）与中文写法。
SEVERITY_ZH: dict[str, str] = {
    "blocking": "阻断",
    "block": "阻断",
    "阻断": "阻断",
    "warning": "警告",
    "warn": "警告",
    "警告": "警告",
    "info": "提示",
    "notice": "提示",
    "提示": "提示",
}

#: 禁用措辞清单（协议 §13.2）。生成文本命中任何一条即判定报告失败。
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "没有效果",
    "证明有效",
    "所以应当投放",
    "应当立即投放",
    "不显著，说明",
    "显著，所以",
)

#: 阻断时禁止出现的因果/决策断言措辞（docs/05 §4.5）。
#: 这些词只在**非阻断**分支的模板里出现，阻断分支走独立模板。
_CAUSAL_WORDS: tuple[str, ...] = (
    "邮件有效",
    "有效果",
    "提高了",
    "降低了",
    "应当投放",
    "建议投放",
    "证明",
)

_NA = "不适用"
_NOT_PROVIDED = "未提供"


# ============================================================================
#  1. 格式化层（全报告唯一的数字显示入口，docs/05 §2.1「显示精度由格式化层统一处理」）
# ============================================================================


def _finite(value: Any) -> bool:
    """判断是否为有限实数（None / NaN / inf / 非数值一律为 False）。"""
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt_rate(value: Any) -> str:
    """比例 → 百分数，保留 4 位小数（任务约定）。

    Examples:
        >>> fmt_rate(0.0128)
        '1.2800%'
        >>> fmt_rate(float("nan"))
        '不适用'
    """
    if not _finite(value):
        return _NA
    return f"{float(value):.4%}"


def fmt_proportion_pair(value: Any, *, decimals: int = 6) -> str:
    """比例 → 「小数（百分数）」两种表示。

    docs/05 §1.6 要求比例类字段同时给出小数与百分比，且二者必须由**同一数值**格式化
    得到——本函数是唯一实现，避免两处各自取整产生不一致。

    Examples:
        >>> fmt_proportion_pair(0.0128)
        '0.012800（1.2800%）'
    """
    if not _finite(value):
        return _NA
    return f"{float(value):.{decimals}f}（{fmt_rate(value)}）"


def _pp_number(value: Any) -> str:
    """比例差 → 百分点数值文本（带符号，4 位小数）。"""
    if not _finite(value):
        return _NA
    return f"{float(value) * 100:+.4f}"


def fmt_pp(value: Any) -> str:
    """比例差（risk_difference）→ 百分点，带符号保留 4 位小数。

    Examples:
        >>> fmt_pp(0.0071)
        '+0.7100 个百分点'
    """
    if not _finite(value):
        return _NA
    return f"{_pp_number(value)} 个百分点"


def fmt_p(value: Any) -> str:
    """p 值 → 科学计数法（4 位小数）。下溢至 0 时如实标注，不写成 0。"""
    if not _finite(value):
        return _NA
    value = float(value)
    if value == 0.0:
        return "< 1e-300（浮点下溢，非真实为 0）"
    return f"{value:.4e}"


def fmt_rr(value: Any) -> str:
    """风险比 / 区间端点 → 4 位小数。"""
    if not _finite(value):
        return _NA
    return f"{float(value):.4f}"


def fmt_number(value: Any, *, decimals: int = 4) -> str:
    """一般数值（检验统计量、最小期望频数、检查项 value）→ 定点小数。"""
    if not _finite(value):
        return _NOT_PROVIDED
    return f"{float(value):.{decimals}f}"


def fmt_count(value: Any) -> str:
    """计数 → 千分位整数。"""
    if not _finite(value):
        return _NOT_PROVIDED
    return f"{int(round(float(value))):,}"


def severity_zh(severity: Any) -> str:
    """严重级别 → 中文。无法识别时原样返回，不吞掉信息。"""
    if severity is None:
        return _NOT_PROVIDED
    text = str(severity).strip()
    return SEVERITY_ZH.get(text.lower(), text)


def metric_label_zh(metric_id: str) -> str:
    """指标标识 → 中文名。未登记时返回「未登记中文名」，不猜、不编造。"""
    return METRIC_LABELS_ZH.get(metric_id, "未登记中文名")


def fmt_signed(value: Any, *, decimals: int = 4) -> str:
    """一般数值 → 带符号定点小数（连续型指标的均值与差异）。

    Examples:
        >>> fmt_signed(0.7698)
        '+0.7698'
    """
    if not _finite(value):
        return _NA
    return f"{float(value):+.{decimals}f}"


def fmt_interval(interval: Any, *, kind: str = "rate") -> str:
    """置信区间 → 文本。``kind`` 取 ``rate`` / ``pp`` / ``rr``。

    无定义（``defined = False``）时输出「无定义」并保留方法与水平，
    与「真实计算得到 0」区分开（docs/05 §2.1）。
    """
    method = getattr(interval, "method", None)
    level = getattr(interval, "level", None)
    defined = bool(getattr(interval, "defined", True))
    method_txt = str(method) if method else "方法未提供"
    level_txt = f"{float(level):.0%}" if _finite(level) else "水平未提供"

    low = getattr(interval, "low", None)
    high = getattr(interval, "high", None)
    if not defined or not (_finite(low) and _finite(high)):
        return f"无定义（{method_txt}，{level_txt}）"

    if kind == "pp":
        body = f"[{_pp_number(low)}, {_pp_number(high)}] 个百分点"
    elif kind == "rr":
        body = f"[{fmt_rr(low)}, {fmt_rr(high)}]"
    else:
        body = f"[{fmt_rate(low)}, {fmt_rate(high)}]"
    return f"{body}（{method_txt}，{level_txt}）"


def _bounds_cross_zero(low: Any, high: Any) -> bool | None:
    """区间上下界是否跨越 0。任一端缺失时返回 None（**不猜**）。"""
    if not (_finite(low) and _finite(high)):
        return None
    return float(low) <= 0.0 <= float(high)


def _interval_crosses_zero(interval: Any) -> bool | None:
    """``Interval`` 对象是否跨越 0。无定义时返回 None（**不猜**）。"""
    if not bool(getattr(interval, "defined", True)):
        return None
    return _bounds_cross_zero(getattr(interval, "low", None), getattr(interval, "high", None))


def find_forbidden_wording(text: str) -> list[tuple[str, str]]:
    """扫描禁用措辞（协议 §13.2）。

    Args:
        text: 待检查文本。

    Returns:
        ``[(命中的禁用措辞, 所在行)]``；空列表表示通过。

    Note:
        命中的行会截断到 80 字符，便于写入运行记录与失败明细。
    """
    hits: list[tuple[str, str]] = []
    for line in text.splitlines():
        for phrase in FORBIDDEN_PHRASES:
            if phrase in line:
                hits.append((phrase, line.strip()[:80]))
    return hits


# ============================================================================
#  2. 行缓冲：区分程序生成文本与外部透传文本
# ============================================================================


@dataclass
class _Doc:
    """行缓冲。

    分开记录「程序生成」与「外部透传」的行，使措辞自检可以：
    程序生成文本命中禁用措辞 → 直接失败（不发布报告）；
    外部透传文本命中 → 原样保留（追溯要求）但在报告末尾显式标注。
    """

    _lines: list[str] = field(default_factory=list)
    _external: list[str] = field(default_factory=list)

    def add(self, *lines: str) -> None:
        self._lines.extend(lines)

    def extend(self, lines: Iterable[str]) -> None:
        self._lines.extend(lines)

    def add_external(self, *lines: str) -> None:
        for line in lines:
            self._lines.append(line)
            self._external.append(line)

    @property
    def text(self) -> str:
        return "\n".join(self._lines)

    @property
    def own_text(self) -> str:
        external = set(self._external)
        return "\n".join(line for line in self._lines if line not in external)


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    """构造 Markdown 表格（转义竖线与换行，避免破坏表格结构）。"""
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
        out.append("| " + " | ".join(cells) + " |")
    return out


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """从对象或映射读取字段（``CheckResult`` 由其他模块提供，只读不改）。"""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ============================================================================
#  3. 输入上下文
# ============================================================================


@dataclass(frozen=True)
class _Context:
    """渲染所需的全部输入（**唯一数据来源**）。"""

    run_id: str
    experiment_key: str
    dataset_version: str
    comparison_key: str
    stats_by_metric: dict[str, Any]
    continuous_by_metric: dict[str, Any]
    checks: list[Any]
    window_days: int
    generated_at: str
    chart_paths: dict[str, Path]
    blocked: bool
    blocking_checks: list[Any]
    failed_checks: list[Any]
    warning_checks: list[Any]
    alphas: list[float]
    #: 组别别名。**只由调用方显式传入**（``treatment_label`` / ``control_label``），
    #: 绝不从统计结果对象推断；为 None 时报告中不出现任何臂别名（保持既有行为）。
    treatment_label: str | None = None
    control_label: str | None = None

    @property
    def metric_ids(self) -> list[str]:
        return list(self.stats_by_metric)

    @property
    def continuous_ids(self) -> list[str]:
        return list(self.continuous_by_metric)

    def alpha_text(self) -> str:
        """α 文本。多个指标 α 不一致时逐个列出，不取其中一个代表全体。"""
        if not self.alphas:
            return _NOT_PROVIDED
        return "、".join(f"{a:g}" for a in self.alphas)

    def check_codes(self, items: Sequence[Any]) -> str:
        codes = [str(_field(c, "code", "未提供检查码")) for c in items]
        return "、".join(f"`{c}`" for c in codes) if codes else "无"


# ============================================================================
#  4. 章节构建
# ============================================================================

#: 阻断时的固定句（docs/05 §4.5 指定措辞，逐字使用）。
_BLOCKED_SENTENCE = "因数据质量问题，本报告不提供因果结论；以下为描述性结果供排查"

#: docs/05 §1.3 要求的第 2 章必填字段中，本函数入参**未覆盖**的部分。
#: 按 §2.1「缺失与为零必须区分」，此处如实列出而不是留空或补默认值。
_MISSING_SOURCE_FIELDS: tuple[str, ...] = (
    "数据集标识 `dataset_id`、来源说明与文件哈希 `file_hash`",
    "数据版本行数 `dataset_row_count`、数据取得时间 `data_retrieved_at`",
    "字段映射版本 `field_mapping_version`",
    "分析对象定义（纳入与排除规则）与 `analysis_population`",
    "纳入分析的随机分配客户总数 `n_assigned_total`（本报告按指标分别给出 k/n）",
    "预期分配比例 `expected_ratios` 及其来源说明 `expected_ratios_source`",
    "观察到的组别比例 `observed_ratio`",
    "实际执行与设计口径的偏差 `distance_from_design`",
)


def _arm_title(base: str, label: str | None) -> str:
    """组别行标题：有显式别名时写成「干预组（Mens E-Mail）」，否则只用 base。

    ``label`` 只可能来自调用方显式传入的参数——本模块**不从统计结果对象推断组别**。
    """
    return f"{base}（{label}）" if label else base


def _failed_flag(value: Any) -> bool:
    """``passed`` 判定：只有**明确为假**才算未通过；``None`` 视为未判定，不算失败。"""
    if value is None:
        return False
    if isinstance(value, bool):
        return not value
    return value == 0


def _format_check_value(value: Any) -> str:
    """检查项统计量 → 文本。缺失如实写「未提供」，不写 0（docs/05 §2.1）。"""
    if value is None:
        return "未提供（缺失不写 0）"
    return fmt_number(value)


def _code_version() -> str:
    """尽力取得代码版本（git 短哈希）。

    只读项目根的 ``.git`` 目录，不启动子进程、不修改任何状态；
    取不到时如实写「未提供」，不编造版本号（AGENTS.md §2.7）。
    """
    try:
        root = Path(__file__).resolve().parents[3]
        head = root / ".git" / "HEAD"
        if not head.is_file():
            return "未提供（未找到 .git/HEAD）"
        content = head.read_text(encoding="utf-8").strip()
        if not content.startswith("ref:"):
            return content[:12]
        ref = content.split(" ", 1)[1].strip()
        ref_file = root / ".git" / ref
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip()[:12]
        packed = root / ".git" / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith(("#", "^")) and line.endswith(" " + ref):
                    return line.split(" ", 1)[0][:12]
        return "未提供（HEAD 指向的引用无法解析）"
    except OSError:
        return "未提供（读取版本信息失败）"


def _package_version() -> str:
    """已安装包版本；以源码路径运行时如实写「未安装」。"""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - 标准库始终可用
        return _NOT_PROVIDED
    try:
        return version("dsa")
    except PackageNotFoundError:
        return "未安装（以源码路径运行）"


def _write_header(doc: _Doc, ctx: _Context) -> None:
    status = "已阻断（BLOCKED）" if ctx.blocked else "正常（NORMAL）"
    causal = (
        "否（存在阻断级异常，由阻断规则程序判定；docs/05 §4.1）"
        if ctx.blocked
        else "是（本次未检出阻断级异常；由阻断规则程序判定）"
    )
    doc.add(f"# 统计报告 · {ctx.experiment_key} / {ctx.comparison_key}", "")
    doc.extend(
        _md_table(
            ["项目", "内容"],
            [
                ("run_id", f"`{ctx.run_id}`"),
                ("实验标识 experiment_key", f"`{ctx.experiment_key}`"),
                ("比较标识 comparison_key", f"`{ctx.comparison_key}`"),
                ("数据版本 dataset_version", f"`{ctx.dataset_version}`"),
                ("结果窗口", f"实验后 {ctx.window_days} 天（固定窗口）"),
                ("显著性水平 α", ctx.alpha_text()),
                ("生成时间", ctx.generated_at),
                ("报告状态", f"**{status}**"),
                ("可输出因果结论", causal),
                ("本报告传入指标数", f"{len(ctx.stats_by_metric)} 个比例型 + {len(ctx.continuous_by_metric)} 个连续型"),
                ("本报告传入检查项数", str(len(ctx.checks))),
            ],
        )
    )
    doc.add("")


def _write_block_banner(doc: _Doc, ctx: _Context) -> None:
    doc.add(
        "---",
        "",
        "## ⛔ 已阻断",
        "",
        f"> **报告状态：已阻断。** 本次运行检出 {len(ctx.blocking_checks)} 项阻断级异常"
        "（`severity = blocking` 且 `passed = false`）。",
        ">",
        "> 停止输出因果结论；描述性结果与诊断信息保留，供排查使用。",
        "> 阻断**不等于**运行失败（docs/05 §4.1），但不得通过重试工具、放宽阈值或更换比较对象绕过。",
        "",
        "### 阻断原因",
        "",
    )
    rows = [
        (
            f"`{_field(c, 'code', '未提供检查码')}`",
            str(_field(c, "name", "未提供检查名")),
            severity_zh(_field(c, "severity")),
            _format_check_value(_field(c, "value")),
            str(_field(c, "detail", "未提供说明")),
        )
        for c in ctx.blocking_checks
    ]
    table = _md_table(["检查码", "名称", "级别", "检查值", "说明"], rows)
    doc.extend(table[:2])
    doc.add_external(*table[2:])
    doc.add(
        "",
        f"> {_BLOCKED_SENTENCE}",
        "",
        "本次仍保留的描述性范围：各组样本量、比例、区间与检验数值（逐处标注「前提不成立，不作为结论」）、"
        "数据质量检查明细与追溯信息；**不含**任何因果结论或投放建议。",
        "需要补充的信息见第 3 章与第 6 章。",
        "",
        "---",
        "",
    )


def _write_ch1_plan(doc: _Doc, ctx: _Context) -> None:
    doc.add("## 1. 分析计划", "", "### 1.1 比较对象", "")
    if ctx.treatment_label or ctx.control_label:
        # 组别别名只在调用方显式传入时出现；缺哪一侧就如实写「未提供」。
        arm_rows: list[tuple[str, str]] = [
            (
                "干预组",
                ctx.treatment_label
                or "未提供（组别取值只由显式参数 `treatment_label` 传入，本模块不代为推断）",
            ),
            (
                "对照组",
                ctx.control_label
                or "未提供（组别取值只由显式参数 `control_label` 传入，本模块不代为推断）",
            ),
            (
                "组别取值来源",
                "调用方显式传入的 `treatment_label` / `control_label`；"
                "本模块**不从统计结果对象推断组别**。",
            ),
        ]
    else:
        arm_rows = [
            (
                "干预组 / 对照组",
                "由传入的结构化结果对象定义（treatment 侧 / control 侧）。"
                "组别取值未在本次输入中提供，故不在此复述，也不代为推断。",
            )
        ]
    doc.extend(
        _md_table(
            ["字段", "内容"],
            [
                ("比较标识", f"`{ctx.comparison_key}`"),
                *arm_rows,
                (
                    "随机化单位 / 分析单位",
                    "客户（本条为 docs/02 §6 的协议口径说明；本次入参未包含单位字段，"
                    "不作为本次输入事实）",
                ),
                (
                    "比较数",
                    "1（本函数一次调用只接受一个 `comparison_key`，比较数由调用结构决定；"
                    "该值不是统计结果，报告不据此推断其他比较）",
                ),
            ],
        )
    )
    doc.add("", "### 1.2 分析指标", "")
    metric_rows = [
        (
            f"`{mid}`",
            metric_label_zh(mid),
            METRIC_ROLES_ZH.get(mid, "未登记（docs/03-指标字典.md 中未找到该标识）"),
            METRIC_NOTES_ZH.get(mid, "未登记（本报告不代为定义口径）"),
        )
        for mid in ctx.metric_ids
    ]
    continuous_rows = [
        (
            f"`{mid}`",
            metric_label_zh(mid),
            METRIC_ROLES_ZH.get(mid, "未登记（docs/03-指标字典.md 中未找到该标识）"),
            METRIC_NOTES_ZH.get(mid, "未登记（本报告不代为定义口径）"),
        )
        for mid in ctx.continuous_ids
    ]
    if metric_rows or continuous_rows:
        doc.extend(
            _md_table(["指标标识", "中文名", "角色", "口径边界"], metric_rows + continuous_rows)
        )
    else:
        doc.add("本次调用未传入任何指标的统计结果（`stats_by_metric` 与 `continuous_by_metric` 均为空）。")
    doc.add("", "### 1.3 结果窗口与显著性水平", "")
    doc.extend(
        _md_table(
            ["字段", "内容"],
            [
                (
                    "结果窗口",
                    f"实验后 {ctx.window_days} 天（固定窗口）。数据无逐次事件时间线，"
                    "不支持任意日期窗口与按天监控（协议 §7）。",
                ),
                (
                    "显著性水平 α",
                    f"{ctx.alpha_text()}（双侧；取自统计结果对象，不取记忆值）",
                ),
                (
                    "多重比较说明",
                    "本次报告只包含 1 个比较。协议 §5.2：主要比较与次要（探索性）比较同时报告时，"
                    "必须说明多重比较问题并给出 Bonferroni 校正后的判定"
                    "（校正水平 = α / 比较数，由程序按输入的 α 与比较数计算；"
                    "本报告不代入未经确认的比较数）。**不得只挑显著的比较或指标汇报**。",
                ),
                (
                    "本计划不支持的内容",
                    "任意日期窗口、按干预后行为筛选人群（如仅看访问者）、中途查看后的上线/停止决策、"
                    "净收益与 ROI（无成本数据）（协议 §14）",
                ),
            ],
        )
    )
    doc.add("")


def _write_ch2_source(doc: _Doc, ctx: _Context) -> None:
    doc.add("## 2. 数据来源与版本", "")
    doc.extend(
        _md_table(
            ["字段", "内容"],
            [
                ("数据版本 dataset_version", f"`{ctx.dataset_version}`"),
                ("运行标识 run_id", f"`{ctx.run_id}`"),
                ("实验标识 experiment_key", f"`{ctx.experiment_key}`"),
                ("比较标识 comparison_key", f"`{ctx.comparison_key}`"),
                ("结果窗口", f"实验后 {ctx.window_days} 天（固定窗口）"),
                (
                    "指标定义来源",
                    "指标标识与中文名镜像 docs/metrics.yaml 与 docs/03-指标字典.md；"
                    "统计口径见 docs/02 §8、§12",
                ),
            ],
        )
    )
    doc.add(
        "",
        "> `dataset_version` 与 `run_id` 为**原样透传**：本模块不对其做解析、拼接、截断或推断"
        "（协议 §15、docs/05 §2.3）。",
        "",
        "本次调用**未提供**的字段（按 docs/05 §1.3 属第 2 章必填项；此处如实列出，"
        "不填默认值、不用其他字段代替）：",
        "",
    )
    doc.extend([f"- {item}" for item in _MISSING_SOURCE_FIELDS])
    doc.add("")


def _write_ch3_checks(doc: _Doc, ctx: _Context) -> None:
    doc.add("## 3. 数据质量检查与阻断项", "")
    if not ctx.checks:
        doc.add(
            "本次调用**未提供任何检查项**（`checks` 为空）。**未执行检查不等于检查通过**："
            "数据可信度在本报告中未经确认（见第 6 章 6.4）。",
            "",
        )
    else:
        rows = []
        for check in ctx.checks:
            passed = _field(check, "passed")
            passed_txt = "通过" if passed is True else ("未通过" if _failed_flag(passed) else "未判定")
            rows.append(
                (
                    f"`{_field(check, 'code', '未提供检查码')}`",
                    str(_field(check, "name", "未提供检查名")),
                    severity_zh(_field(check, "severity")),
                    passed_txt,
                    _format_check_value(_field(check, "value")),
                    str(_field(check, "detail", "未提供说明")),
                )
            )
        table = _md_table(["检查码", "名称", "级别", "是否通过", "检查值", "说明"], rows)
        doc.extend(table[:2])
        doc.add_external(*table[2:])
        doc.add("")

    doc.extend(
        _md_table(
            ["汇总项", "内容"],
            [
                ("检查项合计", f"{len(ctx.checks)} 项"),
                ("未通过项", f"{len(ctx.failed_checks)} 项"),
                ("阻断级未通过项", f"{len(ctx.blocking_checks)} 项"),
                ("警告级未通过项", f"{len(ctx.warning_checks)} 项"),
                (
                    "causal_conclusion_allowed",
                    "否（存在阻断级异常）"
                    if ctx.blocked
                    else "是（本次未检出阻断级异常；由阻断规则程序判定，不由文本判断）",
                ),
                ("阻断级异常清单", ctx.check_codes(ctx.blocking_checks)),
                ("警告级异常清单", ctx.check_codes(ctx.warning_checks)),
            ],
        )
    )
    if ctx.blocked:
        doc.add(
            "",
            "阻断时的描述性输出范围（docs/05 §4.5）：保留计数、比例、区间与检验数值并逐处标注"
            "「前提不成立，不作为结论」，不含因果结论与投放建议。",
        )
    if any("SRM" in str(_field(c, "code", "")).upper() for c in ctx.checks):
        doc.add(
            "",
            "SRM 检查的解释边界（协议 §11.3，强制原样体现）：",
            "",
            "- 未通过 ⇒ **只提示需要诊断，不自动断定异常原因**（可能是分流、日志、导出等多种原因）；",
            "- 通过 ⇒ **不能据此认为随机化完全正确**，只说明样本比例与预期一致，"
            "无法排除其他维度的失衡。",
        )
    doc.add("")


def _significance_cell(st: Any, *, blocked: bool) -> str:
    """显著性判定单元格。**判定只取自结构化结果对象，不由本模块重算。**"""
    alpha = getattr(st, "alpha", None)
    alpha_txt = f"{float(alpha):g}" if _finite(alpha) else _NOT_PROVIDED
    if blocked:
        return "前提不成立，不作判定（阻断时不输出因果结论）"
    test = getattr(st, "test", None)
    p_value = getattr(test, "p_value", None)
    if not _finite(p_value):
        return "无法判定：结构化结果未提供有效 p 值（缺失与为 0 必须区分，不作替代推断）"
    significant = getattr(st, "significant", None)
    if not isinstance(significant, bool):
        return "无法判定：结构化结果未提供 significant 判定（本模块不自行用 p 与 α 比较代替）"
    if significant:
        return f"差异在 α={alpha_txt} 下显著（判定取自结构化结果的 significant 字段）"
    return "未检出统计显著差异；在现有样本量下无法排除小效应存在"


def _direction_cell(st: Any, *, blocked: bool) -> str:
    """效应方向单元格。方向只由 ``risk_difference`` 的符号决定（docs/05 §2.2 步骤 5）。"""
    if blocked:
        return "前提不成立，不作方向判定"
    rd = getattr(st, "risk_difference", None)
    if not _finite(rd):
        return _NOT_PROVIDED
    rd = float(rd)
    if rd > 0:
        return "干预组比例高于对照组（方向由 risk_difference 符号决定）"
    if rd < 0:
        return "干预组比例低于对照组（方向由 risk_difference 符号决定）"
    return "两组比例相等（risk_difference = 0）"


def _metric_rows(mid: str, st: Any, ctx: _Context) -> list[tuple[str, str]]:
    """一个指标一张表的所有字段行。每个数值都取自 ``st``。"""
    test = getattr(st, "test", None)
    risk_difference = getattr(st, "risk_difference", None)
    risk_ratio = getattr(st, "risk_ratio", None)
    lift = getattr(st, "relative_lift", None)

    if not _finite(risk_ratio):
        rr_cell = "不适用（相对变化在当前数据下无定义；按协议 §8.3 只报告绝对差异）"
    else:
        rr_cell = fmt_rr(risk_ratio)

    if lift is None:
        lift_cell = "不适用（结构化结果未提供该值，不写成 0）"
    elif not _finite(lift):
        lift_cell = "不适用（相对变化无定义）"
    else:
        lift_cell = f"{float(lift):+.4%}（定义为风险比 − 1，辅助展示）"

    used_fisher = getattr(test, "used_fisher", None)
    if used_fisher is True:
        fisher_cell = "是（本次输入的期望频数触发了改用条件）"
    elif used_fisher is False:
        fisher_cell = "否（本次输入的期望频数未触发改用条件）"
    else:
        fisher_cell = _NOT_PROVIDED

    treatment_title = _arm_title("干预组", ctx.treatment_label)
    control_title = _arm_title("对照组", ctx.control_label)
    return [
        ("指标标识", f"`{mid}`"),
        ("指标中文名", metric_label_zh(mid)),
        ("指标角色", METRIC_ROLES_ZH.get(mid, "未登记（docs/03-指标字典.md 中未找到该标识）")),
        ("口径边界", METRIC_NOTES_ZH.get(mid, "未登记（本报告不代为定义口径）")),
        ("结构化结果自述指标", f"`{getattr(st, 'metric', None) or _NOT_PROVIDED}`"),
        (
            f"{treatment_title} k / n",
            f"{fmt_count(getattr(st, 'treatment_k', None))} / {fmt_count(getattr(st, 'treatment_n', None))}",
        ),
        (f"{treatment_title}比例", fmt_proportion_pair(getattr(st, "treatment_rate", None))),
        (
            f"{treatment_title}比例区间",
            fmt_interval(getattr(st, "treatment_ci", None), kind="rate"),
        ),
        (
            f"{control_title} k / n",
            f"{fmt_count(getattr(st, 'control_k', None))} / {fmt_count(getattr(st, 'control_n', None))}",
        ),
        (f"{control_title}比例", fmt_proportion_pair(getattr(st, "control_rate", None))),
        (f"{control_title}比例区间", fmt_interval(getattr(st, "control_ci", None), kind="rate")),
        ("绝对差异（风险差 RD）", fmt_pp(risk_difference)),
        ("绝对差异区间", fmt_interval(getattr(st, "risk_difference_ci", None), kind="pp")),
        ("风险比 RR", rr_cell),
        ("风险比区间", fmt_interval(getattr(st, "risk_ratio_ci", None), kind="rr")),
        ("相对提升 relative_lift", lift_cell),
        ("检验方法", str(getattr(test, "method", None) or _NOT_PROVIDED)),
        ("检验统计量", fmt_number(getattr(test, "statistic", None))),
        ("p 值", fmt_p(getattr(test, "p_value", None))),
        ("最小期望频数", fmt_number(getattr(test, "min_expected", None))),
        ("是否改用 Fisher 精确检验", fisher_cell),
        ("显著性判定", _significance_cell(st, blocked=ctx.blocked)),
        ("效应方向", _direction_cell(st, blocked=ctx.blocked)),
    ]


def _continuous_significance_cell(cs: Any, *, blocked: bool) -> str:
    """连续型指标的显著性判定（同样只取自结构化结果对象）。"""
    alpha = getattr(cs, "alpha", None)
    alpha_txt = f"{float(alpha):g}" if _finite(alpha) else _NOT_PROVIDED
    if blocked:
        return "前提不成立，不作判定（阻断时不输出因果结论）"
    p_value = getattr(cs, "p_value", None)
    if not _finite(p_value):
        return "无法判定：结构化结果未提供有效 p 值（缺失与为 0 必须区分，不作替代推断）"
    significant = getattr(cs, "significant", None)
    if not isinstance(significant, bool):
        return "无法判定：结构化结果未提供 significant 判定（本模块不自行用 p 与 α 比较代替）"
    if significant:
        return f"差异在 α={alpha_txt} 下显著（判定取自结构化结果的 significant 字段）"
    return "未检出统计显著差异；在现有样本量下无法排除小效应存在"


def _continuous_ci_text(cs: Any) -> str:
    """连续型指标的区间文本（bootstrap 百分位区间；无端点时如实写「不适用」）。"""
    alpha = getattr(cs, "alpha", None)
    level_txt = f"{1 - float(alpha):.0%}" if _finite(alpha) else "水平未提供"
    ci_low = getattr(cs, "ci_low", None)
    ci_high = getattr(cs, "ci_high", None)
    if not (_finite(ci_low) and _finite(ci_high)):
        return "不适用（结构化结果未提供有效的区间端点，不写成 0）"
    method = getattr(cs, "method", None) or _NOT_PROVIDED
    return (
        f"[{fmt_signed(ci_low)}, {fmt_signed(ci_high)}]（{level_txt}，方法 `{method}`；"
        "置信水平由输入的 α 按 1 − α 计算）"
    )


def _continuous_rows(mid: str, cs: Any, ctx: _Context) -> list[tuple[str, str]]:
    """一个连续型指标一张表的所有字段行。每个数值都取自 ``cs``。"""
    seed = getattr(cs, "bootstrap_seed", None)
    seed_cell = f"`{int(seed)}`（已固定，bootstrap 可复现）" if _finite(seed) else _NOT_PROVIDED

    difference = getattr(cs, "difference", None)
    if ctx.blocked:
        direction_cell = "前提不成立，不作方向判定"
    elif not _finite(difference):
        direction_cell = _NOT_PROVIDED
    elif float(difference) > 0:
        direction_cell = "干预组均值高于对照组（方向由 difference 符号决定）"
    elif float(difference) < 0:
        direction_cell = "干预组均值低于对照组（方向由 difference 符号决定）"
    else:
        direction_cell = "两组均值相等（difference = 0）"

    treatment_n = fmt_count(getattr(cs, "treatment_n", None))
    control_n = fmt_count(getattr(cs, "control_n", None))
    treatment_title = _arm_title("干预组", ctx.treatment_label)
    control_title = _arm_title("对照组", ctx.control_label)
    return [
        ("指标标识", f"`{mid}`"),
        ("指标中文名", metric_label_zh(mid)),
        ("指标角色", METRIC_ROLES_ZH.get(mid, "未登记（docs/03-指标字典.md 中未找到该标识）")),
        (
            "口径边界",
            METRIC_NOTES_ZH.get(mid, "未登记（本报告不代为定义口径）")
            + " 金额类指标的货币单位未在本次入参中提供，故不代为标注货币。",
        ),
        ("指标类型", "连续型（均值差异），非比例型；下方比例型指标的检验方法不适用于本指标"),
        ("结构化结果自述指标", f"`{getattr(cs, 'metric', None) or _NOT_PROVIDED}`"),
        (
            f"{treatment_title}均值（样本量）",
            f"{fmt_number(getattr(cs, 'treatment_mean', None))}（n = {treatment_n}）",
        ),
        (
            f"{control_title}均值（样本量）",
            f"{fmt_number(getattr(cs, 'control_mean', None))}（n = {control_n}）",
        ),
        ("绝对差异（干预组 − 对照组）", fmt_signed(difference)),
        ("绝对差异区间", _continuous_ci_text(cs)),
        ("检验方法", str(getattr(cs, "method", None) or _NOT_PROVIDED)),
        ("p 值", fmt_p(getattr(cs, "p_value", None))),
        ("显著性判定", _continuous_significance_cell(cs, blocked=ctx.blocked)),
        ("效应方向", direction_cell),
        (
            f"{treatment_title}零值占比",
            fmt_proportion_pair(getattr(cs, "treatment_zero_share", None)),
        ),
        (
            f"{control_title}零值占比",
            fmt_proportion_pair(getattr(cs, "control_zero_share", None)),
        ),
        ("零值占比解读提示", _ZERO_INFLATION_NOTE),
        ("bootstrap 随机种子", seed_cell),
    ]


def _write_ch4_stats(doc: _Doc, ctx: _Context) -> None:
    doc.add("## 4. 统计结果", "")
    if ctx.blocked:
        doc.add(
            "> ⚠️ **前提不成立**：以下统计量的计算前提已被第 3 章的阻断级异常破坏，"
            "仅作排查参考，**不得**作为结论或决策依据（docs/05 §4.5）。",
            "",
        )
    if not ctx.stats_by_metric and not ctx.continuous_by_metric:
        doc.add(
            "本次调用未传入任何指标的统计结果（`stats_by_metric` 与 `continuous_by_metric` 均为空）。",
            "",
        )
        return
    doc.add(
        "本章每个数值都取自传入的结构化结果对象（比例型 `ComparisonStats`、"
        "连续型 `ContinuousStats`）；本模块不做任何统计计算，也不补默认值。",
        "",
    )
    index = 0
    for mid in ctx.metric_ids:
        index += 1
        st = ctx.stats_by_metric[mid]
        doc.add(f"### 4.{index} `{mid}` {metric_label_zh(mid)}（比例型指标）", "")
        doc.extend(_md_table(["字段", "数值"], _metric_rows(mid, st, ctx)))
        doc.add("")
    for mid in ctx.continuous_ids:
        index += 1
        cs = ctx.continuous_by_metric[mid]
        doc.add(f"### 4.{index} `{mid}` {metric_label_zh(mid)}（连续型指标）", "")
        doc.extend(_md_table(["字段", "数值"], _continuous_rows(mid, cs, ctx)))
        doc.add("")


def _write_ch5_charts(doc: _Doc, ctx: _Context) -> None:
    doc.add("## 5. 图表引用", "")
    if not ctx.chart_paths:
        doc.add(
            "本次调用**未提供图表路径**（`chart_paths` 为空）。"
            "docs/05 §1.6 要求至少一张效果与区间图，该必填项在本报告中未满足。",
            "",
        )
    else:
        rows = [
            (
                f"`{mid}`",
                metric_label_zh(mid),
                f"`{path}`",
                "存在" if path.exists() else "**缺失（引用无法定位）**",
            )
            for mid, path in ctx.chart_paths.items()
        ]
        doc.extend(_md_table(["指标标识", "中文名", "产物路径", "文件状态"], rows))
        doc.add(
            "",
            "> 图中的数值与本报告第 4 章同源（同一个结构化结果对象），由程序生成，"
            "图中标注与表格一致。",
            "> 图中 0 参考线表示两组无差异；区间方法与检验方法不同，两者结论不一定完全一致，"
            "判定以第 4 章的检验结果为准。",
            "> 图表仅呈现统计结果，不构成投放建议。",
            "",
        )
    metric_list = "、".join(f"`{m}`" for m in ctx.metric_ids + ctx.continuous_ids) or "（空）"
    doc.add(
        f"本报告覆盖的指标：{metric_list}"
        f"（比例型 {len(ctx.metric_ids)} 个、连续型 {len(ctx.continuous_by_metric)} 个）。"
        "**报告全部传入指标，不按显著性筛选**（协议 §13.2）。"
        "比例型与连续型指标的区间与检验方法不同，两者不可直接比较，"
        "各自的判定以第 4 章的对应表格为准。",
        "",
    )
    if not ctx.continuous_by_metric:
        doc.add(
            "> 本次调用未传入连续型指标的统计结果；"
            "若实验登记中包含此类指标（如人均消费），其数值不在本报告内。",
            "",
        )


def _evidence_bullet(mid: str, st: Any) -> str:
    """统计证据段的一条指标结论（非阻断分支）。"""
    label = metric_label_zh(mid)
    alpha = getattr(st, "alpha", None)
    alpha_txt = f"{float(alpha):g}" if _finite(alpha) else _NOT_PROVIDED
    test = getattr(st, "test", None)
    method = str(getattr(test, "method", None) or "方法未提供")
    statistic = fmt_number(getattr(test, "statistic", None))
    p_value = fmt_p(getattr(test, "p_value", None))
    ci_txt = fmt_interval(getattr(st, "risk_difference_ci", None), kind="pp")
    crosses = _interval_crosses_zero(getattr(st, "risk_difference_ci", None))
    crosses_txt = (
        "区间跨越 0" if crosses is True else ("区间不跨越 0" if crosses is False else "区间是否跨越 0 无法判定")
    )
    significant = getattr(st, "significant", None)

    if not isinstance(significant, bool):
        return (
            f"- `{mid}`（{label}）：本次输入未提供结构化的显著性判定，"
            f"因此本报告不陈述显著性结论（检验方法 {method}，统计量 = {statistic}，"
            f"p = {p_value}；绝对差异区间 {ci_txt}，{crosses_txt}）。"
        )
    if significant:
        return (
            f"- `{mid}`（{label}）：差异在 α={alpha_txt}（双侧）下**有统计显著性**"
            f"（{method}，统计量 = {statistic}，p = {p_value}；绝对差异区间 {ci_txt}，"
            f"{crosses_txt}）。该判定只说明差异超出随机波动可解释的范围，"
            "不等同于差异具有业务价值。"
        )
    return (
        f"- `{mid}`（{label}）：**未检出统计显著差异**；在现有样本量"
        f"（干预组 {fmt_count(getattr(st, 'treatment_n', None))}、"
        f"对照组 {fmt_count(getattr(st, 'control_n', None))}）下无法排除小效应存在"
        f"（{method}，统计量 = {statistic}，p = {p_value}；绝对差异区间 {ci_txt}，"
        f"{crosses_txt}）。"
    )


def _effect_bullet(mid: str, st: Any, ctx: _Context) -> list[str]:
    """效果大小段的一条指标结论：绝对差异与相对变化**同时**给出（协议 §13.2）。"""
    label = metric_label_zh(mid)
    rd = getattr(st, "risk_difference", None)
    lines = [
        f"- `{mid}`（{label}）：在实验后 {ctx.window_days} 天窗口内，"
        f"两组比例的绝对差异（风险差）= {fmt_pp(rd)}，区间 "
        f"{fmt_interval(getattr(st, 'risk_difference_ci', None), kind='pp')}；"
        f"相对变化：风险比 RR = {fmt_rr(getattr(st, 'risk_ratio', None))}，"
        f"区间 {fmt_interval(getattr(st, 'risk_ratio_ci', None), kind='rr')}，"
        f"相对提升 = {_lift_text(getattr(st, 'relative_lift', None))}。"
    ]
    if not _finite(getattr(st, "risk_ratio", None)):
        lines.append(
            "  - 相对变化在当前数据下无定义（协议 §8.3），本指标**只报告绝对差异**，"
            "不输出无穷值或极大百分比。"
        )
    return lines


def _lift_text(lift: Any) -> str:
    if lift is None:
        return "不适用（未提供，不写成 0）"
    if not _finite(lift):
        return "不适用（无定义）"
    return f"{float(lift):+.4%}"


def _zero_inflation_sentence(cs: Any) -> str:
    """零值占比的解读句（数值全部取自 ``cs``，百分比下限由程序计算）。"""
    treatment = getattr(cs, "treatment_zero_share", None)
    control = getattr(cs, "control_zero_share", None)
    if not (_finite(treatment) and _finite(control)):
        return (
            "本次输入未提供零值占比，因此无法判断该指标的零膨胀程度；"
            "在零膨胀程度未知时，不得把均值当作普遍客户的水平来解读。"
        )
    floor_pct = int(math.floor(min(float(treatment), float(control)) * 100))
    return (
        f"两组零值占比分别为 {fmt_rate(treatment)} 与 {fmt_rate(control)}，"
        f"即两组均有至少 {floor_pct}% 的客户该指标取值为 0；"
        "因此该均值差异主要由少数非零取值的客户驱动，不能理解为普遍客户的水平变化。"
    )


def _continuous_evidence_bullet(mid: str, cs: Any) -> str:
    """统计证据段的一条连续型指标结论（非阻断分支）。"""
    label = metric_label_zh(mid)
    alpha = getattr(cs, "alpha", None)
    alpha_txt = f"{float(alpha):g}" if _finite(alpha) else _NOT_PROVIDED
    method = str(getattr(cs, "method", None) or "方法未提供")
    p_value = fmt_p(getattr(cs, "p_value", None))
    ci_txt = _continuous_ci_text(cs)
    crosses = _bounds_cross_zero(getattr(cs, "ci_low", None), getattr(cs, "ci_high", None))
    crosses_txt = (
        "区间跨越 0" if crosses is True else ("区间不跨越 0" if crosses is False else "区间是否跨越 0 无法判定")
    )
    significant = getattr(cs, "significant", None)
    if not isinstance(significant, bool):
        return (
            f"- `{mid}`（{label}）：本次输入未提供结构化的显著性判定，"
            f"因此本报告不陈述显著性结论（检验方法 {method}，p = {p_value}；"
            f"绝对差异 {fmt_signed(getattr(cs, 'difference', None))}，区间 {ci_txt}，{crosses_txt}）。"
        )
    if significant:
        return (
            f"- `{mid}`（{label}）：均值差异在 α={alpha_txt}（双侧）下**有统计显著性**"
            f"（{method}，p = {p_value}；绝对差异 {fmt_signed(getattr(cs, 'difference', None))}，"
            f"区间 {ci_txt}，{crosses_txt}）。该判定只说明差异超出随机波动可解释的范围，"
            "不等同于差异具有业务价值。"
        )
    return (
        f"- `{mid}`（{label}）：**未检出统计显著差异**；在现有样本量"
        f"（干预组 {fmt_count(getattr(cs, 'treatment_n', None))}、"
        f"对照组 {fmt_count(getattr(cs, 'control_n', None))}）下无法排除小效应存在"
        f"（{method}，p = {p_value}；绝对差异 {fmt_signed(getattr(cs, 'difference', None))}，"
        f"区间 {ci_txt}，{crosses_txt}）。"
    )


def _continuous_effect_bullet(mid: str, cs: Any, ctx: _Context) -> list[str]:
    """效果大小段的一条连续型指标结论：窗口、绝对差异、零膨胀三者同时给出。"""
    label = metric_label_zh(mid)
    return [
        f"- `{mid}`（{label}）：在实验后 {ctx.window_days} 天窗口内，"
        f"绝对差异（干预组 − 对照组）= {fmt_signed(getattr(cs, 'difference', None))}，"
        f"区间 {_continuous_ci_text(cs)}；该均值按全部被随机分配到该组的客户计算，"
        "**包含未消费客户（其取值记为 0）**。",
        f"  - {_zero_inflation_sentence(cs)}",
    ]


def _write_ch6_conclusion(doc: _Doc, ctx: _Context) -> None:
    doc.add("## 6. 结论", "")
    if ctx.blocked:
        doc.add(f"> {_BLOCKED_SENTENCE}", "")

    # ---- 6.1 统计证据 ----
    doc.add("### 6.1 统计证据", "")
    if ctx.blocked:
        doc.add(
            f"{_BLOCKED_SENTENCE}。",
            "",
            "引发阻断的检查项见第 3 章；第 4 章的数值不构成统计证据，仅作排查用。",
            "",
        )
    elif not ctx.stats_by_metric and not ctx.continuous_by_metric:
        doc.add("本次调用未传入指标统计结果，本章无统计证据可陈述。", "")
    else:
        total = len(ctx.metric_ids) + len(ctx.continuous_ids)
        doc.add(
            f"本报告共报告 {total} 个指标（比例型 {len(ctx.metric_ids)} 个、"
            f"连续型 {len(ctx.continuous_by_metric)} 个），"
            "**主要与次要指标全部报告，不按显著性筛选**（协议 §13.2）。",
            "",
        )
        doc.extend([_evidence_bullet(mid, ctx.stats_by_metric[mid]) for mid in ctx.metric_ids])
        doc.extend(
            [
                _continuous_evidence_bullet(mid, ctx.continuous_by_metric[mid])
                for mid in ctx.continuous_ids
            ]
        )
        doc.add("")

    # ---- 6.2 效果大小 ----
    doc.add("### 6.2 效果大小", "")
    if ctx.blocked:
        doc.add(
            "因果前提不成立，本次不提供效果大小结论。第 4 章中的两组比例、均值与其差值"
            "是观测值本身，其比较前提（随机化、数据完整性）已被第 3 章的阻断项破坏，"
            "仅作排查用。",
            "",
        )
    elif not ctx.stats_by_metric and not ctx.continuous_by_metric:
        doc.add("本次调用未传入指标统计结果，本章无效果大小可陈述。", "")
    else:
        for mid in ctx.metric_ids:
            doc.extend(_effect_bullet(mid, ctx.stats_by_metric[mid], ctx))
        for mid in ctx.continuous_ids:
            doc.extend(_continuous_effect_bullet(mid, ctx.continuous_by_metric[mid], ctx))
        baselines = "、".join(
            f"{metric_label_zh(mid)} {fmt_rate(getattr(ctx.stats_by_metric[mid], 'control_rate', None))}"
            for mid in ctx.metric_ids
        )
        if baselines:
            doc.add(
                "",
                "**绝对差异与相对变化必须同时阅读**（协议 §13.2）：相对变化的数值取决于对照组基线比例"
                f"（本例对照组比例：{baselines}），基线越小时，同一绝对差异会对应越大的相对变化；"
                "因此不得用相对变化单独描述收益幅度，也不得用相对变化掩盖极小的绝对差异。",
                "",
            )
        else:
            doc.add("")

    # ---- 6.3 业务价值 ----
    doc.add("### 6.3 业务价值", "")
    doc.add(
        "本次调用**未提供成本、触达限制与业务阈值数据**，因此本报告：",
        "",
        "- 不评估净收益，不计算 ROI，不给出预算或触达比例建议（协议 §13.1、§14）；",
        "- 只陈述效果大小与其不确定性；是否投放属于业务决策。",
        "",
    )
    if ctx.blocked:
        doc.add("在因果前提不成立的情况下，本次不提供任何业务价值判断。", "")
    else:
        all_items = list(ctx.stats_by_metric.items()) + list(ctx.continuous_by_metric.items())
        significant_ids = [mid for mid, obj in all_items if getattr(obj, "significant", None) is True]
        not_significant_ids = [mid for mid, obj in all_items if getattr(obj, "significant", None) is False]
        undetermined_ids = [
            mid for mid, obj in all_items if not isinstance(getattr(obj, "significant", None), bool)
        ]
        if significant_ids:
            doc.add(
                "对检出统计显著差异的指标（"
                + "、".join(f"`{m}`" for m in significant_ids)
                + f"）：差异在 α={ctx.alpha_text()} 下显著；"
                "是否投放还需结合成本、触达限制与业务阈值。",
                "",
            )
        if not_significant_ids:
            doc.add(
                "对未检出统计显著差异的指标（"
                + "、".join(f"`{m}`" for m in not_significant_ids)
                + "）：在现有样本量下无法排除小效应存在；"
                "是否投放同样需结合成本、触达限制与业务阈值，且本次结果不构成投放依据。",
                "",
            )
        if undetermined_ids:
            doc.add(
                "以下指标本次无法给出显著性判定，因此也不提供业务判断："
                + "、".join(f"`{m}`" for m in undetermined_ids)
                + "。",
                "",
            )
        if not (significant_ids or not_significant_ids or undetermined_ids):
            doc.add("本次没有可用的指标结果，因此不提供业务判断。", "")

    # ---- 6.4 数据可信度 ----
    doc.add("### 6.4 数据可信度", "")
    if not ctx.checks:
        doc.add(
            "本次调用未提供任何数据质量检查结果：数据可信度**未经验证**。"
            "**未执行检查不等于检查通过**，不得据此推断数据质量良好。",
            "",
        )
    else:
        passed_count = sum(1 for c in ctx.checks if _field(c, "passed") is True)
        doc.add(
            f"本次共提供 {len(ctx.checks)} 项检查：通过 {passed_count} 项，"
            f"未通过 {len(ctx.failed_checks)} 项"
            f"（其中阻断级 {len(ctx.blocking_checks)} 项、警告级 {len(ctx.warning_checks)} 项）。",
            "",
        )
        if ctx.blocked:
            doc.add(
                f"阻断级异常清单：{ctx.check_codes(ctx.blocking_checks)}。"
                "按 docs/05 §4.2，这些异常的根因需要人工诊断；程序只负责检出与记录，"
                "不代填业务结论。",
                "",
            )
        else:
            doc.add(
                "本次未检出阻断级异常（依据：传入的检查项清单）。"
                "该结论只说明这些检查未发现问题，**不说明数据完全没有问题**，"
                "也不代替实验设计信息的核对。",
                "",
            )
        if ctx.warning_checks:
            doc.add(
                f"警告级未通过项：{ctx.check_codes(ctx.warning_checks)}，已写入 6.5 的局限。",
                "",
            )
    doc.add(
        f"结论外推范围：仅适用于本实验中被随机分配的人群，以及实验后 {ctx.window_days} 天的固定窗口"
        "（协议 §3、§7）；不得外推到全体注册用户或从未购买的新客。",
        "",
    )

    # ---- 6.5 补充子块 ----
    doc.add(
        "### 6.5 补充：假设、局限与下一步（docs/05 §1.7 要求，不属于上述四段结论）",
        "",
    )
    doc.add("**假设**（结论依赖的前提，均未被本次计算验证）：", "")
    doc.add(
        "- 随机分配按实验设计正确执行，且分析单位与随机化单位一致（均为客户）；",
        "- 结果字段无系统性缺失，观察窗口内结果已完整回收；",
        "- 组间无干扰（一名客户的干预不影响另一名客户的结果）。",
        "",
    )
    doc.add("**局限**：", "")
    limitations = [
        "第 2 章列出的未提供字段（数据来源、行数、哈希、分析对象定义等）缺失，"
        "追溯链不完整；",
        "无成本数据，因此不提供净收益、ROI 或预算结论；",
        f"结果窗口固定为实验后 {ctx.window_days} 天，不支持任意日期窗口；",
        "本报告只覆盖传入本函数的指标；未传入的指标不在本报告内。",
    ]
    if not ctx.continuous_by_metric:
        limitations.append(
            "本次未传入连续型指标（如人均消费）的统计结果，本报告不含其数值，"
            "因此第 6 章的结论未覆盖该类指标。"
        )
    for check in ctx.blocking_checks:
        code = _field(check, "code", "未提供检查码")
        limitations.append(
            f"阻断级异常 `{code}`（{_field(check, 'name', '未提供检查名')}）："
            "由该异常导致的结论限制见第 3 章与 docs/05 §4.2。"
        )
    for check in ctx.warning_checks:
        code = _field(check, "code", "未提供检查码")
        limitations.append(f"警告级未通过检查 `{code}`：影响解释范围，但不阻断描述性输出。")
    doc.extend([f"- {item}" for item in limitations])
    doc.add("", "**下一步**（可执行的验证动作）：", "")
    next_steps = [
        "补齐第 2 章列出的必填字段后重跑，使报告具备完整追溯链；",
        "如需业务判断，补充成本、触达限制与业务阈值，并对阈值做预先声明；",
        "不得以「换窗口、换人群、换口径直到显著」为下一步（协议 §14、§17）。",
    ]
    if not ctx.continuous_by_metric:
        next_steps.insert(
            1,
            "如需覆盖连续型指标（如人均消费），请在调用时传入 `continuous_by_metric` 后重新生成报告；",
        )
    if ctx.blocked:
        next_steps.insert(
            1,
            "按 docs/05 §4.2 对阻断级异常做诊断（程序只报偏离，根因需人工核查），"
            "处置完成后重新运行本分析，而不是放宽阈值或删掉检查项。",
        )
    doc.extend([f"{i}. {item}" for i, item in enumerate(next_steps, start=1)])
    doc.add("")


def _write_ch7_repro(doc: _Doc, ctx: _Context, out_path: Path) -> None:
    ci_methods: set[str] = set()
    test_methods: set[str] = set()
    for st in ctx.stats_by_metric.values():
        for attr in ("treatment_ci", "control_ci", "risk_difference_ci", "risk_ratio_ci"):
            method = getattr(getattr(st, attr, None), "method", None)
            if method:
                ci_methods.add(str(method))
        method = getattr(getattr(st, "test", None), "method", None)
        if method:
            test_methods.add(str(method))
    for cs in ctx.continuous_by_metric.values():
        method = getattr(cs, "method", None)
        if method:
            test_methods.add(str(method))

    seeds = sorted(
        {
            int(getattr(cs, "bootstrap_seed"))
            for cs in ctx.continuous_by_metric.values()
            if _finite(getattr(cs, "bootstrap_seed", None))
        }
    )
    if seeds:
        seed_cell = (
            "bootstrap 随机种子（取自连续型结果对象）："
            + "、".join(f"`{s}`" for s in seeds)
            + "；种子已固定，相同输入必须得到相同区间（协议 §12.4）。"
        )
    else:
        seed_cell = (
            "本次传入的统计量为确定性计算（区间与检验方法由结果对象给出），不涉及随机性；"
            "若上游引入 bootstrap 或置换检验，必须在此登记种子（协议 §12.4）。"
        )

    artifacts = [f"`{out_path}`（本报告）"]
    artifacts.extend(
        f"`{path}`（{'存在' if path.exists() else '缺失'}）" for path in ctx.chart_paths.values()
    )
    doc.add("## 7. 复现信息", "")
    doc.extend(
        _md_table(
            ["字段", "内容"],
            [
                ("run_id", f"`{ctx.run_id}`"),
                ("数据版本 data_version", f"`{ctx.dataset_version}`"),
                ("程序版本 code_version", _code_version()),
                ("包版本 package_version", _package_version()),
                (
                    "区间方法（取自结果对象）",
                    "、".join(f"`{m}`" for m in sorted(ci_methods)) or _NOT_PROVIDED,
                ),
                (
                    "检验方法（取自结果对象）",
                    "、".join(f"`{m}`" for m in sorted(test_methods)) or _NOT_PROVIDED,
                ),
                ("显著性水平 α", ctx.alpha_text()),
                ("结果窗口", f"实验后 {ctx.window_days} 天"),
                ("生成时间 generated_at", ctx.generated_at),
                ("随机种子", seed_cell),
                ("产物清单", "；".join(artifacts)),
                (
                    "重放方式",
                    "以同一份结构化结果对象重新调用 "
                    "`render_stats_report(run_id=..., experiment_key=..., comparison_key=..., "
                    "dataset_version=..., window_days=...)`；本模块无独立 CLI，"
                    "重放必须使用与本次完全相同的输入对象。",
                ),
                (
                    "不可复现项",
                    "第 2 章列出的未提供字段在当前输入下无法复原；"
                    "这些字段缺失时，本报告不具备完整复现能力。",
                ),
            ],
        )
    )
    doc.add(
        "",
        "> 相同输入必须产生相同输出：本模块不做统计计算、不引入随机性；"
        "显示精度由格式化层统一控制（比例 4 位小数百分数、p 值科学计数法、差异用百分点），"
        "解释文本不二次改写数值（协议 §15、docs/05 §2.1）。",
        "",
    )


def _write_selfcheck(doc: _Doc, *, blocked: bool) -> None:
    """渲染前的程序自检：程序生成的文本不得违反措辞禁令。"""
    own_text = doc.own_text
    own_hits = find_forbidden_wording(own_text)
    if own_hits:
        details = "；".join(f"「{phrase}」出现在：{line}" for phrase, line in own_hits)
        raise ValueError(
            "报告正文命中禁用措辞，报告不落盘（协议 §13.2、docs/05 §2.2）：" + details
        )
    if blocked:
        causal_hits = [
            (word, line.strip()[:80])
            for line in own_text.splitlines()
            for word in _CAUSAL_WORDS
            if word in line
        ]
        if causal_hits:
            details = "；".join(f"「{word}」出现在：{line}" for word, line in causal_hits)
            raise ValueError(
                "阻断状态下报告正文出现因果/决策断言措辞，报告不落盘（docs/05 §4.5）：" + details
            )
    leaked = [
        (phrase, line)
        for phrase, line in find_forbidden_wording(doc.text)
        if (phrase, line) not in set(find_forbidden_wording(own_text))
    ]
    if leaked:
        doc.add(
            "",
            "---",
            "",
            "> ⚠️ **措辞自检**：以下由外部传入的文本（检查项名称或说明）命中禁用措辞清单。",
            "> 为满足追溯要求已原样保留，但不得作为结论引用，需回到上游修正来源文本：",
            "",
        )
        doc.extend([f"> - 「{phrase}」出现在：{line}" for phrase, line in leaked])


# ============================================================================
#  5. 对外入口
# ============================================================================


def _build_context(
    *,
    run_id: str,
    experiment_key: str,
    dataset_version: str,
    comparison_key: str,
    stats_by_metric: dict[str, Any],
    continuous_by_metric: dict[str, Any] | None,
    checks: list,
    window_days: int,
    generated_at: str | None,
    chart_paths: dict[str, Path] | None,
    treatment_label: str | None = None,
    control_label: str | None = None,
) -> _Context:
    stats_map = dict(stats_by_metric or {})
    continuous_map = dict(continuous_by_metric or {})
    check_list = list(checks or [])

    def _clean_label(label: str | None) -> str | None:
        """组别别名归一：去空白；空串视为未提供（不写空括号）。"""
        if label is None:
            return None
        text = str(label).strip()
        return text or None

    def severity_of(check: Any) -> str:
        return severity_zh(_field(check, "severity"))

    failed = [c for c in check_list if _failed_flag(_field(c, "passed"))]
    blocking = [c for c in failed if severity_of(c) == "阻断"]
    warnings_failed = [c for c in failed if severity_of(c) == "警告"]
    alphas = sorted(
        {
            float(getattr(st, "alpha"))
            for st in list(stats_map.values()) + list(continuous_map.values())
            if _finite(getattr(st, "alpha", None))
        }
    )
    return _Context(
        run_id=str(run_id),
        experiment_key=str(experiment_key),
        dataset_version=str(dataset_version),
        comparison_key=str(comparison_key),
        stats_by_metric=stats_map,
        continuous_by_metric=continuous_map,
        checks=check_list,
        window_days=int(window_days),
        generated_at=generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        chart_paths={str(k): Path(v) for k, v in (chart_paths or {}).items()},
        blocked=bool(blocking),
        blocking_checks=blocking,
        failed_checks=failed,
        warning_checks=warnings_failed,
        alphas=alphas,
        treatment_label=_clean_label(treatment_label),
        control_label=_clean_label(control_label),
    )


def render_stats_report(
    *,
    run_id: str,
    experiment_key: str,
    dataset_version: str,
    comparison_key: str,
    stats_by_metric: dict[str, Any],
    out_path: Path,
    continuous_by_metric: dict[str, Any] | None = None,
    checks: list = (),
    window_days: int = 14,
    generated_at: str | None = None,
    chart_paths: dict[str, Path] | None = None,
    treatment_label: str | None = None,
    control_label: str | None = None,
) -> Path:
    """生成 Markdown 统计报告，返回报告路径。

    Args:
        run_id: 运行标识，**原样输出**用于追溯。
        experiment_key: 实验标识。
        dataset_version: 数据版本，**原样输出**用于追溯。
        comparison_key: 比较标识（如 ``primary``）。
        stats_by_metric: 比例型指标标识 → ``ComparisonStats``。**报告数值的唯一来源之一。**
        out_path: 报告输出路径。父目录不存在时创建；不依赖项目根。
        continuous_by_metric: 连续型指标标识 → ``ContinuousStats``（如
            ``spend_per_customer``）。为 ``None`` 或空时，报告不含连续型指标，
            并在第 5、6 章如实标注该缺失（不静默省略）。
        checks: ``CheckResult`` 列表（``code`` / ``name`` / ``severity`` / ``passed`` /
            ``detail`` / ``value``）。判定阻断级异常的依据；默认空元组。
        window_days: 结果窗口天数，默认 14。
        generated_at: 生成时间；未传时用 ``datetime.now().astimezone()``。
        chart_paths: 指标标识 → 图表文件路径，仅用于第 5 章引用。
        treatment_label: 干预组组别取值（如 ``"Mens E-Mail"``）。**可选**：传入时第 1 章
            「比较对象」与第 4 章各表的行标题会写「干预组（Mens E-Mail）」；
            不传时报告不出现任何臂别名。该值**只由本参数提供，不从统计结果对象推断**。
        control_label: 对照组组别取值（如 ``"No E-Mail"``），语义同上。

    Returns:
        写入后的报告路径。

    Raises:
        ValueError: 程序生成的正文命中禁用措辞（协议 §13.2），或阻断状态下出现
            因果/决策断言措辞（docs/05 §4.5）。此时**不落盘**。

    Note:
        本函数**只读入参**：不做统计计算、不访问数据库、不读取配置文件，
        因此可以在没有数据库、没有 LLM 的环境下独立验证（AGENTS.md §3.2）。
        比例型与连续型指标分别在独立的表中渲染：两者的区间与检验方法不同，
        不混在一张表里，避免方法被误读。
    """
    ctx = _build_context(
        run_id=run_id,
        experiment_key=experiment_key,
        dataset_version=dataset_version,
        comparison_key=comparison_key,
        stats_by_metric=stats_by_metric,
        continuous_by_metric=continuous_by_metric,
        checks=checks,
        window_days=window_days,
        generated_at=generated_at,
        chart_paths=chart_paths,
        treatment_label=treatment_label,
        control_label=control_label,
    )

    doc = _Doc()
    _write_header(doc, ctx)
    if ctx.blocked:
        _write_block_banner(doc, ctx)
    _write_ch1_plan(doc, ctx)
    _write_ch2_source(doc, ctx)
    _write_ch3_checks(doc, ctx)
    _write_ch4_stats(doc, ctx)
    _write_ch5_charts(doc, ctx)
    _write_ch6_conclusion(doc, ctx)
    _write_ch7_repro(doc, ctx, Path(out_path))
    _write_selfcheck(doc, blocked=ctx.blocked)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc.text + "\n", encoding="utf-8")
    return path
