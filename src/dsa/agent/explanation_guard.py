"""LLM 解释文字的数值与结论守卫（阶段七）。

缺陷回顾（外部审查 F03）
------------------------
> `node_report()` 通过提示词要求"不新增任何数字"，随后直接把模型返回
> 拼接到报告，没有逐项核对或失败处理。
>
> 本轮注入受控回复：
> > 邮件带来 99.9% 的提升，已经证明有效。
>
> 该回复原样出现在最终报告。……它证明当前提示词约束
> **没有转化为可执行的数值与结论检查**。

这正是 `AGENTS.md` §2.1 的落地缺口：铁律说"LLM 解释文本中若出现数字，
必须与结构化结果核对，不一致即判为**失败**，不得放行"，
而代码里只有一句提示词。**提示词是请求，不是约束。**

本模块把那条铁律变成可执行的检查。

设计取舍（必须说清楚，否则会被误当成形式化验证）
----------------------------------------------
这是一组**基于形状的检查**，不是证明：

- 能查出：**凭空造出的效果量**、**百分点/百分比混淆**、**错组别**、
  **无证据的显著性断言**、**与阻断状态矛盾的因果断言**。
- 查不出：量级正确但语义错误的话（例如把"访问率"说成"购买率"
  却引用了正确数字）。这一条由 :func:`check_arm_reference` 部分覆盖。
- **因此它只用于"解释文字"这一层，且失败时可以整段丢弃** ——
  被守卫的文本是可选的润色，丢弃它不影响报告的完整性。
  这也是为什么报告主体必须由模板回填：**只有可丢弃的东西才允许过不严格的检查。**
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "GuardVerdict",
    "allowed_numbers",
    "check_explanation",
    "extract_numbers",
]

#: 相对容差。与 `dsa.models.registry` 的指纹比较同理：
#: 允许模型对数字做合理取舍（`0.6805` vs `0.680501`），
#: 但不允许数量级错误（`0.68` vs `68`）。
RTOL = 5e-3

#: 效果类单位：出现在数字后即视为"效果量断言"，必须与已知效果量相符。
_EFFECT_UNITS = ("%", "％", "个百分点", "个百分比", "pp", "倍")

#: 这些词附近的裸数字也视为效果断言。
_EFFECT_CONTEXT = re.compile(r"提升|提高|增长|降低|下降|减少|差异|效果|增量|多出|高出|少|增加")

#: 计数类断言的量级门槛：≥ 1000 的裸数字只可能是样本量/人数，必须能对上。
_COUNT_THRESHOLD = 1000

_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

#: 无证据的过强结论。
_OVERCLAIM_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"已经证明|已被证明|证明了|得到证明|证实了|已证实"), "使用了「证明/证实」这类确定性表述"),
    (re.compile(r"必然|必定|一定会|肯定会|保证|确保|毫无疑问|毋庸置疑"), "使用了「必然/保证」这类确定性表述"),
    (re.compile(r"必然带来|一定带来|将带来收益|保证收益|确保收益"), "对下一轮收益作了承诺"),
    (re.compile(r"因果(?:关系)?(?:已经)?确立|因果关系成立"), "在没有实验设计说明的情况下断言因果关系确立"),
)

#: 只有显著时才允许出现的断言。
_SIGNIFICANCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"显著(?:性)?(?:提升|提高|增长|增加|差异|效果|地)"),
    re.compile(r"有统计显著性|达到显著|统计上显著"),
)

#: 阻断状态下禁止出现的因果断言。
_CAUSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"邮件(?:是)?有效|证明(?:了)?有效|确实(?:带来|提升|提高)"),
    re.compile(r"因此(?:应该|可以)(?:加大|继续)投放"),
)

#: 结构性常量：报告里天然会出现、且与业务量无关的数字。
#: 刻意保持**很短** —— 每多一条，守卫就松一分。
_STRUCTURAL = frozenset({0.0, 1.0, 2.0, 3.0, 14.0, 95.0, 100.0, 0.05, 0.95})

#: 组别措辞 → 比较标识。用于检查"说的组和算的组是不是同一个"。
_ARM_WORDS: dict[str, tuple[str, ...]] = {
    "primary": ("男装", "男性", "mens", "men"),
    "secondary_womens": ("女装", "女性", "womens", "women"),
}


@dataclass
class GuardVerdict:
    """守卫结论。**必须可序列化**（要写进检查点与运行记录）。"""

    passed: bool
    reason: str = ""
    rejected: bool = False
    unknown_numbers: list[str] = field(default_factory=list)
    overclaims: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    text_len: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "reason": self.reason,
            "rejected": bool(self.rejected),
            "unknown_numbers": list(self.unknown_numbers),
            "overclaims": list(self.overclaims),
            "notes": list(self.notes),
            "text_len": int(self.text_len),
        }


def _kinds(payload: Mapping[str, Any] | None, arms: Sequence[Mapping[str, Any]] | None,
           extra: Iterable[float]) -> dict[str, list[float]]:
    """把结构化结果摊成**分类**的允许值表。

    分类是必要的：把"百分点"和"百分比"混在一个集合里，
    恰恰就查不出审查点名的"百分比与百分点混淆"。
    """
    out: dict[str, list[float]] = {
        "percent": [],   # 只能以 % 出现（组内率、相对提升）
        "pp": [],        # 只能以"个百分点"出现（绝对差异）
        "raw": [],       # 原尺度（可作 % 之外的裸数）
        "count": [],     # 计数
        "other": [],     # α、种子、bootstrap 次数等
    }

    def _add(key: str, value: Any) -> None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(v):
            out[key].append(v)

    for e in (payload or {}).get("effects") or []:
        # 组内率：百分比
        for k in ("treatment_value", "control_value"):
            if e.get(k) is not None:
                _add("percent", float(e[k]) * 100.0)
        # 绝对差异：百分点（原尺度单独记，但**不允许**以 % 出现）
        for k in ("difference", "ci_low", "ci_high"):
            if e.get(k) is not None:
                _add("pp", float(e[k]) * 100.0)
                _add("raw", float(e[k]))
        # 相对提升：百分比
        if e.get("relative_lift") is not None:
            _add("percent", float(e["relative_lift"]) * 100.0)
        # 计数与统计量
        for k in ("treatment_k", "control_k", "treatment_n", "control_n"):
            if e.get(k) is not None:
                _add("count", e[k])
        for k in ("ratio", "ratio_ci_low", "ratio_ci_high"):
            if e.get(k) is not None:
                _add("raw", e[k])
        for k in ("alpha",):
            if e.get(k) is not None:
                _add("other", e[k])
        for k in ("bootstrap_seed",):
            if e.get(k) is not None:
                _add("other", e[k])

    for a in arms or ():
        for k in ("n_customers", "n_converted", "n_visited", "n_outcome_missing"):
            _add("count", a.get(k))
        for k in ("conversion_rate", "visit_rate"):
            if a.get(k) is not None:
                _add("percent", float(a[k]) * 100.0)
        if a.get("spend_per_customer") is not None:
            _add("raw", a["spend_per_customer"])

    for v in (payload or {}).get("population_n", None), (payload or {}).get("limit_n", None):
        _add("count", v)
    for k in ("window_days", "n_boot", "bootstrap_seed"):
        _add("other", (payload or {}).get(k))
    for v in extra:
        _add("other", v)

    return out


def allowed_numbers(
    payload: Mapping[str, Any] | None,
    arms: Sequence[Mapping[str, Any]] | None = None,
    extra: Iterable[float] = (),
) -> dict[str, list[float]]:
    """对外暴露的允许值表（供测试与调试核对守卫的判据）。"""
    return _kinds(payload, arms, extra)


def extract_numbers(text: str) -> list[tuple[str, float, str]]:
    """抽出文本里的数字 → ``[(原文字面, 数值, 紧随单位)]``。

    单位只看**紧随其后**的少量字符：隔了半句话的 `%` 不属于这个数字。
    """
    out: list[tuple[str, float, str]] = []
    for m in _NUM_RE.finditer(text or ""):
        literal = m.group(0)
        tail = text[m.end(): m.end() + 6].lstrip()
        unit = ""
        for u in _EFFECT_UNITS:
            if tail.startswith(u):
                unit = u
                break
        try:
            value = float(literal.replace(",", ""))
        except ValueError:
            continue
        out.append((literal, value, unit))
    return out


def _close(value: float, candidates: Sequence[float]) -> bool:
    return any(
        abs(value - c) <= max(RTOL * max(abs(c), 1.0), 1e-9) for c in candidates
        if math.isfinite(c)
    )


def _classify(value: float, table: Mapping[str, list[float]]) -> str | None:
    """这个数值属于哪一类（可同时属于多类）。返回优先级最高的一类。"""
    for key in ("count", "percent", "pp", "raw", "other"):
        if _close(value, table.get(key) or []):
            return key
    return None


def check_arm_reference(text: str, comparison_key: str) -> str | None:
    """解释文字是否把组别说错。

    Returns:
        出错说明；没错时 None。

    **为什么需要单独检查**：数字全对但组别说反，是这类报告最危险的错误 ——
    "女装邮件让购买率提升 0.68 个百分点"里的每个数字都可能对得上，
    只是它们属于男装。数字守卫**查不出**这一条。
    """
    expected = _ARM_WORDS.get(comparison_key, ())
    if not expected:
        return None
    others = [w for k, ws in _ARM_WORDS.items() if k != comparison_key for w in ws]
    low = (text or "").lower()
    said_wrong = [w for w in others if w.lower() in low]
    said_right = [w for w in expected if w.lower() in low]
    if said_wrong and not said_right:
        return (f"本次比较是 `{comparison_key}`（{expected[0]}），"
                f"但解释文字只提到 {said_wrong}，组别对不上")
    if said_wrong and said_right:
        return (f"解释文字同时提到本次比较的组与其它组（{said_wrong}），"
                "无法确认结论归属哪一组，不做放行")
    return None


def check_explanation(
    text: str,
    *,
    payload: Mapping[str, Any] | None = None,
    arms: Sequence[Mapping[str, Any]] | None = None,
    comparison_key: str = "primary",
    blocked: bool = False,
    extra_allowed: Iterable[float] = (),
) -> GuardVerdict:
    """检查一段 LLM 补写的解释。

    Args:
        text: 模型输出。
        payload: `analyze_experiment_effect` 的结构化返回（允许值的来源）。
        arms: `query_arm_aggregates` 的组聚合（允许值的补充来源）。
        comparison_key: 本次比较，用于组别检查。
        blocked: 数据质量是否已阻断（阻断时禁止因果断言）。
        extra_allowed: 额外的允许值（如计划里的 n_boot、人数）。

    Returns:
        `GuardVerdict`。``passed=False`` 时 ``reason`` 说明第一条命中的原因。
    """
    verdict = GuardVerdict(passed=True, text_len=len(text or ""))
    body = (text or "").strip()
    if not body:
        # 空文本不是"错误解释"，只是没有解释 —— 不能判失败。
        verdict.notes.append("模型未返回解释文字，跳过守卫")
        return verdict

    table = _kinds(payload, arms, extra_allowed)

    # ---- 1. 过强结论 ----
    for pattern, why in _OVERCLAIM_PATTERNS:
        m = pattern.search(body)
        if m:
            verdict.overclaims.append(m.group(0))
            verdict.passed = False
            verdict.rejected = True
            verdict.reason = f"结论过强：{why}（命中「{m.group(0)}」）"
            return verdict

    # ---- 2. 阻断状态下不得给因果结论 ----
    if blocked:
        for pattern in _CAUSAL_PATTERNS:
            m = pattern.search(body)
            if m:
                verdict.rejected = True
                verdict.passed = False
                verdict.reason = (
                    f"本次分析已被阻断，但解释文字给出因果断言（命中「{m.group(0)}」）"
                )
                return verdict

    # ---- 3. 显著性断言必须有依据 ----
    significant = any(bool(e.get("significant")) for e in (payload or {}).get("effects") or [])
    if not significant:
        for pattern in _SIGNIFICANCE_PATTERNS:
            m = pattern.search(body)
            if m:
                verdict.rejected = True
                verdict.passed = False
                verdict.reason = (
                    f"结构化结果**未**支持显著性，但解释文字断言显著"
                    f"（命中「{m.group(0)}」）"
                )
                return verdict

    # ---- 4. 逐个数字核对 ----
    unknown: list[str] = []
    for literal, value, unit in extract_numbers(body):
        if value in _STRUCTURAL:
            continue
        kind = _classify(value, table)

        if kind is None:
            # ≥1000 的裸数字一定是样本量类断言；其它裸数字若落在效果语境里也算断言
            if abs(value) >= _COUNT_THRESHOLD:
                unknown.append(f"{literal}（样本量类，结构化结果中没有这个数）")
            else:
                span = body[max(0, _span_start(body, literal) - 12): _span_start(body, literal) + 12]
                if unit or _EFFECT_CONTEXT.search(span):
                    unknown.append(f"{literal}{unit}（效果量类，结构化结果中没有这个数）")
            continue

        # 百分点 / 百分比混淆：% 后面跟着的数字其实是"个百分点"的值
        if unit in ("%", "％") and kind in ("pp", "raw"):
            if not _close(value, table.get("percent") or []):
                verdict.rejected = True
                verdict.passed = False
                verdict.reason = (
                    f"单位混淆：`{literal}%` 对应的是**绝对差异**"
                    f"（{value:g} 个百分点），不是百分比。"
                    "绝对差异必须写成「个百分点」，相对变化才用「%」"
                )
                return verdict

    if unknown:
        verdict.unknown_numbers = unknown
        verdict.rejected = True
        verdict.passed = False
        verdict.reason = (
            "解释文字出现了结构化结果中不存在的数字："
            + "；".join(unknown[:5])
            + "（AGENTS.md §2.1：不一致即判失败，不得放行）"
        )
        return verdict

    # ---- 5. 组别 ----
    arm_issue = check_arm_reference(body, comparison_key)
    if arm_issue:
        verdict.rejected = True
        verdict.passed = False
        verdict.reason = arm_issue
        return verdict

    verdict.notes.append("数值、单位、组别与结论强度均通过守卫")
    return verdict


def _span_start(text: str, literal: str) -> int:
    """字面量在文本中的位置（取首次出现；重复时取首次，足够定位上下文）。"""
    return text.find(literal)
