"""结构化计划的构造与校验（阶段七）。

**本模块是 F02 的修复本体。**

缺陷回顾（外部审查 F02）
------------------------
> 结构化计划没有表达名单要求，比较对象和能力范围会被忽略。
> ……最后一个请求的计划已经识别为女装比较，执行仍读取男装模型。
> 其余几个请求的比例、人数和排序依据也未进入计划。
> 部分没有「排序」等关键词的同义请求甚至不会触发名单工具。
>
> **修复：** 计划至少包含数据集、实验、比较、任务类型、指标、结果窗口、
> 筛选条件、排序依据、触达比例／人数和模型版本。LLM 可用于理解语义，
> 程序负责核对这些字段与登记能力；**不能用持续追加关键词替代语义与能力校验**。

最后那句话是本次修复的指导思想。项目原来的做法是维护一串关键词：
漏了一种说法就补一条。这在**能力边界**上是危险的 ——
每一次漏判都表现为「静默回答了另一个问题」，而不是报错。
所以这里改成两层：

1. **理解层**（纯程序，可测）：把问题解析成一组**具名字段**；
2. **校验层**（纯程序，可测）：把每个字段拿去和**登记表**核对，
   对不上就产出结构化 issue，而不是猜一个默认值继续跑。

第 2 层是关键：「用户要女装，但只有男装模型」必须变成一条 issue，
而不是一个默认值。
"""

from __future__ import annotations

import re
from typing import Any

from dsa.agent.state import AnalysisPlan, RequestCategory

# ============================================================ 数据集


#: 数据集名 → 关键字。**必须先于"默认 Hillstrom"判断**：
#: 用户明确说了别的数据集时，默认值是错的答案（审查反例 5）。
DATASET_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("retailhero", "retailhero"),
    ("retail hero", "retailhero"),
    ("retail_hero", "retailhero"),
    ("零售英雄", "retailhero"),
    ("hillstrom", "hillstrom"),
    ("希尔斯特罗姆", "hillstrom"),
)

#: 未指定数据集时的默认值。`dataset_source` 会如实标注它是默认而非用户所述。
DEFAULT_DATASET = "hillstrom"

#: 数据集能力缺口说明。**没有登记的随机化机制就不能做因果结论**
#: （docs/metrics.yaml 中 retailhero 的 `randomized_experiment` 为 null；
#: SRM 通过只证明分组比例均衡，不证明随机化存在）。
DATASET_CAPABILITY_NOTES: dict[str, str] = {
    "retailhero": (
        "RetailHero 适配只开放**数据工程与描述性**能力：它的随机化机制没有文档，"
        "SRM 通过只说明分组比例接近预期，**不构成随机化的证据**，"
        "因此不对它输出任何因果/增量结论（AGENTS.md §2.5、docs/metrics.yaml）。"
    ),
}


# ============================================================ 指标

METRIC_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("购买率", "conversion_rate"),
    ("转化率", "conversion_rate"),
    ("购买", "conversion_rate"),
    ("访问率", "visit_rate"),
    ("访问", "visit_rate"),
    ("人均消费", "spend_per_customer"),
    ("消费", "spend_per_customer"),
)

# ============================================================ 比较对象


#: ⚠️ **顺序与边界都重要**：`"mens"` 是 `"womens"` 的子串。
#:
#: 早期用 `kw in text` 逐条匹配，`womens email 购买率` 会先命中 `mens`，
#: 于是"女装"被识别成"男装"，**静默回答了另一个问题**（审查反例 1）。
#: 修法是双重的：纯 ASCII 关键字加 `\b` 词边界，且女装排在男装之前。
COMPARISON_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("女装", "secondary_womens"),
    ("女性", "secondary_womens"),
    ("womens", "secondary_womens"),
    ("women", "secondary_womens"),
    ("男装", "primary"),
    ("男性", "primary"),
    ("mens", "primary"),
    ("men", "primary"),
)


def _kw_match(text: str, keyword: str) -> bool:
    """关键字匹配。**纯 ASCII 关键字用词边界**，中文用子串。

    没有词边界时 `mens` 会命中 `womens`；有了词边界，
    `womens` 里的 `mens` 前面是 `o`（词字符），不构成边界，因此不匹配。
    """
    if not keyword:
        return False
    if keyword.isascii():
        return re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE) is not None
    return keyword in text


def _lookup(text: str, patterns: tuple[tuple[str, str], ...]) -> str | None:
    for kw, value in patterns:
        if _kw_match(text, kw):
            return value
    return None


def _find(text: str, patterns: tuple[tuple[str, str], ...]) -> tuple[str, str] | None:
    for kw, reason in patterns:
        if _kw_match(text, kw):
            return kw, reason
    return None


# ============================================================ 干预后筛选


#: 干预后字段（用于识别"按干预后行为定义人群"）。
#: 值是它对应的数据库列，仅用于给出可读的理由。
POST_TREATMENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("访问过网站", "visit"), ("看过网站", "visit"), ("访问者", "visit"),
    ("访问", "visit"), ("浏览", "visit"), ("打开邮件", "open"), ("点击邮件", "click"),
    ("已购买", "conversion"), ("购买者", "conversion"),
    ("购买", "conversion"), ("转化", "conversion"), ("下单", "conversion"),
    ("消费过", "spend"), ("消费", "spend"), ("花费", "spend"),
)

#: 限制性算子：出现在干预后字段**之前**时，说明用户在按该字段圈人群。
RESTRICTIVE_OPERATORS: tuple[str, ...] = (
    "只保留", "只有", "只看", "只要", "仅保留", "仅看", "仅仅", "仅", "只",
    "限定", "限制", "筛选", "过滤", "排除", "剔除", "去掉", "去除", "限于",
    "必须是", "必须是", "保留",
)

#: 否定词。**只检查它是否紧贴在限制性算子之前**。
#:
#: 为什么不能放宽到"字段之前"：`排除没有访问网站的客户` 里
#: `没有` 紧贴 `访问`，但它的意思是"没访问的人"，**排除他们 = 只留访问者**，
#: 是典型的干预后筛选。把 `没有` 当成否定词就会漏判这一条（审查反例 3）。
NEGATORS: tuple[str, ...] = (
    "不", "不要", "不用", "不是", "不应", "别", "无需", "不必", "不得", "禁止",
    "不能", "切勿", "而非", "并不是",
)

#: 明确声明"分母是全部被分配客户"的措辞 —— 出现即视为用户在纠正口径。
ALL_ASSIGNED_MARKERS: tuple[str, ...] = (
    "全部分配", "全部被随机分配", "所有被随机分配", "全体客户", "全部客户",
    "所有客户", "全体被分配", "按全部", "不筛选", "不做筛选", "不排除",
)

#: 模型分数类措辞 —— 它们**不是**干预后字段，扫描前必须先剔除，
#: 否则「按购买概率排序」会因为含「购买」被误判成干预后筛选。
_MODEL_SCORE_PHRASES: tuple[str, ...] = (
    "购买概率", "购买倾向", "购买可能性", "转化概率", "购买预测", "预测购买",
    "增量排序", "预测增量", "增量得分", "uplift",
)

#: 「…的客户 / …的人」句式：即使没有限制性算子，也表示按该字段圈人群。
_POPULATION_SUFFIX_RE = re.compile(r"(的客户|的人|的顾客|的会员|的用户|的消费者|的那批)")

#: 效果类措辞：用于判断一句话是不是在问效果（决定「的客户」句式是否构成筛选）。
_EFFECT_CONTEXT_RE = re.compile(r"效果|提升|提高|影响|差异|增量|作用|是否有效|有没有用|带来")

_OPERATOR_LOOKBACK = 12
_SUFFIX_LOOKAHEAD = 8


def _is_question_form_negation(head: str, negator: str) -> bool:
    """``head`` 末尾的否定词是否其实是**疑问形式**的一部分。

    ⚠️ 这一条由**回归集 A02** 发现：

    > 能不能只看访问过网站的人？访问者里的购买率才算数吧？

    `能不能` 里含 `不能`，而 `不能` 在否定词表里 —— 于是这句**要求按访问者
    筛选人群**的话被当成了"口径纠正"，**放行**了。这正是分母口径红线要防的事。

    疑问形式（`能不能` / `能否` / `可否`）里的"不"不表示否定，
    它只是中文正反问句的构词。判据：否定词以 `不` 开头，且它前面紧邻
    `能` / `否` / `可`。
    """
    if not negator.startswith("不"):
        return False
    before = head[: len(head) - len(negator)]
    return before.endswith(("能", "否", "可"))


def _strip_model_score_phrases(text: str) -> str:
    out = text
    for phrase in _MODEL_SCORE_PHRASES:
        out = out.replace(phrase, "〓" * len(phrase))
    return out


def detect_post_treatment_filter(question: str) -> tuple[bool, str]:
    """问题是否在**按干预后行为筛选人群**。

    Returns:
        ``(是否命中, 理由)``。理由会原样进入结构化错误，因此必须可读。

    判定规则（两条，任一命中即算）：

    R1 限制性算子在干预后字段前 12 字内出现（``只看访问者`` / ``排除没有访问的``）；
    R2 干预后字段后跟 ``的客户/的人`` 等群体后缀，**且**该问题在问效果。

    否定与口径纠正可以取消判定：``不要只看访问者，请按全部分配客户计算``
    既命中 R1，也命中否定词与「全部分配」标记，因此**不**判为筛选。
    """
    raw = question or ""
    text = _strip_model_score_phrases(raw)
    text = re.sub(r"(?:不限制|不考虑|不用筛选|不按|只看|仅看|限定|筛选|按)[^，。；！？]*?渠道",
                  lambda m:"〓"*len(m.group()),text)
    text = re.sub(r"(?:历史|实验前|干预前)(?:消费|购买|访问)[^，。；！？]*?(?:客户|用户|顾客|人)",
                  lambda m: "〓" * len(m.group(0)), text)
    low = text

    def _cancelled(operator_start: int) -> bool:
        """限制性算子是否被否定，或同句出现了"按全部分配客户"的口径纠正。"""
        head = low[max(0, operator_start - 4):operator_start]
        for n in NEGATORS:
            if head.endswith(n) and not _is_question_form_negation(head, n):
                return True
        return bool(re.search(r"(?:请按|改为|改按|应按)(?:全部|所有|全体)", low))

    for field_kw, col in POST_TREATMENT_FIELDS:
        start = 0
        while True:
            idx = low.find(field_kw, start)
            if idx < 0:
                break
            start = idx + 1

            # R1：前面的限制性算子
            window = re.split(r"[，。；！？]",low[max(0, idx - _OPERATOR_LOOKBACK):idx])[-1]
            for op in RESTRICTIVE_OPERATORS:
                op_idx = window.rfind(op)
                if op_idx < 0:
                    continue
                abs_op = max(0, idx - _OPERATOR_LOOKBACK) + op_idx
                if _cancelled(abs_op):
                    continue
                return True, (
                    f"`{op}{field_kw}` 要求按干预后字段 `{col}` 筛选人群作为分母，"
                    "这会破坏随机化，使「干预效果」与「人群差异」无法区分"
                    "（AGENTS.md §2.2）"
                )

            # R2：后面跟群体后缀，且整句在问效果
            tail = low[idx + len(field_kw): idx + len(field_kw) + _SUFFIX_LOOKAHEAD]
            if _POPULATION_SUFFIX_RE.search(tail) and _EFFECT_CONTEXT_RE.search(low):
                return True, (
                    f"`{field_kw}…` 把人群限定为按干预后字段 `{col}` 筛选出的子集，"
                    "分母必须保留**全部被随机分配**的客户（AGENTS.md §2.2）"
                )
    return False, ""


# ============================================================ 固定窗口


#: 时长单位 → 天数。**「两周」等于 14 天**，与本项目固定窗口一致，
#: 因此它是可接受的表述而不是超范围请求。
_DURATION_UNITS: dict[str, int] = {"天": 1, "日": 1, "周": 7, "星期": 7, "礼拜": 7, "个月": 30, "月": 30}

_CN_NUM: dict[str, int] = {
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

_DURATION_RE = re.compile(
    r"(?P<num>\d+|[一两二三四五六七八九十]+)\s*(?P<unit>个月|星期|礼拜|天|日|周|月)"
)

#: 本项目固定的结果窗口（天）——来自实验登记 `outcome_window_days`。
FIXED_WINDOW_DAYS = 14


def detect_outcome_window(question: str) -> tuple[int | None, str]:
    """问题里提到的结果窗口 → ``(天数, 原文片段)``。

    没有提到时长时返回 ``(None, "")``。
    中文数字（``七天``）与阿拉伯数字（``7天``）都识别。
    """
    windows = outcome_windows(question)
    return windows[0] if windows else (None, "")


# ============================================================ 人群选择


#: 人群选择意图。**比原版宽得多** —— 审查指出
#: 「部分没有『排序』等关键词的同义请求甚至不会触发名单工具」。
#: 只会说「该给谁发邮件」的用户，和会说「按增量排序选前 10%」的用户，
#: 问的是同一件事。
SELECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"名单", r"排序", r"排名", r"优先", r"分值", r"得分",
        r"前\s*\d+\s*(?:名|个|位|人|%|％)", r"top\s*\d+",
        r"选哪些", r"哪些客户", r"哪些人", r"该给谁", r"发给谁", r"投给谁",
        r"给谁发", r"触达谁", r"触达哪些", r"人群", r"圈选", r"筛选出",
        r"选出", r"选客户", r"挑出", r"挑哪", r"最值得", r"最该", r"优先触达",
        r"要发给", r"应该发", r"投放给谁",
        # ⚠️ 下面两条由**冻结验收集 FZ05** 发现（`按购买概率从高到低，
        # 挑 300 个人出来` 被判成"需要澄清"）。
        # 原来的谱系只覆盖「前 N 名」「选出」「挑出」，而用户会说
        # 光杆的「挑」和「N 个人」——**同义的第三种说法**。
        # 这与 L034 是同一类缺口：规则的覆盖面是**列举**出来的，
        # 而用户的说法是**生成**出来的，两者永远不会自动对齐。
        # 所以这里补的是"说法"，而兜底靠的是 `_SIZE_NOUN_RE` 这种**结构**判据。
        r"挑",
        r"\d+\s*个?\s*(?:人|客户|顾客|用户)",
    )
)

#: 「询问数量」的措辞。用于把「有 300 个客户吗」这类**问数量**的问题
#: 与「挑 300 个客户」这类**要名单**的问题分开 ——
#: 两者的字面结构相同（数字 + 量词 + 名词），只有意图不同。
_ASK_COUNT_RE = re.compile(r"多少|几[个位名]|数量|人数是|有几个")


def wants_customer_selection(question: str) -> bool:
    """问题是否在问「该给哪些客户发」（而不是问整体效果或问数量）。"""
    q = question or ""
    if not any(p.search(q) for p in SELECTION_PATTERNS):
        return False
    # 「多少个客户」是在**问数量**，不是在要名单。
    # 但若同时出现了明确的名单动词（名单/排序/该给谁…），仍按名单处理。
    explicit = any(p.search(q) for p in SELECTION_PATTERNS
                   if p.pattern not in (r"挑",
                                        r"\d+\s*个?\s*(?:人|客户|顾客|用户)"))
    if not explicit and _ASK_COUNT_RE.search(q):
        return False
    return True


#: 排序依据关键词 → 规范策略名。
#:
#: ⚠️ 取值必须与 `dsa.models.policy` 的权威策略定义一致
#: （该模块由并行的修复建立；名字不一致会在工具层被 `E_TOOL_001` 拦住，
#: 不会被静默接受 —— 这正是 5.2 缺陷要求的行为）。
RANKING_KEYWORDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"无干预购买概率|对照(?:组)?(?:的)?购买概率|不看邮件(?:也)?会买"), "control_probability"),
    (re.compile(r"购买概率|购买倾向|购买可能性|转化概率|购买预测|propensity"), "purchase_probability"),
    (re.compile(r"增量|提升度|uplift|预测增量|增量得分"), "uplift"),
    (re.compile(r"随机(?:选|挑|排序)"), "random"),
)

#: 未指定排序依据时的默认策略。**必须在计划与报告里显式说明是默认值**，
#: 否则又是一次"静默回答了另一个问题"。
DEFAULT_RANKING = "uplift"

#: 触达比例：`前 10%` / `10% 的客户` / `触达 20 %` / `给 20% 客户发`
#:
#: ⚠️ 前缀**必须可选**。回归集 A04（`如果只能给 20% 客户发邮件，该选谁？`）
#: 用到的是「给 N% 客户」，「给」不在原先的前缀表里，于是比例解析不出来、
#: 计划判为"缺少触达规模"、整条请求失败。
#: 这类缺口的形态与本项目反复出现的一致：**规则的覆盖面是列举出来的，
#: 而用户的说法是生成出来的**。所以这里不再穷举动词前缀。
#:
#: 放宽不会引入误判：`parse_limit` 的结果只在**人群选择**任务上被校验
#: （见 `validate_plan_against_registry`），非选择任务里出现的百分数
#: 只被记进 `limit_pct` 而无人使用。
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|％|个?百分点)", re.IGNORECASE)

#: 触达人数。三种写法，**量词或名词至少要有一个** ——
#: 否则 `前 10%` 会被这里先匹配成"10 个人"，
#: 而 `parse_limit` 是先看人数后看比例的。
#:
#: ⚠️ 第三种写法（`Top 50 客户` / `300 客户`）由冻结验收集发现：
#: 中文里"50 客户"省略量词是常见的，而原正则要求必须有量词。
_N_RE = re.compile(
    r"(?:"
    r"(?:前|top)\s*(?P<a>\d[\d,]*)\s*(?:名|个|位|人|客户|顾客|用户)"
    r"|(?P<b>\d[\d,]*)\s*(?:名|个|位|人)"
    r"|(?P<c>\d[\d,]*)\s*(?:客户|顾客|用户)"
    r")",
    re.IGNORECASE,
)


def parse_limit(question: str) -> tuple[int | None, float | None, str]:
    """解析名单规模 → ``(人数, 比例, 依据原文)``。

    比例与人数**互斥**：同时出现时以人数为准（更具体）。

    ⚠️ 比例**不在这里换算成人**。因为换算需要知道可打分人群的大小，
    而那是数据事实，必须由工具返回（见 `docs/08` §2：
    大结果不猜不编，一切数字来自程序）。因此计划里保留 `limit_pct`，
    由执行节点向工具要 `population_n` 后再算，并把两步都记进报告。
    """
    q = question or ""
    m = _N_RE.search(q)
    if m:
        raw = m.group("a") or m.group("b") or m.group("c") or ""
        n = int(raw.replace(",", ""))
        if n > 0:
            return n, None, m.group(0).strip()
    m = _PCT_RE.search(q)
    if m:
        pct = float(m.group(1)) / 100.0
        if 0 < pct <= 1:
            return None, pct, m.group(0).strip()
    return None, None, ""


# ============================================================ 分类


#: 需要阻断的固定表述（高精度快路径）。
#: 更一般的判定由 :func:`detect_post_treatment_filter` 负责 ——
#: 这里保留的几条是最常见、且无需上下文就能判定的说法。
BLOCKED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("只看访问", "要求按干预后行为（是否访问）筛选人群，会破坏随机化"),
    ("仅访问", "同上"),
    ("访问过网站的人", "同上"),
    ("访问者", "同上"),
    ("看过网站", "同上"),
    ("打开邮件的人", "同上"),
    ("点击邮件的人", "同上"),
    ("只看购买", "按干预后行为筛选人群"),
    ("只看消费", "按干预后行为筛选人群"),
)

#: 超出能力范围的表述（docs/02-分析协议.md §14）。
UNSUPPORTED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("净收益", "本项目没有成本数据，无法计算净收益或投资回报"),
    ("净收入", "同上"),
    ("roi", "本项目没有成本数据，无法计算投资回报"),
    ("roas", "同上"),
    ("投资回报", "同上"),
    ("换窗口", "第一版结果窗口固定为 14 天，不支持动态窗口"),
    ("最近一个月", "第一版不支持动态日期窗口"),
    ("最近一周", "第一版不支持动态日期窗口"),
    ("最近三个月", "第一版不支持动态日期窗口"),
    ("时间范围", "第一版不支持动态日期窗口"),
    ("实时", "第一版不支持实时实验查询"),
    # ⚠️ 以下几条由**独立验收案例**发现（A03）：
    # 原模式只覆盖「实时」「最近一个月」等说法，而用户会说
    # 「每天的效果曲线」「每小时自动刷新」「定时跑」——
    # 同一能力限制的**不同表述**，规则没覆盖就漏判成"需要澄清"。
    # 见 docs/日志.md L034：这类缺口只有**没见过的表述**才能暴露。
    ("每小时", "第一版不支持实时/定时刷新"),
    ("自动刷新", "第一版不支持实时/定时刷新"),
    ("定时", "第一版不支持实时/定时刷新"),
    ("效果曲线", "第一版不支持按日/按小时的效果曲线"),
    ("每天的效果", "第一版不支持动态日期窗口"),
    ("每日效果", "第一版不支持动态日期窗口"),
    ("预测未来", "第一版不做前瞻预测"),
    ("因果图", "第一版不支持因果图推断"),
    ("中介效应", "第一版不支持中介分析"),
    # 客户级不确定性：**刻意不提供**（docs/09 §7）。
    ("预测区间", "本版本刻意不提供客户级预测区间"),
)


def _active_match(text: str, start: int) -> bool:
    head = re.split(r"[，。；！？,;!?]", text[:start])[-1]
    return not re.search(r"(?:不要|无需|不用|不必|别|不)(?:只|再|算|计算|分析|比较|看|考虑)*$", head)

def infer_metrics(question: str) -> list[str]:
    q = _strip_model_score_phrases(question or "")
    hits = []
    for kw, metric in METRIC_KEYWORDS:
        for match in re.finditer(re.escape(kw), q):
            if _active_match(q, match.start()):
                hits.append((match.start(), -len(kw), metric))
    result = []
    for _, _, metric in sorted(hits):
        if metric not in result:
            result.append(metric)
    return result

def infer_metric(question: str) -> str | None:
    metrics = infer_metrics(question)
    return metrics[0] if metrics else None

def requested_arms(question: str) -> list[str]:
    q = question or ""
    arms = []
    for kw, comparison in COMPARISON_KEYWORDS:
        pattern = rf"\b{re.escape(kw)}\b" if kw.isascii() else re.escape(kw)
        for match in re.finditer(pattern, q, re.IGNORECASE):
            if _active_match(q, match.start()):
                arm = "mens" if comparison == "primary" else "womens"
                if arm not in arms:
                    arms.append(arm)
    return arms

def infer_comparison(question: str) -> str | None:
    arms = requested_arms(question)
    if len(arms) > 1:
        return "mens_vs_womens"
    if arms:
        return "primary" if arms[0] == "mens" else "secondary_womens"
    return None

def chinese_number(raw: str) -> int:
    if raw.isdigit():
        return int(raw)
    if "十" in raw:
        left, right = raw.split("十", 1)
        return (10 if not left else _CN_NUM.get(left, 0) * 10) + _CN_NUM.get(right, 0)
    return _CN_NUM.get(raw, 0)

def outcome_windows(question: str) -> list[tuple[int, str]]:
    return [(chinese_number(m.group("num")) * _DURATION_UNITS[m.group("unit")],
             m.group(0)) for m in _DURATION_RE.finditer(question or "")
            if _active_match(question or "", m.start())]

def population_conditions(question: str) -> list[dict[str, Any]]:
    conditions = []
    for clause in re.split(r"[，。；！？,;!?]", question or ""):
        match = re.search(r"只分析|只看|仅分析|仅看|限定|限于|筛选|过滤|排除|剔除|只保留|仅保留", clause)
        feature = re.search(r"渠道|线上|线下|新客|老客|历史|实验前|干预前|年龄|性别|地区|邮编|history|recency|newbie|zip_code|channel", clause, re.I)
        if feature and (match or re.search(r"的(?:客户|人|顾客|用户)", clause)):
            if match and not _active_match(clause, match.start()):
                continue
            conditions.append({"raw": clause, "kind": "pre_treatment", "status": "unsupported"})
        elif match and _active_match(clause, match.start()) and re.search(
                r"的(?:客户|人|顾客|用户)", clause) and not detect_post_treatment_filter(clause)[0]:
            conditions.append({"raw": clause, "kind": "unresolved_population", "status": "clarification"})
    return conditions

def request_constraints(question: str) -> list[dict[str, Any]]:
    q = question or ""
    metrics = infer_metrics(q)
    arms = requested_arms(q)
    windows = outcome_windows(q)
    filters = population_conditions(q)
    post,post_reason=detect_post_treatment_filter(q)
    if post:filters.append({"raw":q,"kind":"post_treatment","status":"unsupported","reason":post_reason})
    n, pct, raw = parse_limit(q)
    selection = wants_customer_selection(q)
    records = [
        {"field":"metrics", "raw":q, "value":metrics,
         "status":"resolved" if metrics or selection else "clarification"},
        {"field":"arms", "raw":q, "value":arms,
         "status":"unsupported" if len(arms)>1 else ("resolved" if arms or selection else "clarification")},
        {"field":"window", "raw":[r for _,r in windows], "value":[d for d,_ in windows],
         "status":"clarification" if len({d for d,_ in windows})>1 else
                  ("unsupported" if windows and windows[0][0]!=14 else "resolved")},
        {"field":"population", "raw":[f["raw"] for f in filters], "value":filters,
         "status":filters[0]["status"] if filters else "resolved"},
        {"field":"ranking", "raw":q if selection else "", "value":infer_ranking(q)[0] if selection else None,
         "status":"resolved"},
        {"field":"limit", "raw":raw, "value":{"n":n,"pct":pct},
         "status":"clarification" if selection and n is None and pct is None else "resolved"},
    ]
    return records


def infer_dataset(question: str) -> tuple[str, str]:
    """问题 → ``(数据集, 来源)``。来源为 ``explicit`` 或 ``default``。"""
    hit = _lookup(question or "", DATASET_KEYWORDS)
    if hit:
        return hit, "explicit"
    return DEFAULT_DATASET, "default"


def infer_ranking(question: str) -> tuple[str, str]:
    """问题 → ``(排序依据, 来源)``。"""
    q = question or ""
    for pattern, kind in RANKING_KEYWORDS:
        if pattern.search(q):
            return kind, "explicit"
    return DEFAULT_RANKING, "default"


def classify_rules(question: str) -> tuple[RequestCategory, list[str], str]:
    """**纯规则**分类。返回 ``(类别, 缺失项, 原因)``。

    顺序是关键：**先判阻断、再判能力、再判数据集、最后判是否需要澄清**。
    若顺序反过来，"只看访问者的购买率"会因为缺少实验标识而被判成
    "需要澄清"，用户补充之后仍会走到按干预后字段筛选的分析上 ——
    这正是蓝图要求「关键异常不能被绕过」要防的情形。
    """
    q = (question or "").strip()
    missing: list[str] = []
    if not q:
        return RequestCategory.NEEDS_CLARIFICATION, ["question"], "问题为空"

    # 1. 阻断优先 —— 但是**先看否定与口径纠正**：
    #    `不要只看访问者，请按全部分配客户计算…` 是在纠正口径，不是违规。
    blocked, reason = detect_post_treatment_filter(q)
    if not blocked:
        hit = _find(q, BLOCKED_PATTERNS)
        if hit and not _phrase_is_negated(q, hit[0]):
            blocked, reason = True, hit[1]
    if blocked:
        return RequestCategory.BLOCKED, [], reason

    # 2. 能力范围（旧表述）
    hit = _find(q, UNSUPPORTED_PATTERNS)
    if hit:
        return RequestCategory.UNSUPPORTED, [], hit[1]

    if len(requested_arms(q)) > 1:
        return RequestCategory.UNSUPPORTED, [], "本版未登记男装与女装的直接比较；不能替换成邮件组与无邮件组。"
    filters = population_conditions(q)
    if filters:
        if any(f["status"] == "unsupported" for f in filters):
            return RequestCategory.UNSUPPORTED, [], "本版不支持实验前特征子群筛选；请求条件保留，不执行全体客户替代分析。"
        return RequestCategory.NEEDS_CLARIFICATION, ["population"], "人群条件无法可靠解析，请确认是否分析全部分配客户。"
    windows = outcome_windows(q)
    if len({days for days, _ in windows}) > 1:
        return RequestCategory.NEEDS_CLARIFICATION, ["window"], "指定了多个不同结果窗口，请确认使用实验固定的14天窗口。"
    if "客单价" in q and "人均消费" not in q and "全部分配客户" not in q:
        return RequestCategory.NEEDS_CLARIFICATION, ["denominator"], "客单价分母不明确；本版仅支持每位被分配客户的人均消费。"

    # 3. 结果窗口：**必须判在"是否需要澄清"之前**。
    #    「比较男装邮件实验后七天购买率」指标和比较都齐全，
    #    若不在此拦住，它会一路跑到执行并用 14 天窗口算出结果 ——
    #    用户拿到的是另一个问题的答案（审查反例 4）。
    days, raw = detect_outcome_window(q)
    if days is not None and days != FIXED_WINDOW_DAYS:
        return RequestCategory.UNSUPPORTED, [], (
            f"问题指定了「{raw}」（约 {days} 天）的结果窗口，"
            f"而本实验的结果窗口**固定为 {FIXED_WINDOW_DAYS} 天**"
            "（docs/02-分析协议.md §8.2）；换窗口需要走协议变更流程"
        )

    # 4. 数据集：用户点名了别的数据集就不能默认 Hillstrom。
    dataset, source = infer_dataset(q)
    if source == "explicit" and dataset in DATASET_CAPABILITY_NOTES:
        return RequestCategory.UNSUPPORTED, [], (
            f"问题指向数据集 `{dataset}`。" + DATASET_CAPABILITY_NOTES[dataset]
        )

    # 5. 是否需要澄清：缺指标或比较对象。
    #
    # ⚠️ **人群选择问题豁免这两项要求**。
    # 「该给哪些客户发邮件」是**排序**问题 —— 它要的是「谁排前面」，
    # 既不需要指定指标（默认按预测增量排），也不需要比较臂
    # （排序不涉及两组对比）。早期版本对它一律要求比较对象，
    # 于是这类问题被判为"需要澄清"，而用户补充比较对象后
    # 得到的仍不是他要的名单。见 docs/日志.md L033。
    if wants_customer_selection(q):
        n, pct, _ = parse_limit(q)
        if n is None and pct is None:
            return RequestCategory.NEEDS_CLARIFICATION, ["limit"], "请指定名单人数或触达比例。"
        return RequestCategory.EXECUTABLE, [], ""

    if infer_metric(q) is None:
        missing.append("metric")
    if infer_comparison(q) is None:
        missing.append("comparison")
    if missing:
        return RequestCategory.NEEDS_CLARIFICATION, missing, "缺少分析条件"

    return RequestCategory.EXECUTABLE, [], ""


def _phrase_is_negated(text: str, phrase: str) -> bool:
    """固定短语是否被否定（``不要只看访问者``）。

    与 :func:`_cancelled` 用**同一个**疑问形式判据 ——
    两处各写一份就会出现"一处修了一处没修"，而那正是本项目反复出现的形态。
    """
    idx = text.find(phrase)
    if idx < 0:
        return False
    head = text[max(0, idx - 4):idx]
    for n in NEGATORS:
        if head.endswith(n) and not _is_question_form_negation(head, n):
            return True
    return any(m in text for m in ALL_ASSIGNED_MARKERS)


# ============================================================ 计划构造


def build_plan(question: str) -> AnalysisPlan:
    """问题 → 结构化计划。**每个字段都有来源标注**。

    计划里刻意保留 `*_source` 字段（``explicit`` / ``default``）：
    "用户说的" 与 "我们替他定的" 是两件事，
    报告必须能把后者说成"默认"，否则就是替用户做了决定还不告诉他。
    """
    q = question or ""
    dataset, dataset_source = infer_dataset(q)
    selection = wants_customer_selection(q)
    metric = infer_metric(q)
    ranking, ranking_source = infer_ranking(q)
    limit_n, limit_pct, limit_raw = parse_limit(q)

    metrics: list[str] = infer_metrics(q)
    if metrics:
        pass
    elif not selection:
        metrics.append("conversion_rate")

    task_type = "select_customers" if selection else "compare_effect"

    return AnalysisPlan(
        dataset_key=dataset,
        dataset_source=dataset_source,
        experiment_key="hillstrom_email_v1" if dataset == "hillstrom" else "",
        comparison_key=infer_comparison(q) or "primary",
        comparison_source="explicit" if infer_comparison(q) else "default",
        task_type=task_type,
        metrics=metrics,
        metric_source="explicit" if metric else ("none" if selection else "default"),
        outcome_window_days=detect_outcome_window(q)[0] or FIXED_WINDOW_DAYS,
        requested_arms=requested_arms(q),
        request_constraints=request_constraints(q),
        filters=population_conditions(q),
        ranking=ranking if selection else None,
        ranking_source=ranking_source if selection else None,
        limit_n=limit_n,
        limit_pct=limit_pct,
        limit_raw=limit_raw,
        model_key=(m.group(1) if (m := re.search(r"(?:模型|model)\s*[:：]?\s*([a-z][a-z0-9_]+)", q, re.I)) else None),
        model_source="explicit" if m else None,
        notes=[],
        question=q,
    )


# ============================================================ 计划校验


def _known_comparison_keys(cfg: dict[str, Any]) -> set[str]:
    return {c.get("comparison_key") for c in (cfg.get("comparisons") or [])}


def validate_plan_against_registry(
    plan: AnalysisPlan, cfg: dict[str, Any] | None, metric_ok: dict[str, bool],
    models: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], AnalysisPlan]:
    """把计划逐字段与**登记能力**核对。

    **本函数是纯函数**：所有外部事实（实验配置、指标是否登记、模型清单）
    都由调用方取好传进来。这样它可以脱离数据库单测，
    而"哪个字段该被拒绝"这件事就有了独立可验证的判据。

    Returns:
        ``(issues, plan)``。``plan`` 可能被补上解析出的 `model_key`。
    """
    issues: list[dict[str, Any]] = []
    plan = plan.model_copy(deep=True)

    def _issue(field: str, code: str, message: str) -> None:
        issues.append({"field": field, "code": code, "message": message})

    # ---- 数据集 ----
    if plan.dataset_key != "hillstrom":
        note = DATASET_CAPABILITY_NOTES.get(
            plan.dataset_key, f"数据集 `{plan.dataset_key}` 未登记因果分析能力"
        )
        _issue("dataset_key", "E_CAP_001", note)
        return issues, plan

    # ---- 实验与比较 ----
    if cfg is None:
        _issue("experiment_key", "E_DB_001", "实验配置不可读，无法核对比较与窗口")
        return issues, plan

    keys = _known_comparison_keys(cfg)
    if plan.comparison_key not in keys:
        _issue("comparison_key", "E_CAP_001",
               f"未登记的预注册比较：{plan.comparison_key}；已登记 {sorted(keys)}")

    days = (cfg.get("experiment") or {}).get("outcome_window_days")
    if days != plan.outcome_window_days:
        _issue("outcome_window_days", "E_CAP_001",
               f"结果窗口固定为 {days} 天，计划为 {plan.outcome_window_days}")

    for condition in plan.request_constraints:
        if condition.get("status") != "resolved":
            _issue(condition["field"], "E_PLAN_001" if condition["status"] == "clarification" else "E_CAP_001",
                   "原始请求条件尚未解决：" + str(condition.get("raw")))
    if plan.filters and not any(i["field"] == "population" for i in issues):
        _issue("population", "E_CAP_001", "本版本不支持人群筛选，不能执行全体客户替代分析。")
    plan.dataset_version = (cfg.get("dataset") or {}).get("version_label") or plan.dataset_version

    # ---- 指标 ----
    for m in plan.metrics:
        if not metric_ok.get(m, False):
            _issue("metrics", "E_CAP_001", f"未登记或不可做因果比较的指标：{m}")

    # ---- 人群选择 ----
    if plan.task_type == "select_customers":
        known = registered_policy_kinds()
        if plan.ranking is None:
            _issue("ranking", "E_PLAN_001", "人群选择缺少排序依据")
        elif plan.ranking not in known:
            _issue("ranking", "E_CAP_001",
                   f"未登记的排序依据：{plan.ranking}；已登记 {list(known)}")

        if plan.limit_n is None and plan.limit_pct is None:
            _issue("limit", "E_PLAN_001",
                   "人群选择缺少触达规模：请给出比例（如「前 10%」）或人数（如「前 300 名」）")

        # **模型与比较必须匹配**。这是审查反例里最严重的一条：
        # 「给女装邮件按增量排序选前 20% 客户」的计划已识别为女装比较，
        # 执行却读取男装模型 —— 用户拿到一份**贴错标签**的名单。
        from dsa.agent.model_resolution import resolve_model
        key,source,model_issues=resolve_model(plan.model_dump(),models)
        plan.model_key,plan.model_source=key,source
        issues.extend(model_issues)

    return issues, plan


def registered_policy_kinds() -> tuple[str, ...]:
    """已登记的策略/排序依据。

    **以 `dsa.models.policy` 的权威定义为准**（5.2 缺陷要求
    「用一个稳定的策略定义贯穿离线评估、分数存储和名单查询」）。
    读不到时退回内置清单 —— 退回值是**保守的**，
    与权威定义不一致会表现为「校验明确拒绝」，而不是静默放行。
    """
    try:
        from dsa.models.policy import POLICY_DEFINITIONS

        keys = tuple(str(k) for k in POLICY_DEFINITIONS)
        if keys:
            return keys
    except Exception:  # noqa: BLE001 —— 导入失败不能影响分类
        pass
    return ("uplift", "purchase_probability", "control_probability", "random")


__all__ = [
    "BLOCKED_PATTERNS",
    "COMPARISON_KEYWORDS",
    "DEFAULT_DATASET",
    "DEFAULT_RANKING",
    "FIXED_WINDOW_DAYS",
    "METRIC_KEYWORDS",
    "UNSUPPORTED_PATTERNS",
    "build_plan",
    "classify_rules",
    "detect_outcome_window",
    "detect_post_treatment_filter",
    "infer_comparison",
    "infer_dataset",
    "infer_metric",
    "infer_ranking",
    "parse_limit",
    "registered_policy_kinds",
    "validate_plan_against_registry",
    "wants_customer_selection",
]
