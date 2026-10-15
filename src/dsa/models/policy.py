"""离线策略评估：在**预先固定**的触达比例下比较各策略的总体增量。

口径依据
--------
- `docs/06-建模与策略评估协议.md` §5（离线策略评估）、§6（预先固定的评估规则）、
  §7（模拟验证）；措辞边界见 `docs/02-分析协议.md` §13。
- 本模块**只做评估，不做任何选择**：触达比例集合、显著性水平、bootstrap 种子全部写死在
  模块常量里。调用方不得用评估结果反过来挑触达比例或换种子（协议 §6.1/§6.4/§6.5）。
- 本模块不含任何成本口径的判断（项目没有成本数据），只报告总体增量的点估计与区间。

估计目标
--------
策略 ``π`` 把每名客户映射成"发邮件 / 不发邮件"：``π(x) = 1`` 表示发。记

    τ(x) = E[Y(1) - Y(0) | X = x]

本模块估计的是**总体增量**（协议 §5.2 的"估计的总体增量（绝对百分点）"）：

    V(π) = E[ π(X) · τ(X) ]

也就是"相对**谁都不发**，按 π 投放的每名客户（人均）绝对增量"。两条对账性质：

* ``π ≡ 1``（``all_treat``）时 ``V = ATE``，正好是主比较的总体效果；
* ``π ≡ 0``（``none``）时 ``V = 0``。

之所以不是"被选中人群的平均增量" ``E[τ | π=1]``：协议 §5.2 的估值式带有 ``|π| / n``
因子，且 §5.1 把 ``all_treat`` 的 fraction 记为 1.0——只有乘上触达比例，
``all_treat`` 才等于 ATE。

估计量（为什么是这个形式）
--------------------------
协议 §5.2 要求"具体公式与区间方法必须在实现阶段用模拟数据核对后固定下来"。
本模块固定使用下面这个形式，理由写在 :func:`snips_estimate` 的 docstring 里：

.. math::

    \\hat V(\\pi) = \\hat P(\\pi{=}1)\\left[
        \\frac{\\sum_i \\pi_i W_i Y_i / e}{\\sum_i \\pi_i W_i / e}
      - \\frac{\\sum_i \\pi_i (1-W_i) Y_i / (1-e)}{\\sum_i \\pi_i (1-W_i) / (1-e)}
    \\right]

即"**两个 Hájek（自归一化）分量之差 × 触达比例**"。在 ``e ≡ 0.5`` 下它退化为
"被选中人群的组间购买率之差 × 触达比例"，正是完全随机化下按设计无偏的经典形式。

区间方法
--------
百分位 bootstrap，**在每次重抽样内部重新执行策略的选取规则**（按打分重排、再取前 q），
而不是固定住原样本选出的那份名单。理由：留出集是从总体抽出的一份样本，
策略是"排序取前 q"这条**规则**，重抽样必须把"名单本身会变"这一层不确定性算进去。
两种做法（固定名单 / 名单重选）的实际区间覆盖率都已在
:mod:`dsa.models.policy_simulation` 里用已知真值的模拟核对过，见该模块 docstring。

策略**差异**（差值区间）
------------------------
⚠️ **不能用"两个策略各自的区间是否重叠"来判断差异。**

两个策略跑在**同一份留出客户**上，两个估计因此是**相关**的。边际区间的重叠与否
不是任何假设检验的判据。一个可手算的反例（见 ``tests/test_stage4_policy_contrasts.py``）：
两个估计的标准误都是 1、相关系数 0.95，则**差值**标准误是
``sqrt(1 + 1 - 2×0.95) = 0.3162``；两条 95% 边际区间 ``[-1.96, 1.96]`` 与
``[-0.96, 2.96]`` 明显重叠，而估计差值为 1 时差值区间 ``[0.3802, 1.6198]`` **排除 0**。

因此本模块直接对差值做区间：:func:`evaluate_policies_with_contrasts` 在**同一次共同
bootstrap 的配对抽样**里计算 :math:`V(a) - V(b)` 的重抽样分布（同一批索引下相减，
不另起一次独立抽样），再取百分位区间，得到 :class:`PolicyContrast`。
主要比较（比例 + 对照）由 :data:`CONTRAST_PRIMARY_FRACTION` /
:data:`CONTRAST_PRIMARY_POLICY` / :data:`CONTRAST_PRIMARY_REFERENCE` **预先固定**；
其余比较一律标记为**探索性**，不做多重比较校正（措辞见
:data:`EXPLORATORY_CONTRAST_NOTE`）。

策略命名（三类概率不得混用）
----------------------------
历史上曾出现"同名策略在两处代表不同模型输出"的错误（缺陷 5.2）。本项目区分三类量：

======================  ==========================================  ==========================
概念                     含义                                          数据来源
======================  ==========================================  ==========================
分配倾向概率             ``P(被分到干预组 | x)``，真值恒为 0.5，          **不来自模型**（由实验设计
（allocation            且**不随 x 变化**（协议 §2.2）                 推导；:data:`ALLOCATION_PROPENSITY`）
 propensity）
无干预购买概率           ``P(购买 | 对照, x)``                          T-Learner 的 ``mu0``
                                                                       （``customer_scores.mu0``）
合并购买预测分数         ``P(购买 | x)``，跨组合并训练                  ``customer_scores.propensity_score``
======================  ==========================================  ==========================

策略名必须**如实**：按合并购买预测排序的策略叫 ``purchase_probability``，
按无干预购买概率排序的叫 ``control_probability``。策略的唯一权威定义在
:data:`POLICY_DEFINITIONS`；离线评估取打分用 :func:`scores_from_frame`，
名单接口排序用 :data:`POLICY_ORDER_COLUMNS` —— 两者来自**同一个**常量，
不允许各自硬编码列名。旧名 ``propensity`` 指向不明（历史上离线评估用 mu0、
名单接口用合并预测），一律**明确拒绝**：:func:`reject_legacy_policy_name`。

措辞边界（协议 §5.4）
--------------------
留出集上的评估是**回顾性**的：名单是演示输出，不得表述成"这些客户将来会怎样"。
每个 :class:`PolicyScore` 的 ``note`` 都带这条提示。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

#: 固定触达比例：协议 §6.1 预先写死，**必须全部报告**，不只报最好的那个。
#: 调用方可以只请求其中一部分（例如核对单个比例），但**不得扩展**成别的取值——
#: 见 :func:`evaluate_policies` 的参数校验。
TREATMENT_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50)

#: 条件化分配概率 e(x) = P(arm = mens | arm ∈ {mens, control})，由实验设计推导，
#: 恒为 0.5 且与 x 无关（协议 §2.2）。**不是估计出来的**，也不是策略打分。
PROPENSITY: float = 0.5

#: 分配倾向概率（**概念 1/3**）：``P(被分到干预组 | x)``。
#: 恒为 0.5、**不随 x 变化**、**不来自模型**（协议 §2.2 由三组等概率条件化推导）。
#: 与 :data:`PROPENSITY` 是同一个数：``PROPENSITY`` 是它在 IPS 估计量里的角色名，
#: 这里给它一个不与"策略打分"混淆的名字。
ALLOCATION_PROPENSITY: float = PROPENSITY

#: :data:`ALLOCATION_PROPENSITY` 的固定说明（测试与报告直接引用，避免各处复述走样）。
ALLOCATION_PROPENSITY_NOTE: str = (
    "分配倾向概率 P(被分到干预组 | x)：由实验设计推导，真值恒为 0.5，"
    "不随 x 变化，不来自模型，也不作为任何策略的打分。"
)

#: 显著性水平（双侧），协议 §12.3。
DEFAULT_ALPHA: float = 0.05

#: bootstrap 重复次数默认值。
DEFAULT_N_BOOTSTRAP: int = 1000

#: 项目固定随机种子（协议 §3.2/§6.5）。一旦确定不再更换。
DEFAULT_SEED: int = 20261011


# ============================================================================
#  策略的**唯一**权威定义
# ============================================================================


@dataclass(frozen=True, slots=True)
class PolicyDefinition:
    """一条策略的权威定义。

    名称、用哪一类概率、对应 `customer_scores` 的哪一列，只在这里写一次。
    离线评估（:func:`scores_from_frame`）与名单接口（:data:`POLICY_ORDER_COLUMNS`）
    都从它派生，不允许各自硬编码列名——这正是缺陷 5.2 的成因。
    """

    name: str
    kind: str  # 'ranked'（按打分排序取前 q）| 'supplement'（补充参照）
    concept: str  # 用到的概率/量的名字（见模块 docstring 的三类概率表）
    score_column: str | None  # customer_scores 里的列（含 cs. 前缀）；None = 无打分列
    score_source: str  # 离线评估里打分数组的来源
    description: str


POLICY_DEFINITIONS: dict[str, PolicyDefinition] = {
    "random": PolicyDefinition(
        name="random",
        kind="ranked",
        concept="no_model",
        score_column="cs.customer_id",
        score_source="固定种子的均匀随机打分（random_scores，不落库）",
        description=(
            "随机选择：不打分，按固定种子随机挑 q 比例的人。"
            "名单接口用 cs.customer_id 给出稳定顺序（评估里是随机抽样）。"
            "它是「不做选择」的对照，也是主要比较的参照策略。"
        ),
    ),
    "purchase_probability": PolicyDefinition(
        name="purchase_probability",
        kind="ranked",
        concept="merged_purchase_probability",
        score_column="cs.propensity_score",
        score_source="跨组合并训练的购买预测模型输出（customer_scores.propensity_score）",
        description=(
            "按合并购买预测分数排序：P(购买 | x)，模型在干预组与对照组上合并训练。"
            "旧名 propensity 同时被用在「离线评估按 mu0」与「名单按本列」两处，"
            "语义不一致，故改名（缺陷 5.2）。"
        ),
    ),
    "control_probability": PolicyDefinition(
        name="control_probability",
        kind="ranked",
        concept="control_purchase_probability",
        score_column="cs.mu0",
        score_source="T-Learner 的对照组结果模型输出（mu0 = P(购买 | 对照, x)）",
        description=(
            "按无干预购买概率排序：本来（不被打扰时）就会买的人排在前面。"
            "作为「按购买概率找人」这条传统做法的如实版本。"
        ),
    ),
    "uplift": PolicyDefinition(
        name="uplift",
        kind="ranked",
        concept="uplift",
        score_column="cs.uplift",
        score_source="T-Learner 的预测增量 tau_hat(x) = mu1 - mu0",
        description="按预测增量排序：预计「被推一把才会买」的人排在前面。",
    ),
    "all_treat": PolicyDefinition(
        name="all_treat",
        kind="supplement",
        concept="no_model",  # 全部触达，与打分无关
        score_column=None,
        score_source="无打分：全部客户都发（记账比例 1.0）",
        description="补充参照：全部触达。资源消耗与固定比例策略不同，只能并列展示。",
    ),
    "none": PolicyDefinition(
        name="none",
        kind="supplement",
        concept="none",
        score_column=None,
        score_source="无打分：谁也不发（记账比例 0.0）",
        description="补充参照：全部不触达。增量按定义恒为 0，区间退化为 [0, 0]。",
    ),
}

#: 需要打分排序的策略（协议 §5.1 的固定集合）。
RANKED_POLICIES: tuple[str, ...] = tuple(
    sorted(n for n, d in POLICY_DEFINITIONS.items() if d.kind == "ranked")
)

#: 打分来自**分数表某一列**的策略（即除 ``random`` 之外的排序策略）。
#: 离线评估从库里取打分时用它；``random`` 的打分由 :func:`random_scores` 现算。
SCORED_POLICIES: tuple[str, ...] = tuple(
    n for n in RANKED_POLICIES if POLICY_DEFINITIONS[n].score_column and n != "random"
)

#: 补充参照策略：资源消耗与固定比例策略不同，只能并列展示、不参与排名（协议 §5.4）。
SUPPLEMENT_POLICIES: tuple[str, ...] = tuple(
    sorted(n for n, d in POLICY_DEFINITIONS.items() if d.kind == "supplement")
)

#: :class:`PolicyScore` 里 ``policy`` 字段的全部合法取值。
POLICY_NAMES: tuple[str, ...] = tuple(sorted(POLICY_DEFINITIONS))

#: 名单接口的排序列表：策略名 → ``customer_scores`` 的列（带 ``cs.`` 前缀）。
#:
#: ⚠️ 这是**唯一**的列映射。``dsa/tools/analysis_tools.py`` 必须从这里导入，
#: 不要另抄一份字面量；离线评估的取列走 :func:`policy_score_columns`（同一个源头）。
POLICY_ORDER_COLUMNS: dict[str, str] = {
    name: d.score_column
    for name, d in POLICY_DEFINITIONS.items()
    if d.kind == "ranked" and d.score_column is not None
}

#: 旧策略名 → 权威名。旧名 ``propensity`` 在历史上指过两个不同的模型输出
#: （离线评估用 mu0、名单接口用合并购买预测），语义不明，**不得静默接受**。
LEGACY_POLICY_NAMES: dict[str, str] = {"propensity": "purchase_probability"}

#: 旧名/改名相关的错误码（与 `docs/08-工具协议.md` 的错误码体系一致）。
LEGACY_POLICY_NAME_CODE: str = "E_TOOL_001"


class LegacyPolicyNameError(ValueError):
    """使用了已废弃的策略名（旧名语义不明，必须由调用方改成权威名）。

    ``code`` 供工具层直接回填错误码（``E_TOOL_001``），不必再解析消息文本。
    """

    code: str = LEGACY_POLICY_NAME_CODE

    def __init__(self, name: str, replacement: str) -> None:
        self.name = name
        self.replacement = replacement
        super().__init__(
            f"[{LEGACY_POLICY_NAME_CODE}] 策略名 {name!r} 已废弃：它在历史上指过两个不同的"
            f"模型输出（离线评估用 mu0、名单接口用合并购买预测），语义不明。"
            f"请改用 {replacement!r}（合并购买预测分数）或 "
            f"{'control_probability'!r}（无干预购买概率 mu0）。"
        )


def reject_legacy_policy_name(name: str) -> None:
    """旧名一律拒绝；合法名原样通过。

    Raises:
        LegacyPolicyNameError: ``name`` 是 :data:`LEGACY_POLICY_NAMES` 里的旧名。
    """
    if name in LEGACY_POLICY_NAMES:
        raise LegacyPolicyNameError(name, LEGACY_POLICY_NAMES[name])


def resolve_policy_name(name: str) -> str:
    """把策略名解析成权威名；旧名与未登记名都报错，**不静默退化**。"""
    reject_legacy_policy_name(name)
    if name not in POLICY_DEFINITIONS:
        raise ValueError(
            f"未登记的策略名 {name!r}；合法取值为 {POLICY_NAMES}。"
            "策略的定义只写在 POLICY_DEFINITIONS 里，不接受临时策略。"
        )
    return name


def policy_score_columns() -> dict[str, str]:
    """策略名 → ``customer_scores`` 的**纯列名**（不带表别名），供离线评估取打分。

    直接从 :data:`POLICY_ORDER_COLUMNS` 派生：名单接口排序用的列与这里取的列
    因此必然是同一个标识符（缺陷 5.2 的验收要求）。
    ``random`` 没有分数列（打分由 :func:`random_scores` 现算），不包含在内。
    """
    return {name: col.split(".")[-1] for name, col in POLICY_ORDER_COLUMNS.items() if name != "random"}


def scores_from_frame(frame: Any, policies: Sequence[str]) -> dict[str, np.ndarray]:
    """从分数表（DataFrame）按权威定义取出各策略的打分数组。

    这是离线评估取打分的**唯一**入口：列名来自 :func:`policy_score_columns`，
    与名单接口的排序列表同源。

    Args:
        frame: 含 ``mu1/mu0/uplift/propensity_score`` 的分数表。
        policies: 策略名序列；``random`` 不接受（打分由 :func:`random_scores` 现算）。

    Returns:
        ``{策略名: 打分数组（float）}``。

    Raises:
        LegacyPolicyNameError: 出现旧名 ``propensity``。
        ValueError: 策略名未登记、要求 ``random``、或分数表缺少所需列。
    """
    columns = policy_score_columns()
    out: dict[str, np.ndarray] = {}
    for name in policies:
        resolve_policy_name(name)
        if name not in columns:
            raise ValueError(
                f"策略 {name!r} 没有分数列（打分由 random_scores 现算），"
                "不能用 scores_from_frame 取分"
            )
        column = columns[name]
        if column not in getattr(frame, "columns", ()):
            raise ValueError(
                f"分数表缺少列 {column!r}（策略 {name!r} 的打分列）。"
                "列名来自 POLICY_ORDER_COLUMNS，不允许各调用方自行改名。"
            )
        values = np.asarray(frame[column], dtype=float)
        if not bool(np.all(np.isfinite(values))):
            raise ValueError(f"策略 {name!r} 的打分列 {column!r} 含 nan/inf")
        out[name] = values
    return out


# ============================================================================
#  预先固定的主要比较（协议 §6：看见结果之前就定下来）
# ============================================================================

#: **主要触达比例**：预先固定为 10%。
#:
#: 理由（与留出集结果无关，全部是看到结果之前就能说清的运营与口径理由）：
#:
#: 1. 十分位是项目里**已经存在**的默认触达深度——`sql/004_policy_queries.sql` 的
#:    ``top_decile_check`` 把 ``:fraction`` 默认成 0.1，沿用它可以避免"报告一套、
#:    核对另一套"。
#: 2. 它是固定比例集合里最小两档之一（0.05 只有约 426 人，名单与区间都太薄），
#:    同时是真实营销活动里最常见的投放深度（十分位人群包）。
#: 3. 与 0.30 相比，10% 更接近"真要做一次触达"的规模；把它定为主要比较，
#:    可以避免用最大的那个比例（区间最窄）来制造显著性——后者正是 §3.4 禁止的做法。
CONTRAST_PRIMARY_FRACTION: float = 0.10

#: **主要对照**：``random``（不做选择）。它是协议 §5.1 要求的基线，
#: 也是"策略到底有没有带来差别"这个问题的自然参照。
CONTRAST_PRIMARY_REFERENCE: str = "random"

#: **主要策略**：``uplift``（按预测增量排序）。它是本项目的目标策略
#: （docs/06 §4.2 的 T-Learner 输出），主要比较问的就是"它相对不做选择有没有差别"。
CONTRAST_PRIMARY_POLICY: str = "uplift"

#: 每个比例下要报告的差值对 ``(策略, 对照)``：计算 ``V(策略) − V(对照)``。
#:
#: 顺序是有意的：三条"各自 vs random"在前（random 代表不做选择），
#: 再是策略之间的两两比较。**每个无序对只出现一次**（有测试锁住）。
#: 补充参照（all_treat / none）资源消耗不同，不参与同比例差值比较（协议 §5.4）。
CONTRAST_PAIRS: tuple[tuple[str, str], ...] = (
    ("uplift", "random"),
    ("purchase_probability", "random"),
    ("control_probability", "random"),
    ("uplift", "purchase_probability"),
    ("uplift", "control_probability"),
    ("purchase_probability", "control_probability"),
)

#: 非主要比较的固定说明：写明**没有**做多重比较校正。
EXPLORATORY_CONTRAST_NOTE: str = (
    "探索性比较：本次没有做多重比较校正（未控制族错误率），"
    "结论只能作为下一步的假设，不能当作已确认的策略差异。"
)

#: 主要比较的固定说明。
PRIMARY_CONTRAST_NOTE: str = (
    "主要比较：触达比例与对照策略在模块常量里预先固定，本次评估不据此调整口径。"
)


def _check_contrast_pairs() -> None:
    """导入即自检：差值对必须只覆盖排序策略、每个无序对恰好一次、且不含补充参照。"""
    seen: set[frozenset[str]] = set()
    for a, b in CONTRAST_PAIRS:
        if a not in RANKED_POLICIES or b not in RANKED_POLICIES:
            raise AssertionError(
                f"CONTRAST_PAIRS 含非排序策略 ({a!r}, {b!r})；"
                f"补充参照资源消耗不同，不参与同比例差值比较（协议 §5.4）"
            )
        if a == b:
            raise AssertionError(f"CONTRAST_PAIRS 含自比较 ({a!r}, {b!r})")
        key = frozenset((a, b))
        if key in seen:
            raise AssertionError(f"CONTRAST_PAIRS 里 {sorted(key)} 出现了两次（符号相反的两条）")
        seen.add(key)
    expected = {frozenset(p) for p in itertools.combinations(RANKED_POLICIES, 2)}
    if seen != expected:
        raise AssertionError(
            f"CONTRAST_PAIRS 未覆盖全部分组：缺少 {sorted(map(sorted, expected - seen))}，"
            f"多出 {sorted(map(sorted, seen - expected))}"
        )
    if (CONTRAST_PRIMARY_POLICY, CONTRAST_PRIMARY_REFERENCE) not in CONTRAST_PAIRS:
        raise AssertionError(
            f"主要比较 ({CONTRAST_PRIMARY_POLICY!r}, {CONTRAST_PRIMARY_REFERENCE!r}) "
            "必须在 CONTRAST_PAIRS 里"
        )
    if CONTRAST_PRIMARY_FRACTION not in TREATMENT_FRACTIONS:
        raise AssertionError(
            f"主要触达比例 {CONTRAST_PRIMARY_FRACTION!r} 不在预先固定的集合 "
            f"{TREATMENT_FRACTIONS} 中（协议 §6.1）"
        )


_check_contrast_pairs()

#: ``all_treat`` / ``none`` 在 :class:`PolicyScore.fraction` 里的记账值。
ALL_TREAT_FRACTION: float = 1.0
NONE_FRACTION: float = 0.0

#: 空策略（谁都不触达）的总体增量：``E[0 · τ(X)] = 0``。
#: 这是**定义值**，不是估出来的数——"不触达任何人"相对"不触达任何人"没有增量，
#: 也不含任何抽样不确定性。
#: 注意 :func:`snips_estimate` 在空策略下仍然如实返回 ``nan``（分母为 0，比值本身无定义）；
#: "按定义填 0"这一步只发生在 :func:`evaluate_policies` 的补充参照行上，
#: 并由 :attr:`PolicyScore.is_definitional` 显式标记，好让报告层把它与估计值区分开。
EMPTY_POLICY_VALUE: float = 0.0

#: 每条策略结果都带上的回顾性提示（口径：协议 §5.4、docs/02-分析协议.md §13）。
RETROSPECTIVE_NOTE: str = (
    "回顾性评估：这是留出集上已经发生的结果，用于比较既定规则，"
    "不是对这些客户将来行为的预测；项目没有成本数据，本结果不含成本口径的判断。"
)


@dataclass(frozen=True, slots=True)
class PolicyScore:
    """一个策略在某个触达比例下的评估结果。"""

    policy: str  # POLICY_NAMES 之一：'random' | 'purchase_probability' | 'control_probability'
    #             | 'uplift' | 'all_treat' | 'none'（定义见 POLICY_DEFINITIONS）
    fraction: float  # 触达比例；all_treat / none 记为 1.0 / 0.0
    n_selected: int  # 实际被选中的客户数
    ips_estimate: float  # SNIPS 估计的总体增量（绝对比例，如 0.0068）
    ci_low: float
    ci_high: float
    alpha: float
    bootstrap_seed: int
    n_bootstrap: int
    # 仅供描述，不作为策略优劣依据（有偏）
    observed_rate_selected: float
    note: str = ""

    @property
    def ci_defined(self) -> bool:
        """区间是否有定义（重抽样里估计量全为 nan 时无定义）。"""
        return not (math.isnan(self.ci_low) or math.isnan(self.ci_high))

    @property
    def is_definitional(self) -> bool:
        """该行是否为**定义值**而不是估计值。

        目前只有 ``none``：谁都不触达时总体增量按定义为 0，不含抽样不确定性，
        因此它的 0 与 [0, 0] 不是从数据估出来的。报告层应当据此把它渲染成参照点，
        并且**不要**拿它的区间宽度与其他策略比较。
        """
        return self.policy == "none" and self.fraction == NONE_FRACTION

    @property
    def estimate_defined(self) -> bool:
        """点估计是否有定义（分母为 0 时无定义）。"""
        return not math.isnan(self.ips_estimate)

    @property
    def ci_width(self) -> float:
        """区间宽度；无定义时为 nan。"""
        return self.ci_high - self.ci_low if self.ci_defined else math.nan


@dataclass(frozen=True, slots=True)
class PolicyContrast:
    """两个策略在**同一个触达比例**下的差值 ``V(policy) − V(reference_policy)`` 及其区间。

    差值取自**同一次共同 bootstrap 的配对抽样**：第 ``b`` 次重抽样里两个策略共用同一批
    索引，差值就是 ``draws[policy][b] − draws[reference][b]``。这样做才把"两个估计相关"
    这件事算进了差值的不确定性（独立抽样会给出偏宽的区间，见
    ``tests/test_stage4_policy_contrasts.py``）。

    区间为百分位区间：对差值重抽样中**两个策略都有定义**的那些次取 2.5%/97.5% 分位。
    """

    policy: str  # 被减的策略（a）
    reference_policy: str  # 作为对照的策略（b）；差值 = V(a) − V(b)
    fraction: float
    difference: float  # 点估计差 = V_hat(a) − V_hat(b)
    ci_low: float
    ci_high: float
    alpha: float
    bootstrap_seed: int
    n_bootstrap: int
    n_pairs_defined: int  # 两个策略都有定义的配对重抽样次数
    is_primary: bool
    note: str = ""

    @property
    def ci_defined(self) -> bool:
        return not (math.isnan(self.ci_low) or math.isnan(self.ci_high))

    @property
    def excludes_zero(self) -> bool:
        """差值区间是否排除 0。**无定义时返回 False**（不把"算不出来"当成有差异）。"""
        if not self.ci_defined or math.isnan(self.difference):
            return False
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low if self.ci_defined else math.nan

    def decision_note(self) -> str:
        """按差值区间给出的**唯一**结论口径（报告层直接用这句）。"""
        if not self.ci_defined or math.isnan(self.difference):
            return "差值区间无定义（重抽样里两个策略没有同时可比的样本），本次不做结论。"
        if self.excludes_zero:
            return (
                f"差值 {self.difference * 100:+.4f} 个百分点，"
                f"{100 * (1 - self.alpha):.0f}% 差值区间排除 0："
                "在这一份留出集上两个策略的总体增量**有可检出的差别**；"
                "这是**单次回顾性评估**的结果，不是对未来的承诺。"
            )
        return (
            f"差值 {self.difference * 100:+.4f} 个百分点，"
            f"{100 * (1 - self.alpha):.0f}% 差值区间包含 0："
            "**当前留出集无法区分这两个策略**（不能据此说任何一个更优，"
            "也不能说两者等价）。"
        )


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """一次策略评估的完整结果：各策略的边际结果 + 两两差值。"""

    scores: tuple[PolicyScore, ...]
    contrasts: tuple[PolicyContrast, ...]
    fractions: tuple[float, ...]
    alpha: float
    bootstrap_seed: int
    n_bootstrap: int


def _as_binary_array(values, *, name: str) -> np.ndarray:
    """把入参规整成 bool 数组，并拒绝非 0/1 的取值。

    宁可在这里报错，也不要把 ``2``、``-1``、``nan`` 静默当成 True——
    那些取值几乎一定意味着上游把某一列接错了。
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，实际 ndim={arr.ndim}")
    if arr.dtype == bool:
        return arr
    if arr.dtype.kind in "iuf":
        if arr.size and not bool(np.all((arr == 0) | (arr == 1))):
            raise ValueError(f"{name} 必须是 0/1 或 bool 数组，实际含 {arr.dtype} 的其他取值")
        return arr.astype(bool)
    raise ValueError(f"{name} 必须是 0/1 或 bool 数组，实际 dtype={arr.dtype}")


def _as_probability(value, *, name: str) -> float:
    prob = float(value)
    if math.isnan(prob) or not 0.0 < prob < 1.0:
        raise ValueError(f"{name} 必须落在开区间 (0, 1) 内，实际为 {value!r}")
    return prob


def _snips_core(sel_treat: np.ndarray, sel_ctrl: np.ndarray, y: np.ndarray, share: float) -> float:
    """**常数 e** 下的 SNIPS 主体（bootstrap 内层用，避免几千次重复校验输入）。

    ``sel_treat`` / ``sel_ctrl`` 分别是"策略要发、并且实际在干预组/对照组"的掩码，
    ``y`` 是 0/1 浮点结果，``share`` 是策略的触达比例 ``P(π=1)``。

    e 为常数时 ``1/e`` 与 ``1/(1-e)`` 在两个 Hájek 分量的分子分母里同时出现、整体约掉，
    于是 :func:`snips_estimate` 的加权形式精确退化为"两臂普通均值之差 × 触达比例"。
    两条路径的等价性有测试覆盖（``test_core_matches_public_estimate_for_constant_propensity``）；
    **只有 e 为常数时才能走这里**，本项目 e ≡ 0.5（协议 §2.2）满足该条件。
    """
    n_treat = int(sel_treat.sum())
    n_ctrl = int(sel_ctrl.sum())
    if share <= 0.0 or n_treat == 0 or n_ctrl == 0:
        # 分母为 0 的情形不可修补：策略没选到人（share = 0），
        # 或选到的人全在同一臂，都没有可比的差值。返回 nan 而不是抛异常。
        return math.nan
    mean_treat = float(y[sel_treat].mean())
    mean_ctrl = float(y[sel_ctrl].mean())
    return share * (mean_treat - mean_ctrl)


def snips_estimate(
    treatment: np.ndarray,
    y: np.ndarray,
    selected_treat: np.ndarray,
    propensity: float = PROPENSITY,
) -> float:
    r"""自归一化逆概率加权（SNIPS）估计：策略 π 触达「selected_treat 为 True」的客户时的总体增量。

    参数
    ----
    treatment:
        实际是否被干预（bool 数组，或 0/1 数组）。1 = 干预组（``Mens E-Mail``）。
    y:
        结果（bool / 0-1 数组），本项目为 ``conversion``。
    selected_treat:
        策略是否会给该客户发邮件（bool 数组）——注意是"是否发"，不是"是否被选中"，
        未发邮件的客户对增量估计无贡献（``π_i = 0``）。
    propensity:
        条件化倾向 e(x)，本项目恒为 0.5（协议 §2.2）。

    公式
    ----
    记 :math:`\pi_i = \mathbb{1}[\text{selected\_treat}_i]`，
    :math:`S_1 = \{i: \pi_i = 1, W_i = 1\}`，:math:`S_0 = \{i: \pi_i = 1, W_i = 0\}`：

    .. math::

        \hat V(\pi) \;=\; \underbrace{\frac{1}{n}\sum_i \pi_i}_{\hat P(\pi=1)}
        \left[
            \underbrace{\frac{\sum_{i \in S_1} Y_i / e}{\sum_{i \in S_1} 1 / e}}_{\hat E[Y(1)\mid \pi=1]}
            -
            \underbrace{\frac{\sum_{i \in S_0} Y_i / (1-e)}{\sum_{i \in S_0} 1 / (1-e)}}_{\hat E[Y(0)\mid \pi=1]}
        \right]

    **为什么是这个形式，而不是"只对匹配上的客户求购买率再乘触达比例"**
    （即协议 §5.2 的示意式，本模块保留为
    :func:`dsa.models.policy_simulation.illustrative_sketch_estimate` 只作反例）：

    1. 增量必须同时用到**两个**反事实分支。只统计"实际动作等于策略动作"的客户，
       在 ``π_i = 0`` 的那部分人里取到的是 ``W_i = 0`` 的**对照**结果，
       把它们和干预组结果混进同一个比值，估出来的是"结果的水平"，不是"增量"，
       而且天然被基线购买率抬高（``all_treat`` 时会直接给出干预组购买率本身）。
    2. 乘上 ``\hat P(\pi=1) = \frac1n\sum_i \pi_i`` 才把"被选中人群的效果"折算成
       **总体人均**增量。这一步是协议 §5.1 给 ``all_treat`` 记 fraction = 1.0 的直接后果：
       ``π ≡ 1`` 时该式退化为 ATE。
    3. 两个分量各自**自归一化**（Hájek 形式）：分母是各自的权重和。相比不做归一化的
       Horvitz–Thompson 差值形式，它在 ``e`` 有偏差、或在选取人群里干预/对照人数
       不平衡时都更稳，且结果天然落在 ``[-share, share]`` 内有界。

    为什么无偏（``e`` 为常数时是**精确**无偏）
    -----------------------------------------
    在 ``e ≡ 0.5`` 下两个 Hájek 分量恰好退化为 ``S_1``、``S_0`` 上的普通均值，于是

    .. math::

        \hat V = \frac{|S|}{n}\left(\bar Y_{S_1} - \bar Y_{S_0}\right)

    固定住选中的客户集合 ``S``（它只由 x 决定）后，随机化保证 ``W`` 在 ``S`` 内与潜在结果
    独立、且 ``P(W=1) = 0.5``，因此 ``\bar Y_{S_1} - \bar Y_{S_0}`` 对
    ``E[\tau \mid \pi=1]`` 条件无偏；再对 ``S`` 取期望即得 ``E[\pi(X)\tau(X)] = V(\pi)``。
    这就是完全随机化实验里"选中人群组间差值 × 选中比例"的经典形式。

    两个附带事实（都有测试覆盖）：

    * 常数 ``e`` 下两个分量里的 ``1/e`` 会在比值里约掉，因此**估计值对 e 的取值不敏感**；
      ``e`` 只影响"我们凭什么认为这个比值无偏"的解释，不影响算术结果。
    * 分母为 0 时返回 ``nan``（不抛异常）：没有选到干预组客户、或没选到对照组客户，
      差值无从计算。``nan`` 会被如实带进报告，不代填任何常量。

    Args:
        treatment: 实际是否被干预。
        y: 结果（0/1）。
        selected_treat: 策略是否会给该客户发邮件。
        propensity: 条件化倾向，默认 0.5。

    Returns:
        总体增量的点估计（绝对比例）；分母为 0 时返回 ``nan``——即"没选到干预组客户"、
        "没选到对照组客户"，或"一个人都没选"这三种情形。

    Raises:
        ValueError: 三个数组长度不一致；含非 0/1 取值；``propensity`` 不在 (0, 1) 内。
    """
    treat = _as_binary_array(treatment, name="treatment")
    selected = _as_binary_array(selected_treat, name="selected_treat")
    outcome = np.asarray(y)
    if outcome.dtype == bool:
        outcome = outcome.astype(float)
    else:
        outcome = _as_binary_array(y, name="y").astype(float)
    if not (treat.shape == selected.shape == outcome.shape):
        raise ValueError(
            f"treatment / y / selected_treat 长度必须一致，实际为 "
            f"{treat.shape[0]} / {outcome.shape[0]} / {selected.shape[0]}"
        )
    e = _as_probability(propensity, name="propensity")
    if outcome.size == 0:
        return math.nan
    # e 为常数时 1/e 在分子分母同时出现，这里直接按比例保留，便于非 0.5 的调用方核对。
    sel_treat = selected & treat
    sel_ctrl = selected & ~treat
    n_treat = int(sel_treat.sum())
    n_ctrl = int(sel_ctrl.sum())
    if n_treat == 0 or n_ctrl == 0:
        return math.nan
    num_treat = float(outcome[sel_treat].sum()) / e
    num_ctrl = float(outcome[sel_ctrl].sum()) / (1.0 - e)
    den_treat = n_treat / e
    den_ctrl = n_ctrl / (1.0 - e)
    share = float(selected.sum()) / float(selected.shape[0])
    return share * (num_treat / den_treat - num_ctrl / den_ctrl)


def random_scores(n: int, *, seed: int = DEFAULT_SEED) -> np.ndarray:
    """random 策略的打分（均匀随机，固定种子）。

    打分越大越优先触达（见 :func:`evaluate_policies`），因此均匀随机打分等价于
    "随机挑 q 比例的人发邮件"。用 ``default_rng(seed)`` 生成，同种子逐位复现，
    不触碰 numpy 全局随机状态。

    Args:
        n: 客户数。
        seed: 随机种子，默认项目固定种子。

    Returns:
        长度为 n、取值在 [0, 1) 的浮点数组。

    Raises:
        ValueError: n 非正。
    """
    if n <= 0:
        raise ValueError(f"n 必须为正整数，实际为 {n!r}")
    return np.random.default_rng(seed).random(n)


def _selected_mask(score: np.ndarray, fraction: float, *, name: str) -> np.ndarray:
    """按打分从高到低取前 ``fraction`` 比例，返回 bool 掩码。

    * 取法：``k = round(fraction * n)``（四舍五入到整数人数）。
    * 排序：``np.argsort(-score, kind="stable")``——**降序、并列时按原位置先后**。
      并列的打破规则必须固定，否则同一份打分在不同运行里会选出不同名单。
    """
    arr = np.asarray(score)
    if arr.ndim != 1:
        raise ValueError(f"{name} 打分必须是一维数组，实际 ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} 打分为空数组")
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} 打分必须是数值数组，实际 dtype={arr.dtype}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} 打分含 nan/inf，排序规则会变得不确定")
    n = arr.shape[0]
    k = int(fraction * n + 0.5)
    k = max(0, min(k, n))
    mask = np.zeros(n, dtype=bool)
    if k == 0:
        return mask
    if k == n:
        mask[:] = True
        return mask
    mask[np.argsort(-arr, kind="stable")[:k]] = True
    return mask


def _bootstrap_draws(
    *,
    y: np.ndarray,
    treatment: np.ndarray,
    specs: list[tuple[str, np.ndarray, float]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """共同 bootstrap 的**重抽样分布**：``{标签: 长度 n_bootstrap 的数组}``。

    ``specs`` 是 ``(标签, 打分, 触达比例)`` 列表；**所有标签共用同一批重抽样索引**
    （公共随机数）。这一点是差值区间的前提：只有索引相同，
    ``draws[a] - draws[b]`` 才是"同一份留出客户上两个策略的差值"的抽样分布；
    各自独立重抽样会把两者之间的相关性丢掉，差值区间会偏宽。

    每次重抽样内部**重新执行选取规则**（``score[idx]`` 再排序取前 q），
    而不是沿用原样本选出的名单——原因见模块 docstring。

    估计量无定义（选不到两臂）的那些次如实填 ``nan``，不做任何填补。
    """
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    draws: dict[str, np.ndarray] = {label: np.empty(n_bootstrap) for label, _, _ in specs}
    y_float = y.astype(float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_b = y_float[idx]
        t_b = treatment[idx]
        for label, score, fraction in specs:
            mask = _selected_mask(score[idx], fraction, name=label)
            draws[label][b] = _snips_core(mask & t_b, mask & ~t_b, y_b, float(mask.mean()))
    return draws


def _percentile_interval(values: np.ndarray, alpha: float) -> tuple[float, float, int]:
    """百分位区间：``(low, high, 无定义次数)``。全为 nan 时返回 ``(nan, nan, n)``。"""
    finite = values[np.isfinite(values)]
    undefined = int(values.size - finite.size)
    if finite.size == 0:
        return math.nan, math.nan, undefined
    low, high = np.percentile(finite, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high), undefined


def _contrast_draws(draws: Mapping[str, np.ndarray], policy: str, reference: str, fraction: float) -> np.ndarray:
    """配对差值重抽样：同一批索引下两个策略的抽样值相减。

    ``nan`` 参与相减的结果仍是 ``nan``（"这一次重抽样里至少有一个策略无定义"），
    由 :func:`_percentile_interval` 统一剔除并计数。
    """
    return draws[f"{policy}@{fraction}"] - draws[f"{reference}@{fraction}"]


def _validate_fractions(fractions: tuple[float, ...]) -> tuple[float, ...]:
    """触达比例必须非空、无重复、且全部来自 :data:`TREATMENT_FRACTIONS`。"""
    if not fractions:
        raise ValueError("fractions 不能为空：协议 §6.1 要求报告预先固定的触达比例")
    seen: set[float] = set()
    for q in fractions:
        q = float(q)
        if q in seen:
            raise ValueError(f"fractions 里有重复取值 {q!r}")
        seen.add(q)
        if q not in TREATMENT_FRACTIONS:
            raise ValueError(
                f"触达比例 {q!r} 不在预先固定的集合 {TREATMENT_FRACTIONS} 中。"
                "比例集合写在模块常量里，调用方不得扩展（协议 §6.1、§6.4）。"
            )
    return tuple(float(q) for q in fractions)


def _evaluate_policies_impl(
    *,
    y: np.ndarray,
    treatment: np.ndarray,
    scores: dict[str, np.ndarray],  # 策略名 -> 打分（越大越优先触达）
    fractions: tuple[float, ...],
    alpha: float,
    n_bootstrap: int,
    bootstrap_seed: int,
    contrast_pairs: tuple[tuple[str, str], ...],
) -> PolicyEvaluation:
    """评估主体：校验输入 -> 共同 bootstrap -> 组装边际结果与差值。

    所有约定（打分方向、``none`` 是定义值、``random`` 自动补、比例不得扩展、
    全部区间与差值共用同一批重抽样索引）见公开入口
    :func:`evaluate_policies_with_contrasts` 的 docstring。
    """
    treat = _as_binary_array(treatment, name="treatment")
    outcome = _as_binary_array(y, name="y").astype(float)
    if treat.shape != outcome.shape:
        raise ValueError(
            f"y 与 treatment 长度必须一致，实际为 {outcome.shape[0]} / {treat.shape[0]}"
        )
    n = treat.shape[0]
    if n == 0:
        raise ValueError("y / treatment 不能为空")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha 必须落在 (0, 1) 内，实际为 {alpha!r}")
    if n_bootstrap < 2:
        raise ValueError(f"n_bootstrap 至少为 2 才能给出百分位区间，实际为 {n_bootstrap!r}")
    fractions = _validate_fractions(tuple(fractions))

    if not scores:
        raise ValueError(f"scores 不能为空；至少需要 {RANKED_POLICIES} 中的一个策略打分")
    unknown = sorted(set(scores) - set(RANKED_POLICIES))
    if unknown:
        raise ValueError(
            f"scores 含未登记的策略名 {unknown}；合法取值为 {RANKED_POLICIES}。"
            "报告口径固定为协议 §5.1 的五种策略，不接受额外策略。"
        )
    score_arrays: dict[str, np.ndarray] = {}
    for name, values in scores.items():
        arr = np.asarray(values)
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(
                f"策略 {name!r} 的打分长度必须等于样本量 {n}，实际 shape={arr.shape}"
            )
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError(f"策略 {name!r} 的打分含 nan/inf")
        score_arrays[name] = arr.astype(float)
    if "random" not in score_arrays:
        # 随机基线不允许缺失：协议 §5.1 把它列为必须比较的策略之一。
        score_arrays["random"] = random_scores(n, seed=DEFAULT_SEED)

    ranked = sorted(score_arrays)
    specs: list[tuple[str, np.ndarray, float]] = []
    masks: dict[tuple[str, float], np.ndarray] = {}
    for name in ranked:
        for q in fractions:
            masks[(name, q)] = _selected_mask(score_arrays[name], q, name=name)
            specs.append((f"{name}@{q}", score_arrays[name], q))
    # 补充参照：all_treat 全部触达、none 全部不触达。
    # 打分传常数数组即可：_selected_mask 在 q=1.0 / 0.0 时分别返回全选 / 全不选。
    zero_scores = np.zeros(n)
    masks[("all_treat", ALL_TREAT_FRACTION)] = np.ones(n, dtype=bool)
    masks[("none", NONE_FRACTION)] = np.zeros(n, dtype=bool)
    specs.append((f"all_treat@{ALL_TREAT_FRACTION}", zero_scores, ALL_TREAT_FRACTION))
    specs.append((f"none@{NONE_FRACTION}", zero_scores, NONE_FRACTION))

    intervals: dict[str, tuple[float, float, int]] = {}
    draws = _bootstrap_draws(
        y=outcome,
        treatment=treat,
        specs=specs,
        n_bootstrap=int(n_bootstrap),
        seed=int(bootstrap_seed),
    )
    for label, values in draws.items():
        intervals[label] = _percentile_interval(values, float(alpha))

    results: list[PolicyScore] = []
    for name in ranked:
        for q in fractions:
            mask = masks[(name, q)]
            n_selected = int(mask.sum())
            low, high, undefined = intervals[f"{name}@{q}"]
            results.append(
                PolicyScore(
                    policy=name,
                    fraction=q,
                    n_selected=n_selected,
                    ips_estimate=snips_estimate(treat, outcome, mask, PROPENSITY),
                    ci_low=low,
                    ci_high=high,
                    alpha=float(alpha),
                    bootstrap_seed=int(bootstrap_seed),
                    n_bootstrap=int(n_bootstrap),
                    observed_rate_selected=(
                        float(outcome[mask].mean()) if n_selected else math.nan
                    ),
                    note=_ranked_note(name, q, n_selected, n, undefined),
                )
            )
    for name, q in (("none", NONE_FRACTION), ("all_treat", ALL_TREAT_FRACTION)):
        mask = masks[(name, q)]
        n_selected = int(mask.sum())
        low, high, undefined = intervals[f"{name}@{q}"]
        if name == "none":
            # 空策略特判：增量按定义为 0，也没有抽样不确定性，因此区间是退化的 [0, 0]。
            # 这里**不动估计量**（snips_estimate 在空策略下依旧如实返回 nan），
            # 只在组装报告行时填入定义值，并用 is_definitional 标记出来。
            low = high = EMPTY_POLICY_VALUE
        results.append(
            PolicyScore(
                policy=name,
                fraction=q,
                n_selected=n_selected,
                ips_estimate=(
                    EMPTY_POLICY_VALUE if name == "none" else snips_estimate(
                        treat, outcome, mask, PROPENSITY
                    )
                ),
                ci_low=low,
                ci_high=high,
                alpha=float(alpha),
                bootstrap_seed=int(bootstrap_seed),
                n_bootstrap=int(n_bootstrap),
                observed_rate_selected=(float(outcome[mask].mean()) if n_selected else math.nan),
                note=_supplement_note(name, n_selected, n, undefined),
            )
        )
    results.sort(key=lambda r: (r.fraction, r.policy))

    contrasts = _build_contrasts(
        draws=draws,
        point_estimates={(r.policy, r.fraction): r.ips_estimate for r in results},
        available=set(ranked),
        fractions=fractions,
        contrast_pairs=contrast_pairs,
        alpha=float(alpha),
        bootstrap_seed=int(bootstrap_seed),
        n_bootstrap=int(n_bootstrap),
    )
    return PolicyEvaluation(
        scores=tuple(results),
        contrasts=contrasts,
        fractions=fractions,
        alpha=float(alpha),
        bootstrap_seed=int(bootstrap_seed),
        n_bootstrap=int(n_bootstrap),
    )


def _build_contrasts(
    *,
    draws: Mapping[str, np.ndarray],
    point_estimates: Mapping[tuple[str, float], float],
    available: set[str],
    fractions: tuple[float, ...],
    contrast_pairs: tuple[tuple[str, str], ...],
    alpha: float,
    bootstrap_seed: int,
    n_bootstrap: int,
) -> tuple[PolicyContrast, ...]:
    """按 :data:`CONTRAST_PAIRS` 组装差值区间（只对**本次有打分**的策略）。"""
    out: list[PolicyContrast] = []
    for q in fractions:
        for policy, reference in contrast_pairs:
            if policy not in available or reference not in available:
                # 本次没有给这个策略打分（例如模拟里只评估 uplift）：不编造差值。
                continue
            diff_draws = _contrast_draws(draws, policy, reference, q)
            low, high, undefined = _percentile_interval(diff_draws, alpha)
            n_pairs_defined = int(diff_draws.size - undefined)
            is_primary = (
                q == CONTRAST_PRIMARY_FRACTION
                and policy == CONTRAST_PRIMARY_POLICY
                and reference == CONTRAST_PRIMARY_REFERENCE
            )
            a = point_estimates[(policy, q)]
            b = point_estimates[(reference, q)]
            difference = a - b if math.isfinite(a) and math.isfinite(b) else math.nan
            contrast = PolicyContrast(
                policy=policy,
                reference_policy=reference,
                fraction=q,
                difference=difference,
                ci_low=low,
                ci_high=high,
                alpha=alpha,
                bootstrap_seed=bootstrap_seed,
                n_bootstrap=n_bootstrap,
                n_pairs_defined=n_pairs_defined,
                is_primary=is_primary,
                note="",
            )
            note = (
                f"差值 = V({policy}) − V({reference})，取自同一次共同 bootstrap 的"
                f"**配对**抽样（{n_bootstrap} 次重抽样中 {n_pairs_defined} 次两个策略同时有定义）。"
                f"{PRIMARY_CONTRAST_NOTE if is_primary else EXPLORATORY_CONTRAST_NOTE}"
                f"{contrast.decision_note()}"
            )
            if undefined:
                note += (
                    f" 有 {undefined} 次重抽样的差值无定义（两个策略没有同时可比的样本），"
                    "这些次不计入区间。"
                )
            note += RETROSPECTIVE_NOTE
            out.append(
                PolicyContrast(
                    policy=policy,
                    reference_policy=reference,
                    fraction=q,
                    difference=difference,
                    ci_low=low,
                    ci_high=high,
                    alpha=alpha,
                    bootstrap_seed=bootstrap_seed,
                    n_bootstrap=n_bootstrap,
                    n_pairs_defined=n_pairs_defined,
                    is_primary=is_primary,
                    note=note,
                )
            )
    out.sort(key=lambda c: (c.fraction, not c.is_primary, c.policy, c.reference_policy))
    return tuple(out)


def evaluate_policies_with_contrasts(
    *,
    y: np.ndarray,
    treatment: np.ndarray,
    scores: dict[str, np.ndarray],  # 策略名 -> 打分（越大越优先触达）
    fractions: tuple[float, ...] = TREATMENT_FRACTIONS,
    alpha: float = DEFAULT_ALPHA,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    bootstrap_seed: int = DEFAULT_SEED,
    contrast_pairs: tuple[tuple[str, str], ...] = CONTRAST_PAIRS,
) -> PolicyEvaluation:
    """在固定触达比例下评估各策略，**并给出策略之间的差值区间**。

    每个 (策略, 比例) 组合给出一条 :class:`PolicyScore`：SNIPS 点估计 + 百分位
    bootstrap 区间。``all_treat`` / ``none`` 作为**补充参照**各出一条（0.0 / 1.0 记账），
    它们的资源消耗与固定比例策略不同，不能与固定比例策略直接排名（协议 §5.4）。

    差值（:class:`PolicyContrast`）在**同一次** bootstrap 内配对相减得到，
    因此它把"两个策略跑在同一份留出客户上、估计相关"这件事算进去了。
    **判断策略差异只能看差值区间**，不能看两条边际区间是否重叠。

    约定
    ----
    * 打分**越大越优先触达**；比例取前 ``round(q * n)`` 名（并列按原位置先后，规则固定）。
    * ``none`` 那一行是**定义值**：谁都不触达时总体增量按定义为 0，区间为退化的 [0, 0]，
      ``is_definitional`` 为 True。注意 :func:`snips_estimate` 对空选择仍然返回 ``nan``
      （分母为 0），"按定义填 0"只发生在这一行的组装里，不是估计量的输出。
    * ``scores`` 的键只能取 :data:`RANKED_POLICIES` 中的名字；``'random'`` 缺失时
      用 :func:`random_scores` 以固定种子补上——协议 §5.1 要求随机基线**必须**在报告里。
    * ``fractions`` 只能取 :data:`TREATMENT_FRACTIONS` 的子集，不得扩展；
      默认值即全部五档，一次调用就满足"全部报告"的要求（协议 §6.1）。
    * 全部边际区间与全部差值共用同一批重抽样索引（公共随机数）。
    * ``contrast_pairs`` 默认即 :data:`CONTRAST_PAIRS`；只对本次**有打分**的策略出差值，
      没打分的策略不会被编造出来。

    Args:
        y: 结果（0/1 或 bool）。本项目为 ``conversion``。
        treatment: 实际是否被干预（0/1 或 bool）。
        scores: 策略名 -> 打分数组。至少一个；缺 ``'random'`` 会自动补。
        fractions: 要评估的触达比例，必须是 :data:`TREATMENT_FRACTIONS` 的子集。
        alpha: 显著性水平，区间为 ``1 - alpha`` 百分位区间。
        n_bootstrap: bootstrap 重复次数。
        bootstrap_seed: bootstrap 种子，默认项目固定种子。
        contrast_pairs: 要报告的 ``(策略, 对照)`` 对。

    Returns:
        :class:`PolicyEvaluation`。``scores`` 按 ``(fraction, policy)`` 升序；
        ``contrasts`` 按 ``(fraction, 主要比较优先, 策略, 对照)`` 升序。

    Raises:
        ValueError: 输入长度不一致、含非 0/1 取值、打分名非法、比例越界或重复、
            ``alpha`` 不在 (0, 1) 内、``n_bootstrap`` 小于 2、差值对含补充参照。
    """
    for policy, reference in contrast_pairs:
        for name in (policy, reference):
            if name not in RANKED_POLICIES:
                raise ValueError(
                    f"差值对里的 {name!r} 不在排序策略 {RANKED_POLICIES} 中。"
                    "补充参照（all_treat/none）资源消耗不同，不参与同比例差值比较"
                    "（协议 §5.4）。"
                )
    return _evaluate_policies_impl(
        y=y,
        treatment=treatment,
        scores=scores,
        fractions=fractions,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        contrast_pairs=contrast_pairs,
    )


def evaluate_policies(
    *,
    y: np.ndarray,
    treatment: np.ndarray,
    scores: dict[str, np.ndarray],
    fractions: tuple[float, ...] = TREATMENT_FRACTIONS,
    alpha: float = DEFAULT_ALPHA,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    bootstrap_seed: int = DEFAULT_SEED,
) -> list[PolicyScore]:
    """只取边际结果的兼容入口（内部走 :func:`evaluate_policies_with_contrasts`）。

    ⚠️ **不要用返回的区间互相重叠来判断策略差异**——那不是一个差异检验。
    需要比较策略时用 :func:`evaluate_policies_with_contrasts` 的差值区间。
    """
    evaluation = evaluate_policies_with_contrasts(
        y=y,
        treatment=treatment,
        scores=scores,
        fractions=fractions,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    return list(evaluation.scores)


def _bootstrap_suffix(undefined: int) -> str:
    if undefined <= 0:
        return ""
    return (
        f" bootstrap 中有 {undefined} 次重抽样比值无定义（该次不计入区间），"
        "区间因此建立在剩余重抽样上。"
    )


def _ranked_note(name: str, q: float, n_selected: int, n: int, undefined: int) -> str:
    """按打分排序的策略的说明文字。"""
    definition = POLICY_DEFINITIONS[name]
    return (
        f"按 {name} 打分降序取前 {q:.0%}（{n_selected}/{n} 人）。"
        f"该策略的打分来源：{definition.score_source}。"
        "估计量为 SNIPS：选中医访/对照两臂的购买率之差 × 触达比例，"
        "e(x)=0.5 由实验设计推导（协议 §2.2），不是估计出来的。"
        f"{_bootstrap_suffix(undefined)}"
        f"observed_rate_selected 只是选中人群的观测购买率，有偏，不作为策略优劣依据。"
        f"策略之间有没有差异，只能看 evaluate_policies_with_contrasts 给出的差值区间。"
        f"{RETROSPECTIVE_NOTE}"
    )


def _supplement_note(name: str, n_selected: int, n: int, undefined: int) -> str:
    """补充参照策略的说明文字。"""
    if name == "none":
        return (
            f"补充参照：全部不触达（0/{n} 人）。定义上相对自身没有差异，增量恒为 0，"
            "区间为退化的 [0, 0]——这一行是定义值而非估计值（见 is_definitional），"
            "不要拿它的区间宽度与其他策略比较。"
            "SNIPS 的分母在空策略下为 0、比值本身无定义，"
            "snips_estimate 对空选择如实返回 nan，报告层才按定义填 0。"
            f"{RETROSPECTIVE_NOTE}"
        )
    return (
        f"补充参照：全部触达（{n_selected}/{n} 人），估计量即总体平均效果。"
        "它消耗的资源与其他比例策略不同，只能并列展示，不能与固定比例策略直接排名。"
        f"{_bootstrap_suffix(undefined)}"
        f"{RETROSPECTIVE_NOTE}"
    )
