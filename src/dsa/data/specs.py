"""数据集与实验的规格定义。

本模块是**可执行的口径来源**：导入程序按这里的规格写入数据库。
人类可读的镜像见 `docs/metrics.yaml` 与 `docs/03-指标字典.md`；
`tests/test_specs_consistency.py` 会断言两者一致，防止口径漂移。

修改本文件中的任何口径都必须同步更新：
- docs/02-分析协议.md（若涉及统计口径）
- docs/03-指标字典.md 与 docs/metrics.yaml（若涉及字段或指标）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

# ============================================================================
#  数据集
# ============================================================================


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """一个不可变的数据版本。"""

    dataset_key: str
    version_label: str
    file_name: str
    file_sha256: str
    row_count: int
    source_url: str
    file_bytes: int | None = None
    grain: str = "customer"
    notes: str | None = None

    def relative_path(self) -> Path:
        """相对于项目 data/raw 的文件路径。"""
        return Path("data") / "raw" / self.file_name


HILLSTROM = DatasetSpec(
    dataset_key="hillstrom",
    version_label="hillstrom@2008-03-20",
    file_name="hillstrom_email_analytics_2008-03-20.csv",
    # 下载自原始发布方，哈希用于复现与防篡改（见 docs/00-数据来源核查记录.md）
    file_sha256="0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece",
    file_bytes=3_964_977,
    row_count=64_000,
    source_url=(
        "http://www.minethatdata.com/"
        "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    ),
    notes=(
        "原始实验说明来自 MineThatData 博客（核查时该页不可达，"
        "改用 TensorFlow Datasets 转载核对，见 docs/00 §3.1）。"
        "源数据含 6562 行字段完全相同的记录，禁止按整行去重。"
    ),
)


# ============================================================================
#  实验
# ============================================================================


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """实验的一个组。"""

    arm_key: str
    segment_value: str          # 与源数据 segment 字段取值一一对应
    role: str                   # treatment | control
    expected_ratio: Decimal     # SRM 检验使用的预期比例


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    """预注册的一次比较。"""

    comparison_key: str
    treatment_arm_key: str
    control_arm_key: str
    kind: str                   # confirmatory | exploratory
    alpha: Decimal
    sided: int
    multiplicity_note: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """一个已结束、窗口固定的随机实验。"""

    experiment_key: str
    name: str
    dataset: DatasetSpec
    outcome_window_days: int
    expected_ratio_source: str
    expected_ratio_note: str
    arms: tuple[ArmSpec, ...]
    comparisons: tuple[ComparisonSpec, ...]
    randomization_unit: str = "customer"
    analysis_unit: str = "customer"
    outcome_window_fixed: bool = True
    notes: str | None = None
    extra: dict = field(default_factory=dict, compare=False)

    def arm(self, arm_key: str) -> ArmSpec:
        for a in self.arms:
            if a.arm_key == arm_key:
                return a
        raise KeyError(f"未知的 arm_key: {arm_key}")

    @property
    def expected_ratio_sum(self) -> Decimal:
        return sum((a.expected_ratio for a in self.arms), Decimal(0))


_THIRD = Decimal("0.3333333333")

HILLSTROM_EMAIL_V1 = ExperimentSpec(
    experiment_key="hillstrom_email_v1",
    name="Hillstrom 邮件营销随机实验",
    dataset=HILLSTROM,
    outcome_window_days=14,
    # 原始实验说明明确写出 1/3 each，因此预期比例有权威来源。
    # 协议 §11.1 禁止从观测比例反推预期比例。
    expected_ratio_source="original_description",
    expected_ratio_note=(
        "原始说明：'1/3 were randomly chosen to receive an e-mail campaign featuring Mens / "
        "1/3 ... Womens / 1/3 ... not receive'。见 docs/00-数据来源核查记录.md §3.3。"
    ),
    arms=(
        ArmSpec("mens", "Mens E-Mail", "treatment", _THIRD),
        ArmSpec("womens", "Womens E-Mail", "treatment", _THIRD),
        ArmSpec("control", "No E-Mail", "control", _THIRD),
    ),
    comparisons=(
        ComparisonSpec(
            comparison_key="primary",
            treatment_arm_key="mens",
            control_arm_key="control",
            kind="confirmatory",
            alpha=Decimal("0.05"),
            sided=2,
        ),
        ComparisonSpec(
            comparison_key="secondary_womens",
            treatment_arm_key="womens",
            control_arm_key="control",
            kind="exploratory",
            alpha=Decimal("0.05"),
            sided=2,
            multiplicity_note=(
                "探索性比较。与主要比较同时报告时须说明多重比较问题，"
                "并给出 Bonferroni 校正后的判定 α = 0.025。"
            ),
        ),
    ),
    notes=(
        "主要比较为 Mens E-Mail vs No E-Mail。选定主要比较时已知两干预组的观测效果，"
        "属已知的选择性偏倚来源，已在 docs/02-分析协议.md §5.3 如实记录。"
        "两个干预组之间的直接对比本版不做。"
    ),
)


# ============================================================================
#  源数据列定义（导入时用于校验与按序读取）
# ============================================================================

#: 源 CSV 的列顺序。导入按位置读取，因此顺序变化必须被发现。
SOURCE_COLUMNS: tuple[str, ...] = (
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
)

#: 实验前特征列（允许作为建模输入）
#:
#: 注意包含 ``history_band``：它是 ``history_segment`` 的**派生列**
#: （由 ``customers`` 表的生成列自动维护，见 sql/001_schema.sql），
#: 来源是实验前的历史消费分箱，因此不构成泄漏。
#: 它给线性模型提供 one-hot 编码无法表达的单调序信息。
#: 与之对应的 POST_TREATMENT_COLUMNS 仍是绝对禁区。
PRE_TREATMENT_COLUMNS: tuple[str, ...] = (
    "recency",
    "history_segment",
    "history",
    "history_band",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
)

#: 干预后结果列（**禁止**作为建模输入）
POST_TREATMENT_COLUMNS: tuple[str, ...] = ("visit", "conversion", "spend")

#: 分组列
ASSIGNMENT_COLUMN = "segment"
