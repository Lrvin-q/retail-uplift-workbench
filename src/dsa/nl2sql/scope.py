"""受控 NL2SQL —— 查询范围与 schema 供给（阶段八增强项）。

本模块回答两个问题：

1. **哪些对象可以被生成出来的 SQL 引用？** —— ``ALLOWED_TABLES`` / ``ALLOWED_VIEWS``
   与 :func:`in_scope`。白名单之外的任何对象（含 ``pg_*`` / ``information_schema``）
   一律拒绝，这是 docs/08-工具协议.md §1「受限」「参数化」在自由 SQL 场景下的等价物。
2. **模型需要知道什么才写得出正确 SQL？** —— :func:`schema_prompt`。
   它必须给出**表粒度、主键、关联键、字段中文含义、禁止使用的字段**，
   以及本项目最容易踩的那个坑：RetailHero 的 ``purchase_sum`` 是**交易级**字段，
   被冗余到每一行商品明细上，直接 ``sum()`` 会虚高（实测 5.81 ~ 11.41 倍）
   且**不报错、不为空** —— 一个静默的错数（见 docs/日志.md L026、
   docs/metrics.yaml 的 ``transaction_amount_redundant``）。

为什么粒度必须写进提示词
------------------------
「一行代表什么」决定了每个聚合函数是否合法：``count(*)`` 在客户级表上是客户数，
在明细级表上是行数。模型不看粒度就会给出「看起来对、实际差 5 倍」的答案，
而这类错误**不会有任何报错**。所以粒度是本模块的一等公民，而不是注释。

铁律（AGENTS.md）
-----------------
- §2.2 分母口径不可动摇：分母 = 全部被随机分配的客户，不得按干预后行为过滤。
- §2.3 特征只能用决策前信息：``visit`` / ``conversion`` / ``spend`` /
  ``treatment_flg`` / ``target`` 不得作为特征或过滤条件。
- §2.5 数据集不混用：Hillstrom（``customers`` / ``outcomes`` …）与
  RetailHero（``rh_*``）是**两个独立数据集**，禁止 JOIN 或跨集拼接客户。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "ALLOWED_TABLES",
    "ALLOWED_VIEWS",
    "HILLSTROM_PREFIX_TABLES",
    "RETAILHERO_PREFIX",
    "DatasetKind",
    "FieldSpec",
    "ForbiddenField",
    "TableSpec",
    "allowed_object_names",
    "dataset_of",
    "forbidden_columns",
    "in_scope",
    "is_system_object",
    "schema_prompt",
    "table_grain",
]


# ============================================================================
#  数据结构
# ============================================================================


#: 粒度取值。**粒度决定聚合函数是否合法**，不是注释里的形容词。
DatasetKind = str  # "hillstrom" | "retailhero" | "meta"

GRAIN_CUSTOMER = "客户级（一行 = 一名客户）"
GRAIN_ASSIGNMENT = "客户-实验级（一行 = 一名客户在一次实验中的分组）"
GRAIN_TXN_PRODUCT = "交易-商品明细级（一行 = 一笔交易中的一个商品）"
GRAIN_TRANSACTION = "交易级（一行 = 一笔交易）"
GRAIN_PRODUCT = "商品级（一行 = 一个 SKU）"
GRAIN_SOURCE_FILE = "源文件级（一行 = 一个源文件）"

#: RetailHero 的命名空间前缀。跨数据集拼接必须靠它识别（AGENTS.md §2.5）。
RETAILHERO_PREFIX = "rh_"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """一个允许引用的字段：类型 + 中文含义 + 必读提醒。"""

    name: str
    type: str
    meaning: str
    role: str = "attribute"
    note: str = ""

    def render(self) -> str:
        bits = [f"`{self.name}`", self.type, self.meaning]
        if self.note:
            bits.append(f"⚠️ {self.note}")
        return " — ".join(bits)


@dataclass(frozen=True, slots=True)
class ForbiddenField:
    """禁止使用的字段：为什么禁、允许的替代写法是什么。"""

    name: str
    reason: str
    instead: str = ""

    def render(self) -> str:
        tail = f"（改用：{self.instead}）" if self.instead else ""
        return f"`{self.name}` —— {self.reason}{tail}"


@dataclass(frozen=True, slots=True)
class TableSpec:
    """白名单里的一个表或视图。

    Attributes:
        name: 表名（全小写，与 PostgreSQL 中的实际对象名一致）。
        kind: ``hillstrom`` / ``retailhero`` / ``meta``。用于数据集不混用检查。
        grain: 一行代表什么。**聚合函数是否合法由它决定。**
        primary_key: 主键（业务键优先，因为它是模型看得懂的关联键）。
        join_keys: 与其他表的关联键。
        description: 这个表是干什么的。
        fields: 允许引用的字段及其中文含义。
        forbidden: 明确禁止使用的字段（干预后信息、标签等）。
    """

    name: str
    kind: DatasetKind
    grain: str
    primary_key: str
    join_keys: tuple[str, ...]
    description: str
    fields: tuple[FieldSpec, ...] = ()
    forbidden: tuple[ForbiddenField, ...] = ()
    is_view: bool = False

    @property
    def is_customer_grain(self) -> bool:
        """一行是否恰好是一名客户（决定 ``count(*)`` 能否当客户数用）。"""
        return self.grain in (GRAIN_CUSTOMER, GRAIN_ASSIGNMENT)

    @property
    def is_multi_row_per_customer(self) -> bool:
        """关联到本对象是否会**放大行数**（关联扩张的判据）。"""
        return self.grain in (GRAIN_TXN_PRODUCT, GRAIN_TRANSACTION)

    def render(self) -> str:
        lines = [
            f"#### {self.name}（{self.grain}）",
            f"- 作用：{self.description}",
            f"- 主键：{self.primary_key}",
            f"- 关联键：{', '.join(self.join_keys) if self.join_keys else '（无）'}",
        ]
        if self.fields:
            lines.append("- 允许字段：")
            lines.extend(f"    - {f.render()}" for f in self.fields)
        if self.forbidden:
            lines.append("- 禁止使用：")
            lines.extend(f"    - {f.render()}" for f in self.forbidden)
        return "\n".join(lines)


# ============================================================================
#  维度定义（与 sql/001_schema.sql、sql/005_retailhero_schema.sql 逐一对应）
# ============================================================================

#: Hillstrom 侧的客户级表。用于「数据集不混用」的判定。
HILLSTROM_PREFIX_TABLES: tuple[str, ...] = (
    "customers",
    "outcomes",
    "assignments",
    "experiments",
    "experiment_arms",
    "experiment_comparisons",
)


_HILLSTROM_FORBIDDEN: tuple[ForbiddenField, ...] = (
    ForbiddenField(
        "visit",
        "干预后行为，仅可作为被统计的结果，**不得**用来筛选分母或作为特征",
        "分子写成 count(*) filter (where o.visited)",
    ),
    ForbiddenField(
        "conversion",
        "干预后行为，仅可作为被统计的结果，**不得**用来筛选分母",
        "分子写成 count(*) filter (where o.converted)",
    ),
    ForbiddenField(
        "spend",
        "干预后金额，仅可作为被统计的结果，**不得**用来筛选分母",
        "人均消费 = coalesce(sum(o.spend),0) / count(*)",
    ),
    ForbiddenField(
        "segment",
        "源数据中的随机分组取值，是**分配**而不是特征；分组统计请用 arm_key/role",
        "join experiment_arms ar on ar.arm_id = a.arm_id 后用 ar.arm_key",
    ),
)


ALLOWED_TABLES: dict[str, TableSpec] = {
    # ------------------------------------------------------------ Hillstrom
    "customers": TableSpec(
        name="customers",
        kind="hillstrom",
        grain=GRAIN_CUSTOMER,
        primary_key="customer_id",
        join_keys=("customer_id", "dataset_id", "customer_key"),
        description="客户维度表，只含实验前信息。**刻意不含 visit/conversion/spend**",
        fields=(
            FieldSpec("customer_id", "bigint", "内部客户 ID，与 assignments/outcomes 关联的键"),
            FieldSpec("customer_key", "text", "带数据集命名空间的业务键，形如 `hillstrom#123`，**禁止跨数据集拼接**"),
            FieldSpec("dataset_id", "bigint", "数据版本外键"),
            FieldSpec("source_row_number", "integer", "源文件行号"),
            FieldSpec("recency", "smallint", "距上次购买的月数，取值 1-12", "pre_treatment_feature"),
            FieldSpec("history_segment", "text", "历史消费金额分箱，7 档如 `1) $0 - $100`", "pre_treatment_feature"),
            FieldSpec("history_band", "smallint", "history_segment 的档位序数 1-7（派生列）", "pre_treatment_feature"),
            FieldSpec("history", "numeric", "历史消费金额", "pre_treatment_feature"),
            FieldSpec("mens", "boolean", "是否购买过男装", "pre_treatment_feature"),
            FieldSpec("womens", "boolean", "是否购买过女装", "pre_treatment_feature"),
            FieldSpec("zip_code", "text", "居住地类型：Surburban/Urban/Rural（原始拼写就是 Surburban，勿改）", "pre_treatment_feature"),
            FieldSpec("newbie", "boolean", "是否新客户", "pre_treatment_feature"),
            FieldSpec("channel", "text", "历史购买渠道：Web/Phone/Multichannel", "pre_treatment_feature"),
        ),
    ),
    "outcomes": TableSpec(
        name="outcomes",
        kind="hillstrom",
        grain=GRAIN_CUSTOMER,
        primary_key="outcome_id",
        join_keys=("experiment_id", "customer_id"),
        description="一名客户在一次实验固定窗口内的结果。未购买时 spend = 0（由 CHECK 保证）",
        fields=(
            FieldSpec("outcome_id", "bigint", "主键"),
            FieldSpec("experiment_id", "bigint", "实验外键"),
            FieldSpec("customer_id", "bigint", "客户外键"),
            FieldSpec("window_days", "smallint", "观察窗口天数，本数据集为 14，固定不可改"),
            FieldSpec("visited", "boolean", "是否访问网站"),
            FieldSpec("converted", "boolean", "是否购买"),
            FieldSpec("spend", "numeric", "消费金额；未购买必为 0"),
        ),
        forbidden=_HILLSTROM_FORBIDDEN[:3],
    ),
    "assignments": TableSpec(
        name="assignments",
        kind="hillstrom",
        grain=GRAIN_ASSIGNMENT,
        primary_key="assignment_id",
        join_keys=("experiment_id", "customer_id", "arm_id"),
        description="一名客户在一次实验中的分组。**指标分母 = 本表按组计数**",
        fields=(
            FieldSpec("assignment_id", "bigint", "主键"),
            FieldSpec("experiment_id", "bigint", "实验外键"),
            FieldSpec("customer_id", "bigint", "客户外键"),
            FieldSpec("arm_id", "bigint", "组外键，对应 experiment_arms.arm_id"),
            FieldSpec("assigned_at", "timestamptz", "写入本表的时间，**不是**实验的实际分配时间"),
        ),
        forbidden=(
            ForbiddenField(
                "（任何干预后条件写进 where）",
                "随机化是按客户做的，按干预后行为过滤分母等于换了一个人群，破坏可比性",
                "过滤只写实验与组别，结果字段只进分子",
            ),
        ),
    ),
    "experiments": TableSpec(
        name="experiments",
        kind="hillstrom",
        grain="实验级（一行 = 一个实验配置）",
        primary_key="experiment_id",
        join_keys=("experiment_id", "dataset_id"),
        description="实验配置。本数据集只有一个：`hillstrom_email_v1`",
        fields=(
            FieldSpec("experiment_id", "bigint", "主键"),
            FieldSpec("experiment_key", "text", "实验标识，如 `hillstrom_email_v1`"),
            FieldSpec("dataset_id", "bigint", "数据版本外键"),
            FieldSpec("name", "text", "实验名称"),
            FieldSpec("randomization_unit", "text", "随机化单位，恒为 customer"),
            FieldSpec("analysis_unit", "text", "分析单位，恒为 customer"),
            FieldSpec("outcome_window_days", "smallint", "结果窗口天数"),
            FieldSpec("outcome_window_fixed", "boolean", "窗口是否固定；true 表示不支持任意日期筛选"),
            FieldSpec("expected_ratio_source", "text", "SRM 预期比例的权威来源，禁止从观测比例反推"),
        ),
    ),
    "experiment_arms": TableSpec(
        name="experiment_arms",
        kind="hillstrom",
        grain="实验组级（一行 = 一个组）",
        primary_key="arm_id",
        join_keys=("experiment_id",),
        description="实验组定义：mens / womens / control",
        fields=(
            FieldSpec("arm_id", "bigint", "主键"),
            FieldSpec("experiment_id", "bigint", "实验外键"),
            FieldSpec("arm_key", "text", "组标识：mens / womens / control"),
            FieldSpec("segment_value", "text", "与源数据 segment 字段一一对应"),
            FieldSpec("role", "text", "treatment 或 control"),
            FieldSpec("expected_ratio", "numeric", "设计上的预期分配比例，仅供 SRM 检验"),
        ),
    ),
    "experiment_comparisons": TableSpec(
        name="experiment_comparisons",
        kind="hillstrom",
        grain="比较级（一行 = 一个预注册比较）",
        primary_key="comparison_id",
        join_keys=("experiment_id", "treatment_arm_id", "control_arm_id"),
        description="预注册的比较：primary（确认性）与 secondary_womens（探索性）",
        fields=(
            FieldSpec("comparison_id", "bigint", "主键"),
            FieldSpec("experiment_id", "bigint", "实验外键"),
            FieldSpec("comparison_key", "text", "比较标识：primary / secondary_womens"),
            FieldSpec("treatment_arm_id", "bigint", "干预组"),
            FieldSpec("control_arm_id", "bigint", "对照组"),
            FieldSpec("kind", "text", "confirmatory 或 exploratory"),
            FieldSpec("alpha", "numeric", "显著性水平"),
            FieldSpec("sided", "smallint", "单侧 1 / 双侧 2"),
        ),
    ),
    # ------------------------------------------------------------ 元数据（只读，只允许少量字段）
    "datasets": TableSpec(
        name="datasets",
        kind="meta",
        grain="数据版本级（一行 = 一个不可变的数据版本）",
        primary_key="dataset_id",
        join_keys=("dataset_id",),
        description="数据版本登记表。用于核对某个版本的来源与行数",
        fields=(
            FieldSpec("dataset_id", "bigint", "主键"),
            FieldSpec("dataset_key", "text", "数据集标识：hillstrom / retailhero"),
            FieldSpec("version_label", "text", "版本标识，如 `hillstrom@2008-03-20`"),
            FieldSpec("row_count", "integer", "登记行数"),
            FieldSpec("grain", "text", "登记粒度"),
        ),
    ),
    # ------------------------------------------------------------ RetailHero
    "rh_clients": TableSpec(
        name="rh_clients",
        kind="retailhero",
        grain=GRAIN_CUSTOMER,
        primary_key="client_id",
        join_keys=("client_id",),
        description="RetailHero 客户维度表，**客户集合的权威来源（400,162 行）**",
        fields=(
            FieldSpec("client_id", "text", "客户标识，10 位十六进制串（不是整数）"),
            FieldSpec("first_issue_date", "timestamptz", "首张卡发放时间"),
            FieldSpec("first_redeem_date", "timestamptz", "首次核销时间"),
            FieldSpec("age", "smallint", "年龄。**有缺失(NULL)与异常值(0/944/1901 等)**，不得用 0 填充，不得静默过滤"),
            FieldSpec("gender", "text", "M/F/U，U 表示未知，保留不插补"),
        ),
    ),
    "rh_purchases": TableSpec(
        name="rh_purchases",
        kind="retailhero",
        grain=GRAIN_TXN_PRODUCT,
        primary_key="(client_id, transaction_id, product_id)",
        join_keys=("client_id", "product_id", "transaction_id"),
        description=(
            "购买明细。**一行是一个商品明细，不是一笔交易**；"
            "交易级字段被冗余到每一行上，聚合前必须按 transaction_id 收敛"
        ),
        fields=(
            FieldSpec("client_id", "text", "客户标识"),
            FieldSpec("transaction_id", "text", "交易标识；同一交易有多行（每行一个商品）"),
            FieldSpec("transaction_datetime", "timestamptz", "交易时间"),
            FieldSpec("product_id", "text", "商品标识"),
            FieldSpec("product_quantity", "numeric", "【明细级】本行商品件数，**可直接 sum**", "line_level_quantity"),
            FieldSpec("trn_sum_from_iss", "numeric", "【本行】来自签发卡的金额部分"),
            FieldSpec("trn_sum_from_red", "numeric", "【本行】来自核销的金额部分；大量为 NULL，NULL 不是 0"),
            FieldSpec("purchase_sum", "numeric", "【交易级】整笔交易金额，**冗余在每一行明细上**", "transaction_level_amount",
                      note="同 transaction_id 的各行取值 100% 相同；直接 sum 会虚高约 5.81 倍（实测到 11.41 倍）且不报错。"
                           "必须先 group by (client_id, transaction_id) 取 max 收敛后再求和"),
            FieldSpec("regular_points_received", "numeric", "【交易级】常规积分获得，同样必须先按交易收敛", "transaction_level_amount"),
            FieldSpec("express_points_received", "numeric", "【交易级】快速积分获得，同样必须先按交易收敛", "transaction_level_amount"),
            FieldSpec("regular_points_spent", "numeric", "【交易级】常规积分消耗，同样必须先按交易收敛", "transaction_level_amount"),
            FieldSpec("express_points_spent", "numeric", "【交易级】快速积分消耗，同样必须先按交易收敛", "transaction_level_amount"),
            FieldSpec("store_id", "text", "门店标识"),
        ),
        forbidden=(
            ForbiddenField(
                "sum(purchase_sum) / sum(regular_points_received) 等交易级字段的直接求和",
                "交易级字段在明细行上重复，直接求和是静默错数（虚高约 5.81 倍）",
                "from rh_transactions（已按 transaction_id 收敛）后再 sum",
            ),
        ),
    ),
    "rh_products": TableSpec(
        name="rh_products",
        kind="retailhero",
        grain=GRAIN_PRODUCT,
        primary_key="product_id",
        join_keys=("product_id",),
        description="商品维度表。level_1..4 是四级品类层级",
        fields=(
            FieldSpec("product_id", "text", "商品标识"),
            FieldSpec("level_1", "text", "一级品类（最粗）"),
            FieldSpec("level_2", "text", "二级品类"),
            FieldSpec("level_3", "text", "三级品类"),
            FieldSpec("level_4", "text", "四级品类（最细）"),
            FieldSpec("segment_id", "numeric", "商品细分标识"),
            FieldSpec("brand_id", "text", "品牌"),
            FieldSpec("vendor_id", "text", "供应商"),
            FieldSpec("netto", "numeric", "净含量"),
            FieldSpec("is_own_trademark", "boolean", "是否自有品牌"),
            FieldSpec("is_alcohol", "boolean", "是否含酒精"),
        ),
    ),
    "rh_uplift_train": TableSpec(
        name="rh_uplift_train",
        kind="retailhero",
        grain=GRAIN_CUSTOMER,
        primary_key="client_id",
        join_keys=("client_id",),
        description="训练集（含干预标记与结果）。**仅用于建模/评估口径核对，不作特征**",
        fields=(
            FieldSpec("client_id", "text", "客户标识"),
            FieldSpec("treatment_flg", "smallint", "干预标记 0/1"),
            FieldSpec("target", "smallint", "购买标签 0/1；观察窗口长度数据中未说明"),
        ),
        forbidden=(
            ForbiddenField("treatment_flg", "干预后信息，**禁止作为特征**；只允许用于分组定义与建模口径核对（AGENTS.md §2.3）",
                           "需要分组时按 treatment_flg 分组计数，不要把它放进 select 的特征列"),
            ForbiddenField("target", "结果标签，**禁止作为特征**（同 §2.3）",
                           "只作为被统计的分子，如 count(*) filter (where target = 1)"),
        ),
    ),
    "rh_uplift_test": TableSpec(
        name="rh_uplift_test",
        kind="retailhero",
        grain=GRAIN_CUSTOMER,
        primary_key="client_id",
        join_keys=("client_id",),
        description="测试集客户清单（无标签）",
        fields=(FieldSpec("client_id", "text", "客户标识"),),
    ),
    "rh_client_features": TableSpec(
        name="rh_client_features",
        kind="retailhero",
        grain=GRAIN_CUSTOMER,
        primary_key="client_id",
        join_keys=("client_id",),
        description=(
            "客户特征快照，**一行 = 一名 rh_clients 里的客户（含无交易客户）**。"
            "total_spend 已按 transaction_id 收敛，是金额的正确入口"
        ),
        fields=(
            FieldSpec("client_id", "text", "客户标识"),
            FieldSpec("n_transactions", "integer", "交易笔数"),
            FieldSpec("total_spend", "numeric", "总消费 = Σ over distinct transactions of max(purchase_sum)，**已去重**"),
            FieldSpec("avg_basket", "numeric", "客单价；无交易时为 NULL（不是 0）"),
            FieldSpec("n_items", "numeric", "商品件数（明细级字段求和，**不需要**按交易收敛）"),
            FieldSpec("n_distinct_products", "integer", "购买过的不同商品数"),
            FieldSpec("n_stores", "integer", "去过的门店数"),
            FieldSpec("first_txn_date", "timestamptz", "首笔交易时间；无交易为 NULL"),
            FieldSpec("last_txn_date", "timestamptz", "末笔交易时间；无交易为 NULL"),
            FieldSpec("days_active", "integer", "活跃天数"),
            FieldSpec("n_top_level1", "integer", "购买涉及的一级品类数"),
            FieldSpec("has_purchase_history", "boolean", "是否有交易记录；false 时上面的计数为 0、时间为 NULL"),
        ),
    ),
    "rh_source_files": TableSpec(
        name="rh_source_files",
        kind="retailhero",
        grain=GRAIN_SOURCE_FILE,
        primary_key="file_key",
        join_keys=(),
        description="源文件清单（行数、字节数、SHA256），用于证明「库里行数 = CSV 行数」",
        fields=(
            FieldSpec("file_key", "text", "文件标识：clients / products / purchases / …"),
            FieldSpec("file_name", "text", "文件名"),
            FieldSpec("row_count", "integer", "源文件行数"),
            FieldSpec("file_sha256", "char(64)", "文件哈希"),
        ),
    ),
}


ALLOWED_VIEWS: dict[str, str] = {
    "v_experiment_customer": (
        "Hillstrom 分析基础视图：**一行 = 一名客户在一次实验中的分组与结果（含实验前特征）**。"
        "已用 LEFT JOIN 关联 outcomes，结果缺失时 visited/converted/spend 为 NULL 且 outcome_missing = true。"
        "已购率/访问率/人均消费的分母都是**本视图按 arm 分组的全部行**，"
        "绝不允许按 visited / converted 过滤（docs/02 §8.1）"
    ),
    "rh_transactions": (
        "RetailHero **交易级收敛视图：一行 = 一笔交易**。"
        "交易级字段（purchase_sum、*_points_*）已用 max 取一行，明细级字段（n_items）已求和。"
        "**任何涉及交易金额/点数的聚合都必须基于本视图**，不得直接对 rh_purchases 求和；"
        "inconsistency_flags > 0 表示组内交易级字段不一致，此时结果不可信"
    ),
}

#: 视图的粒度（供关联扩张判断使用）。视图本身的字段说明在上面的 dict 里，不重复登记。
VIEW_GRAINS: dict[str, str] = {
    "v_experiment_customer": GRAIN_ASSIGNMENT,
    "rh_transactions": GRAIN_TRANSACTION,
}

#: 视图的数据集归属。
VIEW_DATASETS: dict[str, DatasetKind] = {
    "v_experiment_customer": "hillstrom",
    "rh_transactions": "retailhero",
}


# ============================================================================
#  范围判定
# ============================================================================


def allowed_object_names() -> tuple[str, ...]:
    """允许引用的表 + 视图名（已排序，供错误信息与提示词使用）。"""
    return tuple(sorted(set(ALLOWED_TABLES) | set(ALLOWED_VIEWS)))


def is_system_object(name: str) -> bool:
    """是否为系统对象（``pg_*`` 前缀、``pg_catalog`` / ``information_schema`` 模式）。"""
    low = name.lower()
    if low.startswith("pg_"):
        return True
    if low.split(".")[0] in ("pg_catalog", "information_schema"):
        return True
    return False


def normalize_object_name(raw: str) -> str:
    """把引用到的对象名归一化成裸表名。

    - ``public.customers`` → ``customers``
    - ``"Customers"`` → ``Customers``（带引号表示**区分大小写**，原样保留，
      因此它不会命中全小写的白名单 —— 这是刻意的：PostgreSQL 里
      ``"Customers"`` 与 ``customers`` 是两个不同的对象）
    """
    value = raw.strip()
    if not value:
        return ""
    parts = [p for p in value.split(".") if p]
    return parts[-1] if parts else ""


def table_grain(name: str) -> str | None:
    """返回对象的粒度说明；不在范围内返回 ``None``。"""
    key = normalize_object_name(name)
    if key in ALLOWED_TABLES:
        return ALLOWED_TABLES[key].grain
    return VIEW_GRAINS.get(key)


def dataset_of(name: str) -> DatasetKind | None:
    """返回对象所属数据集；不在范围内返回 ``None``。"""
    key = normalize_object_name(name)
    if key in ALLOWED_TABLES:
        return ALLOWED_TABLES[key].kind
    return VIEW_DATASETS.get(key)


def is_multi_row_per_customer(name: str) -> bool:
    """关联到该对象是否可能放大客户行数（交易/明细级）。"""
    key = normalize_object_name(name)
    if key in ALLOWED_TABLES:
        return ALLOWED_TABLES[key].is_multi_row_per_customer
    return VIEW_GRAINS.get(key) == GRAIN_TRANSACTION


def forbidden_columns(name: str) -> tuple[str, ...]:
    """该对象上被明确禁止使用的字段名（不含 ``（任何…）`` 这类说明性条目）。"""
    key = normalize_object_name(name)
    spec = ALLOWED_TABLES.get(key)
    if spec is None:
        return ()
    return tuple(f.name for f in spec.forbidden if not f.name.startswith("（"))


def in_scope(
    sql: str,
    *,
    tables: Sequence[str] | None = None,
) -> tuple[bool, str]:
    """SQL 是否只引用了允许的表 / 视图。

    判定对象是**解析出来的表引用**（sqlglot AST，见
    :func:`dsa.nl2sql.guards.referenced_tables`），不是文本匹配 ——
    把 ``pg_shadow`` 藏在子查询或 CTE 里同样会被抓到。

    Args:
        sql: 待判定 SQL。
        tables: 已解析出的表引用。给出时直接用它，避免重复解析。

    Returns:
        ``(True, "")`` 或 ``(False, 说明为什么越界、以及允许什么)``。
    """
    if tables is not None:
        names = [str(t) for t in tables]
    else:
        from dsa.nl2sql.guards import SqlParseError, referenced_tables

        try:
            names = referenced_tables(sql)
        except SqlParseError as exc:
            return False, f"SQL 无法解析：{exc}"

    allowed = set(allowed_object_names())
    bad: list[str] = []
    for raw in names:
        bare = normalize_object_name(raw)
        if not bare:
            continue
        if bare in allowed:
            continue  # 白名单优先，避免名字巧合被系统表规则误伤
        if is_system_object(raw) or is_system_object(bare):
            bad.append(f"{raw}（系统表/系统模式，不在查询范围内）")
            continue
        bad.append(f"{raw}（不在允许的表/视图白名单内）")

    if bad:
        return False, (
            "SQL 引用了范围外的对象：" + "；".join(sorted(set(bad)))
            + "。允许的对象：" + ", ".join(allowed_object_names())
        )
    if not names:
        # 没有任何表引用的 SELECT（如 select 1）不构成越界，但也没有分析价值。
        return True, ""
    return True, ""


# ============================================================================
#  提示词
# ============================================================================

#: ``purchase_sum`` 那条铁律的原文。**必须出现在提示词里**，
#: 否则模型会写出虚高约 10 倍的 SUM，而且不报错（docs/日志.md L026）。
PURCHASE_SUM_WARNING = (
    "**purchase_sum 是交易级字段，被冗余到该交易的每一行商品明细上**"
    "（实测平均 5.81 行/交易、最高 19 行；同一 transaction_id 的各行取值 100% 相同，"
    "metrics.yaml 记录实测虚高 11.41x / 10.34x）。因此：\n"
    "  · `select sum(purchase_sum) from rh_purchases` —— **错**，虚高约 5.81 ~ 11.41 倍，"
    "**不报错、不为空、看起来完全正常**；\n"
    "  · `select sum(purchase_sum) from rh_transactions` —— **对**，该视图已按 "
    "(client_id, transaction_id) 用 max 收敛，一行 = 一笔交易；\n"
    "  · 不得写成 `sum(distinct purchase_sum)`；\n"
    "  · 明细级字段（`product_quantity`）不受此限制，可以直接按行求和；\n"
    "  · `rh_client_features.total_spend` 已经是收敛后的客户级金额，是最省事的正确入口。\n"
    "  见 docs/日志.md L026、docs/metrics.yaml 的 `transaction_amount_redundant`。"
)

GLOBAL_RULES = (
    "1. **只能生成单条 SELECT**（可以带 WITH）。禁止 DDL/DML、多语句、系统表。\n"
    "2. **分母口径红线（AGENTS.md §2.2）**：购买率/访问率/人均消费的分母是"
    "**全部被随机分配到该组的客户数**。绝对禁止把 `visit` / `visited` / `converted` / "
    "`spend` 写进 `where` 或 `join ... on` 去筛分母 —— 那等于换了一个人群，破坏随机化。\n"
    "   正确写法：分子用 `count(*) filter (where o.converted)`，分母仍用 `count(*)`；\n"
    "   错误写法：`select count(*) from assignments a where o.visited = 1`。\n"
    "3. **粒度决定聚合**：见每张表的「粒度」标注。客户级表上一行 = 一名客户，"
    "`count(*)` 就是客户数；明细级表上一行是一个商品，`count(*)` 不是客户数，"
    "需要客户数时必须 `count(distinct client_id)`。\n"
    "4. **数据集不得混用（AGENTS.md §2.5）**：`rh_*` 表与 Hillstrom 表"
    "（customers/outcomes/assignments/experiments…）**禁止在同一个 SQL 里 JOIN**，"
    "也不得让一个数据集充当另一个的对照组。\n"
    "5. **干预后字段不得作为特征或过滤条件**（AGENTS.md §2.3）：`treatment_flg` / `target` / "
    "`visit` / `converted` / `spend` 只能作为被统计的结果。\n"
    "6. **不要自己算结论数字**：返回 SQL 即可，数值由数据库产生（AGENTS.md §2.1）。\n"
    "7. 结果字段请用带业务含义的别名（如 `conversion_rate`），便于与参数化工具交叉核对。\n"
    "8. 不确定或不支持时**不要猜**：返回 `{\"unsupported\": \"原因\"}`。"
)


def _metrics_block() -> list[str]:
    """从 docs/metrics.yaml 读出指标字典（读取失败不致命）。"""
    try:
        from dsa.config import PROJECT_ROOT
        import yaml

        doc = yaml.safe_load(__import__("importlib.resources", fromlist=["files"]).files("dsa.resources").joinpath("metrics.yaml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 —— 提示词少一段，好过整个模块 import 失败
        return ["（指标字典 docs/metrics.yaml 读取失败，本段省略）"]

    lines: list[str] = []
    metrics = doc.get("metrics") or {}
    for mid, m in metrics.items():
        lines.append(
            f"- `{mid}`（{m.get('name_zh', '')}，粒度 {m.get('grain')}，角色 {m.get('role')}）"
        )
        lines.append(f"    - 分子：{m.get('numerator')}")
        lines.append(f"    - 分母：{m.get('denominator')}")
        for c in m.get("constraints") or []:
            lines.append(f"    - 约束：{c}")
        if m.get("forbidden_filters"):
            lines.append(f"    - 禁止过滤字段：{', '.join(m['forbidden_filters'])}")

    unsupported = doc.get("unsupported_metrics") or []
    if unsupported:
        lines.append("- **明确不提供的指标**（遇到就直接说做不到，不要编 SQL）：")
        lines.extend(f"    - `{u.get('id')}`：{u.get('reason')}" for u in unsupported)
    return lines


def _capabilities_block() -> list[str]:
    try:
        from dsa.config import PROJECT_ROOT
        import yaml

        doc = yaml.safe_load(__import__("importlib.resources", fromlist=["files"]).files("dsa.resources").joinpath("metrics.yaml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    lines: list[str] = []
    for key, ds in (doc.get("datasets") or {}).items():
        caps = ds.get("capabilities") or {}
        yes = [k for k, v in caps.items() if v is True]
        no = [k for k, v in caps.items() if v is False]
        unknown = [k for k, v in caps.items() if v is None]
        lines.append(f"- `{key}`（{ds.get('version')}，粒度 {ds.get('grain')}，{ds.get('rows')} 行）")
        if yes:
            lines.append(f"    - 支持：{', '.join(yes)}")
        if no:
            lines.append(f"    - **不支持**：{', '.join(no)}")
        if unknown:
            lines.append(f"    - **未确认**（不得当作支持）：{', '.join(unknown)}")
    return lines


def schema_prompt(*, include_metrics: bool = True, include_capabilities: bool = True) -> str:
    """给模型的 schema 说明。

    内容按「模型写错 SQL 的方式」组织，而不是按数据库文档的习惯：

    1. 表粒度 —— 决定 ``count(*)`` 是什么；
    2. 主键与关联键 —— 决定 JOIN 怎么写、会不会扩张行数；
    3. 每个允许字段的中文含义 —— 决定 ``where`` 条件选哪个列；
    4. **禁止使用的字段** —— 决定哪些列不能碰（干预后信息、标签）；
    5. 指标字典 —— 分子/分母/约束的权威定义（docs/metrics.yaml）；
    6. 数据集能力 —— 哪些分析这个数据**根本不支持**（避免模型硬编 SQL）。

    Returns:
        可直接拼进 system prompt 的中文说明文本。
    """
    parts: list[str] = [
        "# 可查询的范围",
        "",
        "你只能查询下列表与视图。**任何其他对象（包括 pg_catalog、information_schema、"
        "任何 pg_* 系统表）都不允许引用。**",
        "",
        "## 铁律",
        "",
        "### 铁律 A —— RetailHero 交易金额必须先按交易收敛",
        "",
        PURCHASE_SUM_WARNING,
        "",
        "### 通用规则",
        "",
        GLOBAL_RULES,
        "",
        "## 表清单（注意每张表的**粒度**）",
        "",
    ]

    for name in sorted(ALLOWED_TABLES):
        parts.append(ALLOWED_TABLES[name].render())
        parts.append("")

    parts.append("## 分析视图")
    parts.append("")
    for name in sorted(ALLOWED_VIEWS):
        parts.append(f"#### {name}（{VIEW_GRAINS.get(name, '')}）")
        parts.append(f"- 说明：{ALLOWED_VIEWS[name]}")
        parts.append("")

    if include_metrics:
        parts.append("## 指标字典（分子 / 分母 / 约束）")
        parts.append("")
        parts.extend(_metrics_block())
        parts.append("")

    if include_capabilities:
        parts.append("## 数据集能力（不支持的要直接说明做不到，不要硬写 SQL）")
        parts.append("")
        parts.extend(_capabilities_block())
        parts.append("")

    return "\n".join(parts)


#: 供测试与外部检查：提示词里**必须**出现的若干关键句（缺一即为回归）。
REQUIRED_PROMPT_MARKERS: tuple[str, ...] = (
    "purchase_sum",
    "transaction_id",
    "去重",
    "粒度",
    "分母",
    "treatment_flg",
    "rh_transactions",
    "数据集",
)


def prompt_missing_markers(prompt: str) -> list[str]:
    """返回提示词里缺失的关键标记（空列表表示满足要求）。"""
    return [m for m in REQUIRED_PROMPT_MARKERS if m not in prompt]


def _iter_specs() -> Iterable[TableSpec]:
    return ALLOWED_TABLES.values()


def scope_summary() -> dict[str, Any]:
    """范围摘要（自检 / 日志 / 报错说明用）。"""
    by_kind: dict[str, list[str]] = {}
    for spec in _iter_specs():
        by_kind.setdefault(spec.kind, []).append(spec.name)
    for name in ALLOWED_VIEWS:
        by_kind.setdefault(VIEW_DATASETS.get(name, "meta"), []).append(name)
    return {
        "tables": sorted(ALLOWED_TABLES),
        "views": sorted(ALLOWED_VIEWS),
        "by_dataset": {k: sorted(v) for k, v in sorted(by_kind.items())},
        "multi_row_per_customer": sorted(
            n for n, s in ALLOWED_TABLES.items() if s.is_multi_row_per_customer
        ),
    }
