"""特征定义与预处理管线（阶段四任务二）。

口径来源：`docs/06-建模与策略评估协议.md` §1.1、§1.2、§3.3、§4.3。

两条铁律：

1. **特征只能用实验前信息**：`recency, history_segment, history, history_band,
   mens, womens, zip_code, newbie, channel`。
   `visit / conversion / spend` 是**干预后结果**，进入特征即构成数据泄漏。
2. **预处理只在训练集 `fit`**，再 `transform` 到验证集与留出集。
   本模块只提供「未拟合的预处理构件」；拟合一律由 `dsa.models.tlearner`
   在训练子集上完成，调用方拿到的是已含预处理的 `Pipeline`。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dsa.data.specs import POST_TREATMENT_COLUMNS, PRE_TREATMENT_COLUMNS
from dsa.stats.aggregates import assert_no_post_treatment_in_features

# ============================================================================
#  特征列定义（顺序固定，登记进模型版本）
# ============================================================================

#: 全部实验前特征列，顺序固定。
#:
#: **唯一来源是 `dsa.data.specs.PRE_TREATMENT_COLUMNS`**：这里直接引用而不复制清单，
#: 因为「两处各维护一份特征表」正是口径漂移的来源。
#: 顺序沿用 specs 的顺序（= `docs/06-建模与策略评估协议.md` §1.1 的顺序），
#: 因为模型登记会记录 `feature_columns`，顺序不稳定会让两次训练无法比对。
#:
#: 注：`history_band` 是 `history_segment` 的派生列（`customers` 表的生成列，
#: 由 PostgreSQL 自动维护），来源仍是实验前的历史消费分箱，**不构成泄漏**；
#: 它给线性模型提供 one-hot 无法表达的单调序信息。
FEATURE_COLUMNS: tuple[str, ...] = PRE_TREATMENT_COLUMNS

#: 类别列：字符串类别 + 三个布尔指示变量。
#: 布尔指示变量按**类别**处理（OneHotEncoder）：它们不是连续量，
#: 标准化一个 0/1 变量不改变模型拟合，却让系数失去"原单位"含义。
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "history_segment",
    "zip_code",
    "channel",
    "mens",
    "womens",
    "newbie",
)

#: 数值列：`history_segment` 视为类别，`history_band` 与 `recency` / `history` 视为数值。
NUMERIC_COLUMNS: tuple[str, ...] = ("recency", "history_band", "history")

#: 预处理后新增的列名前缀（供登记时说明）
PREPROCESS_STEP_NAME = "preprocess"
CLASSIFIER_STEP_NAME = "clf"


def _check_column_partition() -> None:
    cat, num = set(CATEGORICAL_COLUMNS), set(NUMERIC_COLUMNS)
    if cat & num:
        raise AssertionError(f"类别列与数值列重叠: {sorted(cat & num)}")
    missing = set(FEATURE_COLUMNS) - (cat | num)
    if missing:
        raise AssertionError(f"以下特征列既未归入类别也未归入数值: {sorted(missing)}")
    extra = (cat | num) - set(FEATURE_COLUMNS)
    if extra:
        raise AssertionError(f"以下列不在实验前特征之内: {sorted(extra)}")

    # 与 specs 必须完全同步：集合相等，且不得有 specs 之外的列
    delta = set(FEATURE_COLUMNS) - set(PRE_TREATMENT_COLUMNS)
    if delta:
        raise AssertionError(
            f"FEATURE_COLUMNS 含 specs.PRE_TREATMENT_COLUMNS 之外的列: {sorted(delta)}。"
            "新特征必须先进入 specs.py，不允许只在模型层私加。"
        )
    dropped = set(PRE_TREATMENT_COLUMNS) - set(FEATURE_COLUMNS)
    if dropped:
        raise AssertionError(f"以下实验前特征被漏掉: {sorted(dropped)}")


# 导入即自检：定义本身出错时要立刻暴露，而不是等到拟合
_check_column_partition()
assert_no_post_treatment_in_features(FEATURE_COLUMNS)


# ============================================================================
#  泄漏防线
# ============================================================================


def assert_no_leakage(columns: Iterable[str]) -> None:
    """若列集合含**干预后字段**则抛 `ValueError`。

    这是 `docs/06` §1.2 要求的程序化防线，**必须在拟合前调用**。
    本函数不复用领域规则，直接对 `POST_TREATMENT_COLUMNS` 取交集，
    因此任何调用点都得到同一结论。

    Args:
        columns: 待检查的列名集合。

    Raises:
        ValueError: 含 `visit` / `conversion` / `spend` 中任意一个。
    """
    leaked = sorted(set(columns) & set(POST_TREATMENT_COLUMNS))
    if leaked:
        raise ValueError(
            f"特征中出现干预后结果字段，构成数据泄漏: {leaked}。"
            f"允许的特征只有: {list(FEATURE_COLUMNS)}"
        )


def assert_only_feature_columns(columns: Iterable[str]) -> None:
    """更严格的一道防线：列集合必须**恰好**是实验前特征列。

    与 `assert_no_leakage` 的分工：
    - `assert_no_leakage` 只拒绝干预后字段，允许 `customer_key` 等元数据列；
    - 本函数要求完全一致，用于「要喂给模型的 X」这种场景。
    """
    assert_no_leakage(columns)
    given = set(columns)
    expected = set(FEATURE_COLUMNS)
    extra = sorted(given - expected)
    missing = sorted(expected - given)
    if extra or missing:
        raise ValueError(
            f"特征列与规格不一致。多余: {extra}；缺失: {missing}。"
            f"规格内的特征: {list(FEATURE_COLUMNS)}"
        )


# ============================================================================
#  预处理构件
# ============================================================================


def build_preprocessor() -> ColumnTransformer:
    """构造**未拟合**的预处理器。

    - 类别列 → `OneHotEncoder(handle_unknown='ignore')`
    - 数值列 → `StandardScaler`

    返回对象尚未 `fit`：调用方必须在训练集上 `fit`（或交给 `Pipeline` 拟合），
    绝不允许在整个数据集上 `fit`（docs/06 §3.3）。
    """
    assert_no_leakage(FEATURE_COLUMNS)
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_COLUMNS),
            ),
            ("num", StandardScaler(), list(NUMERIC_COLUMNS)),
        ],
        # 明确丢弃未声明的列：任何列都不允许被静默带进模型
        remainder="drop",
    )


def build_pipeline(*, classifier) -> Pipeline:
    """把预处理器与基础学习器串成一条 `Pipeline`。

    调用方因此**无法忘记 transform**：拿到 `Pipeline` 就是拿到完整流程。
    """
    return Pipeline(
        [
            (PREPROCESS_STEP_NAME, build_preprocessor()),
            (CLASSIFIER_STEP_NAME, classifier),
        ]
    )


def feature_names_out(pipeline: Pipeline) -> tuple[str, ...]:
    """从**已拟合**的 `Pipeline` 导出预处理后的特征名（登记用）。"""
    pre = pipeline.named_steps[PREPROCESS_STEP_NAME]
    return tuple(str(n) for n in pre.get_feature_names_out())


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """取出实验前特征子表（按固定顺序），缺列即报错。"""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"输入表缺少特征列: {missing}")
    return df.loc[:, list(FEATURE_COLUMNS)]


def used_columns() -> Sequence[str]:
    """模型实际消费的列名（登记用，顺序即输入顺序）。"""
    return FEATURE_COLUMNS


#: 预处理规格的版本号。**改动任何预处理构件时必须同时 +1**，
#: 否则"同一份预处理"的指纹会在无声中变化（或更糟：变了却没变号）。
PREPROCESSING_SPEC_VERSION: int = 1


def preprocessing_config() -> dict:
    """预处理的**规范化**描述，进入模型身份指纹（缺陷 F10）。

    为什么要有这个函数：指纹要覆盖"预处理配置"，但预处理本身是一堆 sklearn 对象，
    直接摊开 ``get_params()`` 会把 sklearn 的默认值变动也算成"配置变了"。
    这里只登记**影响建模语义**的几项，且写成与 sklearn 版本无关的字面量：

    * 哪些列按类别处理、哪些按数值处理（顺序固定）；
    * 类别编码器与数值变换器的种类与关键设置；
    * 未声明的列一律丢弃（``remainder='drop'``）；
    * 预处理只在训练集 ``fit``（docs/06 §3.3）。

    返回值必须可 JSON 序列化，且只含字符串/数字/列表——这样指纹才对
    "字典键序、浮点表示"免疫（见 `dsa.models.registry.canonical_json`）。
    """
    return {
        "spec_version": PREPROCESSING_SPEC_VERSION,
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "numeric_columns": list(NUMERIC_COLUMNS),
        "categorical_encoder": "OneHotEncoder",
        "categorical_encoder_params": {"handle_unknown": "ignore"},
        "numeric_transformer": "StandardScaler",
        "remainder": "drop",
        "fit_scope": "train_only",
    }


__all__ = [
    "CATEGORICAL_COLUMNS",
    "FEATURE_COLUMNS",
    "NUMERIC_COLUMNS",
    "PREPROCESSING_SPEC_VERSION",
    "assert_no_leakage",
    "assert_only_feature_columns",
    "build_pipeline",
    "build_preprocessor",
    "feature_names_out",
    "preprocessing_config",
    "select_features",
    "used_columns",
]
