"""分组聚合的独立 Python 实现。

**用途**：与 SQL 实现交叉核对。本模块**只从原始 CSV 计算**，不查数据库，
因此它与 SQL 是两条独立路径——两者一致才说明口径实现正确。

口径依据：docs/02-分析协议.md §8。
**分母恒为「全部被随机分配到该组的客户数」**，绝不因干预后行为（visit / spend）
而改变。这是本模块与 SQL 版本必须共同遵守的铁律。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dsa.data.specs import (
    ASSIGNMENT_COLUMN,
    HILLSTROM_EMAIL_V1,
    POST_TREATMENT_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class ArmAggregate:
    """一个组的聚合结果。"""

    arm_key: str
    segment_value: str
    role: str
    n_customers: int
    n_outcome_missing: int
    n_converted: int
    n_visited: int
    total_spend: float

    @property
    def conversion_rate(self) -> float:
        return self.n_converted / self.n_customers if self.n_customers else float("nan")

    @property
    def visit_rate(self) -> float:
        return self.n_visited / self.n_customers if self.n_customers else float("nan")

    @property
    def spend_per_customer(self) -> float:
        return self.total_spend / self.n_customers if self.n_customers else float("nan")


def arm_aggregates_from_raw(csv_path: Path) -> dict[str, ArmAggregate]:
    """从原始 CSV 直接计算各组聚合。**不访问数据库。**

    Args:
        csv_path: Hillstrom 原始文件路径。

    Returns:
        以 arm_key 为键的聚合结果。

    Note:
        分母使用该组的**全部行**（= 全部被随机分配的客户）。
        源数据无缺失，因此 n_outcome_missing 恒为 0；
        若将来出现缺失，这里会如实计数而**不会**把缺失并入未购买。
    """
    df = pd.read_csv(csv_path)

    # 防御性断言：分组的原始取值必须与规格一致，否则下面的映射会静默漏掉客户
    seg_to_arm = {a.segment_value: a for a in HILLSTROM_EMAIL_V1.arms}
    unknown = set(df[ASSIGNMENT_COLUMN].unique()) - set(seg_to_arm)
    if unknown:
        raise ValueError(f"源数据出现规格外的分组取值: {sorted(unknown)}")

    out: dict[str, ArmAggregate] = {}
    for segment_value, arm in seg_to_arm.items():
        sub = df[df[ASSIGNMENT_COLUMN] == segment_value]
        out[arm.arm_key] = ArmAggregate(
            arm_key=arm.arm_key,
            segment_value=segment_value,
            role=arm.role,
            # 分母：该组全部客户
            n_customers=int(len(sub)),
            n_outcome_missing=0,
            n_converted=int(sub["conversion"].sum()),
            n_visited=int(sub["visit"].sum()),
            total_spend=float(sub["spend"].sum()),
        )
    return out


def comparison_subset_from_raw(csv_path: Path, treatment: str, control: str) -> pd.DataFrame:
    """按 arm_key 取出主要比较的客户级子集（原始数据，无派生列）。"""
    df = pd.read_csv(csv_path)
    seg_to_arm = {a.segment_value: a.arm_key for a in HILLSTROM_EMAIL_V1.arms}
    df = df.assign(_arm=df[ASSIGNMENT_COLUMN].map(seg_to_arm))
    return df[df["_arm"].isin([treatment, control])].copy()


def assert_no_post_treatment_in_features(columns) -> None:
    """断言给定列集合不含干预后结果字段（数据泄漏防线）。"""
    leaked = set(columns) & set(POST_TREATMENT_COLUMNS)
    if leaked:
        raise ValueError(f"特征中出现干预后结果字段，构成数据泄漏: {sorted(leaked)}")
