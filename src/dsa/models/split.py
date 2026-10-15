"""建模人群取数与数据划分（阶段四任务一）。

口径来源：`docs/06-建模与策略评估协议.md` §2、§3。

三条不可动摇的规则：

1. **建模人群** = 被分配到 `Mens E-Mail` 或 `No E-Mail` 的客户，
   **丢弃** `Womens E-Mail`。这是**按组别取值确定性筛选**，不是随机抽样。
   分母 = `assignments` 行数，绝不做任何干预后筛选（AGENTS.md §2.2）。
2. **三分法按 arm 分层**：train 60% / valid 20% / test 20%，客户级不重叠。
3. **最终留出集不参与任何选择**（不调参、不选策略、不选种子）。

随机数一律使用 `numpy.random.default_rng(seed)`，**不触碰全局随机状态**。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text
from sqlalchemy.sql import bindparam

from dsa.models.features import FEATURE_COLUMNS

# ============================================================================
#  常量（口径来源：docs/06 §2.3、§3.2）
# ============================================================================

#: 默认实验
DEFAULT_EXPERIMENT_KEY = "hillstrom_email_v1"

#: 干预组 arm_key
ARM_MENS = "mens"
#: 对照组 arm_key
ARM_CONTROL = "control"
#: 建模人群使用的两个组；`womens` 被确定性丢弃
MODELING_ARMS: tuple[str, ...] = (ARM_MENS, ARM_CONTROL)

#: 划分随机种子（docs/06 §3.2 固定为 20261011，写入模型登记）
DEFAULT_SEED = 20261011
#: 划分比例：train / valid / test
DEFAULT_RATIOS: tuple[float, float, float] = (0.6, 0.2, 0.2)

#: 协议 §2.3 登记的人群规模，用于数据版本漂移检查（不作为计算结果使用）
EXPECTED_N_MENS = 21_307
EXPECTED_N_CONTROL = 21_306
EXPECTED_N_MODELING = EXPECTED_N_MENS + EXPECTED_N_CONTROL  # 42,613

#: 输出列顺序
FRAME_COLUMNS: tuple[str, ...] = (
    "customer_key",
    "arm",
    "treatment",
    "y",
    *FEATURE_COLUMNS,
)

_SPLIT_NAMES: tuple[str, str, str] = ("train", "valid", "test")


# ============================================================================
#  取数
# ============================================================================

_SQL_MODELING_FRAME = """
select
    v.customer_key,
    v.arm_key,
    v.recency,
    v.history_segment,
    v.history_band,
    v.history,
    v.mens,
    v.womens,
    v.zip_code,
    v.newbie,
    v.channel,
    v.converted,
    v.outcome_missing
from v_experiment_customer v
where v.experiment_key = :experiment_key
  and v.arm_key in :arms
order by v.customer_key
"""


def load_modeling_frame(engine: Engine, *, experiment_key: str = DEFAULT_EXPERIMENT_KEY) -> pd.DataFrame:
    """从数据库取建模人群。

    分母 = `assignments` 行数：**不做任何干预后筛选**。
    结果缺失（`outcome_missing`）不会被静默当成"未购买"，而是**直接报错**——
    缺失结果必须显式处理，不得混入分母（AGENTS.md §2.2）。

    Args:
        engine: SQLAlchemy 引擎。按 AGENTS.md §5，**必须传只读引擎**
            （`dsa.config.get_engine(readonly=True)`）。
        experiment_key: 实验标识。

    Returns:
        DataFrame，列顺序固定为 `FRAME_COLUMNS`：

        - `customer_key`：带数据集命名空间的客户键（唯一）
        - `arm`：`'mens'` 或 `'control'`
        - `treatment`：bool，`arm == 'mens'`
        - `y`：bool，实验后 14 天内是否购买（= `conversion`）
        - 其余为全部**实验前特征列**（`PRE_TREATMENT_COLUMNS`）

    Raises:
        ValueError: 结果缺失、分组取值越界或取不到任何行时。
    """
    stmt = text(_SQL_MODELING_FRAME).bindparams(bindparam("arms", expanding=True))
    with engine.connect() as conn:
        df = pd.read_sql(
            stmt,
            conn,
            params={"experiment_key": experiment_key, "arms": list(MODELING_ARMS)},
        ).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"建模人群为空: experiment_key={experiment_key!r}")

    unexpected_arms = sorted(set(df["arm_key"]) - set(MODELING_ARMS))
    if unexpected_arms:
        raise ValueError(f"出现建模人群之外的组别: {unexpected_arms}")

    missing = int(df["outcome_missing"].sum())
    if missing:
        raise ValueError(
            f"有 {missing} 名客户的干预后结果缺失。"
            "缺失结果不得并入分母，也不得当作未购买，必须显式处理。"
        )

    out = pd.DataFrame(
        {
            "customer_key": df["customer_key"].astype(str),
            "arm": df["arm_key"].astype(str),
            "treatment": df["arm_key"].to_numpy() == ARM_MENS,
            "y": df["converted"].astype(bool),
        }
    )
    for col in FEATURE_COLUMNS:
        out[col] = df[col].to_numpy()

    out = _coerce_dtypes(out)

    if out["customer_key"].duplicated().any():
        raise ValueError("customer_key 出现重复：一名客户只能出现一次")

    return out.loc[:, list(FRAME_COLUMNS)].reset_index(drop=True)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """把数据库类型显式转成稳定的 pandas 类型。

    `numeric` 列经 DBAPI 返回 `Decimal`（object dtype），
    不显式转换会让 StandardScaler 直接报类型错误。
    """
    out = df.copy()
    out["recency"] = np.asarray(out["recency"], dtype=np.int64)
    out["history_band"] = np.asarray(out["history_band"], dtype=np.int64)
    out["history"] = np.asarray(out["history"], dtype=np.float64)
    out["history_segment"] = out["history_segment"].astype(str)
    out["zip_code"] = out["zip_code"].astype(str)
    out["channel"] = out["channel"].astype(str)
    for col in ("mens", "womens", "newbie"):
        out[col] = np.asarray(out[col], dtype=bool)
    return out


# ============================================================================
#  划分
# ============================================================================


@dataclass(frozen=True, slots=True)
class DataSplit:
    """一次三分法的结果。索引为 `df` 的**位置索引**（0-based），互不重叠。"""

    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    seed: int
    ratios: tuple[float, float, float]
    arm_counts: dict[str, dict[str, int]]

    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train_idx.size),
            "valid": int(self.valid_idx.size),
            "test": int(self.test_idx.size),
        }

    def indices(self, name: str) -> np.ndarray:
        if name not in _SPLIT_NAMES:
            raise KeyError(f"未知集合名: {name!r}，应为 {_SPLIT_NAMES}")
        return {"train": self.train_idx, "valid": self.valid_idx, "test": self.test_idx}[name]

    def describe(self) -> str:
        s = self.sizes()
        parts = [f"seed={self.seed}", f"ratios={self.ratios}"]
        for name in _SPLIT_NAMES:
            counts = self.arm_counts[name]
            detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            parts.append(f"{name}={s[name]} ({detail})")
        return " | ".join(parts)

    def to_registry_dict(self) -> dict[str, object]:
        """`docs/06` §8 要求的划分配置（可直接作为 `models.split_config`）。

        含 `seed` / `ratios` / `counts` / `arm_counts`，
        人数全部由 `arm_counts` 如实统计而来，不写死任何数字。
        """
        return {
            "seed": self.seed,
            "ratios": list(self.ratios),
            "counts": self.sizes(),
            "arm_counts": {k: dict(v) for k, v in self.arm_counts.items()},
        }


def make_split(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> DataSplit:
    """按 `arm` 分层的客户级三分法。

    相同 `df`（同一行序）、同一种子必须产生**完全相同**的划分：
    分层顺序固定为 arm 取值升序，随机数只来自 `np.random.default_rng(seed)`。

    Args:
        df: 至少含 `arm` 列的建模人群（`load_modeling_frame` 的输出）。
        seed: 随机种子，默认 20261011（docs/06 §3.2）。
        ratios: (train, valid, test) 比例，必须全为正且和为 1。

    Returns:
        `DataSplit`，三个位置索引互不重叠且并集恰好等于全集。
    """
    _validate_ratios(ratios)
    if "arm" not in df.columns:
        raise ValueError("make_split 需要 'arm' 列用于分层")

    n = len(df)
    if n == 0:
        raise ValueError("make_split 收到空表")

    arm_values = df["arm"].to_numpy()

    # 固定遍历顺序：rng 的消耗顺序必须与数据无关，否则不可复现
    arms = sorted(np.unique(arm_values).tolist())

    rng = np.random.default_rng(seed)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in _SPLIT_NAMES}

    for arm in arms:
        idx = np.flatnonzero(arm_values == arm)
        perm = rng.permutation(idx)
        n_arm = int(perm.size)

        # 每个集合在该层至少 1 个样本
        n_train = min(max(int(round(n_arm * ratios[0])), 1), n_arm - 2)
        n_valid = min(max(int(round(n_arm * ratios[1])), 1), n_arm - n_train - 1)
        n_test = n_arm - n_train - n_valid
        if n_test < 1:  # pragma: no cover - 上面的夹取已保证；留作防御
            raise ValueError(f"层 {arm!r} 样本太少（{n_arm}），无法三等分")

        parts["train"].append(perm[:n_train])
        parts["valid"].append(perm[n_train : n_train + n_valid])
        parts["test"].append(perm[n_train + n_valid :])

    out: dict[str, np.ndarray] = {}
    for name in _SPLIT_NAMES:
        merged = np.concatenate(parts[name]) if parts[name] else np.empty(0, dtype=np.int64)
        out[name] = np.sort(merged.astype(np.int64, copy=False))

    _assert_partition(out["train"], out["valid"], out["test"], n, arms, arm_values)

    arm_counts = {
        name: _count_arms(arm_values, out[name]) for name in _SPLIT_NAMES
    }

    return DataSplit(
        train_idx=out["train"],
        valid_idx=out["valid"],
        test_idx=out["test"],
        seed=int(seed),
        ratios=(float(ratios[0]), float(ratios[1]), float(ratios[2])),
        arm_counts=arm_counts,
    )


def apply_split(df: pd.DataFrame, split: DataSplit) -> dict[str, pd.DataFrame]:
    """按划分切出三个子表（保留原列，索引重置）。

    会先核对划分与表的行数一致，避免把别的表的划分用错地方。
    """
    total = sum(int(split.indices(name).size) for name in _SPLIT_NAMES)
    if total != len(df):
        raise ValueError(f"DataSplit 覆盖 {total} 行，与 df 的 {len(df)} 行不一致")
    return {name: df.iloc[split.indices(name)].reset_index(drop=True) for name in _SPLIT_NAMES}


# ---- 内部校验 --------------------------------------------------------------


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != 3:
        raise ValueError(f"ratios 必须是三元组，收到 {ratios!r}")
    if any(r <= 0 for r in ratios):
        raise ValueError(f"ratios 每项必须为正，收到 {ratios!r}")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios 之和必须为 1，收到 {ratios!r}（和={sum(ratios)}）")


def _assert_partition(
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    n: int,
    arms: list[str],
    arm_values: np.ndarray,
) -> None:
    """互不重叠 + 并集 = 全集 + 每层每集非空。"""
    for a, b, name_a, name_b in (
        (train_idx, valid_idx, "train", "valid"),
        (train_idx, test_idx, "train", "test"),
        (valid_idx, test_idx, "valid", "test"),
    ):
        overlap = np.intersect1d(a, b)
        if overlap.size:
            raise AssertionError(
                f"{name_a} 与 {name_b} 重叠 {overlap.size} 个索引，例如 {overlap[:5].tolist()}"
            )

    union = np.union1d(np.union1d(train_idx, valid_idx), test_idx)
    if union.size != n:
        raise AssertionError(f"三个集合的并集 {union.size} != 全集 {n}")
    if not np.array_equal(union, np.arange(n)):
        raise AssertionError("三个集合的并集不是 0..n-1")

    for name, idx in (("train", train_idx), ("valid", valid_idx), ("test", test_idx)):
        if idx.size == 0:
            raise AssertionError(f"{name} 为空集")
        if idx.min() < 0 or idx.max() >= n:
            raise AssertionError(f"{name} 含越界索引")
        present = set(np.unique(arm_values[idx]).tolist())
        if present != set(arms):
            raise AssertionError(f"{name} 缺少某些层: 期望 {arms}，实际 {sorted(present)}")


def _count_arms(arm_values: np.ndarray, idx: np.ndarray) -> dict[str, int]:
    """如实统计某集合内各组别人数。"""
    counter = Counter(arm_values[idx].tolist())
    return {str(k): int(v) for k, v in sorted(counter.items())}
