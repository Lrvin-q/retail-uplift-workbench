"""T-Learner 增量模型与购买预测基线（阶段四任务三）。

口径来源：`docs/06-建模与策略评估协议.md` §1、§4。

- **增量模型（T-Learner）**：分别在干预组与对照组拟合结果模型，
  `tau_hat(x) = mu1_hat(x) - mu0_hat(x)`。
- **购买预测基线**：预测 `P(Y=1|X=x)`，两组合并训练或只用其中一组
  （§4.1 要求两种都做，取**验证集**更优者——选择动作由调用方完成，
  本模块只提供两种拟合方式，**不替调用方做选择**）。

基础学习器固定 `LogisticRegression`，**不做超参数搜索**（§4.3 第一轮固定一种）。
预处理随 `Pipeline` 一起返回，调用方无法忘记 `transform`。

⚠️ 本模块**不估计个体真实增量**：每名客户只观测到一个干预结果，
`Y(1)` 与 `Y(0)` 不可能同时被观测到（§1.3）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from dsa.data.specs import POST_TREATMENT_COLUMNS
from dsa.models.features import (
    CLASSIFIER_STEP_NAME,
    FEATURE_COLUMNS,
    PREPROCESS_STEP_NAME,
    assert_no_leakage,
    build_pipeline,
    feature_names_out,
)
from dsa.models.split import DEFAULT_SEED
from dsa.stats.aggregates import assert_no_post_treatment_in_features

#: 基础学习器名称（登记用）
BASE_LEARNER = "LogisticRegression"

#: 保证 lbfgs 收敛的迭代上限（购买率 ~1%，样本 2.5 万，2000 足够）
DEFAULT_MAX_ITER = 2000


# ============================================================================
#  内部构件
# ============================================================================


def _now_iso() -> str:
    """UTC ISO-8601 时间戳（登记用）。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_learner(*, seed: int, overrides: Mapping[str, Any]) -> LogisticRegression:
    """构造基础学习器。参数只在这里拼装，`params` 一律由 `get_params()` 反查。

    只显式设置 `max_iter` 与 `random_state`，其余超参数沿用 sklearn 默认值：
    协议 §4.3 要求第一轮固定一种基础学习器、**不做超参数搜索**。
    登记时 `get_params()` 会把默认值一并解析出来（例如 sklearn 1.9 的
    `penalty='deprecated'` 哨兵值）。我们**刻意不写死** `penalty='l2'`：
    sklearn 1.8 起 `penalty` 已废弃、1.10 将移除，写死会让代码在升级后直接报错，
    而 `get_params()` 导出的值始终等于真正生效的值。
    """
    params: dict[str, Any] = {"max_iter": DEFAULT_MAX_ITER, "random_state": seed}
    params.update(dict(overrides))
    return LogisticRegression(**params)


def _learner_params(pipeline: Pipeline) -> dict[str, Any]:
    """从已拟合的 Pipeline 导出基础学习器的**全部**超参数。

    `sklearn` 的 `get_params()` 是唯一来源，不手工维护参数清单，
    避免登记值与实际训练值漂移。
    """
    return dict(pipeline.named_steps[CLASSIFIER_STEP_NAME].get_params(deep=True))


def _guard_feature_columns() -> tuple[str, ...]:
    """拟合/预测前统一走一遍泄漏防线。"""
    # 领域防线（docs/06 §1.2 指定的程序化防线）
    assert_no_post_treatment_in_features(FEATURE_COLUMNS)
    # 模块内防线：直接对 POST_TREATMENT_COLUMNS 取交集
    assert_no_leakage(FEATURE_COLUMNS)
    unknown = [c for c in FEATURE_COLUMNS if c in POST_TREATMENT_COLUMNS]
    if unknown:  # pragma: no cover - 上面两行已覆盖；留作显式兜底
        raise ValueError(f"特征列表含干预后字段: {unknown}")
    return FEATURE_COLUMNS


def _select_X(X: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """取出特征子表（固定顺序）。多余列一律忽略，绝不带入模型。"""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X 必须是 pandas.DataFrame，收到 {type(X).__name__}")
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(f"X 缺少特征列: {missing}")
    return X.loc[:, list(feature_columns)]


def _require_two_classes(y: np.ndarray, label: str) -> None:
    """结果必须是二值且两类都出现，否则 LogisticRegression 无法拟合。"""
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(
            f"{label} 的 y 只有 {classes.tolist()} 一类取值，无法拟合逻辑回归"
            "（购买率极低时请检查分组或时间窗口口径）"
        )


def _as_1d_float(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _predict_mu(pipeline: Pipeline, X: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    """返回 `P(Y=1|X)`，形状 `(n,)`、`float64`。"""
    proba = pipeline.predict_proba(_select_X(X, feature_columns))
    return _as_1d_float(np.asarray(proba)[:, 1])


# ============================================================================
#  T-Learner
# ============================================================================


@dataclass(frozen=True, slots=True)
class FittedTLerner:
    """已拟合的 T-Learner。

    类名保持历史约定拼写 `FittedTLerner`。
    每个 `Pipeline` **已包含预处理**（在各自分组的训练子集上 fit），
    因此 `predict_*` 不会再触碰原始列以外的任何信息。
    """

    model_treatment: Pipeline  # 干预组结果模型  mu1(x) ≈ E[Y|X, W=1]
    model_control: Pipeline  # 对照组结果模型  mu0(x) ≈ E[Y|X, W=0]
    feature_columns: tuple[str, ...]
    base_learner: str
    params: dict
    n_train_treatment: int
    n_train_control: int
    trained_at: str

    # ---- 预测 ----

    def predict_mu1(self, X: pd.DataFrame) -> np.ndarray:
        """`mu1_hat(x) = E[Y|X=x, W=1]` 的估计，形状 `(n,)`。"""
        assert_no_leakage(self.feature_columns)
        return _predict_mu(self.model_treatment, X, self.feature_columns)

    def predict_mu0(self, X: pd.DataFrame) -> np.ndarray:
        """`mu0_hat(x) = E[Y|X=x, W=0]` 的估计，形状 `(n,)`。"""
        assert_no_leakage(self.feature_columns)
        return _predict_mu(self.model_control, X, self.feature_columns)

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """`tau_hat(x) = mu1_hat(x) - mu0_hat(x)`，形状 `(n,)`、`float64`。"""
        return _as_1d_float(self.predict_mu1(X) - self.predict_mu0(X))

    # ---- 登记 ----

    def to_registry_dict(self) -> dict[str, Any]:
        """`docs/06` §8 要求的登记信息（数值全部由程序产生）。"""
        return {
            "model_type": "t_learner",
            "base_learner": self.base_learner,
            "params": dict(self.params),
            "feature_columns": list(self.feature_columns),
            "n_features_in": len(self.feature_columns),
            "n_features_out": len(feature_names_out(self.model_treatment)),
            "n_train_treatment": self.n_train_treatment,
            "n_train_control": self.n_train_control,
            "n_train_total": self.n_train_treatment + self.n_train_control,
            "trained_at": self.trained_at,
        }


def fit_t_learner(
    train_df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    **learner_overrides: Any,
) -> FittedTLerner:
    """在**训练集**上分别拟合干预组与对照组结果模型。

    预处理（OneHot + Scaler）在各自的训练子集上 `fit`，
    **不会**接触验证集与最终留出集。

    Args:
        train_df: 训练集，需含 `treatment`(bool) 与 `y`(bool) 以及全部实验前特征列。
        seed: 随机种子，默认 20261011；同时作为 `random_state`。
        **learner_overrides: 覆盖 `LogisticRegression` 的超参数（不做搜索，仅显式传参）。

    Returns:
        `FittedTLerner`。

    Raises:
        ValueError: 缺少列、某组样本不足、或某组的 `y` 只有一类。
    """
    feature_columns = _guard_feature_columns()

    for col in ("treatment", "y"):
        if col not in train_df.columns:
            raise ValueError(f"train_df 缺少 '{col}' 列")

    X_all = _select_X(train_df, feature_columns)
    w = train_df["treatment"].to_numpy(dtype=bool)
    y = train_df["y"].to_numpy(dtype=np.int64)

    mask1 = w
    mask0 = ~w
    n1, n0 = int(mask1.sum()), int(mask0.sum())
    if n1 < 2 or n0 < 2:
        raise ValueError(f"某一组样本过少: treatment={n1}, control={n0}")

    _require_two_classes(y[mask1], "干预组(treatment=True)")
    _require_two_classes(y[mask0], "对照组(treatment=False)")

    model_treatment = build_pipeline(
        classifier=_build_learner(seed=seed, overrides=learner_overrides)
    )
    model_control = build_pipeline(
        classifier=_build_learner(seed=seed, overrides=learner_overrides)
    )

    model_treatment.fit(X_all.loc[mask1], y[mask1])
    model_control.fit(X_all.loc[mask0], y[mask0])

    params = _learner_params(model_treatment)
    if params != _learner_params(model_control):  # pragma: no cover - 同参数构造，防御性
        raise AssertionError("两个结果模型的超参数不一致，登记会失真")

    return FittedTLerner(
        model_treatment=model_treatment,
        model_control=model_control,
        feature_columns=feature_columns,
        base_learner=BASE_LEARNER,
        params=params,
        n_train_treatment=n1,
        n_train_control=n0,
        trained_at=_now_iso(),
    )


# ============================================================================
#  购买预测基线
# ============================================================================


@dataclass(frozen=True, slots=True)
class FittedOutcomeModel:
    """已拟合的单结果模型，预测 `P(Y=1|X=x)`。

    用途：策略基线之一（按预测购买概率排序）以及对照。
    **绝不用它的 AUC 证明增量模型有效**（docs/06 §4.1）。
    """

    model: Pipeline
    feature_columns: tuple[str, ...]
    base_learner: str
    params: dict
    n_train: int
    trained_at: str

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回 `P(Y=1)`，形状 `(n,)`、`float64`（不是两列的矩阵）。"""
        assert_no_leakage(self.feature_columns)
        return _predict_mu(self.model, X, self.feature_columns)

    def to_registry_dict(self) -> dict[str, Any]:
        return {
            "model_type": "outcome_model",
            "base_learner": self.base_learner,
            "params": dict(self.params),
            "feature_columns": list(self.feature_columns),
            "n_features_in": len(self.feature_columns),
            "n_features_out": len(feature_names_out(self.model)),
            "n_train": self.n_train,
            "trained_at": self.trained_at,
        }


def fit_outcome_model(
    train_df: pd.DataFrame,
    *,
    treatment: bool | None = None,
    seed: int = DEFAULT_SEED,
    **learner_overrides: Any,
) -> FittedOutcomeModel:
    """拟合购买预测基线。

    Args:
        train_df: 训练集，需含 `y`(bool) 与全部实验前特征列。
        treatment: `None` = 两组**合并**训练；`True` = 只用干预组；`False` = 只用对照组。
            两种口径都做、取验证集更优者是**调用方**的选择（docs/06 §4.1），
            本函数不替调用方选择。
        seed: 随机种子，默认 20261011。
        **learner_overrides: 覆盖 `LogisticRegression` 的超参数。

    Returns:
        `FittedOutcomeModel`。

    Raises:
        ValueError: 缺少列、筛选后无样本、或 `y` 只有一类。
    """
    feature_columns = _guard_feature_columns()

    if "y" not in train_df.columns:
        raise ValueError("train_df 缺少 'y' 列")

    frame = train_df
    label = "全部样本(treatment=None)"
    if treatment is not None:
        if "treatment" not in train_df.columns:
            raise ValueError("treatment 不为 None 时需要 'treatment' 列用于筛选")
        mask = train_df["treatment"].to_numpy(dtype=bool)
        frame = train_df.loc[mask if treatment else ~mask]
        label = "干预组(treatment=True)" if treatment else "对照组(treatment=False)"

    if frame.empty:
        raise ValueError(f"{label} 筛选后无样本")

    X = _select_X(frame, feature_columns)
    y = frame["y"].to_numpy(dtype=np.int64)
    _require_two_classes(y, label)

    pipeline = build_pipeline(classifier=_build_learner(seed=seed, overrides=learner_overrides))
    pipeline.fit(X, y)

    return FittedOutcomeModel(
        model=pipeline,
        feature_columns=feature_columns,
        base_learner=BASE_LEARNER,
        params=_learner_params(pipeline),
        n_train=int(len(frame)),
        trained_at=_now_iso(),
    )


__all__ = [
    "BASE_LEARNER",
    "DEFAULT_MAX_ITER",
    "FittedOutcomeModel",
    "FittedTLerner",
    "PREPROCESS_STEP_NAME",
    "fit_outcome_model",
    "fit_t_learner",
]
