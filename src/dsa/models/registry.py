"""模型登记与分数的读写接口（阶段四）。

对应表：`models`（sql/003_models.sql + sql/006_policy_contrasts_and_identity.sql）、
`customer_scores`、`policy_evaluations`、`policy_contrasts`。

设计要点
--------
1. **写操作只接受可写引擎**（`get_engine(readonly=False)`，账号 `dsa_app`）。
   只读账号 `dsa_ro` 处于会话级只读事务，写操作会被数据库直接拒绝——这是刻意保留的防线：
   离线训练负责写，Agent 只负责读（docs/06 §8.1）。
2. **幂等靠身份指纹，不靠 model_key**（缺陷 F10）：`model_key` 相同**且**
   :func:`compute_identity_fingerprint` 相同才幂等返回已有 `model_id`；
   指纹不同一律抛错并列出**具体哪几个字段**不同。见 :func:`register_model`。
3. **批量写**：分数有 42,613 行，用 `COPY FROM STDIN` 灌入临时表后一次性
   `INSERT ... SELECT`，绝不逐行 INSERT。
4. 本模块**不 import** `dsa.models.policy` 的策略对象：策略评估结果只通过 `getattr`
   读取属性，避免训练/评估模块与本模块相互依赖（`POLICY_KINDS` 是策略名的**副本校验**，
   真源在 `dsa.models.policy.POLICY_DEFINITIONS`，两者一致性有测试覆盖）。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping, Sequence
from uuid import UUID

import pandas as pd
from sqlalchemy import Engine, text

__all__ = [
    "ModelRecord",
    "register_model",
    "save_customer_scores",
    "load_model_record",
    "list_models",
    "save_policy_evaluation",
    "load_policy_evaluations",
    "save_policy_contrast",
    "load_policy_contrasts",
    "compute_identity_fingerprint",
    "identity_payload",
    "canonical_json",
    "normalise_value",
    "diff_identity_payloads",
    "backfill_identity_fingerprints",
    "IdentityMismatchError",
    "IDENTITY_FIELDS",
    "MODEL_TYPES",
    "POLICY_KINDS",
    "SCORE_COLUMNS",
    "MODEL_KEY_PATTERN",
]

#: 允许的模型类型，与 003_models.sql 的 models_type_ck 一致
MODEL_TYPES: tuple[str, ...] = ("t_learner", "outcome_model")

#: 允许的策略，与 `dsa.models.policy.POLICY_DEFINITIONS`（唯一真源）一致。
#:
#: ⚠️ 旧名 ``propensity`` **不在**其中：它同时指过 mu0 与合并购买预测两个输出，
#: 语义不明（缺陷 5.2）。历史上的旧行由 sql/006 迁移改名成 ``control_probability``。
POLICY_KINDS: tuple[str, ...] = (
    "random",
    "purchase_probability",
    "control_probability",
    "uplift",
    "all_treat",
    "none",
)

#: 客户分数表的四个分数列；一行至少要有其中一个（见 customer_scores_any_ck）
SCORE_COLUMNS: tuple[str, ...] = ("mu1", "mu0", "uplift", "propensity_score")

#: 分数列的取值上下界，与 customer_scores 的 CHECK 一致
_SCORE_BOUNDS: dict[str, tuple[float, float]] = {
    "mu1": (0.0, 1.0),
    "mu0": (0.0, 1.0),
    "uplift": (-1.0, 1.0),
    "propensity_score": (0.0, 1.0),
}

#: model_key 的合法形式，与 models_key_ck 一致
MODEL_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

#: 干预后字段（docs/06 §1.2）。出现即构成数据泄漏，数据库 CHECK 也会拦。
POST_TREATMENT_COLUMNS = frozenset(
    {"visit", "visited", "conversion", "converted", "spend", "visit_rate", "conversion_rate"}
)

#: 分数写入用的临时表名（会话级，pg_temp 命名空间）
_STAGE_TABLE = "stage_customer_scores"


# ============================================================================
#  模型身份指纹（缺陷 F10）
# ============================================================================
#
#  问题：同一个 model_key 可以接受**不同**的训练配置，接口返回成功与原 model_id，
#  数据库里留的却是旧参数/旧种子/旧代码版本；分数写入又是"冲突不更新"，
#  于是重新训练可能把新策略评估挂到同一个模型键下，版本对应关系被破坏。
#
#  做法：把"这个模型版本到底是什么"写成一个**指纹**。指纹相同 -> 幂等返回旧 id；
#  指纹不同 -> 抛错，并列出具体哪几个字段不同（旧值 vs 新值）。
#
#  指纹必须**稳定**：params 的键序、1 与 1.0、numpy 标量都不能影响结果。

#: 进入指纹的身份字段（顺序即报错信息的顺序）。
IDENTITY_FIELDS: tuple[str, ...] = (
    "model_type",
    "dataset_version",
    "experiment_key",
    "comparison_key",
    "feature_columns",
    "base_learner",
    "params",
    "split_config",
    "code_version",
    "preprocessing",
)

#: 指纹前缀。写进数据库，方便一眼看出这不是随手填的字符串。
FINGERPRINT_PREFIX = "sha256:"


def normalise_value(value: Any) -> Any:
    """把任意入参规整成**可比较、可规范化序列化**的纯 Python 结构。

    规则（这些规则就是"指纹稳定"的全部来源）：

    * ``bool`` 先于 ``int`` 判断（``True`` 不能被当成 1）；
    * 整数保持整数；**整数值的浮点归一成整数**，于是 ``1`` 与 ``1.0`` 等价；
    * 其余浮点用 Python 的 repr（最短往返表示），同值必然同表示；
    * ``Decimal``（psycopg 读 numeric 时会出现）同样归一；
    * 映射按键排序；序列（list/tuple）保持顺序——``feature_columns`` 的顺序
      是模型语义的一部分，不能排序掉；
    * numpy 标量转成对应的 Python 标量；
    * 其他类型直接报错，而不是 ``str()`` 之后悄悄进指纹。
    """
    import numpy as np  # 局部导入：本模块只在这里需要 numpy

    if isinstance(value, bool):
        return value
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"指纹不接受 nan/inf：{value!r}")
        return int(number) if number.is_integer() else number
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, Mapping):
        return {str(k): normalise_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise_value(v) for v in value]
    raise TypeError(
        f"不支持的值类型 {type(value).__name__}（{value!r}）："
        "身份指纹只接受 JSON 标量、映射与序列，避免把不可比较的表示写进指纹"
    )


def canonical_json(value: Any) -> str:
    """规范化 JSON 文本：键排序、无多余空白、整数与整值浮点同表示。"""
    return json.dumps(
        normalise_value(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def identity_payload(
    *,
    model_type: str,
    dataset_version: str,
    experiment_key: str,
    comparison_key: str,
    feature_columns: Sequence[str],
    base_learner: str,
    params: Mapping[str, Any],
    split_config: Mapping[str, Any],
    code_version: str,
    preprocessing: Mapping[str, Any],
) -> dict[str, Any]:
    """身份指纹的**输入载荷**（规范化后的字段字典）。

    单独抽出来是为了让"指纹不同"能按字段定位：报错信息需要旧值/新值的逐字段对照，
    而不只是两个 sha256。
    """
    return {
        "model_type": model_type,
        "dataset_version": dataset_version,
        "experiment_key": experiment_key,
        "comparison_key": comparison_key,
        "feature_columns": [str(c) for c in feature_columns],
        "base_learner": base_learner,
        "params": dict(params),
        "split_config": dict(split_config),
        "code_version": code_version,
        "preprocessing": dict(preprocessing),
    }


def compute_identity_fingerprint(**kwargs: Any) -> str:
    """模型身份指纹：``sha256:<hex>``，由 :func:`identity_payload` 规范化后哈希。

    **完全相同的配置必然得到相同指纹**（键序、1 与 1.0、numpy 标量都不影响），
    任何一项配置变化都会改变指纹。覆盖率有测试：
    ``tests/test_stage4_identity.py``。
    """
    payload = identity_payload(**kwargs)
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return FINGERPRINT_PREFIX + digest


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """把嵌套载荷摊平成 ``路径 -> 值``（映射下钻，序列当叶子）。"""
    out: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(_flatten(value, prefix=f"{path}."))
        else:
            out[path] = value
    return out


def diff_identity_payloads(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """逐字段比较两份载荷，返回 ``{字段路径: (旧值, 新值)}``。

    嵌套的 ``params`` / ``split_config`` / ``preprocessing`` 会下钻到具体键
    （例如 ``params.C``、``split_config.seed``），这样报错信息能直接指出改了哪一项。
    """
    flat_old, flat_new = _flatten(old), _flatten(new)
    diff: dict[str, tuple[Any, Any]] = {}
    for path in sorted(set(flat_old) | set(flat_new)):
        a = flat_old.get(path, "<缺失>")
        b = flat_new.get(path, "<缺失>")
        if normalise_value(a) != normalise_value(b):
            diff[path] = (a, b)
    return diff


class IdentityMismatchError(ValueError):
    """同一 ``model_key`` 被用于**不同**训练配置（指纹不同）。

    ``diff`` 是 ``{字段路径: (旧值, 新值)}``；``fingerprint_old`` / ``fingerprint_new``
    是两份指纹。消息里给出可直接照做的处置：换一个新的 ``model_key``。
    """

    def __init__(
        self,
        model_key: str,
        *,
        model_id: int,
        diff: Mapping[str, tuple[Any, Any]],
        fingerprint_old: str,
        fingerprint_new: str,
        legacy_row: bool = False,
    ) -> None:
        self.model_key = model_key
        self.model_id = model_id
        self.diff = dict(diff)
        self.fingerprint_old = fingerprint_old
        self.fingerprint_new = fingerprint_new
        lines = [
            f"model_key {model_key!r} 已登记为**另一个**模型版本（model_id={model_id}），"
            "本次登记的训练配置与它不同。以下是逐字段对照（旧值 -> 新值）："
        ]
        for path, (a, b) in diff.items():
            lines.append(f"  · {path}: {_short(a)} -> {_short(b)}")
        if legacy_row:
            lines.append(
                "  提示：该行是迁移前登记的旧行（identity_fingerprint 为空或 "
                "preprocessing_config 为空）。请先运行 "
                "`dsa.models.registry.backfill_identity_fingerprints` 回填，再重试。"
            )
        lines.append(
            "处置：**换一个新的 model_key**（如 tlearner_lr_v2）登记本次配置；"
            "或者把本次配置改回与已登记版本完全一致。"
            "本接口不会自动改 key，也不会用旧配置覆盖新配置。"
        )
        super().__init__("\n".join(lines))


def _short(value: Any, limit: int = 120) -> str:
    """报错信息里的值：规范 JSON，过长则截断。"""
    try:
        text_value = canonical_json(value)
    except (TypeError, ValueError):
        text_value = repr(value)
    return text_value if len(text_value) <= limit else text_value[: limit - 3] + "..."


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """模型登记行（只读视图对象）。"""

    model_id: int
    model_key: str
    model_type: str
    dataset_version: str
    experiment_key: str
    comparison_key: str
    feature_columns: tuple[str, ...]
    base_learner: str
    params: dict
    split_config: dict
    code_version: str
    trained_at: str
    preprocessing_config: dict = field(default_factory=dict)
    identity_fingerprint: str | None = None


# ============================================================================
#  内部工具
# ============================================================================


def _as_dict(value: Any) -> dict:
    """jsonb 列 → dict（兼容驱动返回 str 的情况）。"""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes, bytearray)):
        return json.loads(value)
    return dict(value)


def _as_iso(value: Any) -> str:
    """timestamptz 列 → ISO 字符串。"""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _as_number(value: Any) -> float | None:
    """numeric 列 → float（None 保持 None）。

    **NaN / inf 一律记为 NULL**：``dsa.models.policy.PolicyScore`` 用 ``math.nan``
    表示「估计量无定义」（bootstrap 全部重抽样都无定义，或选取人数为 0），
    而 PostgreSQL 的 numeric 认为 ``NaN`` 比任何数都大，会让
    ``ci_low <= ips_estimate <= ci_high`` 这条 CHECK 误判。
    未定义即缺失，缺失在库里就是 NULL。
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _as_text(value: Any) -> str | None:
    """uuid / text 列 → 字符串。"""
    if value is None:
        return None
    if isinstance(value, UUID):
        return str(value)
    return str(value)


_MODEL_SELECT = """
select
    m.model_id,
    m.model_key,
    m.model_type,
    d.version_label as dataset_version,
    e.experiment_key,
    m.comparison_key,
    m.feature_columns,
    m.base_learner,
    m.params,
    m.split_config,
    m.code_version,
    m.trained_at,
    m.preprocessing_config,
    m.identity_fingerprint
from models m
join datasets    d on d.dataset_id    = m.dataset_id
join experiments e on e.experiment_id = m.experiment_id
"""


def _row_to_record(row: Mapping[str, Any]) -> ModelRecord:
    return ModelRecord(
        model_id=int(row["model_id"]),
        model_key=row["model_key"],
        model_type=row["model_type"],
        dataset_version=row["dataset_version"],
        experiment_key=row["experiment_key"],
        comparison_key=row["comparison_key"],
        feature_columns=tuple(row["feature_columns"] or ()),
        base_learner=row["base_learner"],
        params=_as_dict(row["params"]),
        split_config=_as_dict(row["split_config"]),
        code_version=row["code_version"],
        trained_at=_as_iso(row["trained_at"]),
        preprocessing_config=_as_dict(row.get("preprocessing_config")),
        identity_fingerprint=row.get("identity_fingerprint"),
    )


def _check_feature_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    """校验特征列：非空、全为字符串、不含干预后字段。"""
    cols = tuple(str(c) for c in feature_columns)
    if not cols:
        raise ValueError("feature_columns 不能为空")
    leaked = sorted(set(cols) & POST_TREATMENT_COLUMNS)
    if leaked:
        raise ValueError(
            f"feature_columns 含干预后字段 {leaked}，构成数据泄漏（docs/06 §1.2）；"
            "允许的输入只有 recency/history_segment/history/history_band/"
            "mens/womens/zip_code/newbie/channel 及其训练集内派生的预处理列"
        )
    return cols


def _resolve_ids(conn, *, dataset_version: str, experiment_key: str, comparison_key: str) -> tuple[int, int]:
    """把业务键解析为内部 ID，并核对预注册比较是否存在。"""
    dataset_id = conn.execute(
        text("select dataset_id from datasets where version_label = :v"),
        {"v": dataset_version},
    ).scalar_one_or_none()
    if dataset_id is None:
        raise ValueError(f"数据版本未登记：{dataset_version!r}（先执行 scripts/load_data.py）")

    experiment_id = conn.execute(
        text("select experiment_id from experiments where experiment_key = :k"),
        {"k": experiment_key},
    ).scalar_one_or_none()
    if experiment_id is None:
        raise ValueError(f"实验未登记：{experiment_key!r}")

    exists = conn.execute(
        text(
            "select 1 from experiment_comparisons "
            "where experiment_id = :e and comparison_key = :c"
        ),
        {"e": experiment_id, "c": comparison_key},
    ).scalar_one_or_none()
    if exists is None:
        raise ValueError(
            f"实验 {experiment_key!r} 下不存在预注册比较 {comparison_key!r}；"
            "比较必须先在 experiment_comparisons 中登记（docs/02 §11）"
        )
    return int(dataset_id), int(experiment_id)


# ============================================================================
#  写：模型登记
# ============================================================================

_INSERT_MODEL = """
with ins as (
    insert into models (
        model_key, model_type, dataset_id, experiment_id, comparison_key,
        feature_columns, base_learner, params, split_config, code_version, notes,
        preprocessing_config, identity_fingerprint
    ) values (
        :model_key, :model_type, :dataset_id, :experiment_id, :comparison_key,
        :feature_columns, :base_learner, cast(:params as jsonb),
        cast(:split_config as jsonb), :code_version, :notes,
        cast(:preprocessing_config as jsonb), :identity_fingerprint
    )
    on conflict (model_key) do nothing
    returning model_id
)
select model_id from ins
union all
select model_id from models where model_key = :model_key
limit 1
"""


def register_model(
    engine_writable: Engine,
    *,
    model_key: str,
    model_type: str,
    dataset_version: str,
    experiment_key: str,
    comparison_key: str = "primary",
    feature_columns: Sequence[str],
    base_learner: str,
    params: Mapping[str, Any],
    split_config: Mapping[str, Any],
    code_version: str,
    preprocessing: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> int:
    """登记一个模型版本，返回 ``model_id``。

    **幂等的判据是身份指纹，不是 model_key**（缺陷 F10）：

    * 同一个 ``model_key`` 且 :func:`compute_identity_fingerprint` **完全相同** ->
      幂等，返回已有 ``model_id``，不新增行也不改任何已存字段；
    * 同一个 ``model_key`` 但指纹**不同** -> 抛 :class:`IdentityMismatchError`，
      消息里逐字段列出旧值与新值，并提示换一个新的 ``model_key``。
      **不会**静默返回旧 id，也**不会**覆盖旧配置。

    指纹覆盖：``model_type`` / 数据集版本 / 实验 / 比较 / ``feature_columns``（有序）/
    ``base_learner`` / ``params``（规范化 JSON）/ ``split_config``（含 seed、ratios、
    counts）/ ``code_version`` / 预处理配置。

    为什么"分数写入用冲突不更新"在这条不变式下是安全的
    --------------------------------------------------
    分数行的归属键是 ``model_id``（``unique (model_id, customer_id)``），
    而 ``model_id`` 只能由本函数返回。指纹不同的配置**根本拿不到旧 model_id**，
    因此不可能"用新配置的分数去撞旧版本的行"。剩下的 ``do nothing`` 只处理
    "同一个不可变版本重复写同一批分数"这一种情形，跳过是正确的（幂等）。
    这条不变式有测试锁住：``tests/test_stage4_identity.py``。

    Args:
        engine_writable: ``get_engine(readonly=False)``；只读引擎会被数据库拒绝。
        feature_columns: 模型输入列名（含预处理后特征名），不得含干预后字段。
        params: 全部超参数，写入 jsonb。
        split_config: 划分配置，期望含 seed / ratios / counts（docs/06 §8）。
        code_version: git commit。
        preprocessing: 预处理配置；缺省取 ``dsa.models.features.preprocessing_config()``
            （项目固定的那一套）。显式传入不同配置会改变指纹。

    Returns:
        models.model_id

    Raises:
        IdentityMismatchError: 同一 model_key 的训练配置与已登记版本不同。
        ValueError: model_key / model_type 非法、特征含干预后字段、code_version 为空等。
    """
    if not MODEL_KEY_PATTERN.match(model_key or ""):
        raise ValueError(
            f"model_key 不合法：{model_key!r}；要求 ^[a-z][a-z0-9_]*$，如 tlearner_lr_v1"
        )
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type 必须是 {MODEL_TYPES} 之一，收到 {model_type!r}")
    cols = _check_feature_columns(feature_columns)
    if not base_learner:
        raise ValueError("base_learner 不能为空")
    if not code_version:
        raise ValueError("code_version 不能为空（复现所需，见 docs/06 §8）")
    if not isinstance(params, Mapping) or not isinstance(split_config, Mapping):
        raise ValueError("params / split_config 必须是映射（会写成 jsonb 对象）")

    from dsa.models.features import preprocessing_config as _default_preprocessing

    preprocessing_cfg = dict(preprocessing) if preprocessing is not None else _default_preprocessing()
    if not isinstance(preprocessing_cfg, Mapping):
        raise ValueError("preprocessing 必须是映射（会写成 jsonb 对象）")

    fingerprint = compute_identity_fingerprint(
        model_type=model_type,
        dataset_version=dataset_version,
        experiment_key=experiment_key,
        comparison_key=comparison_key,
        feature_columns=cols,
        base_learner=base_learner,
        params=params,
        split_config=split_config,
        code_version=code_version,
        preprocessing=preprocessing_cfg,
    )
    new_payload = identity_payload(
        model_type=model_type,
        dataset_version=dataset_version,
        experiment_key=experiment_key,
        comparison_key=comparison_key,
        feature_columns=cols,
        base_learner=base_learner,
        params=params,
        split_config=split_config,
        code_version=code_version,
        preprocessing=preprocessing_cfg,
    )

    payload = {
        "model_key": model_key,
        "model_type": model_type,
        "comparison_key": comparison_key,
        "feature_columns": list(cols),
        "base_learner": base_learner,
        "params": json.dumps(dict(params), ensure_ascii=False),
        "split_config": json.dumps(dict(split_config), ensure_ascii=False),
        "code_version": code_version,
        "notes": notes,
        "preprocessing_config": json.dumps(preprocessing_cfg, ensure_ascii=False),
        "identity_fingerprint": fingerprint,
    }

    with engine_writable.begin() as conn:
        dataset_id, experiment_id = _resolve_ids(
            conn,
            dataset_version=dataset_version,
            experiment_key=experiment_key,
            comparison_key=comparison_key,
        )
        payload["dataset_id"] = dataset_id
        payload["experiment_id"] = experiment_id

        model_id = conn.execute(text(_INSERT_MODEL), payload).scalar_one()

        # 幂等核对：无论是刚插入还是命中已有行，都用**库里存的内容**重算指纹再比。
        # 重算而不是直接读 identity_fingerprint：这样"指纹列被写坏/未回填"也能被发现。
        row = conn.execute(
            text(
                """
                select
                    m.model_id, m.model_type, m.comparison_key,
                    d.version_label as dataset_version,
                    e.experiment_key,
                    m.feature_columns, m.base_learner, m.params, m.split_config,
                    m.code_version, m.preprocessing_config, m.identity_fingerprint
                from models m
                join datasets    d on d.dataset_id    = m.dataset_id
                join experiments e on e.experiment_id = m.experiment_id
                where m.model_id = :mid
                """
            ),
            {"mid": model_id},
        ).mappings().one()

        stored_payload = identity_payload(
            model_type=row["model_type"],
            dataset_version=row["dataset_version"],
            experiment_key=row["experiment_key"],
            comparison_key=row["comparison_key"],
            feature_columns=tuple(row["feature_columns"] or ()),
            base_learner=row["base_learner"],
            params=_as_dict(row["params"]),
            split_config=_as_dict(row["split_config"]),
            code_version=row["code_version"],
            preprocessing=_as_dict(row["preprocessing_config"]),
        )
        stored_fingerprint = compute_identity_fingerprint(**stored_payload)
        if stored_fingerprint != fingerprint:
            diff = diff_identity_payloads(stored_payload, new_payload)
            if not diff:  # pragma: no cover - 指纹一致时不可能有差异；留作显式防线
                diff = {"<未知>": ("指纹不同但字段比较一致", "请检查哈希实现")}
            raise IdentityMismatchError(
                model_key,
                model_id=int(model_id),
                diff=diff,
                fingerprint_old=stored_fingerprint,
                fingerprint_new=fingerprint,
                legacy_row=(
                    row["identity_fingerprint"] is None
                    or not _as_dict(row["preprocessing_config"])
                ),
            )

    return int(model_id)


# ============================================================================
#  回填身份指纹（迁移既有库）
# ============================================================================


@dataclass(frozen=True, slots=True)
class BackfillResult:
    """一次回填的结果（只读报表，便于日志与核对）。"""

    model_key: str
    model_id: int
    fingerprint: str
    changed: bool
    note: str


def backfill_identity_fingerprints(
    engine_writable: Engine,
    *,
    model_keys: Sequence[str] | None = None,
    dry_run: bool = False,
) -> list[BackfillResult]:
    """为既有模型行**重算**身份指纹与预处理配置（迁移用）。

    可复现性：指纹完全由数据库里**已存的字段**重算——
    ``model_type`` / 数据版本 / 实验 / 比较 / ``feature_columns`` / ``base_learner`` /
    ``params`` / ``split_config`` / ``code_version`` / ``preprocessing_config``，
    不写任何常量。唯一例外：迁移前的旧行没有 ``preprocessing_config``
    （列为空对象），此时用项目固定的预处理配置
    ``dsa.models.features.preprocessing_config()`` 补上，并把它写回数据库——
    补完之后，这一行的指纹就能**只从数据库**重算出来。

    Args:
        engine_writable: 可写引擎。
        model_keys: 只回填这些 key；``None`` 表示全部。
        dry_run: 只算不写（用于先看差异）。

    Returns:
        每行一条 :class:`BackfillResult`。
    """
    from dsa.models.features import preprocessing_config as _default_preprocessing

    sql = """
    select
        m.model_id, m.model_key, m.model_type, m.comparison_key,
        d.version_label as dataset_version,
        e.experiment_key,
        m.feature_columns, m.base_learner, m.params, m.split_config,
        m.code_version, m.preprocessing_config, m.identity_fingerprint
    from models m
    join datasets    d on d.dataset_id    = m.dataset_id
    join experiments e on e.experiment_id = m.experiment_id
    where cast(:model_keys as text[]) is null
       or m.model_key = any(cast(:model_keys as text[]))
    order by m.model_key
    """
    results: list[BackfillResult] = []
    with engine_writable.begin() as conn:
        rows = conn.execute(
            text(sql),
            {"model_keys": None if model_keys is None else list(model_keys)},
        ).mappings().all()
        for row in rows:
            stored_pre = _as_dict(row["preprocessing_config"])
            used_default_pre = not stored_pre
            if used_default_pre:
                stored_pre = _default_preprocessing()
            payload = identity_payload(
                model_type=row["model_type"],
                dataset_version=row["dataset_version"],
                experiment_key=row["experiment_key"],
                comparison_key=row["comparison_key"],
                feature_columns=tuple(row["feature_columns"] or ()),
                base_learner=row["base_learner"],
                params=_as_dict(row["params"]),
                split_config=_as_dict(row["split_config"]),
                code_version=row["code_version"],
                preprocessing=stored_pre,
            )
            fingerprint = compute_identity_fingerprint(**payload)
            changed = (
                row["identity_fingerprint"] != fingerprint or used_default_pre
            )
            note = (
                "重算自库内字段"
                if not used_default_pre
                else "预处理配置缺失，按项目固定配置补齐后重算"
            )
            if changed and not dry_run:
                conn.execute(
                    text(
                        "update models set identity_fingerprint = :fp, "
                        "preprocessing_config = cast(:pre as jsonb) "
                        "where model_id = :mid"
                    ),
                    {
                        "fp": fingerprint,
                        "pre": json.dumps(stored_pre, ensure_ascii=False),
                        "mid": int(row["model_id"]),
                    },
                )
            results.append(
                BackfillResult(
                    model_key=row["model_key"],
                    model_id=int(row["model_id"]),
                    fingerprint=fingerprint,
                    changed=changed,
                    note=note + ("（dry_run，未写库）" if dry_run else ""),
                )
            )
    return results


# ============================================================================
#  写：客户分数（批量 COPY）
# ============================================================================

_CREATE_STAGE = f"""
drop table if exists pg_temp.{_STAGE_TABLE};
create temp table {_STAGE_TABLE} (
    customer_key     text,
    arm_key          text,
    mu1              numeric(9,8),
    mu0              numeric(9,8),
    uplift           numeric(9,8),
    propensity_score numeric(9,8)
) on commit drop;
"""

_COPY_STAGE = f"""
copy {_STAGE_TABLE} (customer_key, arm_key, mu1, mu0, uplift, propensity_score)
from stdin with (format csv)
"""

_INSERT_SCORES = f"""
insert into customer_scores (model_id, customer_id, arm_key, mu1, mu0, uplift, propensity_score)
select
    %(model_id)s,
    cu.customer_id,
    coalesce(nullif(s.arm_key, ''), ar.arm_key),
    s.mu1, s.mu0, s.uplift, s.propensity_score
from {_STAGE_TABLE} s
join customers cu on cu.customer_key = s.customer_key
left join assignments a
       on a.experiment_id = %(experiment_id)s
      and a.customer_id   = cu.customer_id
left join experiment_arms ar on ar.arm_id = a.arm_id
on conflict (model_id, customer_id) do nothing
"""


def _prepare_scores_frame(scores_df: Any) -> pd.DataFrame:
    """校验并整理待写入的分数表。"""
    if not isinstance(scores_df, pd.DataFrame):
        raise TypeError(f"scores_df 必须是 pandas DataFrame，收到 {type(scores_df).__name__}")
    if "customer_key" not in scores_df.columns:
        raise ValueError("scores_df 必须含 customer_key 列（带数据集命名空间的客户业务键）")

    present = [c for c in SCORE_COLUMNS if c in scores_df.columns]
    if not present:
        raise ValueError(f"scores_df 至少要含 {list(SCORE_COLUMNS)} 中的一个分数列")

    frame = pd.DataFrame(index=scores_df.index)
    frame["customer_key"] = scores_df["customer_key"].astype("string")
    if frame["customer_key"].isna().any():
        raise ValueError("scores_df.customer_key 存在空值")
    frame["customer_key"] = frame["customer_key"].astype(str).str.strip()

    if "arm_key" in scores_df.columns:
        frame["arm_key"] = scores_df["arm_key"].astype("string").fillna("").astype(str)
    else:
        # 缺省时由 assignments 反查该客户在模型实验中的组别
        frame["arm_key"] = ""

    for col in SCORE_COLUMNS:
        if col in scores_df.columns:
            series = pd.to_numeric(scores_df[col], errors="raise").astype(float)
            lo, hi = _SCORE_BOUNDS[col]
            bad = series.notna() & ~series.between(lo, hi)
            if bool(bad.any()):
                raise ValueError(
                    f"{col} 有 {int(bad.sum())} 个值越界，必须在 [{lo}, {hi}] 内"
                    f"（与 customer_scores 的 CHECK 一致）"
                )
            frame[col] = series
        else:
            frame[col] = float("nan")

    return frame


def save_customer_scores(engine_writable: Engine, *, model_id: int, scores_df: Any) -> int:
    """批量写入客户分数，返回**本次实际新增**的行数。

    Args:
        engine_writable: ``get_engine(readonly=False)``。
        model_id: 已登记的模型版本。
        scores_df: 需含 ``customer_key``，以及 ``mu1/mu0/uplift/propensity_score``
            中的至少一个（缺列按 NULL 写）。可选 ``arm_key``；不给则由
            ``assignments`` 反查。

    Returns:
        新增行数。**幂等**：已存在的 ``(model_id, customer_id)`` 被
        ``on conflict do nothing`` 跳过，因此重复写入返回 0。

    Note:
        用 ``COPY FROM STDIN`` 灌入临时表后一次性 ``INSERT ... SELECT``。
        42,613 行逐行 INSERT 会很慢，故不采用。
    """
    frame = _prepare_scores_frame(scores_df)
    if frame.empty:
        return 0

    with engine_writable.begin() as conn:
        row = conn.execute(
            text("select experiment_id from models where model_id = :mid"),
            {"mid": model_id},
        ).one_or_none()
        if row is None:
            raise ValueError(f"model_id={model_id} 未登记；请先调用 register_model")
        experiment_id = int(row.experiment_id)

        driver = conn.connection.driver_connection
        with driver.cursor() as cur:
            cur.execute(_CREATE_STAGE)

            if not hasattr(cur, "copy"):
                raise RuntimeError(
                    f"当前驱动 {type(cur).__module__} 不支持 COPY 协议；"
                    "本项目固定使用 postgresql+psycopg（psycopg 3）"
                )
            payload = frame.to_csv(index=False, header=False, na_rep="").encode("utf-8")
            with cur.copy(_COPY_STAGE) as copy:
                copy.write(payload)

            # 分数行的客户必须真实存在：否则是命名空间/键写错了，必须显式暴露
            unmatched = cur.execute(
                f"select count(*) from {_STAGE_TABLE} s "
                "left join customers cu on cu.customer_key = s.customer_key "
                "where cu.customer_id is null"
            ).fetchone()[0]
            if unmatched:
                sample = cur.execute(
                    f"select s.customer_key from {_STAGE_TABLE} s "
                    "left join customers cu on cu.customer_key = s.customer_key "
                    "where cu.customer_id is null limit 3"
                ).fetchall()
                raise ValueError(
                    f"有 {unmatched} 个 customer_key 在 customers 表中不存在"
                    f"（示例 {[r[0] for r in sample]}）；禁止跨数据集拼接客户（AGENTS.md §2.5）"
                )

            cur.execute(
                _INSERT_SCORES,
                {"model_id": int(model_id), "experiment_id": experiment_id},
            )
            written = int(cur.rowcount)

    return written


# ============================================================================
#  读：模型登记
# ============================================================================


def load_model_record(engine_readonly: Engine, model_key: str) -> ModelRecord | None:
    """按 ``model_key`` 读取模型登记；不存在返回 ``None``。"""
    with engine_readonly.connect() as conn:
        row = conn.execute(
            text(_MODEL_SELECT + " where m.model_key = :model_key"),
            {"model_key": model_key},
        ).mappings().one_or_none()
    return _row_to_record(row) if row is not None else None


def list_models(engine_readonly: Engine, *, dataset_version: str | None = None) -> list[ModelRecord]:
    """列出模型登记；给 ``dataset_version`` 时只列该数据版本下的模型。"""
    sql = _MODEL_SELECT + """
where (cast(:dataset_version as text) is null
       or d.version_label = cast(:dataset_version as text))
order by m.trained_at desc, m.model_key
"""
    with engine_readonly.connect() as conn:
        rows = conn.execute(text(sql), {"dataset_version": dataset_version}).mappings().all()
    return [_row_to_record(r) for r in rows]


# ============================================================================
#  写 / 读：策略评估
# ============================================================================

_REQUIRED_RESULT_ATTRS = ("policy", "fraction", "n_selected", "bootstrap_seed", "n_bootstrap")

_UPSERT_POLICY_EVAL = """
insert into policy_evaluations (
    model_id, policy, fraction, n_selected, ips_estimate, ci_low, ci_high,
    alpha, bootstrap_seed, n_bootstrap, observed_rate_selected, run_id
) values (
    :model_id, :policy, :fraction, :n_selected, :ips_estimate, :ci_low, :ci_high,
    :alpha, :bootstrap_seed, :n_bootstrap, :observed_rate_selected, cast(:run_id as uuid)
)
on conflict (model_id, policy, fraction) do update set
    n_selected             = excluded.n_selected,
    ips_estimate           = excluded.ips_estimate,
    ci_low                 = excluded.ci_low,
    ci_high                = excluded.ci_high,
    alpha                  = excluded.alpha,
    bootstrap_seed         = excluded.bootstrap_seed,
    n_bootstrap            = excluded.n_bootstrap,
    observed_rate_selected = excluded.observed_rate_selected,
    run_id                 = excluded.run_id
"""


def save_policy_evaluation(
    engine_writable: Engine,
    *,
    model_id: int,
    run_id: str | None,
    result: Any,
) -> None:
    """登记一条策略评估结果。

    ``result`` 为具备 ``policy/fraction/n_selected/ips_estimate/ci_low/ci_high/
    alpha/bootstrap_seed/n_bootstrap/observed_rate_selected`` 属性的对象
    （阶段四为 ``dsa.models.policy.PolicyScore``）。本模块刻意不 import 它，
    全部用 ``getattr`` 读取，避免评估模块与登记模块相互依赖。

    **未定义记为 NULL**：``PolicyScore`` 用 ``math.nan`` 表示「估计量无定义」
    （点估计无定义、bootstrap 区间无定义、选取人数为 0 时观测购买率无定义），
    这些一律写成 SQL NULL —— 与 ci_low/ci_high/observed_rate_selected 的
    「可为 NULL 表示未计算」语义一致，也避免 numeric 的 NaN 破坏区间 CHECK。

    **幂等**：同一 ``(model_id, policy, fraction)`` 重复登记走 ``do update``，
    行数不增加（重跑评估覆盖旧值）。
    """
    missing = [a for a in _REQUIRED_RESULT_ATTRS if getattr(result, a, None) is None]
    if missing:
        raise ValueError(
            f"result 缺少必需属性 {missing}（这些字段在 policy_evaluations 中是 NOT NULL）"
        )

    policy = getattr(result, "policy")
    if policy not in POLICY_KINDS:
        raise ValueError(f"policy 必须是 {POLICY_KINDS} 之一，收到 {policy!r}")

    fraction = float(getattr(result, "fraction"))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction 必须在 [0,1] 内，收到 {fraction}")

    ips_estimate = _as_number(getattr(result, "ips_estimate", None))
    ci_low = _as_number(getattr(result, "ci_low", None))
    ci_high = _as_number(getattr(result, "ci_high", None))
    if None not in (ips_estimate, ci_low, ci_high) and not (
        ci_low <= ips_estimate <= ci_high  # type: ignore[operator]
    ):
        raise ValueError(
            f"区间必须包住点估计：ci_low={ci_low} <= ips_estimate={ips_estimate} "
            f"<= ci_high={ci_high} 不成立（与 policy_evaluations_ci_ck 一致）"
        )

    alpha = getattr(result, "alpha", None)
    alpha = 0.05 if alpha is None else float(alpha)
    observed = _as_number(getattr(result, "observed_rate_selected", None))

    params = {
        "model_id": int(model_id),
        "policy": policy,
        "fraction": fraction,
        "n_selected": int(getattr(result, "n_selected")),
        "ips_estimate": ips_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": alpha,
        "bootstrap_seed": int(getattr(result, "bootstrap_seed")),
        "n_bootstrap": int(getattr(result, "n_bootstrap")),
        "observed_rate_selected": observed,
        "run_id": None if run_id is None else str(run_id),
    }
    with engine_writable.begin() as conn:
        conn.execute(text(_UPSERT_POLICY_EVAL), params)


def load_policy_evaluations(engine_readonly: Engine, model_key: str) -> list[dict]:
    """读取某模型版本的全部策略评估，按策略与触达比例排序。

    返回值已转为可直接 JSON 序列化的类型（Decimal→float、UUID→str、
    timestamptz→ISO 字符串）。
    """
    sql = """
    select
        pe.eval_id,
        m.model_key,
        m.model_type,
        pe.policy,
        pe.fraction,
        pe.n_selected,
        pe.ips_estimate,
        pe.ci_low,
        pe.ci_high,
        pe.alpha,
        pe.bootstrap_seed,
        pe.n_bootstrap,
        pe.observed_rate_selected,
        pe.run_id,
        pe.created_at
    from policy_evaluations pe
    join models m on m.model_id = pe.model_id
    where m.model_key = :model_key
    order by pe.policy, pe.fraction
    """
    with engine_readonly.connect() as conn:
        rows = conn.execute(text(sql), {"model_key": model_key}).mappings().all()
    return [
        {
            "eval_id": int(r["eval_id"]),
            "model_key": r["model_key"],
            "model_type": r["model_type"],
            "policy": r["policy"],
            "fraction": _as_number(r["fraction"]),
            "n_selected": int(r["n_selected"]),
            "ips_estimate": _as_number(r["ips_estimate"]),
            "ci_low": _as_number(r["ci_low"]),
            "ci_high": _as_number(r["ci_high"]),
            "alpha": _as_number(r["alpha"]),
            "bootstrap_seed": int(r["bootstrap_seed"]),
            "n_bootstrap": int(r["n_bootstrap"]),
            "observed_rate_selected": _as_number(r["observed_rate_selected"]),
            "run_id": _as_text(r["run_id"]),
            "created_at": _as_iso(r["created_at"]),
        }
        for r in rows
    ]


# ============================================================================
#  写 / 读：策略**差值**（F09 —— 差异的判据只能是差值区间）
# ============================================================================

_REQUIRED_CONTRAST_ATTRS = (
    "policy",
    "reference_policy",
    "fraction",
    "difference",
    "bootstrap_seed",
    "n_bootstrap",
)

_UPSERT_POLICY_CONTRAST = """
insert into policy_contrasts (
    model_id, fraction, policy, reference_policy, difference, ci_low, ci_high,
    alpha, bootstrap_seed, n_bootstrap, n_pairs_defined, is_primary, run_id
) values (
    :model_id, :fraction, :policy, :reference_policy, :difference, :ci_low, :ci_high,
    :alpha, :bootstrap_seed, :n_bootstrap, :n_pairs_defined, :is_primary,
    cast(:run_id as uuid)
)
on conflict (model_id, fraction, policy, reference_policy) do update set
    difference      = excluded.difference,
    ci_low          = excluded.ci_low,
    ci_high         = excluded.ci_high,
    alpha           = excluded.alpha,
    bootstrap_seed  = excluded.bootstrap_seed,
    n_bootstrap     = excluded.n_bootstrap,
    n_pairs_defined = excluded.n_pairs_defined,
    is_primary      = excluded.is_primary,
    run_id          = excluded.run_id
"""


def save_policy_contrast(
    engine_writable: Engine,
    *,
    model_id: int,
    run_id: str | None,
    contrast: Any,
) -> None:
    """登记一条策略差值（``V(policy) − V(reference_policy)``）结果。

    ``contrast`` 为具备 ``policy/reference_policy/fraction/difference/ci_low/ci_high/
    alpha/bootstrap_seed/n_bootstrap/n_pairs_defined/is_primary`` 属性的对象
    （阶段四为 ``dsa.models.policy.PolicyContrast``）。本模块刻意不 import 它，
    全部用 ``getattr`` 读取，避免评估模块与登记模块相互依赖。

    **未定义记为 NULL**：同 :func:`save_policy_evaluation`，``math.nan`` 一律写 NULL，
    避免 numeric 的 NaN 破坏 ``ci_low <= difference <= ci_high`` 这条 CHECK。

    **幂等**：同一 ``(model_id, fraction, policy, reference_policy)`` 走 ``do update``，
    重跑评估覆盖旧值，行数不增加。
    """
    missing = [a for a in _REQUIRED_CONTRAST_ATTRS if getattr(contrast, a, None) is None]
    if missing:
        raise ValueError(
            f"contrast 缺少必需属性 {missing}（这些字段在 policy_contrasts 中是 NOT NULL）"
        )
    policy = getattr(contrast, "policy")
    reference = getattr(contrast, "reference_policy")
    for name in (policy, reference):
        if name not in POLICY_KINDS:
            raise ValueError(f"policy 必须是 {POLICY_KINDS} 之一，收到 {name!r}")
    if policy == reference:
        raise ValueError(f"差值的两侧不能是同一个策略：{policy!r}")

    fraction = float(getattr(contrast, "fraction"))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction 必须在 [0,1] 内，收到 {fraction}")

    difference = _as_number(getattr(contrast, "difference"))
    ci_low = _as_number(getattr(contrast, "ci_low"))
    ci_high = _as_number(getattr(contrast, "ci_high"))
    if None not in (difference, ci_low, ci_high) and not (
        ci_low <= difference <= ci_high  # type: ignore[operator]
    ):
        raise ValueError(
            f"差值区间必须包住差值点估计：ci_low={ci_low} <= difference={difference} "
            f"<= ci_high={ci_high} 不成立（与 policy_contrasts_ci_ck 一致）"
        )

    alpha = getattr(contrast, "alpha", None)
    alpha = 0.05 if alpha is None else float(alpha)
    params = {
        "model_id": int(model_id),
        "fraction": fraction,
        "policy": policy,
        "reference_policy": reference,
        "difference": difference,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": alpha,
        "bootstrap_seed": int(getattr(contrast, "bootstrap_seed")),
        "n_bootstrap": int(getattr(contrast, "n_bootstrap")),
        "n_pairs_defined": int(getattr(contrast, "n_pairs_defined", 0)),
        "is_primary": bool(getattr(contrast, "is_primary", False)),
        "run_id": None if run_id is None else str(run_id),
    }
    with engine_writable.begin() as conn:
        conn.execute(text(_UPSERT_POLICY_CONTRAST), params)


def load_policy_contrasts(engine_readonly: Engine, model_key: str) -> list[dict]:
    """读取某模型版本的全部策略差值结果，按比例与是否为主要比较排序。"""
    sql = """
    select
        pc.contrast_id,
        m.model_key,
        m.model_type,
        pc.fraction,
        pc.policy,
        pc.reference_policy,
        pc.difference,
        pc.ci_low,
        pc.ci_high,
        pc.alpha,
        pc.bootstrap_seed,
        pc.n_bootstrap,
        pc.n_pairs_defined,
        pc.is_primary,
        pc.run_id,
        pc.created_at
    from policy_contrasts pc
    join models m on m.model_id = pc.model_id
    where m.model_key = :model_key
    order by pc.fraction, pc.is_primary desc, pc.policy, pc.reference_policy
    """
    with engine_readonly.connect() as conn:
        rows = conn.execute(text(sql), {"model_key": model_key}).mappings().all()
    return [
        {
            "contrast_id": int(r["contrast_id"]),
            "model_key": r["model_key"],
            "model_type": r["model_type"],
            "fraction": _as_number(r["fraction"]),
            "policy": r["policy"],
            "reference_policy": r["reference_policy"],
            "difference": _as_number(r["difference"]),
            "ci_low": _as_number(r["ci_low"]),
            "ci_high": _as_number(r["ci_high"]),
            "alpha": _as_number(r["alpha"]),
            "bootstrap_seed": int(r["bootstrap_seed"]),
            "n_bootstrap": int(r["n_bootstrap"]),
            "n_pairs_defined": int(r["n_pairs_defined"]),
            "is_primary": bool(r["is_primary"]),
            "run_id": _as_text(r["run_id"]),
            "created_at": _as_iso(r["created_at"]),
        }
        for r in rows
    ]
