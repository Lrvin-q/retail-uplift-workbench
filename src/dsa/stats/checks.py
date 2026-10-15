"""数据质量检查：把「这份数据能不能用来做因果分析」变成可复核的检查项。

依据
----
- `docs/02-分析协议.md` §9（缺失值）、§10（样本排除）、§11（SRM 检查）
- `docs/05-报告结构与异常分类.md` §3.3（异常码字典：`E_MISS_001` / `E_COVER_001` /
  `E_ID_001` / `E_ID_002` / `E_JOIN_001` / `E_GROUP_001` / `E_SRM_001`）
- `AGENTS.md` §2 铁律

三条铁律在本模块里的落点
------------------------
1. **分母恒为「全部被随机分配到该组的客户数」。**
   覆盖率的分母取自 `assignments`（应有个数），**不是** `outcomes` 的行数，
   更不是访问者数或购买者数。本模块只读检查，任何分母都不按
   `visited` / `converted` / `spend` 等**干预后**字段过滤。
2. **结果缺失绝不等于未购买。**
   缺失一律单独计数（`n_missing`）并写进 `detail`；覆盖率不足按异常码报告，
   绝不把缺失并入「未购买」，也绝不静默剔除。
3. **数值由程序计算。**
   每个 `CheckResult.detail` 里的数字都来自 SQL 聚合或 `scipy` 计算，
   不写死、不推测、不由 LLM 生成。

SRM 的两条解释边界（协议 §11.3，报告中必须原样体现）
----------------------------------------------------
- **SRM 通过 ≠ 随机化正确。** 它只说明各组的样本比例与**登记的设计比例**一致，
  无法检验随机化在其他维度上是否失衡，也无法排除系统性偏差。
- **SRM 异常 ≠ 已知原因。** 本模块只报告偏离（观测/预期计数、卡方统计量、
  自由度、p 值），**不得**自动断定是分流 bug、日志丢失还是数据导出问题；
  异常只表示「需要诊断」，可阻断因果结论，但不能替代诊断。

预期比例来源（协议 §11.1）
--------------------------
SRM 的预期比例**只从 `experiment_arms.expected_ratio` 读取**（其权威性由
`experiments.expected_ratio_source` 声明）。**严禁由观测计数反推预期比例**——
那样做会让 SRM 检查永远通过，失去意义。本模块在 `experiments.expected_ratio_source`
为空或实验未登记预期比例时返回 `passed=False` 并说明「无法检查：缺少预期比例」，
**不退化为按观测比例计算**。

异常处理原则
------------
检查函数内部**不吞异常**：数据库不可用、SQL 执行失败时异常直接向上传播。
把「连不上库」伪装成「检查未通过」会掩盖根因（对应 `E_DB_001` 与 `E_DB_002`
在报告层的区分）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from scipy import stats
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

# ============================================================================
#  模块常量：阈值与检查清单集中在此，报告中必须可见，不得散落在文本里
# ============================================================================

#: `E_GROUP_001` 的每组最小样本量阈值
MIN_ARM_SIZE: int = 1000

#: `E_COVER_001`：覆盖率低于此值为 blocking；`[该值, 1.0)` 为 warning
COVERAGE_BLOCKING_THRESHOLD: float = 0.99

#: `E_SRM_001` 的检验水平（协议 §11.2：SRM 用比常规更严格的阈值）
SRM_ALPHA: float = 0.001

#: `E_MISS_001` 检查的 customers 关键字段。
#: ⚠️ 下面的 SQL 由本常量**生成**，保证「查询的字段」与「detail 里说明的字段」不会脱节。
KEY_CUSTOMER_FIELDS: tuple[str, ...] = (
    "customer_key",
    "source_row_number",
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
)

#: 检查项的中文名（与 docs/05 §3.3 的异常码字典一致）
CHECK_NAMES: dict[str, str] = {
    "E_MISS_001": "客户表关键字段缺失",
    "E_COVER_001": "结果字段覆盖率",
    "E_ID_001": "内部客户 ID 唯一",
    "E_ID_002": "源行号一对一且连续",
    "E_JOIN_001": "关联后行数不膨胀",
    "E_GROUP_001": "每组样本量充足",
    "E_SRM_001": "样本分配比例失衡（SRM 卡方拟合优度检验）",
    "E_GROUP_002": "每个比较的两个组都有数据",
}

#: 通过时保留的登记级别（未通过时取本次结果的实际级别，见各检查函数说明）
CHECK_SEVERITY: dict[str, str] = {
    "E_MISS_001": "blocking",
    "E_COVER_001": "blocking",
    "E_ID_001": "blocking",
    "E_ID_002": "blocking",
    "E_JOIN_001": "blocking",
    "E_GROUP_001": "warning",
    "E_SRM_001": "blocking",
    "E_GROUP_002": "blocking",
}

#: `run_quality_checks` 的返回顺序（固定，便于报告与回归测试逐项比对）
CHECK_ORDER: tuple[str, ...] = (
    "E_MISS_001",
    "E_COVER_001",
    "E_ID_001",
    "E_ID_002",
    "E_JOIN_001",
    "E_GROUP_001",
    "E_SRM_001",
    "E_GROUP_002",
)


# ============================================================================
#  结果结构
# ============================================================================


@dataclass(frozen=True, slots=True)
class CheckResult:
    """一项数据质量检查的结果。

    Attributes:
        code: 稳定异常码（如 `E_SRM_001`），可被文档、日志、评测集与报告共同引用。
        name: 中文名。
        severity: 严重级别 `"blocking"` | `"warning"` | `"info"`。
            **未通过时**是本异常的**实际**级别（`E_COVER_001` 在 `[0.99, 1.0)` 时降为
            `warning`）；**通过时**保留该检查的登记级别，表示「这是阻断级检查项，本次已通过」。
            下游判定阻断应使用 `not passed and severity == "blocking"`。
        passed: `True` 表示检查通过（无异常）。
        detail: 人可读说明，**含实际数值**（计数、比率、统计量等）。
        value: 关键数值（p 值、覆盖率、比值等），无则 `None`。
    """

    code: str
    name: str
    severity: str
    passed: bool
    detail: str
    value: float | None = None


# ============================================================================
#  纯函数：SRM（不访问数据库，便于直接测试）
# ============================================================================


def _validate_srm_inputs(
    observed: dict[str, int], expected_ratio: dict[str, float]
) -> None:
    """校验 SRM 输入，不合法即抛 `ValueError`（不静默降级）。

    这里的严格性是刻意的：少一组、比例缺失、比例非正都会让 SRM 失去意义，
    静默跳过或退化成「按观测比例算」会让检查永远通过。
    """
    if not observed:
        raise ValueError("SRM 输入为空：没有观测计数")
    if len(observed) < 2:
        raise ValueError(f"SRM 至少需要 2 个组，收到 {len(observed)} 组")
    if set(observed) != set(expected_ratio):
        missing = sorted(set(observed) - set(expected_ratio))
        extra = sorted(set(expected_ratio) - set(observed))
        raise ValueError(f"SRM 预期比例与观测组不匹配：缺少 {missing}，多出 {extra}")
    total = sum(observed.values())
    if total <= 0:
        raise ValueError("SRM 观测总数为 0，无法检验")
    for key, ratio in expected_ratio.items():
        if ratio is None or not math.isfinite(float(ratio)) or float(ratio) <= 0:
            raise ValueError(f"SRM 预期比例非法：组 {key} 的比例为 {ratio!r}")


def srm_expected_counts(
    observed: dict[str, int], expected_ratio: dict[str, float]
) -> dict[str, float]:
    """按**登记比例**计算各组预期计数：`expected = ratio / sum(ratio) * total`。

    **为什么必须归一化**：`experiment_arms.expected_ratio` 是 `numeric(9,8)`，
    1/3 存为 `0.33333333`，三组之和是 `0.99999999` 而不是 1。若不归一化，
    预期计数之和会与观测总数不等，`scipy.stats.chisquare` 会直接拒绝执行。
    归一化只做缩放，**不改变比例关系**，因此不会把登记比例变成观测比例。

    Args:
        observed: `{组标识: 观测计数}`，必须覆盖全部注册组（含 0 样本的组）。
        expected_ratio: `{组标识: 登记的设计比例}`，来自 `experiment_arms.expected_ratio`。

    Returns:
        与 `observed` 同键、同顺序的预期计数字典。

    Raises:
        ValueError: 组不匹配、组数 < 2、观测总数为 0 或比例非正。
    """
    _validate_srm_inputs(observed, expected_ratio)
    total = float(sum(observed.values()))
    ratio_total = sum(float(expected_ratio[k]) for k in observed)
    return {k: float(expected_ratio[k]) / ratio_total * total for k in observed}


def srm_test(
    observed: dict[str, int], expected_ratio: dict[str, float]
) -> tuple[float, float, int]:
    """SRM 卡方拟合优度检验。**纯计算，不访问数据库。**

    原假设：观测计数符合登记的设计比例（`docs/02-分析协议.md` §11.2）。

    Args:
        observed: `{组标识: 观测计数}`——各组**全部被随机分配**的客户数，
            不得按干预后字段（`visited` / `converted` / `spend`）过滤。
        expected_ratio: `{组标识: 登记比例}`，来自 `experiment_arms.expected_ratio`。
            **不得由观测计数反推。**

    Returns:
        `(卡方统计量, p 值, 自由度)`；自由度为组数 - 1。

    Raises:
        ValueError: 输入不合法（见 `srm_expected_counts`）。

    Note:
        边界（协议 §11.3）：p 值大**不能**证明随机化正确；p 值小**不能**断定原因。
    """
    expected = srm_expected_counts(observed, expected_ratio)
    keys = list(observed)
    result = stats.chisquare(
        f_obs=[float(observed[k]) for k in keys],
        f_exp=[expected[k] for k in keys],
    )
    return float(result.statistic), float(result.pvalue), len(keys) - 1


def srm_check(
    observed: dict[str, int],
    expected_ratio: dict[str, float] | None,
    *,
    ratio_source: str | None = None,
) -> CheckResult:
    """SRM 检查的纯函数部分：登记比例进，`CheckResult` 出。**不访问数据库。**

    缺少预期比例时返回 `passed=False` 且 detail 说明「无法检查：缺少预期比例」，
    **绝不**退化为按观测比例计算。

    Args:
        observed: `{组标识: 观测计数}`。
        expected_ratio: `{组标识: 登记比例}`；为 `None` 或空表示未登记。
        ratio_source: `experiments.expected_ratio_source`，写进 detail 便于追溯。
    """
    code = "E_SRM_001"
    name = CHECK_NAMES[code]
    severity = CHECK_SEVERITY[code]
    source_text = f"预期比例来源 = {ratio_source}" if ratio_source else "预期比例来源未登记"

    if not expected_ratio:
        return CheckResult(
            code=code,
            name=name,
            severity=severity,
            passed=False,
            detail=f"无法检查：缺少预期比例（{source_text}）；不得按观测比例反推预期比例（协议 §11.1）",
            value=None,
        )

    try:
        expected = srm_expected_counts(observed, expected_ratio)
    except ValueError as exc:
        return CheckResult(
            code=code,
            name=name,
            severity=severity,
            passed=False,
            detail=f"无法检查：缺少预期比例或输入不合法（{exc}）；{source_text}",
            value=None,
        )

    chi2, p_value, df = srm_test(observed, expected_ratio)
    passed = p_value >= SRM_ALPHA

    detail = (
        f"观测计数 {_fmt_counts(observed)}；"
        f"预期计数 {_fmt_counts(expected, digits=2)}；"
        f"卡方统计量 {chi2:.6f}；自由度 {df}；p 值 {p_value:.6f}；"
        f"检验水平 α = {SRM_ALPHA}；"
        f"登记比例 {_fmt_counts(expected_ratio, digits=8)}；{source_text}"
    )
    if ratio_source == "derived":
        detail += "；⚠️ 来源为 derived，存在「由观测反推」的风险，需人工确认其权威性"
    if passed:
        detail += "；SRM 通过只说明样本比例与登记比例一致，不能证明随机化正确"
    else:
        detail += "；SRM 异常只提示需要诊断，不得自动断定原因"

    return CheckResult(
        code=code,
        name=name,
        severity=severity,
        passed=passed,
        detail=detail,
        value=p_value,
    )


# ============================================================================
#  纯函数：结果覆盖率
# ============================================================================


def coverage_check(n_expected: int, n_present: int) -> CheckResult:
    """结果覆盖率检查：`覆盖率 = 有结果的客户数 / 应有个数`。

    **分母口径（铁律 1）**：`n_expected` 是「全部被随机分配到该实验的客户数」，
    即 `assignments` 的行数，**不是** `outcomes` 的行数，更不是访问者/购买者数。

    **缺失口径（铁律 2）**：`n_present < n_expected` 时，差额按「结果缺失」单独计数并
    写进 detail；缺失**绝不**等同于「未购买」。当前源数据实测无缺失，
    但本函数的缺失分支是面向派生表与后续数据版本的正式逻辑，不是死代码。

    分级（见 `COVERAGE_BLOCKING_THRESHOLD`）：
    - `覆盖率 == 1.0` → 通过，severity 保留登记的 `blocking`；
    - `[0.99, 1.0)` → 未通过，severity = `warning`；
    - `< 0.99` → 未通过，severity = `blocking`。

    Args:
        n_expected: 应有个数（= 该实验被随机分配的客户数）。
        n_present: 有结果的客户数。

    Raises:
        ValueError: 计数为负，或 `n_present > n_expected`（有结果数不可能超过应有个数）。
    """
    if n_expected < 0 or n_present < 0:
        raise ValueError(f"计数不得为负：n_expected={n_expected}, n_present={n_present}")
    if n_present > n_expected:
        raise ValueError(
            f"有结果数 {n_present} 超过应有个数 {n_expected}，说明分母口径或关联有问题"
        )

    code = "E_COVER_001"
    name = CHECK_NAMES[code]

    if n_expected == 0:
        return CheckResult(
            code=code,
            name=name,
            severity=CHECK_SEVERITY[code],
            passed=False,
            detail="应有个数为 0（该实验没有被随机分配的客户），无法计算覆盖率",
            value=None,
        )

    n_missing = n_expected - n_present
    coverage = n_present / n_expected

    if n_missing == 0:
        passed, severity = True, "blocking"
    elif coverage < COVERAGE_BLOCKING_THRESHOLD:
        passed, severity = False, "blocking"
    else:
        passed, severity = False, "warning"

    detail = (
        f"应有个数 {n_expected}，有结果个数 {n_present}，缺失个数 {n_missing}，"
        f"覆盖率 {coverage:.6f}；"
        f"缺失单独计数，绝不并入「未购买」（协议 §9）；"
        f"阈值：覆盖率 < {COVERAGE_BLOCKING_THRESHOLD} 为 blocking，"
        f"[{COVERAGE_BLOCKING_THRESHOLD}, 1.0) 为 warning"
    )
    return CheckResult(
        code=code, name=name, severity=severity, passed=passed, detail=detail, value=coverage
    )


# ============================================================================
#  SQL（只读；分母一律取自 assignments，绝不按干预后字段过滤）
# ============================================================================

_SQL_MISSING_KEY_FIELDS = f"""
select
    count(*)                                                  as n_total,
    count(*) filter (where {' or '.join(f'cu.{c} is null' for c in KEY_CUSTOMER_FIELDS)})
                                                              as n_missing
from customers cu
join datasets d on d.dataset_id = cu.dataset_id
where d.version_label = :dataset_version
"""

# 覆盖率：分母 = assignments 行数（全部被随机分配的客户），LEFT JOIN 保留缺失
_SQL_OUTCOME_COVERAGE = """
select
    count(*)                    as n_expected,
    count(o.outcome_id)         as n_present
from assignments a
join experiments e on e.experiment_id = a.experiment_id
left join outcomes o on o.experiment_id = a.experiment_id
                    and o.customer_id   = a.customer_id
where e.experiment_key = :experiment_key
"""

_SQL_IDENTITY = """
select
    count(*)                             as n_rows,
    count(distinct cu.customer_id)       as n_ids,
    count(distinct cu.source_row_number) as n_source_rows,
    count(distinct cu.customer_key)      as n_keys,
    min(cu.source_row_number)            as min_row,
    max(cu.source_row_number)            as max_row
from customers cu
join datasets d on d.dataset_id = cu.dataset_id
where d.version_label = :dataset_version
"""

# 关联膨胀：分母是 assignments 行数，分子是同一实验在视图中的行数
_SQL_JOIN_RATIO = """
select
    (select count(*)
       from assignments a
       join experiments e on e.experiment_id = a.experiment_id
      where e.experiment_key = :experiment_key)      as n_assigned,
    (select count(*)
       from v_experiment_customer
      where experiment_key = :experiment_key)        as n_view
"""

# 各组观测样本量：以 experiment_arms 为基准 LEFT JOIN，**空组也会出现（计 0）**。
# 空组若被 INNER JOIN 丢掉，SRM 的观测计数就不完整，检验会失真。
_SQL_ARM_OBSERVED = """
select
    ar.arm_id,
    ar.arm_key,
    ar.expected_ratio,
    count(a.assignment_id) as n_observed
from experiment_arms ar
join experiments e on e.experiment_id = ar.experiment_id
left join assignments a on a.arm_id = ar.arm_id
                      and a.experiment_id = ar.experiment_id
where e.experiment_key = :experiment_key
group by ar.arm_id, ar.arm_key, ar.expected_ratio
order by ar.arm_id
"""

_SQL_SRM_CONFIG = """
select expected_ratio_source, expected_ratio_note
from experiments
where experiment_key = :experiment_key
"""

_SQL_COMPARISONS = """
select
    ec.comparison_key,
    ec.kind,
    tr.arm_key as treatment_arm_key,
    ct.arm_key as control_arm_key
from experiment_comparisons ec
join experiments     e  on e.experiment_id  = ec.experiment_id
join experiment_arms tr on tr.arm_id        = ec.treatment_arm_id
join experiment_arms ct on ct.arm_id        = ec.control_arm_id
where e.experiment_key = :experiment_key
order by ec.comparison_id
"""


# ============================================================================
#  内部辅助
# ============================================================================


@dataclass(frozen=True, slots=True)
class _ArmObservation:
    """一个注册组的观测情况（内部结构，不属于模块公共契约）。"""

    arm_id: int
    arm_key: str
    expected_ratio: float | None
    n_observed: int


def _fmt_counts(values: dict[str, Any], digits: int = 0) -> str:
    """把 `{组: 数值}` 渲染成 `mens=21307, womens=21387` 形式（detail 用）。"""
    return ", ".join(f"{k}={float(v):.{digits}f}" for k, v in values.items())


def _fetch_arm_observations(conn: Connection, experiment_key: str) -> list[_ArmObservation]:
    """读取该实验的**全部注册组**及其观测样本量（含 0 样本的组），按 arm_id 排序。"""
    rows = conn.execute(
        text(_SQL_ARM_OBSERVED), {"experiment_key": experiment_key}
    ).mappings().all()
    return [
        _ArmObservation(
            arm_id=int(r["arm_id"]),
            arm_key=str(r["arm_key"]),
            expected_ratio=(
                float(r["expected_ratio"]) if r["expected_ratio"] is not None else None
            ),
            n_observed=int(r["n_observed"]),
        )
        for r in rows
    ]


def _fetch_coverage_counts(conn: Connection, experiment_key: str) -> tuple[int, int]:
    """返回 `(应有个数, 有结果个数)`。应有个数 = 该实验被随机分配的客户数。"""
    row = conn.execute(
        text(_SQL_OUTCOME_COVERAGE), {"experiment_key": experiment_key}
    ).mappings().one()
    return int(row["n_expected"]), int(row["n_present"])


# ============================================================================
#  各项检查（每项一个 CheckResult；数据库异常向上传播，不吞）
# ============================================================================


def _check_missing_key_fields(conn: Connection, dataset_version: str) -> CheckResult:
    """`E_MISS_001` 客户表关键字段缺失（blocking，通过条件：无 NULL）。"""
    row = conn.execute(
        text(_SQL_MISSING_KEY_FIELDS), {"dataset_version": dataset_version}
    ).mappings().one()
    n_total, n_missing = int(row["n_total"]), int(row["n_missing"])

    if n_total == 0:
        return CheckResult(
            code="E_MISS_001",
            name=CHECK_NAMES["E_MISS_001"],
            severity=CHECK_SEVERITY["E_MISS_001"],
            passed=False,
            detail=(
                f"数据版本 {dataset_version} 在 customers 中没有登记任何客户，"
                f"无法检查关键字段缺失"
            ),
            value=None,
        )

    missing_rate = n_missing / n_total
    return CheckResult(
        code="E_MISS_001",
        name=CHECK_NAMES["E_MISS_001"],
        severity=CHECK_SEVERITY["E_MISS_001"],
        passed=n_missing == 0,
        detail=(
            f"数据版本 {dataset_version}：客户行数 {n_total}，关键字段含 NULL 的行数 "
            f"{n_missing}，缺失率 {missing_rate:.6f}；"
            f"检查字段 {', '.join(KEY_CUSTOMER_FIELDS)}"
        ),
        value=missing_rate,
    )


def _check_outcome_coverage(conn: Connection, experiment_key: str) -> CheckResult:
    """`E_COVER_001` 结果覆盖率（blocking；0.99~1.0 为 warning）。"""
    n_expected, n_present = _fetch_coverage_counts(conn, experiment_key)
    return coverage_check(n_expected, n_present)


def _check_customer_id_unique(conn: Connection, dataset_version: str) -> CheckResult:
    """`E_ID_001` 内部客户 ID 唯一（blocking；通过条件：count(*) = count(distinct id)）。"""
    row = conn.execute(
        text(_SQL_IDENTITY), {"dataset_version": dataset_version}
    ).mappings().one()
    n_rows = int(row["n_rows"])
    n_ids = int(row["n_ids"])

    if n_rows == 0:
        return CheckResult(
            code="E_ID_001",
            name=CHECK_NAMES["E_ID_001"],
            severity=CHECK_SEVERITY["E_ID_001"],
            passed=False,
            detail=f"数据版本 {dataset_version} 在 customers 中没有登记任何客户，无法检查 ID 唯一性",
            value=None,
        )

    duplicates = n_rows - n_ids
    return CheckResult(
        code="E_ID_001",
        name=CHECK_NAMES["E_ID_001"],
        severity=CHECK_SEVERITY["E_ID_001"],
        passed=duplicates == 0,
        detail=(
            f"数据版本 {dataset_version}：行数 {n_rows}，distinct customer_id {n_ids}，"
            f"重复客户数 {duplicates}；"
            f"distinct 源行号 {int(row['n_source_rows'])}，"
            f"distinct customer_key {int(row['n_keys'])}"
        ),
        value=float(duplicates),
    )


def _check_source_row_mapping(conn: Connection, dataset_version: str) -> CheckResult:
    """`E_ID_002` 源行号一对一且连续（blocking；行号唯一、min=1、max=count）。"""
    row = conn.execute(
        text(_SQL_IDENTITY), {"dataset_version": dataset_version}
    ).mappings().one()
    n_rows = int(row["n_rows"])
    n_source_rows = int(row["n_source_rows"])
    min_row = int(row["min_row"]) if row["min_row"] is not None else None
    max_row = int(row["max_row"]) if row["max_row"] is not None else None

    if n_rows == 0 or min_row is None or max_row is None:
        return CheckResult(
            code="E_ID_002",
            name=CHECK_NAMES["E_ID_002"],
            severity=CHECK_SEVERITY["E_ID_002"],
            passed=False,
            detail=f"数据版本 {dataset_version} 在 customers 中没有登记任何客户，无法检查源行号映射",
            value=None,
        )

    # 区间内部缺号数：源行号应恰好覆盖 1..n，任何缺口都说明导入不完整
    gaps = (max_row - min_row + 1) - n_source_rows
    passed = n_source_rows == n_rows and min_row == 1 and max_row == n_rows
    return CheckResult(
        code="E_ID_002",
        name=CHECK_NAMES["E_ID_002"],
        severity=CHECK_SEVERITY["E_ID_002"],
        passed=passed,
        detail=(
            f"数据版本 {dataset_version}：行数 {n_rows}，distinct 源行号 {n_source_rows}，"
            f"min {min_row}，max {max_row}，区间内缺口数 {gaps}"
        ),
        value=float(gaps),
    )


def _check_join_no_inflation(conn: Connection, experiment_key: str) -> CheckResult:
    """`E_JOIN_001` 关联后行数不膨胀（blocking；视图行数 = 分组行数，比值应为 1.0）。"""
    row = conn.execute(
        text(_SQL_JOIN_RATIO), {"experiment_key": experiment_key}
    ).mappings().one()
    n_assigned = int(row["n_assigned"])
    n_view = int(row["n_view"])

    if n_assigned == 0:
        return CheckResult(
            code="E_JOIN_001",
            name=CHECK_NAMES["E_JOIN_001"],
            severity=CHECK_SEVERITY["E_JOIN_001"],
            passed=False,
            detail=f"实验 {experiment_key} 在 assignments 中没有分组记录，无法检查关联是否膨胀",
            value=None,
        )

    ratio = n_view / n_assigned
    if n_view > n_assigned:
        verdict = "发现行数膨胀（一对多关联会放大分母）"
    elif n_view < n_assigned:
        verdict = "视图行数少于分组行数：关联丢失了行（例如缺失结果被 INNER JOIN 丢掉）"
    else:
        verdict = "未发现行数膨胀"
    return CheckResult(
        code="E_JOIN_001",
        name=CHECK_NAMES["E_JOIN_001"],
        severity=CHECK_SEVERITY["E_JOIN_001"],
        passed=n_view == n_assigned,
        detail=(
            f"实验 {experiment_key}：分组行数 {n_assigned}，视图行数 {n_view}，"
            f"比值 {ratio:.6f}；{verdict}"
        ),
        value=ratio,
    )


def _check_arm_size(conn: Connection, experiment_key: str) -> CheckResult:
    """`E_GROUP_001` 每组样本量充足（warning；每组 >= `MIN_ARM_SIZE`）。"""
    arms = _fetch_arm_observations(conn, experiment_key)
    if not arms:
        return CheckResult(
            code="E_GROUP_001",
            name=CHECK_NAMES["E_GROUP_001"],
            severity=CHECK_SEVERITY["E_GROUP_001"],
            passed=False,
            detail=f"实验 {experiment_key} 未登记任何实验组（experiment_arms 为空），无法检查样本量",
            value=None,
        )

    counts = {a.arm_key: a.n_observed for a in arms}
    min_n = min(counts.values())
    return CheckResult(
        code="E_GROUP_001",
        name=CHECK_NAMES["E_GROUP_001"],
        severity=CHECK_SEVERITY["E_GROUP_001"],
        passed=min_n >= MIN_ARM_SIZE,
        detail=(
            f"实验 {experiment_key}：各组分配客户数 {_fmt_counts(counts)}；"
            f"阈值 >= {MIN_ARM_SIZE}；最小值 {min_n}"
        ),
        value=float(min_n),
    )


def _check_srm(conn: Connection, experiment_key: str) -> CheckResult:
    """`E_SRM_001` SRM 卡方拟合优度检验（blocking；p >= `SRM_ALPHA`）。

    预期比例只来自 `experiment_arms.expected_ratio`；`experiments.expected_ratio_source`
    为空或实验未登记预期比例时返回 `passed=False` 并说明「无法检查：缺少预期比例」。
    """
    config = conn.execute(
        text(_SQL_SRM_CONFIG), {"experiment_key": experiment_key}
    ).mappings().first()
    if config is None:
        return CheckResult(
            code="E_SRM_001",
            name=CHECK_NAMES["E_SRM_001"],
            severity=CHECK_SEVERITY["E_SRM_001"],
            passed=False,
            detail=(
                f"无法检查：缺少预期比例（实验 {experiment_key} 未在 experiments 中登记）"
            ),
            value=None,
        )

    source = (config["expected_ratio_source"] or "").strip()
    if not source:
        return CheckResult(
            code="E_SRM_001",
            name=CHECK_NAMES["E_SRM_001"],
            severity=CHECK_SEVERITY["E_SRM_001"],
            passed=False,
            detail=(
                f"无法检查：缺少预期比例（实验 {experiment_key} 的 "
                f"expected_ratio_source 为空）；不得按观测比例反推预期比例（协议 §11.1）"
            ),
            value=None,
        )

    arms = _fetch_arm_observations(conn, experiment_key)
    if not arms:
        return CheckResult(
            code="E_SRM_001",
            name=CHECK_NAMES["E_SRM_001"],
            severity=CHECK_SEVERITY["E_SRM_001"],
            passed=False,
            detail=(
                f"无法检查：缺少预期比例（实验 {experiment_key} 未登记任何组，"
                f"experiment_arms 为空）"
            ),
            value=None,
        )

    observed = {a.arm_key: a.n_observed for a in arms}
    ratios = {
        a.arm_key: (float(a.expected_ratio) if a.expected_ratio is not None else float("nan"))
        for a in arms
    }
    return srm_check(observed, ratios, ratio_source=source)


def _check_comparison_arms(conn: Connection, experiment_key: str) -> CheckResult:
    """`E_GROUP_002` 每个比较的两个组都有数据（blocking；两组样本量均 > 0）。"""
    comparisons = conn.execute(
        text(_SQL_COMPARISONS), {"experiment_key": experiment_key}
    ).mappings().all()
    arms = {a.arm_key: a.n_observed for a in _fetch_arm_observations(conn, experiment_key)}

    if not comparisons:
        return CheckResult(
            code="E_GROUP_002",
            name=CHECK_NAMES["E_GROUP_002"],
            severity=CHECK_SEVERITY["E_GROUP_002"],
            passed=False,
            detail=(
                f"实验 {experiment_key} 未登记任何预注册比较（experiment_comparisons 为空），"
                f"无法检查比较双方是否都有数据"
            ),
            value=None,
        )

    parts: list[str] = []
    min_n: int | None = None
    passed = True
    for c in comparisons:
        t_key, c_key = str(c["treatment_arm_key"]), str(c["control_arm_key"])
        # 视图/分组里没有该组时按 0 计——空组必须如实暴露，不能被跳过
        t_n, c_n = arms.get(t_key, 0), arms.get(c_key, 0)
        if t_n <= 0 or c_n <= 0:
            passed = False
        for n in (t_n, c_n):
            min_n = n if min_n is None else min(min_n, n)
        parts.append(
            f"{c['comparison_key']}（{c['kind']}）：treatment={t_key} {t_n}，control={c_key} {c_n}"
        )

    return CheckResult(
        code="E_GROUP_002",
        name=CHECK_NAMES["E_GROUP_002"],
        severity=CHECK_SEVERITY["E_GROUP_002"],
        passed=passed,
        detail=(
            f"实验 {experiment_key}：" + "；".join(parts) + "；要求每个比较的两组分配客户数均 > 0"
        ),
        value=float(min_n) if min_n is not None else None,
    )


# ============================================================================
#  入口
# ============================================================================


def run_quality_checks(
    engine: Engine,
    experiment_key: str = "hillstrom_email_v1",
    dataset_version: str = "hillstrom@2008-03-20",
) -> list[CheckResult]:
    """运行全部数据质量检查，返回结果列表（顺序固定为 `CHECK_ORDER`）。

    Args:
        engine: SQLAlchemy Engine，**必须是只读引擎**（`dsa.config.get_engine(readonly=True)`）。
        experiment_key: 实验标识，用于覆盖率、关联、分组与 SRM 检查。
        dataset_version: 数据版本标识，用于关键字段缺失与客户标识检查。

    Returns:
        8 个 `CheckResult`，顺序为
        `E_MISS_001, E_COVER_001, E_ID_001, E_ID_002, E_JOIN_001, E_GROUP_001,
        E_SRM_001, E_GROUP_002`。
        下游判定「是否阻断因果结论」应使用 `not passed and severity == "blocking"`。

    Raises:
        sqlalchemy.exc.SQLAlchemyError: 数据库不可用或查询出错时**直接抛出**。
            本函数不吞异常：把「连不上库」记成「检查未通过」会掩盖根因，
            连接类错误（`E_DB_001`）与检查不通过是完全不同的处置路径。

    Note:
        SRM 检查的两条边界（协议 §11.3）必须随结果一并传达：

        - **SRM 通过不能证明随机化正确**——它只说明样本比例与登记的设计比例一致，
          无法检验随机化在其他维度上是否失衡，也无法排除系统性偏差；
        - **SRM 异常不得自动断定原因**——偏离可能是分流 bug、日志丢失、数据导出
          问题等，程序只负责报偏离，原因诊断需要人工介入。

        本函数不修改任何数据，也不改变分母口径：所有分母都取自 `assignments`
        （全部被随机分配的客户），绝不按 `visited` / `converted` / `spend` 过滤。
    """
    with engine.connect() as conn:
        results = [
            _check_missing_key_fields(conn, dataset_version),
            _check_outcome_coverage(conn, experiment_key),
            _check_customer_id_unique(conn, dataset_version),
            _check_source_row_mapping(conn, dataset_version),
            _check_join_no_inflation(conn, experiment_key),
            _check_arm_size(conn, experiment_key),
            _check_srm(conn, experiment_key),
            _check_comparison_arms(conn, experiment_key),
        ]

    codes = tuple(r.code for r in results)
    if codes != CHECK_ORDER:  # pragma: no cover - 结构性防线
        raise AssertionError(f"检查顺序与 CHECK_ORDER 不一致：{codes}")
    return results
