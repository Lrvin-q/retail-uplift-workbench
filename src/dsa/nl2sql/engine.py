"""受控 NL2SQL 引擎：生成 → 校验 → 执行 → **结果核对** → 有限修复。

对应《实现蓝图.md》阶段八任务 B：

| 蓝图任务 | 本模块的实现 |
|---|---|
| 3 设计结构化查询需求，再生成候选 SQL | :func:`build_query_spec` / :func:`route_question` → :func:`generate_sql` |
| 7 按执行错误有限修复；**不以"成功执行"替代结果正确性** | :func:`answer_via_sql` 的修复循环 + :func:`compare_with_reference` |
| 8 使用能区分错误查询的测试数据，与参考答案比较 | :func:`compare_with_reference` |
| 9 不明确或超范围 → 澄清或能力限制；可回退参数化工具 | ``CAPABILITY`` / ``PRECONDITION`` + :func:`_delegate` |

## 为什么需要这个模块

本项目最容易犯、后果最严重的错误是**分母口径**（AGENTS.md §2.2）：
购买率的分母必须是**全部被随机分配到该组的客户**，绝不能按 ``visit`` /
``conversion`` 这类**干预后**行为过滤。一条自由生成的 SQL 很容易违反它，
而且**不会报错** —— 结果看起来完全正常，只是错了。

于是本模块的设计要点是：

1. **执行成功 ≠ 结果正确** —— 必须与参数化工具的参考答案核对
   （:func:`compare_with_reference`）；核不上就判失败，绝不返回 ``ok=True``。
2. **危险语句用 AST 判定**（:func:`dsa.nl2sql.guards.parse_single_select`），
   不靠"文本是否以 SELECT 开头"。
3. **语义问题在执行前拦截**（:func:`dsa.nl2sql.guards.check_semantics`）。
4. **模板优先、模型兜底**：``is_mock()`` 为真时**只走确定性模板**，
   绝不假装模型给了 SQL —— mock 的职责是让链路跑通，不是让结果看起来合理
   （docs/09-Agent协议.md §7）。
5. **拿不准就回退**到已验证的参数化工具（:func:`_delegate`），而不是硬猜 SQL。

## 关于"未核对"

`AGENTS.md` §2.7 要求「已验证 / 失败 / 未验证」必须分开说。本模块的做法：

- 有参考答案且逐项一致 → ``reference_check['matches'] is True``（已验证）；
- 有参考答案但不一致，或形状对不上导致核不上 → ``False``（**判为失败**，
  并用 ``checks['comparable']`` 区分"核对失败"与"无法核对"）；
- **压根没有参考答案**（例如 RetailHero 描述性汇总）→ ``None``，
  并在 ``summary`` 与 ``result_check_note`` 里显式标注**未验证**。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Sequence

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from dsa.config import get_engine
from dsa.nl2sql import guards as G
from dsa.nl2sql import scope as S
from dsa.tools.registry import TOOLS, call_tool
from dsa.tools.schemas import (
    MAX_INLINE_ROWS,
    ToolErrorCode,
    ToolResult,
    error_result,
)

__all__ = [
    "MAX_REPAIR",
    "MAX_ROWS",
    "RTOL",
    "TIMEOUT_MS",
    "TOOL_NAME",
    "ExecOutcome",
    "QuerySpec",
    "Route",
    "ValidationReport",
    "Verdict",
    "answer_via_sql",
    "build_query_spec",
    "compare_with_reference",
    "execute_sql",
    "find_template",
    "generate_sql",
    "route_question",
    "validate",
    "vet_sql",
]

#: 工具名（写进 ``ToolResult.tool``，便于运行记录与报告追溯）。
TOOL_NAME = "answer_via_sql"

#: 修复轮次上限。蓝图阶段八任务 7 明确要求**有限**修复。
MAX_REPAIR: int = 2

#: 单次查询最多返回多少行（防止一条 SQL 把上下文打爆）。
MAX_ROWS: int = G.MAX_ROWS_DEFAULT

#: 语句超时（毫秒），与 docs/08 §4.1 一致。
TIMEOUT_MS: int = G.STATEMENT_TIMEOUT_MS_DEFAULT

#: 结果比对的相对容忍度。**预先声明，不得事后放宽**（AGENTS.md §2.6）。
#: 两条路径用同样的浮点运算，所以容差可以取得很紧。
RTOL: float = 1e-9
_ATOL: float = 1e-9

#: 参与比对的可比字段（越多越好：字段少的时候"恰好相等"太容易发生）。
_COMPARABLE_FIELDS: tuple[str, ...] = (
    "n_customers",
    "n_converted",
    "n_visited",
    "n_outcome_missing",
    "metric_value",
    "conversion_rate",
    "visit_rate",
    "total_spend",
    "spend_per_customer",
)

_DEFAULT_EXPERIMENT = "hillstrom_email_v1"
_DEFAULT_DATASET = "hillstrom"


# ============================================================================
#  结构化查询需求（蓝图任务 3）
# ============================================================================


@dataclass(frozen=True, slots=True)
class QuerySpec:
    """结构化查询需求。**先有它，再生成 SQL**（蓝图任务 3）。

    「先结构化、再生成」而不是「直接让模型写 SQL」，理由是：
    结构化需求**可被校验**（指标在白名单里吗？比较对象存在吗？），
    而一段自由 SQL 只能靠事后解析去猜它想干什么。
    """

    metric: str
    comparison: str | None = None
    experiment_key: str = _DEFAULT_EXPERIMENT
    dataset: str = _DEFAULT_DATASET
    tables: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        parts = [f"指标={self.metric}"]
        if self.comparison:
            parts.append(f"比较={self.comparison}")
        parts.append(f"实验={self.experiment_key}")
        return "；".join(parts)


#: 问题 → 指标的识别模式（取值必须在 docs/metrics.yaml 的 metrics 段里）。
_METRIC_PATTERNS: tuple[tuple[str, str], ...] = (
    ("购买者人均消费", "spend_per_buyer"),
    ("购买率", "conversion_rate"),
    ("转化率", "conversion_rate"),
    ("访问率", "visit_rate"),
    ("人均消费", "spend_per_customer"),
    ("客单价", "spend_per_customer"),
)

#: 问题 → 预注册比较的识别模式（取值必须在 docs/metrics.yaml 的 comparisons 里）。
_COMPARISON_PATTERNS: tuple[tuple[str, str], ...] = (
    ("男装", "primary"),
    ("女装", "secondary_womens"),
    ("无邮件", "primary"),
    ("对照组", "primary"),
)

#: 该数据集上可用的实验（与 tools.analysis_tools.KNOWN_EXPERIMENTS 一致）。
_KNOWN_EXPERIMENTS: tuple[str, ...] = (_DEFAULT_EXPERIMENT,)


def build_query_spec(question: str) -> QuerySpec | None:
    """把自然语言问题转成结构化需求；**无法确定时返回 ``None``**（由调用方澄清）。

    **不猜测**：缺指标或缺比较对象时返回 ``None``，而不是填一个默认值 ——
    默认值会让系统"看起来回答了"，而实际答的是另一个问题。
    （"哪一组更好"这类问题本来就必须指明比较对象；而"各组购买率是多少"
    属于描述性汇总，不走本函数，由 :func:`route_question` 的计数模板处理。）
    """
    q = question or ""
    metric = next((v for k, v in _METRIC_PATTERNS if k in q), None)
    if metric is None:
        return None
    comparison = next((v for k, v in _COMPARISON_PATTERNS if k in q), None)
    if comparison is None:
        return None
    experiment = next((key for key in _KNOWN_EXPERIMENTS if key in q), _DEFAULT_EXPERIMENT)
    return QuerySpec(
        metric=metric,
        comparison=comparison,
        experiment_key=experiment,
        dataset=_DEFAULT_DATASET,
        tables=("assignments", "experiment_arms", "outcomes", "experiments"),
    )


#: 指标 → 聚合表达式。**分子里放干预后条件，分母恒为 ``count(*)``**（AGENTS.md §2.2）。
_METRIC_EXPR: dict[str, str] = {
    "conversion_rate": "count(*) filter (where o.converted)::numeric / count(*)",
    "visit_rate": "count(*) filter (where o.visited)::numeric / count(*)",
    "spend_per_customer": "coalesce(sum(o.spend), 0) / count(*)",
    # 条件于干预后结果，仅作描述（metrics.yaml: descriptive_only）
    "spend_per_buyer": "coalesce(sum(o.spend), 0) / nullif(count(*) filter (where o.converted), 0)",
}


def _arm_from(experiment_key: str, select_list: str) -> str:
    """按组聚合的 SQL 骨架。

    两处细节决定分母口径是否正确：

    - ``left join outcomes``：**没产生结果的客户仍留在分母里**；
    - ``where`` 里**只有**实验定位条件，干预后条件一律放进 ``filter (where ...)``。
    """
    return (
        "select ar.arm_key as arm_key,\n"
        f"       {select_list}\n"
        "from assignments a\n"
        "join experiment_arms ar on ar.arm_id = a.arm_id\n"
        "left join outcomes o\n"
        "       on o.customer_id = a.customer_id\n"
        "      and o.experiment_id = a.experiment_id\n"
        "where a.experiment_id = (\n"
        "    select experiment_id from experiments\n"
        f"    where experiment_key = '{experiment_key}'\n"
        ")\n"
        "group by ar.arm_key\n"
        "order by ar.arm_key"
    )


def generate_sql(spec: QuerySpec) -> str:
    """由结构化需求生成候选 SQL。

    **只从白名单取值拼接**：``spec.metric`` 来自 :data:`_METRIC_PATTERNS`
    （取值均在 docs/metrics.yaml 登记），``spec.experiment_key`` 来自
    :data:`_KNOWN_EXPERIMENTS`，因此生成结果的结构是**可预测**的，
    不会出现"模型编了一个字段名"。

    ⚠️ **列名与参考答案对齐**（``arm_key`` + 指标列名），否则
    :func:`compare_with_reference` 会因「两侧没有可比较的同名列」而判为**失败** ——
    这是**有意的**：核不上就不能算通过。
    """
    if spec.metric not in _METRIC_EXPR:
        raise ValueError(f"未登记的指标：{spec.metric!r}")
    if spec.experiment_key not in _KNOWN_EXPERIMENTS:
        raise ValueError(f"未登记的实验：{spec.experiment_key!r}")
    select_list = f"count(*) as n_customers,\n       {_METRIC_EXPR[spec.metric]} as {spec.metric}"
    return _arm_from(spec.experiment_key, select_list)


def _count_sql(experiment_key: str, columns: Sequence[str]) -> str:
    """按组计数的 SQL（模板路径用；不算比率，比率由核对层按同一口径补算）。"""
    projections = {
        "n_customers": "count(*) as n_customers",
        "n_converted": "count(*) filter (where o.converted) as n_converted",
        "n_visited": "count(*) filter (where o.visited) as n_visited",
        "n_outcome_missing": "count(*) filter (where o.outcome_id is null) as n_outcome_missing",
        "total_spend": "coalesce(sum(o.spend), 0) as total_spend",
    }
    unknown = [c for c in columns if c not in projections]
    if unknown:
        raise ValueError(f"未登记的聚合列：{unknown}")
    return _arm_from(experiment_key, ",\n       ".join(projections[c] for c in columns))


# ============================================================================
#  路由
# ============================================================================


@dataclass(frozen=True, slots=True)
class Route:
    """问题 → 处理路径的判定结果。

    ``kind``：

    - ``sql``：有可用模板，直接给候选 SQL（确定性）；
    - ``tool``：该问题由参数化工具回答更可靠（蓝图任务 9 的回退路径）；
    - ``unsupported``：超出能力范围 → ``E_CAP_001``；
    - ``clarify``：信息不足 → ``E_PLAN_001``。
    """

    kind: str
    name: str = ""
    sql: str = ""
    tool: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    reference_tool: str = ""
    reference_params: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    detail: str = ""


_EXPERIMENT_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("hillstrom_email_v1", "hillstrom_email", "hillstrom", "邮件实验"), _DEFAULT_EXPERIMENT),
)

#: 明确不支持的问题类型 → 为什么（依据 docs/metrics.yaml 的 ``unsupported_metrics``）。
_UNSUPPORTED_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("净收益", "净利润", "毛利", "利润率"),
     "无成本与毛利数据（metrics.yaml: net_revenue / roi 明确不提供）"),
    (("roi", "投资回报"),
     "无成本与毛利数据（metrics.yaml: roi 明确不提供）"),
    (("终身价值", "ltv"),
     "仅有两周窗口数据（metrics.yaml: customer_ltv 明确不提供）"),
    (("留存率", "留存"),
     "窗口过短且为单次快照（metrics.yaml: retention_rate 明确不提供）"),
    (("归因", "触点", "多触点"),
     "无触点链路数据（metrics.yaml: multi_touch_attribution 明确不提供）"),
    (("按天", "每日", "日趋势", "时间序列", "走势"),
     "Hillstrom 无日期字段与事件时间线（metrics.yaml: daily_conversion_trend 明确不提供）"),
    (("个体增量", "每个人的增量", "单个客户的增量"),
     "每名客户只观测一个干预结果，个体增量标签原理上不可得"),
    (("预测", "未来一个月", "明天", "下个月"),
     "本模块只做受控的**回顾性查询**，不做预测；预测请走 dsa.models 的建模流程"),
)

#: 「这个数据集能做什么」类问题 → 走 describe_capability 参数化工具。
_CAPABILITY_PATTERNS: tuple[str, ...] = (
    "支持哪些", "支持什么", "能做什么", "可以做哪些", "有什么能力", "能力范围", "有哪些分析",
)

_ARM_PATTERNS: tuple[str, ...] = (
    "各组", "每个组", "组别", "分组", "购买率", "访问率", "人均", "样本量",
    "多少客户", "人数", "分配", "对照组", "干预组", "arm",
)


def _find_experiment(question: str) -> str | None:
    """从问题里认出实验标识；认不出返回 ``None``（由调用方决定澄清）。"""
    low = question.lower()
    for aliases, key in _EXPERIMENT_ALIASES:
        for alias in aliases:
            if alias.lower() in low:
                return key
    return None


def route_question(question: str) -> Route:
    """把自然语言问题判定成路由。

    **确定性、不依赖 LLM**：这是「计算模块先脱离 LLM 验证」（AGENTS.md §3.2）
    在本模块的落点 —— 规则路由可以离线断言，模型只用于规则覆盖不到的长尾。
    """
    q = (question or "").strip()
    low = q.lower()
    if not q:
        return Route(
            kind="clarify", message="问题为空",
            detail="请描述要分析什么，例如「hillstrom 实验各组的购买率是多少」",
        )

    # ---- 1. 明确不支持（先判，省得让模型白跑一趟）----
    for patterns, reason in _UNSUPPORTED_PATTERNS:
        for pat in patterns:
            if pat in low:
                return Route(kind="unsupported",
                             message=f"该问题超出能力范围：{reason}", detail=pat)

    # ---- 2. 能力询问 → 参数化工具 ----
    if any(pat in q for pat in _CAPABILITY_PATTERNS):
        dataset = "retailhero" if ("retailhero" in low or "rh_" in low) else "hillstrom"
        return Route(kind="tool", name="describe_capability", tool="describe_capability",
                     params={"dataset_key": dataset}, message="这类问题由参数化工具回答")

    experiment = _find_experiment(q)

    # ---- 3. 实验配置 / 指标定义 → 参数化工具 ----
    if any(pat in q for pat in ("实验配置", "实验设计", "预期比例", "结果窗口", "组别定义", "随机化")):
        if experiment is None:
            return Route(kind="clarify", message="未指明是哪个实验",
                         detail=f"当前已登记实验：{_DEFAULT_EXPERIMENT}；请指明后再查配置")
        return Route(kind="tool", name="get_experiment_config", tool="get_experiment_config",
                     params={"experiment_key": experiment}, message="实验配置由参数化工具回答")
    if "指标定义" in q or "指标字典" in q or ("口径" in q and "定义" in q):
        metric_id = next((v for k, v in _METRIC_PATTERNS if k in q), "conversion_rate")
        return Route(kind="tool", name="get_metric_definition", tool="get_metric_definition",
                     params={"metric_id": metric_id}, message="指标定义由参数化工具回答")

    # ---- 4. 分组聚合类问题 → 确定性模板 SQL，并用参数化工具做参考答案 ----
    if any(pat in q for pat in _ARM_PATTERNS):
        if experiment is None:
            return Route(
                kind="clarify", message="未指明是哪个实验",
                detail=(f"请指明实验。当前已登记实验：{_DEFAULT_EXPERIMENT}"
                        "（例如「hillstrom 实验各组的购买率是多少」）"),
            )
        spec = build_query_spec(q)
        if spec is not None and spec.metric in _METRIC_EXPR:
            if spec.metric == "spend_per_buyer":
                return Route(
                    kind="unsupported",
                    message=("购买者人均消费（spend_per_buyer）条件于干预后结果，"
                             "metrics.yaml 标为 descriptive_only，禁止用于组间比较；"
                             "请改问「人均消费」（分母为全部被分配客户）"),
                    detail="spend_per_buyer",
                )
            if spec.comparison:  # 明确指向某个比较 → 生成 SQL + 用参数化工具核对
                return Route(
                    kind="sql", name=f"metric:{spec.metric}", sql=generate_sql(spec),
                    reference_tool="query_arm_aggregates",
                    reference_params={"experiment_key": spec.experiment_key},
                    message=f"指标 {spec.metric}（实验 {spec.experiment_key}）",
                )
        columns = ["n_customers"]
        if any(pat in q for pat in ("购买率", "转化率", "购买", "conversion")):
            columns.append("n_converted")
        if any(pat in q for pat in ("访问率", "访问", "visit")):
            columns.append("n_visited")
        if any(pat in q for pat in ("人均", "消费", "金额", "spend")):
            columns.append("total_spend")
        if len(columns) == 1 and not any(pat in q for pat in ("样本量", "人数", "多少客户", "分组大小")):
            columns.append("n_converted")  # 泛指"各组情况"时，至少给一个可核对的指标
        return Route(
            kind="sql", name="arm_counts", sql=_count_sql(experiment, columns),
            reference_tool="query_arm_aggregates",
            reference_params={"experiment_key": experiment},
            message=f"按组聚合（实验 {experiment}）",
        )

    # ---- 5. RetailHero 描述性汇总（无参数化工具对照 → 结果标注为未核对）----
    if any(pat in q for pat in ("交易", "购买明细", "消费总额", "总消费", "retailhero")):
        return Route(
            kind="sql", name="retailhero_transaction_totals",
            sql=("select count(*) as n_transactions,\n"
                 "       count(distinct client_id) as n_customers,\n"
                 "       sum(purchase_sum) as total_spend,\n"
                 "       sum(n_items) as total_items\n"
                 "from rh_transactions"),
            message="RetailHero 交易级汇总（已按 transaction_id 收敛）",
        )

    # ---- 6. 认不出意图 → 请用户澄清 ----
    return Route(
        kind="clarify", message="无法确定要分析什么",
        detail=("请补充实验与指标，例如：\n"
                "  · 「hillstrom 实验各组的购买率是多少」\n"
                "  · 「各组样本量与访问率」\n"
                "  · 「这个数据集支持哪些分析」"),
    )


def find_template(spec: QuerySpec) -> str | None:
    """有合适模板时返回**参数化工具名**（蓝图任务 9 的回退目标）。

    ⚠️ 这里的模板是**工具**，不是 SQL 字符串：工具的参数受控、SQL 写死，
    风险等级低于任何自由生成的 SQL。
    """
    if spec.dataset == _DEFAULT_DATASET:
        return "query_arm_aggregates"
    return None


# ============================================================================
#  执行
# ============================================================================


@dataclass(frozen=True, slots=True)
class ExecOutcome:
    """一次 SQL 执行的结果（成功或失败都用它表达，不抛异常）。"""

    ok: bool
    sql: str
    columns: tuple[str, ...] = ()
    rows: tuple[dict, ...] = ()
    code: ToolErrorCode = ToolErrorCode.UNKNOWN
    message: str = ""
    detail: str = ""
    duration_ms: int = 0
    repairable: bool = False

    @property
    def row_count(self) -> int:
        return len(self.rows)


def _classify_db_error(exc: BaseException) -> tuple[ToolErrorCode, bool]:
    """把数据库异常映射到错误码，并给出「是否值得重生成 SQL」。

    只有**语法/对象名类**错误值得让模型改一次（``42601`` 语法错误、
    ``42703`` 列不存在、``42P01`` 表不存在、``42883`` 函数不存在…）；
    权限、连接、超时类重试同样的语句没有意义。
    """
    sqlstate = getattr(getattr(exc, "orig", None), "sqlstate", None)
    if sqlstate == "57014":  # query_canceled / statement_timeout
        return ToolErrorCode.TIMEOUT, False
    if sqlstate in ("42501", "28P01", "28000", "3D000"):  # 权限 / 认证 / 库不存在
        return ToolErrorCode.DB_ERROR, False
    if sqlstate in ("42601", "42703", "42P01", "42883", "42804", "22P02"):
        return ToolErrorCode.DB_ERROR, True
    if isinstance(exc, SQLAlchemyError):
        return ToolErrorCode.DB_ERROR, False
    return ToolErrorCode.UNKNOWN, False


def execute_sql(
    sql: str,
    engine: Any,
    *,
    max_rows: int = MAX_ROWS,
    timeout_ms: int = TIMEOUT_MS,
) -> ExecOutcome:
    """在**只读**引擎上执行 SQL，带 ``LIMIT`` 与事务级 ``statement_timeout``。

    ``statement_timeout`` 必须用 ``SET LOCAL`` 在**同一个连接的同一个事务**里设置，
    写进 SQL 文本不生效 —— 它是 PostgreSQL 的事务级参数。
    """
    try:
        limited = G.enforce_limits(sql, max_rows=max_rows, timeout_ms=timeout_ms)
    except ValueError as exc:
        return ExecOutcome(ok=False, sql=sql, code=ToolErrorCode.INVALID_PARAM,
                           message=str(exc), repairable=True)

    started = time.perf_counter()
    try:
        with engine.begin() as conn:
            conn.execute(text(f"set local statement_timeout = {int(timeout_ms)}"))
            result = conn.execute(text(limited))
            columns = tuple(result.keys())
            rows = tuple(dict(r) for r in result.mappings().all())
        return ExecOutcome(
            ok=True, sql=limited, columns=columns, rows=rows,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:  # noqa: BLE001 —— 必须分类后返回，不向上抛
        code, repairable = _classify_db_error(exc)
        sqlstate = getattr(getattr(exc, "orig", None), "sqlstate", None) or ""
        return ExecOutcome(
            ok=False, sql=limited, code=code,
            message=f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}",
            detail=f"sqlstate={sqlstate}", repairable=repairable,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def _execute(sql: str, engine: Any) -> list[dict]:
    """执行并返回行列表；失败抛 ``RuntimeError``。

    供测试与诊断直接调用（``answer_via_sql`` 走 :func:`execute_sql`，不抛异常）。
    """
    outcome = execute_sql(sql, engine)
    if not outcome.ok:
        raise RuntimeError(f"{outcome.code.value}: {outcome.message} {outcome.detail}")
    return [dict(r) for r in outcome.rows]


# ============================================================================
#  校验链
# ============================================================================


@dataclass(frozen=True, slots=True)
class Verdict:
    """一条候选 SQL 的校验结论。"""

    ok: bool
    stage: str = ""
    code: ToolErrorCode = ToolErrorCode.UNKNOWN
    reason: str = ""
    issues: tuple[dict, ...] = ()
    warnings: tuple[dict, ...] = ()

    @property
    def blocking_issues(self) -> tuple[dict, ...]:
        return tuple(G.semantic_blockers(self.issues))


def _verdict_code(stage: str) -> ToolErrorCode:
    """校验失败 → 错误码。

    - 越界（表/视图白名单、系统表）→ ``CAPABILITY``：是"系统没有这个能力"；
    - 危险语句 / 解析失败 → ``CAPABILITY``：受控模块只支持合规的只读查询；
    - 分母红线 → ``PRECONDITION``：是业务口径不成立，不是参数写错了。
    """
    if stage == "semantics-denominator":
        return ToolErrorCode.PRECONDITION
    return ToolErrorCode.CAPABILITY


def vet_sql(sql: str) -> Verdict:
    """四重校验：范围 → 解析（AST）→ 危险语句 → 语义。**任一不过即不得执行。**"""
    tables = None
    try:
        tables = G.referenced_tables(sql)
    except G.SqlParseError as exc:
        return Verdict(False, "parse", ToolErrorCode.CAPABILITY, f"SQL 无法解析：{exc}")

    ok, why = S.in_scope(sql, tables=tables)
    if not ok:
        return Verdict(False, "scope", ToolErrorCode.CAPABILITY, why)

    ok, why = G.parse_single_select(sql)
    if not ok:
        return Verdict(False, "parse", ToolErrorCode.CAPABILITY, why)

    ok, why = G.check_dangerous(sql)
    if not ok:
        return Verdict(False, "dangerous", ToolErrorCode.CAPABILITY, why)

    issues = G.check_semantics(sql)
    blocking = G.semantic_blockers(issues)
    warnings = tuple(i for i in issues if i not in blocking)
    if blocking:
        codes = {i["code"] for i in blocking}
        stage = "semantics"
        if G.CheckCode.DENOMINATOR_POST_TREATMENT in codes:
            stage = "semantics-denominator"
        elif G.CheckCode.AMOUNT_NOT_DEDUPED in codes:
            stage = "semantics-amount"
        elif codes & {G.CheckCode.DATASET_MIXED, G.CheckCode.CROSS_DATASET_JOIN}:
            stage = "semantics-dataset"
        reason = "；".join(f"[{i['code']}] {i['message']}" for i in blocking)
        return Verdict(False, stage, _verdict_code(stage), reason, tuple(issues), warnings)

    return Verdict(True, "", ToolErrorCode.UNKNOWN, "", tuple(issues), warnings)


@dataclass
class ValidationReport:
    """校验链的完整结果。**每一步都记录**，便于解释"为什么拒绝了"。"""

    stages: list[tuple[str, bool, str]] = field(default_factory=list)
    semantic_issues: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(ok for _name, ok, _msg in self.stages)

    @property
    def first_failure(self) -> tuple[str, str] | None:
        for name, ok, msg in self.stages:
            if not ok:
                return name, msg
        return None

    def summary(self) -> str:
        return "；".join(f"{n}={'通过' if ok else '拒绝'}" for n, ok, _m in self.stages)


def validate(sql: str) -> ValidationReport:
    """完整校验链：范围 → 单条 SELECT（AST）→ 危险语句 → 语义。

    **顺序有意如此**：先范围（最便宜，且最能说明"这不是我能答的"），
    再结构，再危险语句，最后语义（最贵，需要理解意图）。
    """
    rep = ValidationReport()

    ok, msg = S.in_scope(sql)
    rep.stages.append(("范围", ok, msg))
    if not ok:
        return rep

    ok, msg = G.parse_single_select(sql)
    rep.stages.append(("单条SELECT(AST)", ok, msg))
    if not ok:
        return rep

    ok, msg = G.check_dangerous(sql)
    rep.stages.append(("危险语句", ok, msg))
    if not ok:
        return rep

    issues = G.check_semantics(sql)
    rep.semantic_issues = issues
    blocking = G.semantic_blockers(issues)
    rep.stages.append((
        "语义", not blocking,
        "；".join(f"[{i['code']}] {i['message']}" for i in blocking),
    ))
    return rep


# ============================================================================
#  结果核对（蓝图任务 8；本模块的纪律落点）
# ============================================================================


def _as_rows(result: Any) -> list[dict] | None:
    """从「工具结果 / SQL 结果 / 裸列表」里取出行列表。"""
    if isinstance(result, ToolResult):
        result = result.data
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict)]
    if not isinstance(result, dict):
        return None
    rows = result.get("rows")
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    return None


def _normalize_arms(result: Any) -> list[dict] | None:
    """把三种形状归一成「按组」的列表。

    - 参数化工具 ``query_arm_aggregates`` → ``ToolResult.data['arms']``
      （已带 ``arm_key`` 与派生比率）；
    - 受控生成的 SQL → ``{'rows': [...]}``（每行一个组）；
    - **裸行列表** → 直接当行用（便于测试与手工核对）。
    """
    if isinstance(result, ToolResult):
        if not result.ok:
            return None
        result = result.data
    if isinstance(result, list):
        source = [r for r in result if isinstance(r, dict)]
    elif isinstance(result, dict):
        arms = result.get("arms")
        if isinstance(arms, list) and arms and isinstance(arms[0], dict):
            source = [r for r in arms if isinstance(r, dict)]
        else:
            rows = _as_rows(result)
            if not rows:
                return None
            source = rows
    else:
        return None

    if not source or "arm_key" not in source[0]:
        return None

    out: list[dict] = []
    for row in source:
        arm_key = row.get("arm_key")
        if arm_key is None:
            return None
        item = dict(row)
        item["arm_key"] = str(arm_key)
        # 缺派生比率时按同一口径补算，让"能比"的字段尽量多
        n = item.get("n_customers")
        if "conversion_rate" not in item and isinstance(n, (int, float)) and n and item.get("n_converted") is not None:
            item["conversion_rate"] = float(item["n_converted"]) / float(n)
        if "visit_rate" not in item and isinstance(n, (int, float)) and n and item.get("n_visited") is not None:
            item["visit_rate"] = float(item["n_visited"]) / float(n)
        if "spend_per_customer" not in item and isinstance(n, (int, float)) and n and item.get("total_spend") is not None:
            item["spend_per_customer"] = float(item["total_spend"]) / float(n)
        out.append(item)
    return out


def _numbers_match(a: Any, b: Any) -> bool:
    """数值比对。**非数值一律判不匹配** —— 宁可说"不一致"，也不含糊过去。

    ⚠️ **必须接受 `Decimal`**。psycopg 对 ``::numeric`` 表达式返回 `Decimal`，
    而 `Decimal` **不是** `int`/`float` 的实例 —— 早期版本写成
    ``isinstance(a, (int, float))`` 会直接返回 `False`，
    于是**所有 Decimal 与 float 的比对都是假失败**：

        SQL=Decimal('0.00572608654839012485')  工具=0.005726086548390125
        -> 报「不一致」，而 rel_diff 算出来是 0.0（两者是同一个数）

    实测这条让端到端核对报「3 处差异，最大相对差 0.00%」——
    **假失败**。它比"假通过"安全（不会放过错误结果），
    但会让正确的结果被判错，同样必须修。
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if not isinstance(a, (int, float, Decimal)) or \
       not isinstance(b, (int, float, Decimal)):
        return False
    diff = abs(float(a) - float(b))
    return diff <= _ATOL + RTOL * max(abs(float(a)), abs(float(b)))


def compare_with_reference(sql_result: Any, reference_result: Any) -> dict:
    """把 SQL 结果与**参考答案**（通常来自参数化工具）比较。

    蓝图任务 7 的关键要求：**「不以'成功执行'替代结果正确性」**。
    本函数存在的意义就是：**执行成功但结果不一致时，判为失败。**

    比较策略：按 ``arm_key`` 对齐，在**两侧同名的数值列**上逐个比较
    （相对容忍度 :data:`RTOL`）。组的集合不一致、或存在数值差异 → 不匹配。

    ⚠️ **「无法核对」不等于「核对通过」**。
    早期实现按列名遍历，两边列名不同时全部 ``continue`` ——
    **一个可比的列都没比，却返回 ``matches=True``**。那是「执行成功 ≠ 结果正确」
    的变体：把"没检查"当成了"检查通过"。现在这种情况一律返回 ``matches=False``，
    并在 ``detail`` 里写明是**无法核对**（``checks['comparable']`` 为 ``False``），
    让调用方能把"核对失败"与"核不上"分开表述。

    Args:
        sql_result: 受控生成 SQL 的结果：``{'columns': [...], 'rows': [...]}``、
            ``{'arms': [...]}``、:class:`ToolResult`，或**裸行列表**。
        reference_result: 参考答案（参数化工具的结果），形状同上。

    Returns:
        ``{"matches": bool | None, "detail": str, "diffs": [...], "compared": int,
        "checks": {...}}``

        - ``True``：共同字段逐个数值一致（**确实比过**，``compared`` > 0）；
        - ``False``：数值不一致 / 组集合不一致 / **无法核对** —— 调用方必须判为失败；
        - ``None``：参考答案本身为空（参数化工具没返回任何组）。
    """
    sql_arms = _normalize_arms(sql_result)
    ref_arms = _normalize_arms(reference_result)

    if sql_arms is None or ref_arms is None:
        return {
            "matches": False,
            "detail": ("无法核对：结果形状无法与参考答案对齐（缺少 arm_key 之类的分组键）"
                       "—— **核不上不等于核对通过**"),
            "diffs": [],
            "compared": 0,
            "checks": {"comparable": False},
        }
    if not ref_arms:
        return {
            "matches": None,
            "detail": "参考答案为空（参数化工具没返回任何组），本次结果**未核对**",
            "diffs": [],
            "compared": 0,
            "checks": {"comparable": False},
        }

    sql_by_arm = {a["arm_key"]: a for a in sql_arms}
    ref_by_arm = {a["arm_key"]: a for a in ref_arms}
    diffs: list[dict] = []
    compared_fields: set[str] = set()
    compared = 0

    if set(sql_by_arm) != set(ref_by_arm):
        diffs.append({
            "arm_key": "*", "field": "arm_set",
            "sql_value": sorted(sql_by_arm), "reference_value": sorted(ref_by_arm),
            "note": "组的集合不一致（多组、少组或键名不同）",
        })

    for arm_key in sorted(set(sql_by_arm) & set(ref_by_arm)):
        for fname in _COMPARABLE_FIELDS:
            s_val = sql_by_arm[arm_key].get(fname)
            r_val = ref_by_arm[arm_key].get(fname)
            if s_val is None or r_val is None:
                continue
            compared_fields.add(fname)
            compared += 1
            if not _numbers_match(s_val, r_val):
                try:
                    rel = abs(float(s_val) - float(r_val)) / max(abs(float(r_val)), 1e-12)
                except (TypeError, ValueError):
                    rel = None
                diffs.append({
                    "arm_key": arm_key, "field": fname,
                    "sql_value": s_val, "reference_value": r_val, "rel_diff": rel,
                })

    checks = {
        "comparable": True,
        "arm_sets_equal": set(sql_by_arm) == set(ref_by_arm),
        "fields_compared": sorted(compared_fields),
        "numbers_compared": compared,
        "arms_compared": len(set(sql_by_arm) & set(ref_by_arm)),
    }

    if compared == 0 and not diffs:
        # 形状对得上（都有 arm_key），但**一个数值都没比过** —— 这就是"无法核对"。
        # ``checks['comparable']`` 必须跟着变成 False，否则调用方会以为比过了。
        return {
            "matches": False,
            "detail": ("无法核对：两侧没有可比较的同名列"
                       f"（SQL 侧列={sorted({c for r in sql_arms for c in r})[:8]}，"
                       f"参考侧列={sorted({c for r in ref_arms for c in r})[:8]}）"
                       "—— **核不上不等于核对通过**"),
            "diffs": [],
            "compared": 0,
            "checks": {**checks, "comparable": False},
        }

    if diffs:
        worst = max((d for d in diffs if d.get("rel_diff") is not None),
                    key=lambda d: d["rel_diff"], default=None)
        head = "；".join(
            f"{d['arm_key']}.{d['field']}: SQL={d['sql_value']} vs 工具={d['reference_value']}"
            for d in diffs[:3]
        )
        return {
            "matches": False,
            "detail": (f"结果与参数化工具的参考答案不一致（{len(diffs)} 处差异）→ **判为失败**：{head}"
                       + (f"；最大相对差 {worst['rel_diff']:.2%}" if worst else "")),
            "diffs": diffs,
            "compared": compared,
            "checks": checks,
        }

    return {
        "matches": True,
        "detail": (f"与参数化工具参考答案一致（{checks['arms_compared']} 个组、"
                   f"{compared} 个数值；字段：{', '.join(sorted(compared_fields))}）"),
        "diffs": [],
        "compared": compared,
        "checks": checks,
    }


# ============================================================================
#  生成（模板 / 模型）
# ============================================================================


def _parse_sql_payload(content: str) -> dict | None:
    """从模型输出里抠出 JSON（容忍 ```json 围栏与前后废话）。"""
    raw = (content or "").strip()
    if not raw:
        return None
    fence = re.search(r"```(?:json|sql)?\s*(.+?)```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    try:
        loaded = json.loads(raw)
        return loaded if isinstance(loaded, dict) else None
    except (ValueError, TypeError):
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if 0 <= start < end:
        try:
            loaded = json.loads(raw[start : end + 1])
            return loaded if isinstance(loaded, dict) else None
        except (ValueError, TypeError):
            return None
    return None


def _llm_sql(question: str, *, repair_context: str = "") -> tuple[str | None, str]:
    """让模型给出候选 SQL。返回 ``(sql | None, 说明)``。

    **mock 模式不生成 SQL**：mock 的职责是让链路跑通，不是假装模型有能力
    （docs/09-Agent协议.md §7）。mock 时一律走规则模板。
    """
    from dsa.agent.llm import get_llm, is_mock

    if is_mock():
        return None, "mock 模式：不调用模型，改走规则模板"

    system = (
        S.schema_prompt()
        + "\n\n# 输出格式\n"
        "只输出一个 JSON 对象，不要解释、不要 Markdown 代码围栏：\n"
        '  {"sql": "<单条 SELECT>"}\n'
        '若问题无法用一条 SELECT 回答，输出：{"unsupported": "<原因>"}\n'
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    if repair_context:
        messages.append({"role": "user", "content": repair_context})

    try:
        reply = get_llm(temperature=0.0).invoke(messages)
    except Exception as exc:  # noqa: BLE001 —— 模型不可用不能让整个工具塌掉
        return None, f"模型调用失败：{type(exc).__name__}: {exc}"

    payload = _parse_sql_payload(getattr(reply, "content", "") or "")
    if isinstance(payload, dict) and payload.get("unsupported"):
        return None, f"模型判定无法用 SQL 回答：{payload['unsupported']}"
    if isinstance(payload, dict) and isinstance(payload.get("sql"), str) and payload["sql"].strip():
        return payload["sql"].strip(), "由模型生成"
    return None, "模型输出无法解析出 SQL"


def _repair_prompt(sql: str, failure: str, *, kind: str) -> str:
    """把失败原因写回给模型，构成一次**有针对性**的修复请求。"""
    return (
        f"上一次生成的 SQL 未被接受（{kind}）：\n{sql}\n\n"
        f"失败原因：\n{failure}\n\n"
        "请重新生成**一条** SELECT，并确保：分母不按干预后字段过滤、"
        "交易级金额先按 transaction_id 收敛、不跨数据集 JOIN、只引用白名单里的表。"
    )


# ============================================================================
#  主流程
# ============================================================================


def _available_capabilities() -> dict:
    """本模块可回答的能力清单（越界/澄清时随错误一起返回，让用户知道能问什么）。"""
    return {
        "supported": {
            "hillstrom": [
                "各组样本量 / 购买率 / 访问率 / 人均消费（query_arm_aggregates）",
                "实验配置：组别、结果窗口、预期分配比例（get_experiment_config）",
                "数据质量检查（run_quality_checks）",
                "指标定义（get_metric_definition）",
                "策略评估结果（get_policy_evaluations）",
                "客户名单（get_customer_scores）",
            ],
            "retailhero": [
                "交易级汇总（rh_transactions，已按 transaction_id 收敛）",
                "客户特征快照（rh_client_features）",
                "数据集能力说明（describe_capability）",
            ],
        },
        "not_supported": [
            "净收益 / ROI / 毛利（无成本数据）",
            "客户终身价值、留存率（窗口与快照限制）",
            "多触点归因（无触点链路）",
            "按日趋势（Hillstrom 无日期字段）",
            "个体增量标签（原理上不可得）",
            "任何写操作、DDL、系统表查询",
        ],
        "example_questions": [
            "hillstrom 实验各组的购买率是多少",
            "各组样本量与访问率",
            "这个数据集支持哪些分析",
            "购买率这个指标怎么定义",
        ],
        "parameterized_tools": sorted(TOOLS),
    }


def _capability_result(message: str, *, detail: str = "", duration_ms: int = 0,
                       data: dict | None = None) -> ToolResult:
    payload: dict = {
        "available_capabilities": _available_capabilities(),
        "hint": "可以改用上面的示例问法，或直接调用对应的参数化工具",
    }
    if data:
        payload.update(data)
    return error_result(TOOL_NAME, ToolErrorCode.CAPABILITY, message, detail,
                        data=payload, duration_ms=duration_ms)


def _precondition_result(message: str, *, detail: str = "", duration_ms: int = 0,
                         data: dict | None = None) -> ToolResult:
    payload: dict = {
        "needed": detail or "请补充缺失的业务条件",
        "available_capabilities": _available_capabilities(),
    }
    if data:
        payload.update(data)
    return error_result(TOOL_NAME, ToolErrorCode.PRECONDITION, message, detail,
                        data=payload, duration_ms=duration_ms)


def _delegate(tool: str, params: dict, engine: Any) -> ToolResult:
    """回退到参数化工具（蓝图任务 9）。

    返回的 ``ToolResult`` 用本模块的名字，同时把被路由到的工具与它的原始结果
    放进 ``data``，保证"这次答案其实是哪个工具给的"可追溯。
    """
    delegated = call_tool(tool, engine, **params)
    return ToolResult(
        ok=delegated.ok,
        tool=TOOL_NAME,
        data={
            "via": "parameterized_tool",
            "routed_tool": tool,
            "routed_params": dict(params),
            "routed_ok": delegated.ok,
            "routed": delegated.to_dict(),
            "available_capabilities": _available_capabilities(),
        },
        refs=list(delegated.refs),
        summary=(f"回退到参数化工具 {tool}："
                 + (delegated.summary or ("成功" if delegated.ok else delegated.error.message))),
        error=delegated.error,
        duration_ms=delegated.duration_ms,
        run_id=delegated.run_id,
    )


def _reference_block(route: Route, engine: Any) -> tuple[dict | None, str]:
    """取参考答案（来自参数化工具）。返回 ``(块 | None, 说明)``。"""
    if not route.reference_tool:
        return None, ("该问题没有对应的参数化工具，结果**未核对**"
                      "（AGENTS.md §2.7：未验证必须显式标注）")
    if route.reference_tool not in TOOLS:
        return None, f"参考答案工具未登记：{route.reference_tool}"
    res = call_tool(route.reference_tool, engine, **route.reference_params)
    if not res.ok:
        return None, f"参考答案工具 {route.reference_tool} 失败：{res.error.message}"
    return ({"tool": route.reference_tool, "params": dict(route.reference_params),
             "data": res.to_dict().get("data")}, "")


def _reference_for(spec: QuerySpec, engine: Any) -> list[dict] | None:
    """取 ``query_arm_aggregates`` 的参考答案（按组列表）；取不到返回 ``None``。"""
    res = call_tool("query_arm_aggregates", engine, experiment_key=spec.experiment_key)
    if not res.ok:
        return None
    arms = (res.data or {}).get("arms") or []
    return [dict(a) for a in arms] or None


def answer_via_sql(
    question: str,
    *,
    engine: Any = None,
    use_llm: bool = True,
    prefer_template: bool = False,
) -> ToolResult:
    """完整流程：范围判定 → 生成候选 SQL → 校验 → 执行 → 结果核对 → 有限修复。

    Args:
        question: 自然语言问题。
        engine: SQLAlchemy 引擎；缺省用 :func:`dsa.config.get_engine` 的**只读**引擎。
        use_llm: 是否允许调用模型生成/修复 SQL。``False`` 时只用确定性模板
            （测试与离线验证走这条路径）。
        prefer_template: ``True`` 时，凡有合适模板的问题**直接回退参数化工具**
            （蓝图任务 9 的字面读法）。缺省 ``False``：仍走受控生成 + 与工具
            参考答案核对 —— 那才是本模块存在的意义，也才暴露得出"执行成功但结果错误"。

    Returns:
        :class:`~dsa.tools.schemas.ToolResult`：

        - 成功：``ok=True``，``data['sql']`` 是**实际执行**的 SQL，
          ``data['reference_check']['matches']`` 说明是否与参数化工具核对一致；
        - 超范围：``E_CAP_001``，``data['available_capabilities']`` 说明可用能力；
        - 信息不足：``E_PLAN_001``，说明缺什么；
        - 数据库错误 / 超时：``E_DB_001`` / ``E_DB_003``。

    **结果正确性优先于"执行成功"**：执行成功但结果与参考答案不一致时返回
    ``ok=False``（``E_PLAN_001``），并在 ``data['reference_check']`` 里给出差异明细。
    """
    started = time.perf_counter()

    def elapsed() -> int:
        return int((time.perf_counter() - started) * 1000)

    if not isinstance(question, str) or not question.strip():
        return error_result(TOOL_NAME, ToolErrorCode.INVALID_PARAM,
                            "question 必须是非空字符串", f"收到 {question!r}")

    route = route_question(question)
    attempts: list[dict] = []

    # ---- 1. 超范围 ----
    if route.kind == "unsupported":
        return _capability_result(route.message, detail=route.detail,
                                  duration_ms=elapsed(), data={"stage": "route"})

    # ---- 2. 需要澄清 ----
    if route.kind == "clarify":
        return _precondition_result(route.message, detail=route.detail,
                                    duration_ms=elapsed(), data={"stage": "route"})

    if engine is None:
        engine = get_engine(readonly=True)

    # ---- 3. 参数化工具优先（这些问题工具答得比 SQL 更可靠）----
    if route.kind == "tool":
        return _delegate(route.tool, route.params, engine)

    # ---- 3b. 显式要求模板优先时，直接回退参数化工具（蓝图任务 9）----
    if prefer_template:
        spec = build_query_spec(question)
        tool = find_template(spec) if spec is not None else None
        if tool:
            return _delegate(tool, {"experiment_key": spec.experiment_key}, engine)

    # ---- 4. 候选 SQL：确定性模板优先，模型兜底 ----
    candidate = route.sql
    source = f"规则模板[{route.name}]" if candidate else ""
    llm_note = ""
    if not candidate:
        if use_llm:
            candidate, llm_note = _llm_sql(question)
            source = "模型生成" if candidate else ""
        else:
            llm_note = "use_llm=False：未调用模型"

    if not candidate:
        return _capability_result(
            "无法为该问题生成可核验的 SQL",
            detail=llm_note or "没有匹配的规则模板，且未获得模型输出",
            duration_ms=elapsed(), data={"stage": "generate"},
        )

    # ---- 5. 校验 → 执行 → 核对 → 有限修复 ----
    repair = 0
    while True:
        verdict = vet_sql(candidate)
        if not verdict.ok:
            attempts.append({"round": repair, "sql": candidate, "stage": verdict.stage,
                             "ok": False, "reason": verdict.reason})
            if repair >= MAX_REPAIR:
                return error_result(
                    TOOL_NAME, verdict.code,
                    f"生成的 SQL 未通过校验（已修复 {repair} 次仍未通过）：{verdict.reason}",
                    detail=f"阶段={verdict.stage}",
                    data={"question": question, "attempts": attempts,
                          "issues": list(verdict.issues),
                          "available_capabilities": _available_capabilities()},
                    duration_ms=elapsed(),
                )
            repair += 1
            nxt, note = (_llm_sql(question, repair_context=_repair_prompt(
                candidate, verdict.reason, kind=verdict.stage))
                if use_llm else (None, "use_llm=False：不调用模型修复"))
            if nxt:
                candidate, source = nxt, "模型生成（修复）"
                continue
            return error_result(
                TOOL_NAME, verdict.code,
                f"生成的 SQL 未通过校验，且无法修复：{verdict.reason}",
                detail=f"阶段={verdict.stage}；{note}",
                data={"question": question, "attempts": attempts,
                      "issues": list(verdict.issues),
                      "available_capabilities": _available_capabilities()},
                duration_ms=elapsed(),
            )

        outcome = execute_sql(candidate, engine)
        if not outcome.ok:
            attempts.append({"round": repair, "sql": outcome.sql, "stage": "execute",
                             "ok": False, "reason": outcome.message})
            if repair < MAX_REPAIR and outcome.repairable and use_llm:
                repair += 1
                nxt, llm_note = _llm_sql(question, repair_context=_repair_prompt(
                    outcome.sql, outcome.message, kind="执行报错"))
                if nxt:
                    candidate, source = nxt, "模型生成（修复）"
                    continue
            return error_result(
                TOOL_NAME, outcome.code,
                f"SQL 执行失败：{outcome.message}",
                detail=f"{outcome.detail}；已尝试 {repair} 次修复",
                data={"question": question, "sql": outcome.sql, "attempts": attempts,
                      "reference_check": {"matches": None, "detail": "未执行到核对阶段"}},
                duration_ms=elapsed(),
            )

        # ---- 结果核对：**执行成功 ≠ 结果正确** ----
        reference, ref_note = _reference_block(route, engine)
        check = (
            compare_with_reference(
                {"columns": list(outcome.columns), "rows": [dict(r) for r in outcome.rows]},
                reference["data"],
            )
            if reference
            else {"matches": None, "detail": ref_note, "diffs": [], "compared": 0,
                  "checks": {"comparable": False}}
        )

        if reference is not None and check["matches"] is not True:
            attempts.append({"round": repair, "sql": outcome.sql, "stage": "result_check",
                             "ok": False, "reason": check["detail"]})
            if repair < MAX_REPAIR and use_llm:
                repair += 1
                nxt, llm_note = _llm_sql(question, repair_context=_repair_prompt(
                    outcome.sql, check["detail"], kind="结果与参考答案不一致"))
                if nxt:
                    candidate, source = nxt, "模型生成（修复）"
                    continue
            # **不得以"成功执行"替代结果正确性**：核不上即失败
            return error_result(
                TOOL_NAME, ToolErrorCode.PRECONDITION,
                f"SQL 执行成功，但结果未通过核对，已判为失败：{check['detail']}",
                detail=(f"已尝试 {repair} 次修复；{llm_note}" if llm_note else f"已尝试 {repair} 次修复"),
                data={
                    "question": question, "generated_sql": candidate, "executed_sql": outcome.sql,
                    "source": source, "attempts": attempts, "reference_check": check,
                    "reference": {"tool": reference["tool"], "params": reference["params"]},
                    "rows": [dict(r) for r in outcome.rows[:MAX_INLINE_ROWS]],
                    "columns": list(outcome.columns),
                },
                duration_ms=elapsed(),
            )

        # ---- 成功 ----
        rows = [dict(r) for r in outcome.rows]
        truncated = len(rows) > MAX_INLINE_ROWS
        data: dict[str, Any] = {
            "question": question,
            "generated_sql": candidate,
            "sql": outcome.sql,
            "source": source,
            "columns": list(outcome.columns),
            "row_count": outcome.row_count,
            "rows": rows[:MAX_INLINE_ROWS],
            "row_truncated": truncated,
            "max_rows_limit": MAX_ROWS,
            "statement_timeout_ms": TIMEOUT_MS,
            "repairs": repair,
            "attempts": attempts,
            "reference_check": check,
            "verification": ("verified" if check["matches"] is True
                             else "unverified" if check["matches"] is None
                             else "mismatch"),
            "semantic_issues": list(verdict.issues),
            "semantic_warnings": [i for i in verdict.issues if i.get("severity") == "warning"],
            "result_check_note": (
                "已与参数化工具参考答案逐项核对" if check["matches"] is True
                else "未与参考答案核对（无对应参数化工具）—— 按 AGENTS.md §2.7 标注为未验证"
            ),
        }
        if reference:
            data["reference"] = {"tool": reference["tool"], "params": reference["params"]}
        if llm_note:
            data["llm_note"] = llm_note

        summary = (
            f"{outcome.row_count} 行；来源={source}；"
            + ("结果已与参数化工具核对一致" if check["matches"] is True
               else "结果**未核对**（见 result_check_note）")
        )
        return ToolResult(ok=True, tool=TOOL_NAME, data=data, summary=summary,
                          duration_ms=elapsed())
