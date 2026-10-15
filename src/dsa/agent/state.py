"""Agent 状态与结构化输出。

对应 docs/09-Agent协议.md §4。所有结构都必须能 JSON 序列化，
因为状态要写入检查点并在重启后恢复。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, Field

#: 每个工具节点的最大重试次数（docs/09 §5.2）
MAX_RETRY = 2
#: 模型格式错误的最大修复次数
MAX_REPAIR = 1


class RequestCategory(str, Enum):
    """四类请求（docs/04-问题集与验收样例.md §1）。"""

    EXECUTABLE = "executable"
    NEEDS_CLARIFICATION = "clarification"
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"


class AnalysisPlan(BaseModel):
    """结构化分析计划。**由 validate 节点做程序校验，不信任 LLM 输出。**

    字段清单来自一次外部审查的修复要求（F02）：计划必须能表达
    **数据集、实验、比较、任务类型、指标、结果窗口、筛选条件、
    排序依据、触达比例／人数、模型版本**。

    为什么每个"用户没明说"的字段都要带一个 `*_source`：

    用户说的和系统替他定的，是两件事。原实现的计划只有四个字段，
    缺的字段就在执行时**默默取默认值** —— 于是「女装」的请求
    拿到了「男装」的名单，而报告上一个字都没提这件事。
    把来源显式写进计划，报告才能把默认值如实地称作"默认"。
    """

    #: 数据集：``hillstrom`` / ``retailhero``
    dataset_key: str = "hillstrom"
    #: ``explicit``（用户点名）/ ``default``（未指定，按默认）
    dataset_source: str = "default"

    dataset_version: str | None = None
    requested_arms: list[str] = Field(default_factory=list)
    request_constraints: list[dict[str, Any]] = Field(default_factory=list)
    experiment_key: str = "hillstrom_email_v1"
    #: 预注册比较标识：``primary``（男装）/ ``secondary_womens``（女装）
    comparison_key: str = "primary"
    comparison_source: str = "default"

    #: 任务类型：``compare_effect``（问效果）/ ``select_customers``（要名单）
    task_type: str = "compare_effect"

    metrics: list[str] = Field(default_factory=list)
    metric_source: str = "default"

    #: 结果窗口（天）。本实验**固定 14 天**，不是可调参数。
    outcome_window_days: int = 14
    #: 请求里出现的筛选条件。**本版本不做任何筛选**，
    #: 保留该字段是为了让「有筛选要求」这件事可见而不是被丢掉。
    filters: list[dict[str, Any]] = Field(default_factory=list)

    #: 名单排序依据（仅 `select_customers`）：策略名，取值由
    #: `dsa.models.policy` 的权威定义决定（见 5.2 缺陷修复）。
    ranking: str | None = None
    ranking_source: str | None = None
    #: 触达人数（与 `limit_pct` 互斥，同时给出时以人数为准）
    limit_n: int | None = None
    #: 触达比例（0–1）。**不在计划里换算成人** ——
    #: 换算需要可打分人群的规模，那是数据事实，由工具返回后现算。
    limit_pct: float | None = None
    #: 用户原话里表示规模的片段，用于报告里回显"你要的是这个"
    limit_raw: str = ""

    #: 名单使用的模型。由 validate 从**登记表**解析，不由 LLM 指定。
    model_key: str | None = None
    model_source: str | None = None

    #: 计划阶段留下的说明（如"排序依据未指定，按默认"）
    notes: list[str] = Field(default_factory=list)
    question: str = ""


class ClarificationRequest(BaseModel):
    missing: list[str] = Field(default_factory=list)
    question: str = ""


class ValidationIssue(BaseModel):
    field: str
    code: str
    message: str


class AgentState(TypedDict, total=False):
    """LangGraph 状态。

    ``total=False`` 是刻意的：节点只返回自己产生的增量，
    LangGraph 负责合并。要求每个节点返回全量状态会让节点之间产生
    不必要的耦合，也会让检查点体积无谓增大。
    """

    question: str
    original_question: str
    request_constraints: list[dict[str, Any]]
    category: str
    plan: dict[str, Any] | None

    #: 缺失的条件项（给 clarify 用）
    missing: list[str]
    #: **已问过的问题**。用于避免重复提问，也是 HITL 恢复的依据。
    clarifications: list[dict[str, Any]]
    #: 用户对澄清的回答
    answers: list[dict[str, Any]]

    tool_results: list[dict[str, Any]]
    checks: list[dict[str, Any]]

    blocked: bool
    blocking_codes: list[str]

    report: str | None
    errors: list[dict[str, Any]]

    #: 各节点的尝试次数，**有限重试的唯一依据**
    attempts: dict[str, int]
    usage: list[dict[str, Any]]

    #: **最近一次 execute 的结果**（F05）。
    #:
    #: 为什么必须与 `errors` 分开：`errors` 是**审计日志**，要保留全部历史；
    #: 而路由需要的是**当前状态**。旧实现用 `errors` 同时承担两件事，
    #: 于是一次暂时失败会永久污染这次运行 ——
    #: 第 2、3 次都成功了，仍然被判定为失败。
    #: 「审计」与「状态」是两种信息，混用会让两者都失准。
    last_execute_ok: bool
    last_execute_retriable: bool
    last_execute_failures: list[dict[str, Any]]

    #: 名单规模换算依据（比例 → 人数）。写进报告使换算可核对。
    selection_basis: dict[str, Any] | None
    #: LLM 解释文字的守卫结论（F03）。
    explanation_guard: dict[str, Any] | None

    #: 校验问题（validate 产出）
    issues: list[dict[str, Any]]
    #: 追踪标识
    run_id: str | None
    thread_id: str | None


def initial_state(question: str, *, run_id: str | None = None,
                  thread_id: str | None = None) -> AgentState:
    """构造初始状态。字段齐全可避免下游到处写 ``state.get(...) or []``。"""
    return AgentState(
        question=question,
        original_question=question,
        category="",
        plan=None,
        missing=[],
        clarifications=[],
        answers=[],
        tool_results=[],
        checks=[],
        blocked=False,
        blocking_codes=[],
        report=None,
        errors=[],
        attempts={},
        usage=[],
        last_execute_ok=True,
        last_execute_retriable=False,
        last_execute_failures=[],
        selection_basis=None,
        explanation_guard=None,
        issues=[],
        run_id=run_id,
        thread_id=thread_id,
    )
