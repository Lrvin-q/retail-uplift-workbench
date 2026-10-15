"""工具注册表：名称 → 函数 + 输入 schema。

供 Agent 与服务层发现和调用工具（docs/08-工具协议.md §4）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from dsa.tools.analysis_tools import (
    DEPRECATED_POLICY_KINDS,
    POLICY_ORDER_COLUMNS,
    TOOL_FUNCS,
)
from dsa.tools.schemas import ToolErrorCode, ToolResult, error_result


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    func: Callable[..., ToolResult]
    description: str
    input_schema: dict


def _s(**props: str) -> dict:
    """构造简易 input_schema。required 为全部参数。"""
    return {
        "type": "object",
        "properties": {k: {"type": v} for k, v in props.items()},
        "required": list(props),
        "additionalProperties": False,
    }


TOOLS: dict[str, ToolSpec] = {
    "get_experiment_config": ToolSpec(
        "get_experiment_config", TOOL_FUNCS["get_experiment_config"],
        "读取实验配置：组别、结果窗口、预期分配比例来源、预注册比较",
        _s(experiment_key="string"),
    ),
    "get_metric_definition": ToolSpec(
        "get_metric_definition", TOOL_FUNCS["get_metric_definition"],
        "读取指标定义（分子、分母、单位、约束）",
        _s(metric_id="string"),
    ),
    "query_arm_aggregates": ToolSpec(
        "query_arm_aggregates", TOOL_FUNCS["query_arm_aggregates"],
        "各组聚合：样本量、购买率、访问率、人均消费。分母为全部被随机分配的客户",
        _s(experiment_key="string"),
    ),
    "run_quality_checks": ToolSpec(
        "run_quality_checks", TOOL_FUNCS["run_quality_checks"],
        "运行数据质量检查并返回阻断判定",
        _s(experiment_key="string", dataset_version="string"),
    ),
    "describe_capability": ToolSpec(
        "describe_capability", TOOL_FUNCS["describe_capability"],
        "说明该数据适配支持与不支持的分析类型",
        _s(dataset_key="string"),
    ),
    "get_model_registry": ToolSpec(
        "get_model_registry", TOOL_FUNCS["get_model_registry"],
        "读取模型登记：特征列、基础学习器、参数、划分配置、代码版本",
        {"type": "object",
         "properties": {"dataset_version": {"type": ["string", "null"]}},
         "required": [], "additionalProperties": False},
    ),
    "get_policy_evaluations": ToolSpec(
        "get_policy_evaluations", TOOL_FUNCS["get_policy_evaluations"],
        "读取已验证的策略评估结果（各策略各触达比例的增量估计与区间）",
        _s(model_key="string"),
    ),
    "get_customer_scores": ToolSpec(
        "get_customer_scores", TOOL_FUNCS["get_customer_scores"],
        "按策略取客户名单前 N 名；大名单写产物并返回引用，不全部内联。"
        "排序依据取自 dsa.models.policy 的权威策略定义；"
        "已废弃的旧名会被显式拒绝（不静默接受）",
        {"type": "object",
         "properties": {
             "model_key": {"type": "string"},
             # ⚠️ 取值**不能写死在这里**：它们来自 `dsa.models.policy` 的
             # 权威定义（缺陷 5.2 要求"一处定义、两处派生"）。
             # 写死一份字面量就等于又制造了一个会漂移的副本。
             #
             # 已废弃的旧名**不列进 enum**：列出来就等于邀请调用方去用它。
             # 但工具函数仍会对它给出带解释的拒绝
             # （`DEPRECATED_POLICY_KINDS`）—— 用于绕过 schema 校验的内部调用，
             # 使旧名无论从哪条路进来都得到同一个明确答复。
             "policy_kind": {"type": "string",
                             "enum": sorted(POLICY_ORDER_COLUMNS)},
             "limit_n": {"type": "integer", "minimum": 1, "maximum": 200000},
             "run_id": {"type": ["string", "null"]},
         },
         "required": ["model_key"], "additionalProperties": False},
    ),
}

# ---------------------------------------------------------------------------
# 实验效果分析工具（F01 修复）。
#
# **为什么拆到 effect_tools 模块再合并**：效果工具与效果载荷、报告渲染
# 是一组（载荷在 `dsa.stats.effect`，渲染在 `dsa.report.effect_view`），
# 放在一起便于「同一份计算产物只被包装一次」这件事被看出来。
# 合并写在这里，是为保持 `TOOLS` 仍是**唯一**的工具清单 ——
# 若让调用方分别 import 两个字典，registry 就失去了"唯一真相"的地位。
# ---------------------------------------------------------------------------
from dsa.tools.effect_tools import (  # noqa: E402
    EFFECT_TOOLS,
    TOOL_ANALYZE_EFFECT,
    TOOL_SCORING_POPULATION,
)

TOOLS[TOOL_ANALYZE_EFFECT] = ToolSpec(
    TOOL_ANALYZE_EFFECT,
    EFFECT_TOOLS[TOOL_ANALYZE_EFFECT],
    "计算实验效果：干预组与对照组的绝对差异、置信区间、风险比与检验。"
    "按指定比较与指标返回；不可做因果比较的指标会被明确拒绝",
    {"type": "object",
     "properties": {
         "experiment_key": {"type": "string"},
         "comparison_key": {"type": "string"},
         "metrics": {"type": ["array", "null"], "items": {"type": "string"}},
         "n_boot": {"type": "integer", "minimum": 100, "maximum": 20000},
         "bootstrap_seed": {"type": "integer"},
         "run_id": {"type": ["string", "null"]},
     },
     "required": ["experiment_key", "comparison_key"],
     "additionalProperties": False},
)

TOOLS[TOOL_SCORING_POPULATION] = ToolSpec(
    TOOL_SCORING_POPULATION,
    EFFECT_TOOLS[TOOL_SCORING_POPULATION],
    "某模型可打分人群的规模与可用打分口径。按比例取名单时，"
    "比例乘的就是这里返回的人数（分母由程序给出，不由 LLM 估）",
    _s(model_key="string"),
)


def call_tool(name: str, engine: Any, *, run_id: str | None = None, **kwargs: Any) -> ToolResult:
    """调用工具。

    **未知工具名返回 ``INVALID_PARAM`` 的 ToolResult，不抛异常**——
    调用方（LLM 或服务层）应能通过返回值区分，而不是靠捕获异常。

    ``run_id`` 是**控制参数**：它默认不会转发给工具函数
    （大多数工具签名里没有它），只被挂到返回的 ``ToolResult.run_id`` 上，
    使服务层能把这次调用记入 ``query_executions`` 并关联到所属运行。

    ⚠️ **但声明了 ``run_id`` 形参的工具必须收到它**（F06 修复点）。
    写产物的工具（`get_customer_scores`、`analyze_experiment_effect`）
    需要 `run_id` 来决定产物落在哪个目录下。早期版本一律不转发，
    于是这些工具**永远拿不到运行标识**，产物全部落在 `adhoc/` 下 ——
    而导出的依据是 `result_refs`，于是"导出包包含实际产物"这条验收
    在结构上就不可能满足。（这也是为什么它们的签名里有 `run_id`、
    schema 里也列了它，却从来没有生效过。）

    转发与否由**函数签名**决定，而不是靠一张手工维护的名单 ——
    后者会随着新工具加入而静默失效。
    """
    spec = TOOLS.get(name)
    if spec is None:
        return error_result(
            str(name),
            ToolErrorCode.INVALID_PARAM,
            f"未知工具：{name!r}",
            detail=f"已登记工具：{sorted(TOOLS)}",
        )

    call_kwargs = dict(kwargs)
    if run_id is not None and _accepts_run_id(spec.func):
        call_kwargs["run_id"] = run_id

    result = spec.func(engine, **call_kwargs)

    if run_id is not None and result.run_id is None:
        result = replace(result, run_id=run_id)
    return result


def _accepts_run_id(func: Callable[..., ToolResult]) -> bool:
    """工具函数是否声明了 ``run_id`` 形参。"""
    import inspect

    try:
        return "run_id" in inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover —— 内建函数等
        return False


def list_tools() -> list[dict]:
    """返回工具清单（供 Agent / 服务层发现）。"""
    return [
        {"name": s.name, "description": s.description, "input_schema": s.input_schema}
        for s in TOOLS.values()
    ]
