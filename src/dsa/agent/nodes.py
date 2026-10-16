"""Agent 各节点实现。

设计要点（docs/09-Agent协议.md §3）：
- ``classify`` / ``validate`` / ``execute`` / ``check`` / ``analyze`` / ``finalize``
  **不依赖 LLM 的判断能力**：分类用规则，校验用登记表。
  这是「关键异常不会因为 LLM 继续调用而被绕过」的技术保证。
- 当前计划由规则解析和已登记能力生成；最终报告由结构化工具结果和程序模板生成。
- LLM 接口与历史解释守卫保留兼容；自由模型解释不进入最终报告。
- **数值一律来自工具结果**，本模块不生成任何业务数字。

本模块在一次外部审查后的三处结构性改动
--------------------------------------
1. **执行节点消费计划**（F01/F02）。原实现固定调用
   `query_arm_aggregates` + `get_experiment_config`，用户问的指标、
   比较对象和名单规模**都到不了执行层**，于是"三个不同的问题拿到同一份报告"。
   现在执行节点从 `plan` 读字段并据此选工具与参数。
2. **报告消费结构化效果**（F01）。效果差、区间、检验来自
   `analyze_experiment_effect` 工具的结构化返回，报告只做排版。
3. **最终结论程序生成**（R01）。历史解释守卫保留用于旧记录；
   当前最终数字、方向和局限说明直接消费已验证结构化结果。
"""

from __future__ import annotations

import json
from typing import Any

from dsa.agent.planning import (
    BLOCKED_PATTERNS,
    COMPARISON_KEYWORDS,
    METRIC_KEYWORDS,
    UNSUPPORTED_PATTERNS,
    build_plan,
    classify_rules,
    infer_comparison,
    infer_metric,
    wants_customer_selection,
)
from dsa.agent.state import (
    MAX_REPAIR,
    MAX_RETRY,
    AgentState,
    RequestCategory,
)

__all__ = [
    "BLOCKED_PATTERNS",
    "COMPARISON_KEYWORDS",
    "METRIC_KEYWORDS",
    "NODE_FUNCS",
    "UNSUPPORTED_PATTERNS",
    "classify_rules",
    "infer_comparison",
    "infer_metric",
    "node_analyze",
    "node_check",
    "node_clarify",
    "node_classify",
    "node_execute",
    "node_finalize",
    "node_plan",
    "node_report",
    "node_repair",
    "node_validate",
    "wants_customer_selection",
]


def node_classify(state: AgentState) -> dict[str, Any]:
    """分类。**规则优先**；规则无法判定时才尝试 LLM，且 LLM 不能推翻阻断判定。"""
    question = state.get("question", "")
    category, missing, reason = classify_rules(question)

    # LLM 只允许在规则判为 EXECUTABLE 之后补充细化，**不得**把
    # BLOCKED / UNSUPPORTED 改判为可执行。
    if category is RequestCategory.EXECUTABLE:
        try:
            from dsa.agent import llm

            if llm.should_call_model():
                llm.get_llm()  # 真实模式下可用于意图细化（不改变类别）
        except Exception:  # noqa: BLE001 —— LLM 不可用不影响规则结论
            pass

    from dsa.agent.planning import request_constraints
    out: dict[str, Any] = {"category": category.value, "missing": missing,
                           "request_constraints":request_constraints(question)}
    if reason:
        out["errors"] = list(state.get("errors") or []) + [
            {"code": "E_PLAN_001" if category is RequestCategory.NEEDS_CLARIFICATION
             else "E_CAP_001", "message": reason, "node": "classify"}
        ]
    return out


def node_clarify(state: AgentState) -> dict[str, Any]:
    """生成澄清问题。**同一轮内已问过的不再重复问**（docs/09 §5.1）。

    ⚠️ 「已问过」的判据是**这一轮里问过**，不是"历史上问过"。

    为什么必须区分：澄清是单轮的（问一次、答一次），但用户可以只答一部分。
    如果判据是"历史上问过"，那么第一轮问了 {指标, 比较对象}、用户只答了指标，
    第二轮就会因为"比较对象问过了"而直接放弃（`E_PLAN_001`）——
    用户明明在配合，系统却因为自己问过一遍就拒绝继续。

    反过来，若完全不看去重，用户什么都不答时系统会无限重复同一个问题。
    因此判据用**回答条数**做轮次标记：一条澄清记录记下它被问出时
    `answers` 的长度；只有当 `answers` 长度**没变**时，才算"本轮已问过"。
    这样"重复提问"与"拒绝继续"两个坏结果都不会出现。
    """
    missing = list(state.get("missing") or [])
    answers = list(state.get("answers") or [])
    answered = {a.get("key") for a in answers}
    # 本轮已问过 = 记录里的答案条数等于当前答案条数
    asked_this_round = {
        c.get("missing_key") for c in (state.get("clarifications") or [])
        if c.get("answers_len", 0) == len(answers)
    }
    todo = list(dict.fromkeys(m for m in missing if m not in asked_this_round))

    if not todo:
        # 所有缺失项都问过了仍在原地 —— 明确失败，不无限循环
        return {
            "report": None,
            "errors": list(state.get("errors") or []) + [
                {"code": "E_PLAN_001", "node": "clarify",
                 "message": "缺失信息已全部询问过，仍未获得可执行条件"},
            ],
        }

    labels = {"metric": "关注哪个指标（购买率 / 访问率 / 人均消费）",
              "comparison": "要比较哪个邮件组与无邮件组（男装 / 女装）",
              "question": "请描述你要分析的问题",
              "window": "确认采用实验固定的14天窗口",
              "denominator": "确认是否使用全部被分配客户的人均消费",
              "population": "确认是否分析全部被分配客户",
              "limit": "请指定名单人数或比例（例如前100名或前10%）",
              "model_key": "请指定候选中的已登记模型"}
    questions = [labels.get(m, f"请补充 {m}") for m in todo]
    text = "为完成分析，还需要确认：" + "；".join(questions) + "。"

    # 用 LLM 润色措辞（可选，失败不影响提问）
    try:
        from dsa.agent import llm

        if llm.should_call_model():
            reply = llm.get_llm().invoke([{"role": "user", "content": text}])
            if getattr(reply, "content", "").strip():
                text = reply.content.strip()
    except Exception:  # noqa: BLE001
        pass

    return {
        "clarifications": list(state.get("clarifications") or []) + [
            {"missing_key": m, "question": q, "answers_len": len(answers)}
            for m, q in zip(todo, questions)
        ],
        "report": text,
    }


def node_plan(state: AgentState) -> dict[str, Any]:
    """生成结构化计划。

    计划字段**全部由 :func:`dsa.agent.planning.build_plan` 用程序解析**，
    覆盖数据集、实验、比较、任务类型、指标、窗口、筛选、排序依据、
    触达比例／人数。LLM 不参与字段取值 ——
    「用户要什么」必须是可测的，否则能力校验就没有可靠的输入。
    """
    plan = build_plan(state.get("question", ""))
    if plan.dataset_source == "default":
        plan.notes.append(
            f"问题未指定数据集，按默认 `{plan.dataset_key}` 处理"
        )
    if plan.task_type == "select_customers" and plan.ranking_source == "default":
        plan.notes.append(
            f"未指定排序依据，按默认 `{plan.ranking}`（预测增量）排序"
        )
    return {"plan": plan.model_dump()}


def node_validate(state: AgentState) -> dict[str, Any]:
    """**纯程序校验**：数据集、比较、指标、窗口、名单规模与模型版本。

    本节点**绝不调用 LLM** —— 这是「校验不可被绕过」的技术保证。

    校验逻辑本体在 :func:`dsa.agent.planning.validate_plan_against_registry`
    （纯函数，可脱离数据库单测）；本节点只负责把外部事实取来喂给它：

    - 实验配置（组别、窗口、预注册比较）
    - 指标是否登记
    - 已登记模型清单（**用于确认"要女装名单时有没有女装模型"**）

    本节点可能**回填 `plan`**（解析出 `model_key`），
    因此返回值里带 `plan` —— 执行节点必须用回填后的计划。
    """
    from dsa.agent.planning import validate_plan_against_registry
    from dsa.agent.state import AnalysisPlan
    from dsa.config import get_engine
    from dsa.tools.registry import call_tool

    engine = get_engine(readonly=True)
    raw = dict(state.get("plan") or {})
    try:
        plan = AnalysisPlan(**raw)
    except Exception as exc:  # noqa: BLE001 —— 计划结构不对就是校验失败
        return {"issues": [{"field": "plan", "code": "E_TOOL_001",
                            "message": f"计划结构不合法：{type(exc).__name__}: {exc}"}]}

    # ---- 取外部事实（失败不抛出，交给纯函数判成 issue）----
    cfg: dict[str, Any] | None = None
    if plan.experiment_key:
        r = call_tool("get_experiment_config", engine, experiment_key=plan.experiment_key)
        if r.ok:
            cfg = r.data
        else:
            cfg = None

    metric_ok: dict[str, bool] = {}
    for m in plan.metrics:
        r = call_tool("get_metric_definition", engine, metric_id=m)
        metric_ok[m] = bool(r.ok)

    models: list[dict[str, Any]] = []
    if plan.task_type == "select_customers":
        reg = call_tool("get_model_registry", engine)
        if reg.ok:
            models = list((reg.data or {}).get("models") or [])
        else:
            return {"issues": [{"field": "model_key", "code": reg.error.code.value,
                                "message": reg.error.message}]}

    issues, plan = validate_plan_against_registry(plan, cfg, metric_ok, models)
    out={"issues":issues,"plan":plan.model_dump()}
    if issues and all(i.get("code")=="E_PLAN_001" for i in issues):
        out.update(category="clarification",missing=list(dict.fromkeys(i["field"] for i in issues)))
    return out


def _effective_limit(plan: dict[str, Any], engine: Any) -> tuple[int | None, dict[str, Any]]:
    """把计划里的触达规模换算成 `limit_n`，并返回**换算依据**。

    比例必须乘上一个真实人数才能变成 `limit_n`。这个人数由
    `get_scoring_population` 工具给出（程序事实），
    **不在提示词里估、也不写死常量**。

    原实现固定 `limit_n=100`，于是「前 10%」和「前 20%」拿到同一份名单 ——
    审查把它列为 F02 的一部分。

    Returns:
        ``(limit_n, basis)``。``basis`` 会被写进报告，使「10% 是多少人」
        这件事可被读者核对。
    """
    from dsa.tools.registry import call_tool

    limit_n = plan.get("limit_n")
    limit_pct = plan.get("limit_pct")
    model_key = plan.get("model_key")
    basis: dict[str, Any] = {"limit_raw": plan.get("limit_raw") or "", "source": "explicit_n"}

    if limit_n is not None:
        basis["limit_n"] = int(limit_n)
        return int(limit_n), basis

    if limit_pct is None or not model_key:
        return None, basis

    r = call_tool("get_scoring_population", engine, model_key=model_key)
    if not r.ok:
        basis["error"] = r.error.message
        return None, basis

    population_n = int((r.data or {}).get("population_n") or 0)
    if population_n <= 0:
        basis["error"] = "可打分人群为 0"
        return None, basis

    # 四舍五入到整数人；至少 1 人，至多全量。
    n = max(1, min(population_n, int(round(limit_pct * population_n))))
    basis.update({
        "source": "explicit_pct",
        "limit_pct": float(limit_pct),
        "population_n": population_n,
        "limit_n": n,
    })
    return n, basis


def node_execute(state: AgentState) -> dict[str, Any]:
    """调用工具取数。**参数来自计划**；失败有限重试；每次调用都记入 tool_results。

    与旧实现的差别（F01/F02 的修复点）：

    | | 旧 | 新 |
    |---|---|---|
    | 效果计算 | 不调用 | `analyze_experiment_effect`（比较与指标来自计划）|
    | 名单规模 | 固定 100 | 计划的 `limit_n` / `limit_pct`，比例按**真实人数**换算 |
    | 名单排序 | 固定 `uplift` | 计划的 `ranking` |
    | 名单模型 | 固定 `tlearner_lr_v1` | 计划校验后回填的 `model_key` |
    """
    from dsa.config import get_engine
    from dsa.tools.registry import call_tool

    plan = dict(state.get("plan") or {})
    engine = get_engine(readonly=True)
    run_id = state.get("run_id")
    results = list(state.get("tool_results") or [])
    selection_basis: dict[str, Any] | None = None
    attempts = dict(state.get("attempts") or {})

    calls: list[tuple[str, dict[str, Any]]] = []
    if plan.get("task_type") == "select_customers":
        calls.append(("get_experiment_config",
                      {"experiment_key": plan.get("experiment_key")}))
    else:
        metrics = list(plan.get("metrics") or [])
        calls.append(("analyze_experiment_effect",
                      {"experiment_key": plan.get("experiment_key"),
                       "comparison_key": plan.get("comparison_key"),
                       "metrics": metrics}))
        # 描述性聚合仍然取：它是**分子/分母的可见证据**，
        # 效果量是"算出来的"，聚合是"看得见的原始计数"。
        calls.append(("query_arm_aggregates",
                      {"experiment_key": plan.get("experiment_key")}))

    n = attempts.get("execute", 0) + 1
    attempts["execute"] = n
    failures: list[dict[str, Any]] = []

    def _call(tool: str, kwargs: dict[str, Any]) -> Any:
        """调用一次工具，并记录**它实际执行的 SQL**。"""
        r, cap = call_tool_captured(tool, engine, run_id=run_id, **kwargs)
        results.append(r.to_dict())
        _record_query(run_id, tool, kwargs, r, cap)
        return r

    for tool, kwargs in calls:
        r = _call(tool, kwargs)
        if not r.ok:
            failures.append({"tool": tool, "code": r.error.code.value,
                             "message": r.error.message,
                             "retriable": r.error.code.value in ("E_DB_001", "E_DB_003")})

    # 名单：只有计划校验通过、且确实要名单时才取
    if plan.get("task_type") == "select_customers" and not any(
        f["tool"] == "get_experiment_config" for f in failures
    ):
        limit_n, selection_basis = _effective_limit(plan, engine)
        if limit_n is None:
            failures.append({
                "tool": "get_customer_scores", "code": "E_PLAN_001",
                "message": "无法确定名单规模：" + str(selection_basis.get("error") or "缺少比例或人数"),
                "retriable": False,
            })
        else:
            skwargs = {"model_key": plan.get("model_key"),
                       "policy_kind": plan.get("ranking") or "uplift",
                       "limit_n": limit_n}
            s = _call("get_customer_scores", skwargs)
            if not s.ok:
                failures.append({"tool": "get_customer_scores", "code": s.error.code.value,
                                 "message": s.error.message,
                                 "retriable": s.error.code.value in ("E_DB_001", "E_DB_003")})

    out: dict[str, Any] = {"tool_results": results, "attempts": attempts}

    # ⚠️ **当次尝试的结果必须单独记录**（F05）。
    #
    # 旧实现的路由函数扫描 `errors` 里**全部历史**执行错误：
    # 第 1 次超时 → 重试；第 2 次成功 → 路由仍看到那条历史超时 → 再重试；
    # 第 3 次成功 → 仍然看到它 → 结束并报告「分析未能完成」。
    # 于是**一次暂时失败永久污染了这次运行**。
    #
    # 修法是把"审计日志"和"当前状态"分开：`errors` 继续累积全部历史
    # （审计需要），路由只看 `last_execute_*`（状态判断需要）。
    # 同一个词在两层含义上复用，就会让两层都失准 —— 与 L030 同族。
    out["last_execute_ok"] = not failures
    out["last_execute_retriable"] = any(f["retriable"] for f in failures)
    out["last_execute_failures"] = failures
    if selection_basis is not None:
        out["selection_basis"] = selection_basis

    if failures:
        out["errors"] = list(state.get("errors") or []) + [
            {**f, "node": "execute"} for f in failures
        ]
        # ⚠️ 这里**不设 blocked**。
        #
        # `blocked` 的语义是「**数据质量检查未通过，不能输出因果结论**」
        # （见 docs/09-Agent协议.md §5.1 与 docs/05 的阻断规则）。
        # 而「工具不支持这个请求」是**执行失败**，不是数据质量问题 ——
        # 后者仍可输出描述性结果，前者连数都没取到。
        #
        # 早期版本在这里对 E_CAP_001 设了 blocked=True，评测立刻抓到两处后果：
        #   ① 该走「明确失败」的案例走了「阻断」路径，fail_gracefully 失败
        #   ② 阻断的 precision 从 1.00 掉到 0.667 —— 出现了不该有的阻断
        # 见 docs/日志.md L030。**同一个词在两层含义上被复用，就会让两层都失准。**
    return out


def call_tool_captured(tool: str, engine: Any, *,
                       run_id: str | None = None, **kwargs: Any) -> tuple[Any, Any]:
    """调用工具并**捕获它实际执行的 SQL**。

    所有调用工具的节点都必须走这里 —— 否则 `query_executions.sql_text`
    会退化成"未捕获"占位符，而那个占位符**看起来**像"这个工具不需要 SQL"。
    实测就发生过：`run_quality_checks` 明明执行了 SQL，却因为调用点忘了
    包捕获窗口，被记成"该工具本次未执行 SQL"。

    Returns:
        ``(ToolResult, SqlCapture)``。捕获窗口只包住这一次调用：
        engine 是缓存的单例，窗口开大了会把别的调用也算进来。
    """
    from dsa.agent.sql_capture import capture_sql
    from dsa.tools.registry import call_tool

    with capture_sql(engine) as cap:
        result = call_tool(tool, engine, run_id=run_id, **kwargs)
    return result, cap


def _record_query(run_id: str | None, tool: str, params: dict, result: Any,
                  capture: Any = None) -> None:
    """把一次工具调用写进 ``query_executions``。

    **为什么需要这个函数**：`run_records.record_query` 早就实现且有测试，
    但没有任何实际工作负载调用它 —— 实测该表**长期为 0 行**，
    意味着「工具参数与耗时」这项追溯信息事实上从未被保存。
    「有函数、有测试」不等于「有数据」，这是本项目的一个真实教训。

    **`capture` 参数是第二次修正**（F06）。第一版虽然接上了，
    但 `sql_text` 写的是 ``-- tool: <名字>`` 这样一句注释，
    `row_count` 又从不存在的 ``data["rows"]`` 推导 ——
    表里那两列在**聚合类**返回上永远是空的。
    现在 SQL 与行数都由 :class:`dsa.agent.sql_capture.SqlCapture`
    从数据库实际执行中取得。取不到就写 NULL，**不编一个 0**。

    **失败不抛出**：运行记录是**辅助**能力，它的失败不应该让主分析崩掉。
    但也不静默 —— 打印一行警告，让问题可见。
    """
    if not run_id:
        return
    try:
        from dsa.config import get_engine
        from dsa.tools.run_records import deterministic_seq, record_query

        sql_text = None
        row_count = None
        bound_params: dict[str, Any] = dict(params)
        captured = capture is not None
        if captured:
            sql_text = capture.sql_text
            row_count = capture.row_count
            if capture.params:
                bound_params = {**bound_params, "db_params": capture.params}
        if not sql_text:
            # ⚠️ **必须区分三种情况**，它们的含义完全不同：
            #
            #   (a) 捕获到 SQL            → 写真实语句
            #   (b) 捕获了，但确实没执行 SQL → "该工具本次未执行 SQL"
            #   (c) **没有启用捕获**        → "未捕获"
            #
            # 第一版把 (b) 和 (c) 写成同一句话。实测立刻出问题：
            # `run_quality_checks` 与 `get_policy_evaluations` 明明执行了 SQL，
            # 却因为调用点没传 capture，被记成"该工具本次未执行 SQL" ——
            # 表里那句话**是错的**，而且读起来像是"这个工具不需要 SQL"，
            # 于是没人会去查它为什么没有 SQL。
            #
            # 这与 F06 被点名的原始缺陷是同一形态：**表里的文字与事实不符**。
            sql_text = (
                "-- (该工具本次未执行 SQL：读取配置文件或纯计算)"
                if captured else
                "-- (未捕获：该调用点未启用 SQL 捕获，不代表没有执行 SQL)"
            )

        write_engine = get_engine(readonly=False)
        record_query(
            write_engine,
            run_id=run_id,
            seq=deterministic_seq(tool, params),
            tool_name=tool,
            sql_text=sql_text,
            params=bound_params,
            row_count=row_count,
            duration_ms=result.duration_ms,
            status="ok" if result.ok else "error",
            error_code=None if result.ok else result.error.code.value,
            error_detail=None if result.ok else result.error.message,
        )
    except Exception as exc:  # noqa: BLE001 —— 辅助能力失败不阻断主流程
        print(f"  ⚠️ 运行记录写入失败（不影响分析结果）: {type(exc).__name__}: {exc}")


def node_check(state: AgentState) -> dict[str, Any]:
    """质量检查与阻断判定。**有 blocking 失败即阻断**，后续不得输出因果结论。"""
    from dsa.config import get_engine
    from dsa.tools.registry import call_tool

    engine = get_engine(readonly=True)
    r, cap = call_tool_captured(
        "run_quality_checks", engine, run_id=state.get("run_id"),
        experiment_key="hillstrom_email_v1",
        dataset_version="hillstrom@2008-03-20")
    _record_query(state.get("run_id"), "run_quality_checks",
                  {"experiment_key": "hillstrom_email_v1",
                   "dataset_version": "hillstrom@2008-03-20"}, r, cap)

    if not r.ok:
        return {
            "checks": [],
            "blocked": True,
            "blocking_codes": ["E_DB_001"],
            "errors": list(state.get("errors") or []) + [
                {"code": r.error.code.value, "message": r.error.message, "node": "check"}
            ],
        }

    checks = r.data.get("checks", [])
    codes = list(r.data.get("blocking_codes", []))
    return {
        "checks": checks,
        "blocked": bool(codes) or not r.data.get("causal_conclusion_allowed", False),
        "blocking_codes": codes,
    }


def _resolve_model_for_plan(plan: dict[str, Any], engine: Any) -> str | None:
    """找出与计划比较对象匹配的已登记模型。

    **不再硬编码 `tlearner_lr_v1`。** 旧实现在两个节点里写死了这个 key，
    于是「女装」的请求会拿到挂在男装模型下的策略评估 —— 数据不假，
    但**标签是错的**，这比缺数据更难发现（F02）。

    多个候选时选**最近登记的**（`trained_at` 最新，退化为 `model_key` 最大）。
    为什么不用 `model_key` 单独排序：`tlearner_lr_v10` 会排在 `tlearner_lr_v2`
    前面，于是"最新的模型"在选择上反而输给字典序 —— 这类排序假设
    在版本号进入两位数时会静默出错。
    """
    from dsa.tools.registry import call_tool
    from dsa.agent.model_resolution import resolve_model
    reg=call_tool("get_model_registry",engine)
    if not reg.ok:
        return None
    key,_,_=resolve_model(plan,(reg.data or {}).get("models") or [])
    return key


def node_analyze(state: AgentState) -> dict[str, Any]:
    """统计 / 模型分析。不可用时记入 errors，**不静默跳过**。

    只做一件旧实现没做对的事：**按计划的比较对象**找模型。
    名单本身由 `execute` 节点取（它才看得到计划里的规模与排序依据）；
    本节点负责策略评估这一层的补充证据。
    """
    from dsa.config import get_engine

    engine = get_engine(readonly=True)
    plan = dict(state.get("plan") or {})
    results = list(state.get("tool_results") or [])

    # 名单已在 execute 取过就不再重复取 —— 一次运行里同一件事只做一次。
    already_listed = any(t.get("tool") == "get_customer_scores"
                         for t in (state.get("tool_results") or []))

    model_key = _resolve_model_for_plan(plan, engine)
    out: dict[str, Any] = {}

    if model_key is None:
        out["errors"] = list(state.get("errors") or []) + [{
            "code": "E_CAP_001", "node": "analyze",
            "message": f"比较 `{plan.get('comparison_key')}` 没有已登记模型，"
                       "不提供策略评估（**不会用其它比较的模型代替**）",
        }]
        out["tool_results"] = results
        return out

    r, cap = call_tool_captured("get_policy_evaluations", engine,
                                run_id=state.get("run_id"), model_key=model_key)
    results.append(r.to_dict())
    _record_query(state.get("run_id"), "get_policy_evaluations",
                  {"model_key": model_key}, r, cap)

    # 名单：仅当问题确实在问「该给谁发」且 execute 尚未取到时才补取。
    if not already_listed and wants_customer_selection(state.get("question", "")):
        skwargs = {"model_key": model_key,
                   "policy_kind": plan.get("ranking") or "uplift",
                   "limit_n": int(plan.get("limit_n") or 100)}
        s, scap = call_tool_captured("get_customer_scores", engine,
                                     run_id=state.get("run_id"), **skwargs)
        results.append(s.to_dict())
        _record_query(state.get("run_id"), "get_customer_scores", skwargs, s, scap)

    out["tool_results"] = results
    if not r.ok:
        out["errors"] = list(state.get("errors") or []) + [
            {"code": r.error.code.value, "message": r.error.message, "node": "analyze",
             "note": "策略评估不可用，报告降级为仅描述性统计"}
        ]
    return out


def _fmt_int(v: Any) -> str:
    return f"{int(v):,}" if isinstance(v, (int, float)) else "—"


def _plan_summary(plan: dict[str, Any]) -> list[str]:
    """把计划回显给读者。**"我们替你定了什么"必须可见。**"""
    if not plan:
        return []
    label = {"primary": "男装邮件", "secondary_womens": "女装邮件"}
    rows = [
        ("数据集", f"`{plan.get('dataset_key')}`"
                   + ("（问题未指定，按默认）" if plan.get("dataset_source") == "default" else "")),
        ("实验", f"`{plan.get('experiment_key')}`"),
        ("比较对象", label.get(plan.get("comparison_key"), plan.get("comparison_key"))
                     + ("（未指定，按默认男装）" if plan.get("comparison_source") == "default" else "")),
        ("任务类型", "人群名单" if plan.get("task_type") == "select_customers" else "效果比较"),
        ("结果窗口", f"实验后 {plan.get('outcome_window_days')} 天（固定）"),
    ]
    metrics = plan.get("metrics") or []
    if metrics:
        from dsa.stats.effect import metric_name

        rows.append(("指标", "、".join(metric_name(m) for m in metrics)
                     + ("（问题未指定，按默认购买率）" if plan.get("metric_source") == "default" else "")))
    if plan.get("task_type") == "select_customers":
        rows.append(("排序依据", f"`{plan.get('ranking')}`"
                     + ("（未指定，按默认预测增量）" if plan.get("ranking_source") == "default" else "")))
        if plan.get("limit_raw"):
            rows.append(("触达规模", f"问题原话「{plan.get('limit_raw')}」"))
        rows.append(("模型", f"`{plan.get('model_key') or '未解析'}`（来自登记表）"))
    if plan.get("filters"):
        rows.append(("筛选条件", json.dumps(plan["filters"], ensure_ascii=False)))
    lines = ["| 项 | 取值 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return lines


def _guarded_explanation(llm: Any, body: str, state: AgentState) -> tuple[str, dict[str, Any]]:
    """报告仅由计算结果渲染。旧启发式守卫不再授权自由文本进入报告。"""
    return "", {"mode": "deterministic", "passed": True, "rejected": False,
                "free_explanation_enabled": False,
                "reason": "数字、方向与显著性均由程序从结构化结果生成。"}


def node_report(state: AgentState) -> dict[str, Any]:
    """生成报告。**所有数字由程序从 tool_results 回填**（AGENTS.md §2.1）。

    三个「只做排版、不做计算」的区块：

    1. **计划回显** —— 让"我们替你定了什么"可见（F02）。
       原实现把默认值默默用掉，用户无从发现"女装的请求走了男装模型"。
    2. **效果估计** —— 来自 `analyze_experiment_effect` 的结构化返回，
       由 :func:`dsa.report.effect_view.render_effect_section` 渲染（F01）。
    3. **描述性统计** —— 分子/分母的原始计数，是效果量的可见证据。

    所有结论由程序生成。历史解释守卫保留兼容接口，不接收自由模型解释。
    """
    from dsa.agent import llm

    plan = dict(state.get("plan") or {})
    arms: list[dict[str, Any]] = []
    effect_payload: dict[str, Any] | None = None
    for tr in state.get("tool_results") or []:
        if tr.get("tool") == "query_arm_aggregates" and tr.get("ok") and not arms:
            arms = (tr.get("data") or {}).get("arms", [])
        if tr.get("tool") == "analyze_experiment_effect" and tr.get("ok") and effect_payload is None:
            effect_payload = tr.get("data") or {}

    lines: list[str] = []
    lines.append(f"## 问题\n\n{state.get('question', '')}\n")

    if state.get("blocked"):
        codes = "、".join(state.get("blocking_codes") or []) or "数据质量检查未通过"
        lines.append(
            "## 结论\n\n"
            f"**本次分析被阻断（{codes}），不给出因果结论。**\n\n"
            "阻断的原因使「干预效果」与「人群差异」无法区分。\n"
            "下方描述性数字仍然可用，但**不得**据此判断邮件是否有效。\n"
        )

    plan_lines = _plan_summary(plan)
    if plan_lines:
        lines.append("## 分析计划\n")
        lines.extend(plan_lines)
        lines.append("")
        for note in plan.get("notes") or []:
            lines.append(f"> 注：{note}\n")

    # ---- 效果估计（F01 的核心修复）----
    if effect_payload:
        from dsa.report.effect_view import render_effect_section

        section = render_effect_section(effect_payload)
        if section:
            lines.append(section)

    # ---- 描述性统计：分子/分母的可见证据 ----
    if arms:
        lines.append("## 各组描述性统计\n")
        lines.append("| 组别 | 样本量 | 购买人数 | 购买率 | 访问人数 | 访问率 |")
        lines.append("|---|---|---|---|---|---|")
        for a in arms:
            rate = a.get("conversion_rate")
            vrate = a.get("visit_rate")
            lines.append(
                f"| {a.get('segment_value') or a.get('arm_key')} | "
                f"{_fmt_int(a.get('n_customers'))} | {_fmt_int(a.get('n_converted'))} | "
                f"{rate:.6%} | {_fmt_int(a.get('n_visited'))} | {vrate:.6%} |"
                if isinstance(rate, (int, float)) and isinstance(vrate, (int, float))
                else f"| {a.get('arm_key')} | {_fmt_int(a.get('n_customers'))} | — | — | — | — |"
            )
        lines.append(
            "\n> 分母为**全部被随机分配到该组的客户数**，"
            "不按任何干预后行为筛选。\n"
        )

    # ---- 人群名单 ----
    for tr in state.get("tool_results") or []:
        if tr.get("tool") != "get_customer_scores" or not tr.get("ok"):
            continue
        data = tr.get("data") or {}
        rows = data.get("rows") or []
        basis = dict(state.get("selection_basis") or {})
        lines.append("\n## 人群名单\n")

        # 「10% 是多少人」必须可核对：把比例、分母、结果三个数都写出来。
        if basis.get("source") == "explicit_pct":
            lines.append(
                f"按问题原话「{basis.get('limit_raw') or '按比例'}」取名单："
                f"可打分人群 **{int(basis.get('population_n') or 0):,}** 人 × "
                f"{float(basis.get('limit_pct') or 0):.4%} = "
                f"**{int(basis.get('limit_n') or 0):,}** 人（四舍五入到整数人）。\n"
            )
        lines.append(
            f"排序依据：`{plan.get('ranking')}`"
            + ("（问题未指定，按默认预测增量）" if plan.get("ranking_source") == "default" else "")
            + f"　｜　模型：`{plan.get('model_key')}`　｜　实际返回 **{data.get('row_count')}** 名\n"
        )
        sort_col = {"uplift": "uplift", "purchase_probability": "propensity_score",
                    "control_probability": "mu0", "random": "customer_key"}.get(
                        str(plan.get("ranking")), "uplift")
        lines.append(f"| 客户 | {sort_col} |")
        lines.append("|---|---|")
        for row in rows[:10]:
            val = row.get(sort_col, row.get("uplift"))
            lines.append(f"| {row.get('customer_key')} | "
                         f"{val:+.8f} |" if isinstance(val, (int, float))
                         else f"| {row.get('customer_key')} | — |")
        lines.append(f"\n共 {data.get('row_count')} 名客户；完整名单见下方证据引用。")
        for ref in tr.get("refs") or []:
            lines.append(f"- 产物：`artifacts/{ref}`")
        lines.append(
            "\n> ⚠️ 名单为**回顾性评估输出**，不构成对下一轮投放效果的承诺；"
            "本报告**不提供客户级预测区间**。\n"
        )

    # ---- 证据与追溯 ----
    run_id = state.get("run_id")
    if not state.get("blocked"):
        refs: list[str] = []
        for tr in state.get("tool_results") or []:
            refs.extend(tr.get("refs") or [])
        lines.append("## 证据与追溯\n")
        lines.append(f"- 运行标识：`{run_id or '（未登记）'}`")
        if effect_payload:
            lines.append(
                f"- 效果估计来源：`analyze_experiment_effect` 工具的结构化返回"
                f"（分析内部标识 `{effect_payload.get('analysis_run_id')}`）"
            )
            sc = effect_payload.get("stats_config") or {}
            lines.append(
                "- 统计口径：比例区间 `{}`、差值区间 `{}`、风险比 `{}`、检验 `{}`".format(
                    sc.get("proportion_interval"), sc.get("difference_interval"),
                    sc.get("ratio_interval"), sc.get("hypothesis_test"))
            )
            lines.append(
                f"- 数据版本：`{effect_payload.get('dataset_version')}`　｜　"
                f"代码版本：`{effect_payload.get('code_version')}`"
            )
        for ref in refs:
            lines.append(f"- 产物引用：`artifacts/{ref}`")
        lines.append("")

    body = "\n".join(lines)

    # ---- LLM 解释 + 守卫（F03）----
    if not state.get("blocked"):
        extra, guard = _guarded_explanation(llm, body, state)
        if extra:
            body += "\n\n## 解释\n\n" + extra + "\n"
        if guard.get("rejected"):
            body += (
                "\n> ⚠️ 模型补写的解释**未通过数值与结论守卫**，已丢弃。\n"
                f"> 原因：{guard.get('reason')}\n"
                "> 上方报告全部由程序生成的数字构成，未受该问题影响。\n"
            )

    return {"report": body, "explanation_guard": guard if not state.get("blocked") else {"mode":"deterministic", "passed":True, "free_explanation_enabled":False}}


def node_repair(state: AgentState) -> dict[str, Any]:
    """格式/参数错误的**有限**修复。超限即明确失败。"""
    attempts = dict(state.get("attempts") or {})
    n = attempts.get("repair", 0) + 1
    attempts["repair"] = n
    out: dict[str, Any] = {"attempts": attempts}
    if n > MAX_REPAIR:
        out["errors"] = list(state.get("errors") or []) + [
            {"code": "E_TOOL_001", "node": "repair",
             "message": f"已尝试修复 {MAX_REPAIR} 次仍未通过，放弃"}
        ]
        out["report"] = None
    return out


def _failure_report(state: AgentState) -> str:
    """能力不足 / 校验不通过时的**明确**说明。

    为什么不能只说一句「分析未能完成」：审查的验收条件要求
    「未知数据集、窗口和筛选条件有明确处置」「女装名单但只有男装模型 →
    明确能力限制」。一句泛泛的失败既没说是哪一项不支持，
    也没说用户可以怎么改，读者只能去翻结构化错误 ——
    而结构化错误的读者是程序，不是人。

    这里把 `issues`（校验问题）与 `errors`（执行/分类错误）
    按来源分组列出，并给出**可执行的下一步**。
    """
    lines: list[str] = ["## 结论\n", "**本次请求未能执行。**\n"]

    issues = list(state.get("issues") or [])
    if issues:
        lines.append("### 校验未通过\n")
        lines.append("| 字段 | 错误码 | 说明 |")
        lines.append("|---|---|---|")
        for it in issues:
            lines.append(f"| `{it.get('field')}` | `{it.get('code')}` | "
                         f"{it.get('message')} |")
        lines.append("")

    codes = {i.get("code") for i in issues}
    errs = [e for e in (state.get("errors") or []) if e.get("node") in ("execute", "classify")]
    if errs:
        lines.append("### 执行错误\n")
        lines.append("| 工具 | 错误码 | 说明 |")
        lines.append("|---|---|---|")
        for e in errs:
            lines.append(f"| `{e.get('tool') or e.get('node')}` | `{e.get('code')}` | "
                         f"{e.get('message')} |")
        lines.append("")
        codes |= {e.get("code") for e in errs}

    if "E_CAP_001" in codes:
        lines.append(
            "### 该怎么办\n\n"
            "这是**能力范围**问题，重试不会改变结论。可选：\n\n"
            "- 换成已支持的数据集 / 比较对象 / 指标后重问；\n"
            "- 若要新增能力（例如为女装比较训练模型），"
            "需先跑 `scripts/train_and_evaluate.py` 登记模型；\n"
            "- 查阅 `docs/01-功能清单与非目标.md` 了解当前版本的支持边界。\n"
        )
    elif "E_PLAN_001" in codes:
        lines.append(
            "### 该怎么办\n\n"
            "这是**条件不足**问题。请补充缺失的条件（如关注哪个指标、"
            "要比较哪个邮件组与无邮件组，或名单的触达比例/人数）后重问。\n"
        )
    else:
        lines.append(
            "### 该怎么办\n\n"
            "请查看上方结构化错误；若是数据库类错误（`E_DB_*`），"
            "重试通常有效。\n"
        )

    run_id = state.get("run_id")
    if run_id:
        lines.append(f"\n---\n\n- 运行标识：`{run_id}`（可据此查 `query_executions` 与产物）")
    return "\n".join(lines)


def node_finalize(state: AgentState) -> dict[str, Any]:
    """汇总最终状态。给出**明确**的结束原因，不留模糊状态。"""
    category = state.get("category")
    report = state.get("report")
    out: dict[str, Any] = {}

    if state.get("blocked") and report is None:
        out["report"] = node_report(state)["report"]

    if report is None and not out.get("report"):
        if category == RequestCategory.UNSUPPORTED.value:
            reasons = [e.get("message") for e in (state.get("errors") or [])
                       if e.get("code") == "E_CAP_001"]
            out["report"] = ("本次请求超出第一版能力范围，未执行分析。\n\n"
                             + ("原因：" + "；".join(filter(None, reasons)) + "\n" if reasons else ""))
        elif category == RequestCategory.NEEDS_CLARIFICATION.value:
            out["report"] = state.get("report") or "需要补充信息后才能继续。"
        elif category == RequestCategory.BLOCKED.value:
            reasons = [e.get("message") for e in (state.get("errors") or [])
                       if e.get("code") == "E_CAP_001"]
            out["report"] = ("本次请求被阻断，未执行分析。\n\n"
                             + ("原因：" + "；".join(filter(None, reasons)) + "\n" if reasons else ""))
        elif state.get("issues") or not state.get("last_execute_ok", True):
            # 校验未通过 / **最近一次**执行失败：**逐条**说明，不留一句"分析未能完成"。
            #
            # ⚠️ 判据是 `last_execute_ok`，**不是**扫描 `errors` ——
            # 同 F05：`errors` 是审计日志，一次暂时超时后重试成功，
            # 它里面仍然有那条超时。用历史判断当前状态会让"成功"被写成"失败"。
            out["report"] = _failure_report(state)
        else:
            out["report"] = "分析未能完成，请查看 errors 字段中的结构化错误。"

    if state.get("run_id"):
        out["usage"] = list(state.get("usage") or [])
    return out


#: 节点名 → 函数（graph.py 用它装配）
NODE_FUNCS = {
    "classify": node_classify,
    "clarify": node_clarify,
    "plan": node_plan,
    "validate": node_validate,
    "execute": node_execute,
    "check": node_check,
    "analyze": node_analyze,
    "report": node_report,
    "repair": node_repair,
    "finalize": node_finalize,
}
