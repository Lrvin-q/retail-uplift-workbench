"""LangGraph 状态图装配。

结构见 docs/09-Agent协议.md §2。条件边的判定**全部是纯程序**，
不依赖 LLM —— 否则「阻断不被绕过」无法保证。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from dsa.agent.nodes import NODE_FUNCS
from dsa.agent.state import MAX_REPAIR, AgentState, RequestCategory

NODE_NAMES = (
    "classify", "clarify", "plan", "validate",
    "execute", "check", "analyze", "report", "repair", "finalize",
)

__all__ = [
    "NODE_NAMES",
    "awaiting_fields",
    "build_graph",
    "is_awaiting_user",
    "resume_with_answers",
    "route_after_check",
    "route_after_classify",
    "route_after_execute",
    "route_after_repair",
    "route_after_validate",
    "run_once",
    "run_status_for",
]


# ============================================================ 条件边


def route_after_classify(state: AgentState) -> str:
    """四类分流。**先阻断、再能力、最后澄清**的顺序由 classify 保证。"""
    cat = state.get("category")
    if cat == RequestCategory.EXECUTABLE.value:
        return "plan"
    if cat == RequestCategory.NEEDS_CLARIFICATION.value:
        return "clarify"
    return "finalize"  # BLOCKED / UNSUPPORTED / 未知一律收敛到 finalize


def route_after_validate(state: AgentState) -> str:
    if state.get("category")=="clarification":
        return "clarify"
    return "execute" if not state.get("issues") else "repair"


def route_after_repair(state: AgentState) -> str:
    """修复成功回 plan 重试；超限直接 finalize（**有限**修复）。"""
    n = (state.get("attempts") or {}).get("repair", 0)
    if n > MAX_REPAIR:
        return "finalize"
    return "plan"


def route_after_check(state: AgentState) -> str:
    """阻断即收敛；否则继续分析。"""
    return "finalize" if state.get("blocked") else "analyze"


def route_after_execute(state: AgentState) -> str:
    """执行失败：**只依据最近一次尝试**决定重试还是收敛。

    缺陷回顾（外部审查 F05）
    ------------------------
    > 一次暂时失败会污染后续重试，成功后仍报告失败。
    > 第 1 次超时 → 重试；第 2 次成功 → 仍重试；第 3 次成功 → 结束并报告"分析未能完成"。

    根因：旧实现在这里扫描 `errors` 里**全部历史**执行错误。
    而 `errors` 是**审计日志**（必须保留历史），不是"当前是否失败"。

    修法是把两种信息分开：

    - `errors` 继续累积全部历史 —— 审计需要它，用户排查需要它；
    - `last_execute_ok` 只反映**最后一次**尝试 —— 路由只允许看它。

    这与 L030（`blocked` 一词被两层含义复用）是同一族问题：
    **一个字段承担两种语义，两层都会失准。**
    """
    from dsa.agent.state import MAX_RETRY

    # 最近一次尝试是否成功。**不看历史。**
    if state.get("last_execute_ok", True):
        return "check"

    attempts = (state.get("attempts") or {}).get("execute", 0)
    # 不可重试的错误（参数错误 / 能力范围）立即停止：重试不会改变结论。
    if not state.get("last_execute_retriable", False):
        return "finalize"
    if attempts <= MAX_RETRY:
        return "execute"
    return "finalize"


# ============================================================ 装配


def build_graph(*, checkpointer: Any = None, use_llm: bool = True) -> Any:
    """构造并编译状态图。

    Args:
        checkpointer: LangGraph 检查点存储器；None 表示不持久化（仅测试用）。
        use_llm: 为 False 时强制 mock，**保证纯程序路径可独立验证**
            （AGENTS.md §3.2「计算模块先脱离 LLM 验证」）。
    """
    if not use_llm:
        import os

        os.environ["DSA_LLM_MODE"] = "mock"

    g = StateGraph(AgentState)
    for name in NODE_NAMES:
        g.add_node(name, NODE_FUNCS[name])

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_after_classify,
                            {"plan": "plan", "clarify": "clarify", "finalize": "finalize"})
    g.add_edge("clarify", END)
    g.add_edge("plan", "validate")
    g.add_conditional_edges("validate", route_after_validate,
                            {"execute": "execute", "repair": "repair", "clarify":"clarify"})
    g.add_conditional_edges("repair", route_after_repair,
                            {"plan": "plan", "finalize": "finalize"})
    g.add_conditional_edges("execute", route_after_execute,
                            {"check": "check", "execute": "execute", "finalize": "finalize"})
    g.add_conditional_edges("check", route_after_check,
                            {"analyze": "analyze", "finalize": "finalize"})
    g.add_edge("analyze", "report")
    g.add_edge("report", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)


#: 终态映射：Agent 状态 → `analysis_runs.status`（数据库约束只允许
#: `pending/running/succeeded/blocked/failed`）。
#:
#: ⚠️ **等待用户不是失败，也不是成功。** 它保持 `pending`，
#: 于是 `finished_at` 为空 —— 这正是「这次还没跑完」的准确表示。
#: 把等待记成 `failed` 会让统计里出现大量假失败；
#: 记成 `succeeded` 则会把"没回答"算成"答完了"。
_STATE_TO_RUN_STATUS: dict[str, str] = {
    "clarification": "pending",
    "blocked": "blocked",
    "unsupported": "failed",
    "executable": "succeeded",
}


def run_status_for(state: dict) -> str:
    """最终状态 → 运行状态。

    ⚠️ **可执行 ≠ 成功。** 一个可执行的请求如果校验没过、或者工具执行失败，
    它的报告是一段"失败说明"——文案完整、字数不少，
    但它**没有回答用户的问题**。旧判据（"报告非空即完成"）会把这种情况
    记成成功，于是完成率这个指标失去了意义（审查 F08 第 3 条）。
    """
    from dsa.agent.outcome import execution_outcome
    outcome, _ = execution_outcome(state)
    return {"succeeded":"succeeded", "awaiting_user":"pending",
            "refused":"blocked" if state.get("blocked") or state.get("category")=="blocked" else "failed",
            "capability_limited":"failed", "failed":"failed"}[outcome]


def _register_run(question: str, *, run_id: str | None) -> str | None:
    """创建运行登记，返回 run_id；**失败不中断分析**（但会打印原因）。

    缺陷回顾（外部审查 F06）
    ------------------------
    > 本轮三次正常交互的 `run_id` 均为 `None`，报告显示"未登记"。
    > 工作台同样没有在启动时创建业务运行记录；
    > **有检查点不等于有完整业务追溯。**

    检查点记的是"图执行到哪一步"，运行登记记的是"这次业务分析是什么、
    用了什么、产出了什么"。两者不是一回事，前者不能替代后者。
    """
    try:
        from dsa.config import get_engine
        from dsa.tools.run_records import create_run

        return create_run(
            get_engine(readonly=False),
            experiment_key="hillstrom_email_v1",
            comparison_key="primary",
            question=question,
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001 —— 登记失败不应让分析跑不起来
        print(f"  ⚠️ 运行登记创建失败（分析继续，但该次运行不可按 run_id 追溯）: "
              f"{type(exc).__name__}: {exc}")
        return None


def _finish_run(run_id: str | None, state: dict) -> None:
    """回填运行终态、产物引用与统计参数。**失败不抛出。**"""
    if not run_id:
        return
    try:
        from dsa.config import get_engine
        from dsa.tools.run_records import update_run_status

        refs: list[str] = []
        stats_config: dict = {}
        plan = dict(state.get("plan") or {})
        for tr in state.get("tool_results") or []:
            refs.extend(tr.get("refs") or [])
            if tr.get("tool") == "analyze_experiment_effect" and tr.get("ok"):
                stats_config = dict((tr.get("data") or {}).get("stats_config") or {})

        from dsa.config import ARTIFACTS_DIR
        from dsa.tools.run_records import to_jsonable
        import json
        run_dir=ARTIFACTS_DIR / str(run_id)
        run_dir.mkdir(parents=True,exist_ok=True)
        report_ref=f"{run_id}/report.md"
        state_ref=f"{run_id}/state.json"
        state["report_ref"]=report_ref
        state.setdefault("explanation_guard",{"mode":"rule_response",
                         "free_explanation_enabled":False})
        for name,content in (
            ("report.md", str(state.get("report") or "本次运行未生成报告。")),
            ("state.json", json.dumps(to_jsonable(dict(state)),ensure_ascii=False,indent=2)),
        ):
            temp=run_dir/(name+".tmp")
            temp.write_text(content,encoding="utf-8")
            temp.replace(run_dir/name)
        refs.extend([state_ref])
        refs=list(dict.fromkeys(refs))
        update_run_status(
            get_engine(readonly=False),
            run_id,
            status=run_status_for(state),
            error_code=(state.get("errors") or [{}])[-1].get("code")
            if state.get("errors") else None,
            result_refs=refs or None,
            report_ref=report_ref,
        )
        # 计划与统计参数：写进运行登记，使「重启后可查计划和报告」成立。
        _write_run_context(run_id, plan, stats_config)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ 运行终态回填失败: {type(exc).__name__}: {exc}")


def _write_run_context(run_id: str, plan: dict, stats_config: dict) -> None:
    """把**计划**与统计参数写回 `analysis_runs`，并**校正比较对象**。

    两件事必须在这里补写：

    1. **计划**里的 `model_key` 是 `validate` 节点回填的，登记开始时还不存在。
       不补写，"重启后查计划"看到的是一份**缺字段的计划**。
    2. **比较对象**。登记发生在 `plan` 节点**之前**，那时还不知道用户问的是
       男装还是女装，只能填一个默认值 `primary`。
       如果就这样留下，一次女装分析的运行记录会写着 `primary` ——
       与 F02 同类：**登记表说的和实际算的不是同一件事**，
       而且它恰好是最容易被引用的一栏（报告追溯、导出清单都读它）。

    ⚠️ 初值仍然是 `primary`，因为不能等到计划出来才登记：
    计划之前就崩掉的运行也必须留下记录。所以这里的选择是
    **先登记一个乐观的默认值，再在终态时校正**，而不是不登记。
    """
    import json

    from sqlalchemy import text

    from dsa.config import get_engine

    comparison = str(plan.get("comparison_key") or "").strip() or None
    with get_engine(readonly=False).begin() as conn:
        conn.execute(
            text(
                "update analysis_runs set plan = cast(:plan as jsonb), "
                "       stats_config = coalesce(cast(:sc as jsonb), stats_config), "
                "       comparison_key = coalesce(:cmp, comparison_key) "
                "where run_id = :rid"
            ),
            {
                "rid": run_id,
                "plan": json.dumps(plan, ensure_ascii=False),
                "sc": json.dumps(stats_config, ensure_ascii=False) if stats_config else None,
                "cmp": comparison,
            },
        )


def run_once(question: str, *, thread_id: str | None = None,
             checkpointer: Any = None, use_llm: bool = False,
             run_id: str | None = None, register: bool = True) -> dict:
    """跑一次完整分析，返回最终状态。供脚本与工作台调用。

    Args:
        register: 是否创建 `analysis_runs` 登记。默认 True ——
            「默认交互没有完整运行记录」正是被审查指出的问题（F06），
            所以登记是**默认行为**，不登记才是需要显式声明的例外。
        run_id: 调用方指定的运行标识（幂等重放用）。为 None 时由 `_register_run` 生成。
    """
    import uuid

    from dsa.agent.state import initial_state

    app = build_graph(checkpointer=checkpointer, use_llm=use_llm)
    tid = thread_id or str(uuid.uuid4())

    rid = run_id
    if register:
        rid = _register_run(question, run_id=run_id) or run_id

    state = initial_state(question, run_id=rid, thread_id=tid)
    config = {"configurable": {"thread_id": tid}}
    final = app.invoke(state, config=config)
    if register:
        _finish_run(rid, final)
    return final


#: 澄清回答的标签映射：把用户答的「值」翻译成问题里能识别的词。
_ANSWER_HINTS = {
    "conversion_rate": "购买率",
    "visit_rate": "访问率",
    "spend_per_customer": "人均消费",
    "primary": "男装邮件",
    "secondary_womens": "女装邮件",
}


def resume_with_answers(
    thread_id: str,
    answers: dict[str, str],
    *,
    checkpointer: Any = None,
    use_llm: bool = False,
    run_id: str | None = None,
) -> dict:
    """用户回答澄清后**继续**执行。

    ⚠️ **实现方式与限制**（必须说清楚，不能让人以为这是原生的暂停/继续）：

    本图在 `clarify` 之后直接到 ``END``，**没有使用 LangGraph 原生的
    ``interrupt``**。因此"继续"不是从暂停点恢复执行栈，而是
    **带着上一轮的澄清记录重新跑一遍** —— 新状态继承
    ``clarifications`` 与 ``answers``，使 `classify` 能利用这些信息。

    为什么这样做而不是用 ``interrupt``：
    - 本图的澄清是**单轮**的（问一次、答一次、再执行），
      不需要保留中间执行栈；
    - 重新跑一遍让**路径完整可见**，比恢复半个执行栈更易审计。

    **但必须承认的代价**：如果将来澄清需要多轮，
    或者 `classify` 之前有昂贵的副作用，重跑就会重复那些代价。
    真要做多轮，应当改用 ``interrupt``。

    缺陷回顾（外部审查 F04）
    ------------------------
    > 复现步骤：输入"女装邮件效果怎么样" → 系统只询问指标 →
    > 调用恢复函数，回答 `metric=conversion_rate` →
    > **恢复后的问题变成"（补充：购买率）"，系统再次询问男装还是女装。**
    > 原因是 `checkpointer.list()` 返回检查点记录，原状态在
    > `checkpoint['channel_values']` 中；代码读取 `getattr(item, 'values', None)`，
    > 得到空值，原问题和澄清记录丢失。

    取状态的逻辑现在收敛到 :func:`dsa.agent.persistence.read_thread_state`
    （三种取法依次尝试），并且**取不到就如实报错**而不是当作"没有上一轮"——
    后者会把一次读取失败伪装成一次全新的提问。

    Args:
        answers: ``{"metric": "conversion_rate", "comparison": "primary"}`` 这类映射。
            值会被翻译成自然语言提示词，拼进新的问题里。

    Returns:
        继续执行后的最终状态；其中 ``answers`` 保留了用户回答以便追溯。

    Raises:
        ValueError: 该 thread 没有可恢复的状态。**不静默重开一轮** ——
            那正是本缺陷的表现形式。
    """
    from dsa.agent.persistence import read_thread_state
    from dsa.agent.state import AgentState, initial_state

    cp = checkpointer
    prev: dict = read_thread_state(cp, thread_id) if cp is not None else {}

    if not prev:
        raise ValueError(
            f"线程 {thread_id} 没有可恢复的状态："
            "要么该会话没有跑过，要么检查点后端读不到它。"
            "**不会以空状态重新开始** —— 那会让用户以为系统记得他答过什么。"
        )

    if prev.get("thread_id") != thread_id:
        raise ValueError("检查点的线程标识与恢复请求不匹配")
    answers={k:str(v).strip() for k,v in answers.items() if str(v).strip()}
    if answers.get("limit"):
        import re
        v=answers["limit"]
        if re.fullmatch(r"\d+",v):
            answers["limit"]="前"+v+"名"
        elif re.fullmatch(r"\d+(?:\.\d+)?[%％]",v):
            answers["limit"]="前"+v
    base_q = str(prev.get("question") or "").strip()
    if answers.get("window"):
        if str(answers["window"]) != "14":
            raise ValueError("本版只支持实验固定的14天窗口")
        from dsa.agent.planning import _DURATION_RE
        base_q = _DURATION_RE.sub("", base_q)
    if answers.get("population") == "all_assigned":
        from dsa.agent.planning import population_conditions
        for condition in population_conditions(base_q):
            base_q = base_q.replace(condition["raw"], "")
    hint_labels = {**_ANSWER_HINTS, "14":"实验后十四天",
                   "spend_per_customer":"人均消费", "all_assigned":"全部分配客户"}
    hints = [("模型 " + str(v)) if k == "model_key" else hint_labels.get(str(v), str(v))
             for k,v in (answers or {}).items() if str(v).strip()]
    question = base_q + ("（补充：" + "、".join(hints) + "）" if hints else "")

    # 用**同一个** thread_id 继续，并把上一轮的澄清记录与本次回答带上。
    state = initial_state(question, run_id=run_id or prev.get("run_id"),
                          thread_id=thread_id)
    state = AgentState(**{**state,
                          "original_question": prev.get("original_question") or prev.get("question"),
                          "clarifications": prev.get("clarifications") or [],
                          "answers": list(prev.get("answers") or []) + [
                              {"key": k, "value": v} for k, v in (answers or {}).items()
                          ]})

    app = build_graph(checkpointer=cp, use_llm=use_llm)
    final = app.invoke(state, config={"configurable": {"thread_id": thread_id}})
    # 恢复也算一次运行推进：状态要回填，否则等待中的运行永远停在 pending。
    _finish_run(final.get("run_id") or state.get("run_id"), final)
    return final


def awaiting_fields(state: dict) -> list[str]:
    """该状态在等用户补哪些字段（`clarification` 时才非空）。

    工作台据此决定"显示哪几个回答框"。**由状态决定界面，而不是让界面猜**。
    """
    if not is_awaiting_user(state):
        return []
    if state.get("missing"):
        return list(dict.fromkeys(state["missing"]))
    answered = {a.get("key") for a in (state.get("answers") or [])}
    out: list[str] = []
    for c in state.get("clarifications") or []:
        key = c.get("missing_key")
        if key and key not in answered and key not in out:
            out.append(key)
    return out


def is_awaiting_user(state: dict) -> bool:
    """该状态是否在**等待用户回答**。

    蓝图要求「**不把等待当成失败，也不默认同意**」。
    因此需要与"失败"明确区分：

    - 等待用户：``category == clarification`` 且已提出问题（``report`` 非空）
    - 失败：有 ``errors`` 且没有可执行的下一步

    混为一谈会导致两种错误处置：把等待当失败（提前放弃），
    或把失败当等待（无限追问）。
    """
    cat = state.get("category")
    has_question = bool((state.get("report") or "").strip())
    return cat == "clarification" and has_question and not state.get("blocked")
