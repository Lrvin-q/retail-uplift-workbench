"""基础工作台（Streamlit）。

蓝图要求「展示问题、计划、状态、报告与证据入口」。

**两条设计约束**：
1. **import 时不连数据库** —— Streamlit 会反复重跑脚本，
   在 import 期建立连接会让页面无法加载（惰性初始化，见 `_ensure()`）。
2. **顶部必须显示当前模型模式** —— 蓝图验收条件规定
   「没有真实模型调用证据时，版本只能标记为 mock 演示」，
   所以 mock 状态必须显眼且不可忽略。
"""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from dsa.agent import llm
from dsa.agent.persistence import (
    checkpointer_kind,
    make_checkpointer,
    new_thread_config,
)
from dsa.config import ARTIFACTS_DIR

_APP: Any = None
_CP: Any = None


def _ensure() -> tuple[Any, Any]:
    """惰性构建图与检查点。**不在 import 期执行。**"""
    global _APP, _CP
    if _APP is None:
        from dsa.agent.graph import build_graph

        _CP = make_checkpointer()
        _APP = build_graph(checkpointer=_CP, use_llm=True)
    return _APP, _CP


#: 尚未实现的能力。**必须在界面上显式列出**。
#:
#: 依据《实现蓝图.md》阶段九：「对尚未完成的能力**明确标识**，
#: 避免界面入口承诺未实现功能。」
#:
#: 为什么单列这个常量而不是散在 UI 代码里：
#: 一份"当前不做什么"的清单应当**可被审阅、可被更新**，
#: 并且与蓝图里的未勾选项保持同步。散在 if 分支里就没人会去维护它。
#: 每一项都注明"为什么没做"，而不只是"没做"。
#:
#: ⚠️ 措辞必须区分「**模块存在**」与「**入口接通**」。
#: 原先把受控 NL2SQL 写成"未实施"，而它已经有独立代码与 109 条测试；
#: 同时它确实**没有**接进本页面。两句话都对，但只有一句出现在界面上 ——
#: 于是界面说的和代码里有的不一致。现在写成"模块已实现，产品入口未接通"。
UNIMPLEMENTED: tuple[tuple[str, str], ...] = (
    ("受控 NL2SQL 的自由提问入口",
     "独立模块**已实现**（`src/dsa/nl2sql/`，109 条测试），"
     "但**尚未接入本页面**；页面上的问题走的是规则计划 + 受限工具路径"),
    ("按日期/按小时的效果曲线", "第一版不支持动态时间窗口与实时刷新"),
    ("净收益 / ROI / ROAS", "没有成本数据，无法计算"),
    ("RetailHero 的因果结论", "该数据集的随机化机制无文档，仅开放数据工程与描述性功能"),
    ("LangGraph 原生中断恢复", "澄清可多轮补充；当前使用带记录重跑"),
    ("客户级预测区间", "**刻意不提供**——回顾性模型分数不足以支撑客户级不确定性"),
)


def capability_notice() -> None:
    """在侧栏列出**尚未实现**的能力。"""
    st.warning(
        "⚠️ **当前版本不提供以下能力**（避免界面承诺未实现的功能）\n\n"
        + "\n".join(f"- **{name}** —— {why}" for name, why in UNIMPLEMENTED)
    )


def mode_banner() -> None:
    """模型模式横幅。**mock 必须一眼可见。**"""
    info = llm.provider_info()
    if info["mode"] == "mock":
        st.error(
            "🟡 **MOCK 演示模式**\n\n"
            "本次未调用真实模型；分析由数据库和统计程序计算，"
            "**不代表真实模型能力，不得用于模型评测结论**。\n\n"
            f"（未检测到可用的 {llm.API_KEY_ENV}）"
        )
    else:
        st.success(
            f"🔴 **真实模型**：{info['provider']} / {info['model']}"
        )


def paired_contrast_rows(rows: list[dict]) -> list[dict]:
    """只解释同一重采样下的配对差值区间。"""
    out=[]
    for r in rows:
        lo,hi=r.get("ci_low"),r.get("ci_high")
        conclusion=("未登记差值区间" if lo is None or hi is None else
                    "差值区间大于0" if lo>0 else "差值区间小于0" if hi<0 else
                    "差值区间包含0，证据不足")
        out.append({"策略":r.get("policy"),"参照策略":r.get("reference_policy"),
                    "触达比例":r.get("fraction"),"比较角色":"主要" if r.get("is_primary") else "探索性",
                    "差值":r.get("difference"),"区间下界":lo,"区间上界":hi,"结论":conclusion})
    return out


def model_panel() -> None:
    """模型与策略评估面板。

    蓝图阶段九要求统一页面接入「**模型评估**」。
    这里展示已登记的模型与其策略评估结果，**数据全部来自数据库**，
    不在界面里硬编码任何数字。
    """
    with st.expander("模型与策略评估（来自登记表）", expanded=False):
        try:
            from dsa.config import get_engine
            from dsa.tools.registry import call_tool

            eng = get_engine(readonly=True)
            reg = call_tool("get_model_registry", eng)
            if not reg.ok:
                st.info(f"模型登记不可用：{reg.error.message}")
                return
            models = reg.data.get("models") or []
            if not models:
                st.info("尚无已登记的模型。先跑 `scripts/train_and_evaluate.py`。")
                return

            st.write(f"已登记 **{len(models)}** 个模型")
            for m in models:
                key = m.get("model_key")
                st.markdown(
                    f"**{key}** ｜ 类型 `{m.get('model_type')}` ｜ "
                    f"学习器 `{m.get('base_learner')}` ｜ "
                    f"特征 {len(m.get('feature_columns') or [])} 个 ｜ "
                    f"代码版本 `{str(m.get('code_version') or '')[:8]}`"
                )
                ev = call_tool("get_policy_evaluations", eng, model_key=key)
                if not ev.ok:
                    st.caption(f"策略评估不可用：{ev.error.message}")
                    continue
                rows = ev.data.get("evaluations") or []
                if not rows:
                    st.caption("暂无策略评估结果。")
                    continue
                st.caption(
                    "各策略各触达比例的增量估计与区间。"
                    "策略差异另见下方配对差值区间；`none` 是按定义为 0 的参照点。"
                )
                st.dataframe(
                    [{
                        "策略": r.get("policy"),
                        "触达比例": r.get("fraction"),
                        "触达人数": r.get("n_selected"),
                        "增量估计": r.get("ips_estimate"),
                        "区间下界": r.get("ci_low"),
                        "区间上界": r.get("ci_high"),
                    } for r in rows],
                    use_container_width=True,
                )
                from sqlalchemy import text
                with eng.connect() as conn:
                    contrasts=[dict(row) for row in conn.execute(text(
                        "select c.* from policy_contrasts c join uplift_models m using(model_id) "
                        "where m.model_key=:key order by c.is_primary desc,c.fraction,c.policy"),
                        {"key":key}).mappings()]
                st.caption("策略差值采用配对区间。包含0表示证据不足；探索性比较不作确证结论。")
                st.dataframe(paired_contrast_rows(contrasts),use_container_width=True)
        except Exception as exc:  # noqa: BLE001 —— 页面必须展示失败而非白屏
            st.warning(f"模型面板加载失败：{type(exc).__name__}: {exc}")


def _start_run(question: str) -> str | None:
    """创建业务运行登记。**页面发起的提问必须有非空运行 ID**（审查 F06）。

    这条是审查的验收条件之一：「从页面发起的问题有非空运行 ID；
    重启后可查计划和报告；导出包包含实际产物；重放有新的执行证据和结构化差异结果」。

    登记失败**不阻断分析**，但界面上会显式提示 ——
    静默降级会让用户以为"这次有追溯"，而实际上没有。
    """
    from dsa.agent.graph import _register_run

    return _register_run(question, run_id=None)


def resume_panel(state: dict, cp: Any, tid: str) -> None:
    """澄清回答入口（审查 F04）。

    没有这个入口时，工作台无法完成「提问 → 被追问 → 补充条件 → 完成原任务」
    这条流程 —— 而它是产品可用性最基本的闭环之一。
    审查的验收条件写的是「**用户能从工作台完成这条流程**」。

    界面**由状态决定**：等哪些字段就显示哪几个回答框
    （`awaiting_fields`），而不是列一堆固定选项让用户猜。
    """
    from dsa.agent.graph import awaiting_fields, resume_with_answers

    if state.get("thread_id") != tid:
        st.error("待澄清任务不属于当前会话，请切换到对应会话。")
        return
    fields = awaiting_fields(state)
    if not fields:
        return

    labels = {
        "metric": "关注哪个指标",
        "comparison": "比较哪个邮件组与无邮件组",
        "question": "请描述你要分析的问题",
        "window": "确认结果窗口",
        "denominator": "确认消费指标分母",
        "population": "确认分析人群",
        "limit": "名单人数或比例",
        "model_key": "指定已登记模型",
    }
    options = {
        "metric": {"购买率": "conversion_rate", "访问率": "visit_rate",
                   "人均消费": "spend_per_customer"},
        "comparison": {"男装邮件": "primary", "女装邮件": "secondary_womens"},
        "window": {"实验固定的14天": "14"},
        "denominator": {"全部被分配客户的人均消费": "spend_per_customer"},
        "population": {"全部被分配客户": "all_assigned"},
    }

    st.subheader("需要你补充信息")
    st.info(state.get("report") or "请补充缺失条件。")
    with st.form("clarify_form"):
        answers: dict[str, str] = {}
        cols = st.columns(len(fields))
        for col, field in zip(cols, fields):
            with col:
                if field in options:
                    choice = st.selectbox(labels.get(field, field), list(options[field]))
                    answers[field] = options[field][choice]
                else:
                    answers[field] = st.text_input(labels.get(field, field),
                                                  key=f"clarify_{tid}_{field}")
        submitted = st.form_submit_button("补充并继续", type="primary")

    if not submitted:
        return

    with st.spinner("继续分析…"):
        try:
            final = resume_with_answers(tid, answers, checkpointer=cp, use_llm=True)
        except Exception as exc:  # noqa: BLE001 —— 页面必须展示失败而非白屏
            st.error(f"继续失败：{type(exc).__name__}: {exc}")
            return
    st.session_state["last_state"] = final
    st.rerun()


def run_evidence_panel() -> None:
    """运行证据面板：按 run_id 导出与重放（审查 F06）。

    界面调用的是**命令行同一套函数**（`scripts.export_run.py`），
    不是另写一份逻辑 —— 否则"页面能导出"与"命令行能导出"会各自演化，
    而其中一份迟早是错的。
    """
    from dsa.tools.run_export import cmd_export, cmd_replay

    with st.expander("运行导出与重放（按运行 ID）", expanded=False):
        st.caption(
            "导出 = 打包该运行的全部产物与清单；"
            "效果重算 = 用**登记的参数**重新计算效果，并逐字段比对结果。"
            "重放不是事件回放：它验证「同样配置能否再产生同样结果」。"
        )
        rid = st.text_input("运行 ID", key="evidence_run_id").strip()
        if not rid:
            st.caption("填入上方报告里的运行标识后可用。")
            return
        c1, c2 = st.columns(2)
        if c1.button("导出该运行"):
            code = cmd_export(rid)
            st.caption(f"退出码 {code}"
                       + ("" if code == 0 else "（非 0 表示导出不完整或失败）"))
        if c2.button("重放该运行"):
            code = cmd_replay(rid)
            st.caption(f"退出码 {code}（0 一致 / 2 未执行 / 3 有差异）")
        st.caption(f"产物目录：`{ARTIFACTS_DIR}`")


def _render_result(state: dict, cp: Any, tid: str) -> None:
    cat = state.get("category", "—")
    label = {
        "executable": "✅ 可执行",
        "clarification": "❓ 需要澄清",
        "blocked": "⛔ 已阻断",
        "unsupported": "🚫 超出能力范围",
    }.get(cat, cat)
    st.subheader(f"状态：{label}")

    if state.get("blocked"):
        st.warning(
            "本次分析被阻断，**不输出因果结论**。"
            f"阻断项：{'、'.join(state.get('blocking_codes') or []) or '（见检查明细）'}"
        )

    with st.expander("分析计划", expanded=False):
        st.json(state.get("plan") or {})
        st.json(state.get("request_constraints") or [])

    if state.get("issues"):
        with st.expander("校验问题", expanded=True):
            st.json(state["issues"])

    with st.expander(f"工具调用（{len(state.get('tool_results') or [])} 次）", expanded=False):
        for tr in state.get("tool_results") or []:
            icon = "✅" if tr.get("ok") else "❌"
            st.write(f"{icon} `{tr.get('tool')}` — {tr.get('summary') or ''}")
            if not tr.get("ok") and tr.get("error"):
                st.caption(f"错误码 {tr['error'].get('code')}：{tr['error'].get('message')}")
            for ref in tr.get("refs") or []:
                st.caption(f"📎 证据文件：`artifacts/{ref}`")

    if state.get("checks"):
        with st.expander("数据质量检查", expanded=bool(state.get("blocked"))):
            st.json(state["checks"])

    if state.get("report"):
        st.subheader("报告")
        st.markdown(state["report"])

    st.subheader("证据入口")
    c1, c2, c3 = st.columns(3)
    c1.metric("运行标识", state.get("run_id") or "—")
    c2.metric("会话标识", tid[:8])
    c3.metric("检查点后端", checkpointer_kind())
    if state.get("errors"):
        with st.expander("结构化错误", expanded=True):
            st.json(state["errors"])


def main() -> None:
    st.set_page_config(page_title="零售营销增量决策助手", page_icon="📊", layout="wide")
    st.title("零售营销增量决策助手")
    mode_banner()

    with st.sidebar:
        st.header("会话")
        if "thread_id" not in st.session_state:
            st.session_state["thread_id"] = str(uuid.uuid4())
        st.code(st.session_state["thread_id"], language=None)
        if st.button("新建会话"):
            for key in list(st.session_state):
                if key in {"last_state", "evidence_run_id", "analysis_question"} or key.startswith("clarify_"):
                    del st.session_state[key]
            st.session_state["thread_id"] = str(uuid.uuid4())
            st.rerun()
        st.divider()
        st.caption(
            "本工具输出为**回顾性分析**，不构成对下一轮投放效果的承诺。"
        )
        st.caption("没有成本数据，因此不提供净收益/投资回报类结论。")
        st.divider()
        capability_notice()

    question = st.text_input(
        "请输入分析问题",
        placeholder="例如：男装邮件有没有提高购买率？", key="analysis_question",
    )
    examples = [
        "男装邮件有没有提高购买率？",
        "女装邮件对访问率的影响",
        "只看访问过网站的人的购买率",
        "净收益是多少",
        "看看效果怎么样",
    ]
    st.caption("示例：" + " ｜ ".join(f"`{e}`" for e in examples))

    if st.button("开始分析", type="primary") and question.strip():
        app, cp = _ensure()
        tid = st.session_state["thread_id"]
        from dsa.agent.state import initial_state

        # **先登记运行，再执行**（审查 F06）：
        # 「从页面发起的问题有非空运行 ID」是验收条件；
        # 登记晚于执行会让"跑挂了的运行"没有任何记录。
        run_id = _start_run(question)
        if run_id is None:
            st.warning(
                "⚠️ 本次运行**未能登记**（详见控制台警告），"
                "因此无法按运行 ID 导出与重放。分析本身不受影响。"
            )

        with st.spinner("分析中…"):
            try:
                state = app.invoke(
                    initial_state(question, run_id=run_id, thread_id=tid),
                    config=new_thread_config(tid),
                )
            except Exception as exc:  # noqa: BLE001 —— 页面必须展示失败而非白屏
                st.error(f"分析失败：{type(exc).__name__}: {exc}")
                st.stop()
        # 回填终态：等待用户时保持 pending，完成时写 succeeded/failed/blocked
        from dsa.agent.graph import _finish_run

        _finish_run(run_id, state)
        st.session_state["last_state"] = state
    # 状态展示独立于本次点击了哪个按钮。
    # 有上一轮状态且**正在等待用户**时，给出回答入口（F04）。
    # 除此之外显示提示 —— 不要在没有可恢复任务时展示一个空表单。
    last = st.session_state.get("last_state")
    if last:
        _app2, _cp2 = _ensure()
        tid = st.session_state["thread_id"]
        if last.get("thread_id") == tid:
            _render_result(last, _cp2, tid)
            resume_panel(last, _cp2, tid)
        else:
            st.warning("历史状态与当前会话不匹配，请重新提问。")
    else:
        st.info("点击「开始分析」执行。")

    # 运行导出与重放（F06）—— 与命令行共用同一套函数
    run_evidence_panel()

    # 模型评估面板与问题无关，始终可查（蓝图要求六项接入统一页面）
    model_panel()


if __name__ == "__main__":  # pragma: no cover
    main()
