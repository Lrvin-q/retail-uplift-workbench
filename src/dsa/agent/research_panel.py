"""Streamlit view of immutable aggregate research; never connects services."""
from __future__ import annotations
import streamlit as st
from dsa.research_portal import (
    decision_scenario, load_catalog, load_exploration_ledger, render_decision,
)


def _effect(value):
    return f"{10000*value['estimate']:.2f} [{10000*value['ci_low']:.2f}, {10000*value['ci_high']:.2f}]"


def render_research_panel():
    catalog = load_catalog()
    st.subheader("研究证据与业务决策")
    st.caption("读取随安装包保存的冻结汇总；无需数据库、模型 API 或原始客户文件。")
    studies = {s["id"]: s for s in catalog["studies"]}
    key = st.selectbox("选择研究", list(studies), index=list(studies).index("starbucks"),
                       format_func=lambda k: studies[k]["label"], key="research_study")
    study = studies[key]
    st.info(study["summary"])
    st.caption("证据等级：" + study["evidence_level"] + "；" + study["product_support"])
    if study["policies"]:
        st.markdown("每万名符合条件客户的新增购买客户数；预算固定为 10%，区间为本轮固定策略评价区间。")
        st.dataframe([{
            "方案": p["label"], "角色": "冻结主策略" if p["role"] == "primary_frozen" else "预先指定参照",
            "相对随机：点估计 [95%区间]": _effect(p["gain_vs_random"]),
            "相对不促销：点估计 [95%区间]": _effect(p["value_vs_none"]),
        } for p in study["policies"]], hide_index=True)
        st.warning("专门方法未证实优于购买预测。参照策略不因留出集点估计较高而被重新选为胜者；匿名特征不能解释购买周期。")
        st.markdown("#### 成本情景与行动")
        st.caption("金额统一使用同一种币种；毛利是每位新增购买客户的贡献毛利。维护费是每万客户、每次活动相对参照额外发生的总费用。")
        cols = st.columns(3)
        margin = cols[0].number_input("新增购买客户贡献毛利", min_value=0., value=3., key="research_margin")
        fee = cols[1].number_input("每次触达成本", min_value=0., value=.15, key="research_fee")
        maintenance = cols[2].number_input("额外实施维护总费", min_value=0., value=0., key="research_maintenance")
        mode = st.radio("触达约束", ["可不花满的上限", "强制配额"], key="research_budget_mode")
        results = [decision_scenario(p, margin=margin, contact_cost=fee,
                                     maintenance_cost=maintenance,
                                     budget_mode="fixed" if mode == "强制配额" else "cap")
                   for p in study["policies"]]
        st.dataframe([{
            "方案": p["label"],
            "相对不促销净贡献": round(r["net_vs_none"]["estimate"], 2),
            "相对不促销区间": f"[{r['net_vs_none']['ci_low']:.2f}, {r['net_vs_none']['ci_high']:.2f}]",
            "相对随机额外净贡献": round(r["net_vs_random"]["estimate"], 2),
            "相对随机区间": f"[{r['net_vs_random']['ci_low']:.2f}, {r['net_vs_random']['ci_high']:.2f}]",
        } for p, r in zip(study["policies"], results)], hide_index=True)
        chosen = next(r for r in results if r["role"] == "primary_frozen")
        st.warning(render_decision(chosen))
        st.caption(chosen["scope"] + "共同触达成本仅在同预算比较中抵消；不能在相对不促销时省略。")
        for note in study["assumptions"]:
            st.caption(note)
        with st.expander("与简单方案的配对差值（次要比较，97.5%区间）"):
            st.dataframe([{
                "比较": r["comparison"], "每万客户差值": r["estimate"] * 10000,
                "区间下界": r["ci_low"] * 10000, "区间上界": r["ci_high"] * 10000,
                "Holm p": r["p_holm"],
            } for r in study["secondary"]], hide_index=True)
    source = study["source"]
    st.caption("冻结源文件：" + source["path"])
    st.code("SHA-256: " + source["sha256"], language=None)
    with st.expander("探索过程与证据限制"):
        ledger = load_exploration_ledger()
        st.warning(ledger["project_disclosure"])
        st.caption(ledger["scope"])
        st.caption(ledger["counts_warning"])
        st.dataframe([{"记录": e["id"], "阶段": e["phase"],
                       "说明": e["interpretation"]} for e in ledger["entries"]], hide_index=True)
        st.json(ledger)
    st.caption("下一步：核实毛利、送达、退订与投诉，并在新队列预先固定策略实验。当前入口不训练、不生成投放名单。")
