"""Synthetic, offline adapter around the real SQL/statistics/model/planning modules.

No database connector, model API or .env file is read by this entry point.
The production PostgreSQL workflow remains in dsa.agent.graph.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
import html
import json
from pathlib import Path
import sqlite3
from typing import TypedDict

import numpy as np
import pandas as pd
# This entry point is offline even when the caller previously enabled tracing.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
from langgraph.graph import StateGraph, START, END

from dsa.agent.planning import build_plan, classify_rules
from dsa.models.split import make_split, apply_split
from dsa.models.tlearner import fit_t_learner, fit_outcome_model
from dsa.models.policy import evaluate_policies_with_contrasts, random_scores
from dsa.stats.proportions import analyze_comparison

SQL = """WITH transactions AS (
    SELECT customer_id, transaction_id, MAX(total_cents) AS total_cents
    FROM purchase_lines GROUP BY customer_id, transaction_id
), customer_totals AS (
    SELECT customer_id, COUNT(*) AS n_transactions, SUM(total_cents) AS spend_cents
    FROM transactions GROUP BY customer_id
)
SELECT a.customer_id, a.treatment,
       CASE WHEN c.n_transactions > 0 THEN 1 ELSE 0 END AS purchased,
       COALESCE(c.spend_cents, 0) AS spend_cents
FROM assignments a LEFT JOIN customer_totals c USING (customer_id)
ORDER BY a.customer_id"""


def aggregate_customer_outcomes(assignments, lines):
    """Transaction total repeats on each item line; sum once per transaction.

    This is the SYNTHETIC fixture's documented grain, not an assumption about
    arbitrary transaction data. Reject contradictory totals before aggregation.
    """
    with sqlite3.connect(":memory:") as connection:
        connection.executescript("""
        CREATE TABLE assignments(customer_id TEXT PRIMARY KEY, treatment INTEGER NOT NULL CHECK(treatment IN (0,1)));
        CREATE TABLE purchase_lines(customer_id TEXT, transaction_id TEXT, product_id TEXT, total_cents INTEGER NOT NULL CHECK(total_cents >= 0));
        """)
        connection.executemany("INSERT INTO assignments VALUES (?,?)", assignments)
        connection.executemany("INSERT INTO purchase_lines VALUES (?,?,?,?)", lines)
        inconsistent = connection.execute("""SELECT 1 FROM purchase_lines
            GROUP BY customer_id, transaction_id HAVING MIN(total_cents) <> MAX(total_cents) LIMIT 1""").fetchone()
        if inconsistent:
            raise ValueError("conflicting transaction totals")
        unknown = connection.execute("""SELECT 1 FROM purchase_lines p LEFT JOIN assignments a
            USING(customer_id) WHERE a.customer_id IS NULL LIMIT 1""").fetchone()
        if unknown:
            raise ValueError("unassigned purchase customer")
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(SQL)]


class RequestState(TypedDict, total=False):
    question: str
    stats: dict
    status: str
    reason: str
    missing_fields: list[str]
    plan: dict
    report: str
    data_kind: str


def run_request(question, stats):
    """Small LangGraph adapter: real rule parser, synthetic comparison binding.

    It intentionally cannot query production data, issue customer lists, or
    perform archive/checkpoint recovery. Unsupported conditions are not ignored.
    """
    def interpret(state):
        category, missing, reason = classify_rules(state["question"])
        plan = build_plan(state["question"])
        status = {"executable": "answer", "clarification": "clarify"}.get(category.value, "refuse")
        if status == "answer" and (
            plan.task_type != "compare_effect" or plan.comparison_key != "primary"
            or plan.metrics != ["conversion_rate"]
        ):
            status, reason = "refuse", "离线演示仅绑定合成数据的男装主题邮件对无邮件、14天购买率；其他功能见完整产品说明。"
        return {"status": status, "missing_fields": missing, "reason": reason,
                "plan": {**plan.model_dump(mode="json"), "dataset_key": "synthetic_demo",
                         "dataset_source": "demo_binding", "experiment_key": "synthetic_email_demo_v1"},
                "data_kind": "synthetic"}

    def report(state):
        s=state["stats"]; ci=s["risk_difference_ci"]
        return {"report": (
            "合成数据演示（不是 Hillstrom 实测）：以全部分配客户为分母，"
            f"邮件组 {s['treatment_k']}/{s['treatment_n']}，"
            f"无邮件组 {s['control_k']}/{s['control_n']}。"
            f"购买率差 {100*s['risk_difference']:.2f} 个百分点，"
            f"95% 区间 [{100*ci['low']:.2f}, {100*ci['high']:.2f}] 个百分点。"
        )}

    graph=StateGraph(RequestState)
    graph.add_node("interpret",interpret);graph.add_node("render",report)
    graph.add_edge(START,"interpret")
    graph.add_conditional_edges("interpret",lambda s: "render" if s["status"]=="answer" else END)
    graph.add_edge("render",END)
    return graph.compile().invoke({"question": question,"stats": stats})


def _json_safe(value):
    """Represent undefined values as null, never invent zero or serialize NaN."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_demo(*, n=2000, seed=20261015):
    """Generate one known synthetic mechanism; fit on training rows only.

    The fixed 20% quota, seed and learner are for pedagogy, never selected to
    obtain a favorable test result. Validation rows are reserved but not tuned.
    """
    if n < 400:
        raise ValueError("use at least 400 synthetic customers")
    rng=np.random.default_rng(seed)
    frame=pd.DataFrame({
        "customer_id":[f"synthetic-{i:06d}" for i in range(n)],
        "recency":rng.integers(1,13,n), "history":rng.lognormal(4.3,.7,n),
        "history_segment":rng.choice(["1) $0 - $100","2) $100 - $200"],n),
        "history_band":rng.integers(1,3,n), "mens":rng.integers(0,2,n).astype(bool),
        "womens":rng.integers(0,2,n).astype(bool),"newbie":rng.integers(0,2,n).astype(bool),
        "zip_code":rng.choice(["Urban","Rural"],n),"channel":rng.choice(["Phone","Web"],n)})
    t=rng.integers(0,2,n)
    # DGP uses only pre-treatment fields; higher responsiveness in recent customers.
    mu0=.06+.12*(frame.recency.to_numpy()<=4)
    tau=.02+.24*(frame.recency.to_numpy()>7)
    y=(rng.random(n)<mu0+t*tau).astype(int)
    frame["treatment"]=t.astype(bool);frame["arm"]=np.where(t,"mens_email","no_email");frame["y"]=y.astype(bool)
    assigned=list(zip(frame.customer_id,t.tolist()))
    lines=[]
    for i in np.flatnonzero(y):
        # A two-item transaction: total repeated on both lines, not additive.
        amount=int(rng.integers(1000,10001))
        lines.extend([(frame.customer_id.iloc[i],f"tx-{i}","item-1",amount),
                      (frame.customer_id.iloc[i],f"tx-{i}","item-2",amount)])
    rows=aggregate_customer_outcomes(assigned,lines)
    actual=np.array([r["purchased"] for r in rows])
    sql_python_match=bool(np.array_equal(actual,y))
    if not sql_python_match:
        raise AssertionError("SQL and independent Python labels disagree")
    itt=asdict(analyze_comparison(treatment_k=int(y[t==1].sum()),treatment_n=int((t==1).sum()),
          control_k=int(y[t==0].sum()),control_n=int((t==0).sum())))
    split=make_split(frame,seed=seed)
    parts=apply_split(frame,split)
    model=fit_t_learner(parts["train"],seed=seed)
    baseline=fit_outcome_model(parts["train"],seed=seed)
    test=parts["test"]
    scores={"uplift":model.predict_uplift(test),
            "purchase_probability":baseline.predict_proba(test),
            "random":random_scores(len(test),seed=seed)}
    result=evaluate_policies_with_contrasts(y=test.y.to_numpy(int),treatment=test.treatment.to_numpy(bool),
             scores=scores,fractions=(.2,),n_bootstrap=200,bootstrap_seed=seed)
    selected=next(x for x in result.scores if x.policy=="uplift" and x.fraction==.2)
    payload = {
       "schema_version":1,"data_kind":"synthetic","seed":seed,
       "warning":"仅为机制已知的合成数据演示，不是 Hillstrom/Criteo 结果或真实业务收益证明。",
       "causal_business_gain_verified":False,
       "counts":{"assigned":n,"purchase_lines":len(lines),"purchasers":int(y.sum()),"sql_python_match":sql_python_match},
       "itt":itt,"split":split.to_registry_dict(),
       "split_ids":{k:frame.iloc[split.indices(k)].customer_id.tolist() for k in ("train","valid","test")},
       "model":{"kind":"T-Learner LogisticRegression","n_train":model.n_train_treatment+model.n_train_control,
                "validation_use":"reserved; no tuning","test_use":"evaluation only"},
       "selection":{"fraction":.2,"n_selected":selected.n_selected,"n_test":len(test)},
       "policy_scores":[asdict(x) for x in result.scores],
       "paired_contrasts":[asdict(x) for x in result.contrasts],
       "workflow":run_request("男装邮件十四天购买率",itt),
    }
    return _json_safe(payload)


def main():
    parser=argparse.ArgumentParser(description="Offline synthetic demo. No API or PostgreSQL required.")
    parser.add_argument("--out",type=Path,default=Path("artifacts/demo"))
    parser.add_argument("--n",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=20261015)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Output exists. Choose a new --out directory; existing evidence is preserved.")
    result=run_demo(n=args.n,seed=args.seed)
    cases=[run_request(q,result["itt"]) for q in (
        "男装邮件十四天购买率","邮件效果怎么样","男装邮件七天购买率",
        "男装与女装购买率直接比较","只看访问者的男装邮件购买率")]
    # All displayed numbers come from the computed structured payload.
    s=result["itt"]; ci=s["risk_difference_ci"]
    lines=["# 合成营销实验：离线演示","","> "+result["warning"],"",
       "## SQL 与实验口径","",
       f"全部分配客户 {args.n} 人；SQL/Python 结果核对：{result['counts']['sql_python_match']}。",
       result["workflow"]["report"],"",
       "## 固定触达比例的模型比较","",
       "训练/验证/测试互斥；预处理仅在训练集拟合。触达比例固定20%，验证集保留，本演示不调参。",
       "以下为SNIPS策略总体增量及配对差值，单位为百分点，分母是全部测试客户。随机基线是一份固定随机排序。",
       "区间对已训练模型条件化，未包含训练不确定性。","",
       "|策略|选中人数|总体增量估计|95%区间|","|---|---:|---:|---|" ]
    for x in result["policy_scores"]:
        lines.append(f"|{x['policy']}|{x['n_selected']}|{100*x['ips_estimate']:.2f}|[{100*x['ci_low']:.2f}, {100*x['ci_high']:.2f}]|")
    lines.extend(["","|配对比较|差值|95%区间|","|---|---:|---|"])
    for x in result["paired_contrasts"]:
        lines.append(f"|{x['policy']} − {x['reference_policy']}|{100*x['difference']:.2f}|[{100*x['ci_low']:.2f}, {100*x['ci_high']:.2f}]|")
    lines.extend(["","## 请求处理","","|问题|状态|说明|","|---|---|---|"])
    for x in cases:
        lines.append(f"|{x['question']}|{x['status']}|{x.get('reason') or '程序模板生成结果'}|")
    lines.extend(["","## 可复核产物","","`result.json` 为数值与划分来源；`workflow_cases.json` 保存各问题的计划和终态；`customer_outcomes.sql` 为实际执行的 SQLite 查询。",
      "离线适配器不提供真实数据检索、完整数据库归档、名单服务或澄清恢复。完整产品代码见 dsa.agent.graph。"])
    report="\n".join(lines)+"\n"
    args.out.mkdir(parents=True)
    (args.out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (args.out/"workflow_cases.json").write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding="utf-8")
    (args.out/"customer_outcomes.sql").write_text(SQL+";\n",encoding="utf-8")
    (args.out/"report.md").write_text(report,encoding="utf-8")
    (args.out/"report.html").write_text(
       '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>合成实验演示</title>'
       '<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.8 system-ui;background:#f4f7fb;color:#16324f}pre{white-space:pre-wrap;background:white;padding:28px;border-radius:12px}</style>'
       '<h1>营销实验与选人策略 · 合成数据演示</h1><pre>'+html.escape(report)+'</pre></html>',encoding="utf-8")
    print(f"Synthetic demo completed: {args.out.resolve()}")
    print(f"ITT difference: {100*s['risk_difference']:.2f} pp; 95% CI [{100*ci['low']:.2f}, {100*ci['high']:.2f}] pp")


if __name__=="__main__":
    main()
