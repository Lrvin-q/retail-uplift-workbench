import json
import subprocess
import sys
import numpy as np
import pytest
from dsa.portfolio_demo import aggregate_customer_outcomes, run_demo, run_request

# A naive line-level join inflates amounts and may drop customers with no purchase.
def test_sql_customer_grain_keeps_nonbuyers_and_deduplicates_transaction_amounts():
    assigned=[("a",1),("b",1),("c",0),("d",0)]
    # Two product lines from the SAME transaction, each repeats its total amount.
    lines=[("a","tx1","p1",1200),("a","tx1","p2",1200),("c","tx2","p1",500)]
    assert aggregate_customer_outcomes(assigned,lines)==[
        {"customer_id":"a","treatment":1,"purchased":1,"spend_cents":1200},
        {"customer_id":"b","treatment":1,"purchased":0,"spend_cents":0},
        {"customer_id":"c","treatment":0,"purchased":1,"spend_cents":500},
        {"customer_id":"d","treatment":0,"purchased":0,"spend_cents":0}]

def test_conflicting_totals_are_blocked_instead_of_picking_arbitrary_amount():
    with pytest.raises(ValueError,match="conflicting"):
        aggregate_customer_outcomes([("a",1)],[("a","x","p1",100),("a","x","p2",200)])

@pytest.mark.parametrize("question,status",[
    ("男装邮件十四天购买率","answer"),
    ("邮件效果怎么样","clarify"),
    ("男装邮件七天购买率","refuse"),
    ("男装与女装购买率直接比较","refuse"),
    ("只看访问者的男装邮件购买率","refuse"),
    ("女装邮件十四天购买率","refuse"),
    ("男装邮件十四天访问率","refuse"),
    ("按增量选100个客户","refuse"),
])
def test_offline_adapter_never_answers_another_question(question,status):
    hand={"treatment_n":4,"treatment_k":2,"control_n":4,"control_k":1,
          "treatment_rate":.5,"control_rate":.25,"risk_difference":.25,
          "risk_difference_ci":{"low":-.2,"high":.6}}
    result=run_request(question,hand)
    assert result["status"]==status
    if status=="answer":
        assert "25.00" in result["report"]
        assert "synthetic" in result["data_kind"]
    else:
        assert "report" not in result


def test_demo_sql_python_counts_and_holdout_are_consistent_and_reproducible():
    result=run_demo(n=1000,seed=17)
    again=run_demo(n=1000,seed=17)
    assert result==again
    assert result["data_kind"]=="synthetic"
    assert result["counts"]["assigned"]==1000
    assert result["counts"]["sql_python_match"] is True
    split=result["split_ids"]
    sets=[set(split[x]) for x in ("train","valid","test")]
    assert len(set.union(*sets))==1000
    assert not (sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])
    assert result["itt"]["treatment_n"]+result["itt"]["control_n"]==1000
    assert result["model"]["n_train"]==len(sets[0])
    assert result["selection"]["n_selected"]==round(len(sets[2])*.2)
    assert result["workflow"]["status"]=="answer"
    assert result["causal_business_gain_verified"] is False


def test_cli_runs_with_no_private_environment_database_or_api(tmp_path):
    import os
    env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ("DATABASE_URL","DEEPSEEK","PYTHONPATH","DSA_"))}
    env.update(DSA_ENV_FILE=str(tmp_path/"absent.env"),DSA_WORKSPACE_ROOT=str(tmp_path),PYTHONDONTWRITEBYTECODE="1")
    out=tmp_path/"demo"
    p=subprocess.run([sys.executable,"-m","dsa.portfolio_demo","--out",str(out)],cwd=tmp_path,env=env,capture_output=True,text=True,encoding="utf-8",timeout=90)
    assert p.returncode==0,p.stderr
    payload=json.loads((out/"result.json").read_text(encoding="utf-8"))
    assert payload["counts"]["assigned"]==2000
    assert (out/"report.md").is_file()
    assert (out/"workflow_cases.json").is_file()
    assert (out/"report.html").is_file()


def test_demo_cannot_open_network_or_read_database_settings(monkeypatch):
    import socket
    import dsa.config
    def blocked(*args,**kwargs):
        raise AssertionError("offline demo attempted an external dependency")
    monkeypatch.setattr(socket.socket,"connect",blocked)
    monkeypatch.setattr(socket,"create_connection",blocked)
    monkeypatch.setattr(dsa.config,"get_settings",blocked)
    assert run_demo(n=400,seed=37)["counts"]["assigned"]==400


def test_structured_output_has_no_nonstandard_nan_values():
    # The none-policy observed rate is undefined, not a made-up zero or JSON NaN.
    result=run_demo(n=400,seed=37)
    json.dumps(result,allow_nan=False)
    none_policy=next(x for x in result["policy_scores"] if x["policy"]=="none")
    assert none_policy["observed_rate_selected"] is None
