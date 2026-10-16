"""Business boundaries must not mistake relative improvement for a profitable action."""
import copy,json,math
import pytest
from dsa.research_portal import decision_scenario,load_catalog,load_exploration_ledger,main,render_decision

@pytest.fixture
def policy():
    # Hand-calculable: 20 added buyers, CI [10,30] per 10,000; random gain 10.
    return {'id':'fixture','label':'测试方案','role':'primary_frozen','fraction':.1,
      'value_vs_none':{'estimate':.002,'ci_low':.001,'ci_high':.003,'alpha':.05},
      'gain_vs_random':{'estimate':.001,'ci_low':-.0005,'ci_high':.0025,'alpha':.05}}

def test_positive_increment_can_require_no_contact_under_cost_cap(policy):
    result=decision_scenario(policy,margin=3,contact_cost=.15)
    assert result['net_vs_none']['estimate']==pytest.approx(-90)
    assert result['net_vs_none']['ci_low']==pytest.approx(-120)
    assert result['net_vs_none']['ci_high']==pytest.approx(-60)
    assert result['net_vs_random']['estimate']==pytest.approx(30)
    assert result['action']=='do_not_contact'
    assert '不投放' in render_decision(result)

def test_mandatory_quota_cannot_be_silently_changed_to_no_contact(policy):
    r=decision_scenario(policy,margin=3,contact_cost=.15,budget_mode='fixed')
    assert r['action']=='escalate_mandatory_budget'
    assert r['budget_mode']=='fixed'
    assert r['contacts']==1000

def test_same_budget_contact_fee_cancels_only_in_relative_comparison(policy):
    a=decision_scenario(policy,contact_cost=0,maintenance_cost=10)
    b=decision_scenario(policy,contact_cost=.15,maintenance_cost=10)
    assert a['net_vs_random']==b['net_vs_random']
    assert b['net_vs_none']['estimate']==pytest.approx(a['net_vs_none']['estimate']-150)
    assert b['net_vs_random']['estimate']==pytest.approx(20)

def test_profitability_uncertainty_does_not_trigger_rollout(policy):
    r=decision_scenario(policy,margin=10,contact_cost=.15)
    assert r['net_vs_none']['ci_low']<0<r['net_vs_none']['ci_high']
    assert r['action']=='collect_economic_evidence'
    rich=decision_scenario(policy,margin=30,contact_cost=.15)
    assert rich['action']=='candidate_for_policy_trial'
    assert '上线' not in rich['action']

@pytest.mark.parametrize('changes',[{'margin':-1},{'margin':float('nan')},{'contact_cost':float('inf')},{'maintenance_cost':-1},{'population':0},{'population':1.5},{'budget_mode':'best_p_value'}])
def test_invalid_economic_assumptions_are_rejected(policy,changes):
    with pytest.raises(ValueError):decision_scenario(policy,**changes)

def test_bad_evidence_is_rejected_instead_of_displaying_a_profit(policy):
    invalid=copy.deepcopy(policy);invalid['value_vs_none']['ci_low']=.01
    with pytest.raises(ValueError):decision_scenario(invalid)
    invalid=copy.deepcopy(policy);invalid['fraction']=1.5
    with pytest.raises(ValueError):decision_scenario(invalid)

def test_frozen_resource_loads_without_original_reports_or_database(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    c=load_catalog()
    assert {'hillstrom','retailhero','criteo','starbucks'}=={s['id'] for s in c['studies']}
    s=next(s for s in c['studies'] if s['id']=='starbucks')
    chosen=next(p for p in s['policies'] if p['role']=='primary_frozen')
    r=decision_scenario(chosen)
    assert r['net_vs_none']['estimate']==pytest.approx(-93.08174525688742)
    assert r['action']=='do_not_contact'

def test_ledger_includes_womens_shared_controls_and_no_project_alpha_claim():
    ledger=load_exploration_ledger()
    ids={r['id'] for r in ledger['entries']}
    assert {'hillstrom_mens_expanded','hillstrom_womens_expanded','criteo','starbucks'}<=ids
    w=next(r for r in ledger['entries'] if r['id']=='hillstrom_womens_expanded')
    assert w['independent_replication'] is False
    assert w['executed_comparisons']==110
    assert ledger['project_wide_type1_control'] is False

def test_cli_uses_packaged_aggregate_and_keeps_purchase_as_reference(capsys,monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    assert main(['--study','starbucks','--margin','3','--contact-cost','.15','--json'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['decision']['action']=='do_not_contact'
    assert data['policy']['id']=='shared_s005'
    assert data['policy']['role']=='primary_frozen'
