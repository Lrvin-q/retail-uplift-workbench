import numpy as np
import pytest
from dsa.research.research_core import top_mask,dr_scores,policy_value,snips,paired_bootstrap,arm_bootstrap_indices,future_sample_size

def test_top_mask_stable_ties_and_exact_budget():
    assert top_mask([4,4,2,1],.5).tolist()==[True,True,False,False]
    assert top_mask([1,2,3,4],.25).tolist()==[False,False,False,True]

@pytest.mark.parametrize("scores,q", [([1,float("nan")],.5),([1,2],1.1),([], .5)])
def test_top_mask_rejects_invalid_values(scores,q):
    with pytest.raises(ValueError): top_mask(scores,q)

def test_aipw_hand_calculated_residual_signs():
    g=dr_scores([1,0,0,1],[1,0,1,0],[.1]*4,[.3]*4)
    np.testing.assert_allclose(g,[1.6,.4,-.4,-1.6],atol=1e-14)

def test_aipw_rejects_invalid_propensity_and_posthoc_nonbinary_outcomes():
    with pytest.raises(ValueError): dr_scores([1],[1],[.1],[.2],e=0)
    with pytest.raises(ValueError): dr_scores([2],[1],[.1],[.2])

def test_policy_value_uses_whole_assigned_population_denominator():
    assert policy_value(np.array([1.6,.4,-.4,-1.6]),np.array([1,0,0,0]))==pytest.approx(.4)
    assert policy_value(np.array([1.6,.4,-.4,-1.6]),np.zeros(4))==0

def test_snips_hand_calculation_and_unobserved_arm():
    # selected two arms: (1/2 - 0/2) * (4/8) = .25
    y=np.array([1,0,0,0,0,0,0,0]); t=np.array([1,1,0,0,1,1,0,0])
    assert snips(y,t,np.array([1,1,1,1,0,0,0,0],bool))==pytest.approx(.25)
    assert np.isnan(snips(y,t,t.astype(bool)))

def test_paired_bootstrap_identical_policies_have_exact_zero_difference():
    y=np.tile([1,0,0,1],10); t=np.tile([1,0,1,0],10); score=np.arange(40)
    a=paired_bootstrap(y,t,{"a":score,"b":score},[.5],40,7)
    assert np.nanmax(np.abs(a["snips"]["a@0.5"]-a["snips"]["b@0.5"]))==0
    b=paired_bootstrap(y,t,{"a":score,"b":score},[.5],40,7)
    np.testing.assert_equal(a["snips"]["a@0.5"],b["snips"]["a@0.5"])

def test_training_bootstrap_preserves_arms_not_outcome_counts():
    t=np.array([1,1,0,0,0],bool)
    idx=arm_bootstrap_indices(t,np.random.default_rng(1))
    assert len(idx)==5 and int(t[idx].sum())==2
    assert set(idx)<=set(range(5))

def test_future_sample_size_reasonable_known_two_proportion_example():
    # .10 vs .12, 80%, two-sided .05: approximately 3,840 per arm.
    n=future_sample_size(.1,.02,.8)
    assert 3700<n<4000
    assert future_sample_size(.1,.01,.8)>n*3.5
    with pytest.raises(ValueError): future_sample_size(.1,0,.8)


def test_equal_scores_follow_customer_order_even_when_resampled():
    # Same score; original customer 2 precedes customer 9 despite resample order.
    assert top_mask([.3,.3,.1],1/3,tie_keys=[9,2,4]).tolist()==[False,True,False]
