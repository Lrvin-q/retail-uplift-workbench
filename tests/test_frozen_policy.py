import numpy as np, pytest
from dsa.research.frozen_policy import freeze_cut,apply_cut,matched_gain,tie_hash
def test_cut_ties_stable_under_row_reordering():
    scores=np.ones(100);ids=np.arange(100)
    cut=freeze_cut(scores,ids,.1)
    a=apply_cut(scores,ids,cut)
    assert a.sum()==10
    ix=np.arange(100)[::-1]
    assert np.array_equal(a[ix],apply_cut(scores[ix],ids[ix],cut))
def test_frozen_cut_not_test_cohort_reranking():
    cut=freeze_cut([1.,2.,3.,4.],np.arange(4),.5)
    assert apply_cut([9.,10.,11.],np.array([20,21,22]),cut).sum()==3
def test_gain_matches_hand_covariance_and_influence_variance():
    y=np.array([1,0,0,1.]);t=np.array([1,0,1,0]);mask=np.array([1,1,0,0],bool)
    # e=.8 => Gamma=(1.075,.7,-.175,-4.3), m0=.1,m1=.3
    g=np.array([1.075,.7,-.175,-4.3])
    r=matched_gain(y,t,np.full(4,.1),np.full(4,.3),mask,e=.8)
    q=.5;d=np.mean((mask-q)*g);influence=(mask-q)*(g-g.mean())-d
    assert r["gain"]==pytest.approx(d)
    assert r["se"]==pytest.approx(np.std(influence,ddof=1)/2)
    assert r["actual_fraction"]==.5
def test_same_budget_empty_or_all_policy_has_zero_gain():
    for mask in [np.zeros(4,bool),np.ones(4,bool)]:
        r=matched_gain([1,0,0,1],[1,0,1,0],[.1]*4,[.3]*4,mask,e=.85)
        assert r["gain"]==0 and r["se"]==0
def test_reject_posttreatment_nonbinary_outcome():
    with pytest.raises(ValueError):matched_gain([2,0],[1,0],[.1]*2,[.3]*2,[1,0],e=.85)

def test_vector_propensity_uses_customer_specific_probabilities():
    from dsa.research.research_core import dr_scores
    np.testing.assert_allclose(dr_scores([1,0],[1,0],[.1,.1],[.3,.3],e=np.array([.8,.6])),[1.075,.45])
    a=matched_gain([1,0,0,1],[1,0,1,0],[.1]*4,[.3]*4,[1,1,0,0],e=.85)
    b=matched_gain([1,0,0,1],[1,0,1,0],[.1]*4,[.3]*4,[1,1,0,0],e=np.full(4,.85))
    assert a==b
@pytest.mark.parametrize("e", [np.array([.8]),np.array([.8,0.])])
def test_vector_propensity_rejects_wrong_shape_and_positivity(e):
    from dsa.research.research_core import dr_scores
    with pytest.raises(ValueError):dr_scores([1,0],[1,0],[.1,.1],[.3,.3],e=e)
