import numpy as np
import pandas as pd
import pytest
from dsa.research.exploration_models import interaction_design,fit_candidate

def synthetic_frame():
    n=1600;rng=np.random.default_rng(71);t=rng.random(n)<.5;x=rng.random(n)<.5
    y=rng.random(n)<(.05+t*(.04+.40*x))
    return pd.DataFrame(dict(customer_key=[f"toy#{i:06d}" for i in range(n)],arm=np.where(t,"mens","control"),treatment=t,y=y,
       recency=rng.integers(1,13,n),history_segment=np.where(x,"high","low"),history_band=np.where(x,2,1),
       history=np.where(x,300.,50.),mens=rng.random(n)<.5,womens=rng.random(n)<.5,zip_code=["Urban"]*n,
       newbie=x,channel=["Web"]*n))

def test_interaction_design_centered_and_scaled_by_hand():
    x=interaction_design(np.array([[1,2],[3,4]]),np.array([0,1]),.1)
    np.testing.assert_allclose(x,[[1,2,-.5,-.05,-.1],[3,4,.5,.15,.2]])

@pytest.mark.parametrize("spec",[{"kind":"t_lr"},{"kind":"s_lr","interaction_scale":.1},{"kind":"s_hgb","min_samples_leaf":50},
 {"kind":"x_hgb","min_samples_leaf":50},{"kind":"dr_ridge","alpha":100}])
def test_candidate_predicts_without_reading_evaluation_labels(spec):
    f=synthetic_frame();g=2*(2*f.treatment.astype(float)-1)*(f.y.astype(float)-.15)
    m=fit_candidate(f,spec,dr_target=g)
    s=m.predict(f)
    changed=f.copy();changed["y"]=~changed.y;changed["treatment"]=~changed.treatment
    np.testing.assert_allclose(s,m.predict(changed),atol=1e-12)
    assert s.shape==(len(f),) and np.isfinite(s).all()
    assert np.mean(s)>0
