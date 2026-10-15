"""Standalone, audited estimators for the exploratory report (no product writes)."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

def _vector(a, name):
    a=np.asarray(a,dtype=float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite nonempty vector")
    return a

def top_mask(scores,fraction,tie_keys=None):
    s=_vector(scores,"scores")
    if not np.isfinite(fraction) or not 0<=fraction<=1: raise ValueError("fraction outside [0,1]")
    mask=np.zeros(len(s),bool)
    keys=np.arange(len(s)) if tie_keys is None else _vector(tie_keys,"tie_keys")
    if len(keys)!=len(s): raise ValueError("tie key length mismatch")
    mask[np.lexsort((keys,-s))[:int(round(float(fraction)*len(s)))]]=True
    return mask

def dr_scores(y,t,mu0,mu1,e=.5):
    y,t,m0,m1=[_vector(a,n) for a,n in zip([y,t,mu0,mu1],["y","t","mu0","mu1"])]
    if not len(y)==len(t)==len(m0)==len(m1): raise ValueError("length mismatch")
    if not np.isin(y,[0,1]).all() or not np.isin(t,[0,1]).all(): raise ValueError("nonbinary y/t")
    e=np.asarray(e,dtype=float)
    if e.ndim>1 or (e.ndim==1 and e.shape!=y.shape): raise ValueError("propensity shape mismatch")
    if not np.isfinite(e).all() or ((e<=0)|(e>=1)).any(): raise ValueError("invalid propensity")
    if ((m0<0)|(m0>1)|(m1<0)|(m1>1)).any(): raise ValueError("invalid probability")
    return m1-m0+t/e*(y-m1)-(1-t)/(1-e)*(y-m0)

def policy_value(gamma,policy):
    g=_vector(gamma,"gamma"); p=_vector(policy,"policy")
    if len(g)!=len(p) or ((p<0)|(p>1)).any(): raise ValueError("invalid policy")
    return float(np.mean(g*p))

def snips(y,t,policy):
    y=np.asarray(y,float);t=np.asarray(t,bool);s=np.asarray(policy,bool)
    if not len(y)==len(t)==len(s): raise ValueError("length mismatch")
    if not s.any(): return 0.
    a=s&t;b=s&~t
    if not a.any() or not b.any(): return float("nan")
    return float(s.mean()*(y[a].mean()-y[b].mean()))

def arm_bootstrap_indices(t,rng):
    t=np.asarray(t,bool)
    return np.concatenate([rng.choice(np.flatnonzero(t==a),int((t==a).sum()),replace=True) for a in [False,True]])

def paired_bootstrap(y,t,scores,fractions,n_bootstrap,seed,mu0=None,mu1=None,tie_by_original_index=True):
    y=np.asarray(y,float);t=np.asarray(t,bool);n=len(y)
    if n<2 or n_bootstrap<2: raise ValueError("too few samples/draws")
    names=[(f"{name}@{q:g}",np.asarray(s,float),q) for name,s in scores.items() for q in fractions]
    out={"snips":{k:np.empty(n_bootstrap) for k,_,_ in names},"aipw":{}}
    use_dr=mu0 is not None and mu1 is not None
    g=dr_scores(y,t,mu0,mu1) if use_dr else None
    if use_dr:
        out["aipw"]={k:np.empty(n_bootstrap) for k,_,_ in names}
        out["aipw"]["ate"]=np.empty(n_bootstrap)
        for q in fractions: out["aipw"][f"expected_random@{q:g}"]=np.empty(n_bootstrap)
    rng=np.random.default_rng(seed)
    for b in range(n_bootstrap):
        idx=rng.integers(0,n,n);yb=y[idx];tb=t[idx]
        if use_dr:
            gb=g[idx];out["aipw"]["ate"][b]=gb.mean()
            for q in fractions: out["aipw"][f"expected_random@{q:g}"][b]=(round(q*n)/n)*gb.mean()
        for key,s,q in names:
            mask=top_mask(s[idx],q,tie_keys=idx if tie_by_original_index else None)
            out["snips"][key][b]=snips(yb,tb,mask)
            if use_dr: out["aipw"][key][b]=np.mean(mask*gb)
    return out

def summarize_draws(point,draws,alpha=.05):
    d=np.asarray(draws,float);f=d[np.isfinite(d)]
    return dict(estimate=float(point),ci_low=float(np.quantile(f,alpha/2)) if len(f) else None,
                ci_high=float(np.quantile(f,1-alpha/2)) if len(f) else None,
                n_defined=int(len(f)),n_undefined=int(len(d)-len(f)),alpha=alpha)

def wald_summary(values,alpha=.05):
    a=_vector(values,"values");m=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a)))
    z=norm.ppf(1-alpha/2)
    return dict(estimate=m,se=se,ci_low=m-z*se,ci_high=m+z*se)

def future_sample_size(p0,delta,power,alpha=.05):
    if not 0<p0<1 or not 0<p0+delta<1 or delta<=0 or not 0<power<1:
        raise ValueError("invalid rates/delta/power")
    effect=abs(proportion_effectsize(p0+delta,p0))
    return int(math.ceil(NormalIndPower().solve_power(effect_size=effect,alpha=alpha,power=power,ratio=1,alternative="two-sided")))
