"""Frozen threshold policies and IID covariance-effect inference.

For a rule pi frozen using development data, Delta = Cov(pi(X), tau(X)).
With known e and independently trained nuisance predictions, Gamma has E[Gamma|X]=tau.
The plug-in Delta uses realized q_n, and IF=(pi-q)*(Gamma-EGamma)-Delta.
This differs from Hillstrom's empirical re-ranking bootstrap. It estimates a published
population contrast for a fixed threshold rule; it does NOT give a strict finite-sample guarantee.
"""
import numpy as np
from scipy.stats import norm
from .research_core import dr_scores

def tie_hash(rowid):
    x=np.asarray(rowid,dtype=np.uint64)+np.uint64(20261021)
    with np.errstate(over="ignore"):
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return x^(x>>np.uint64(31))

def freeze_cut(scores,rowid,q):
    s=np.asarray(scores,float);ids=np.asarray(rowid)
    if len(s)==0 or not np.isfinite(s).all() or not 0<q<1 or len(ids)!=len(s):raise ValueError("invalid cut inputs")
    m=int(round(len(s)*q))
    if not 0<m<len(s):raise ValueError("degenerate development quota")
    h=tie_hash(ids);order=np.lexsort((h,-s));last=order[m-1]
    return dict(score=float(s[last]),tie_hash=int(h[last]),nominal_fraction=float(q),validation_selected=m,validation_n=len(s))

def apply_cut(scores,rowid,cut):
    s=np.asarray(scores,float);h=tie_hash(rowid)
    if len(s)!=len(h) or not np.isfinite(s).all():raise ValueError("invalid scores")
    return (s>cut["score"])|((s==cut["score"])&(h<=np.uint64(cut["tie_hash"])))

def matched_gain(y,t,m0,m1,mask,e=.85,alpha=.05,family=1):
    g=dr_scores(y,t,m0,m1,e=e);p=np.asarray(mask,bool)
    if p.shape!=g.shape or len(g)<2:raise ValueError("mask/sample mismatch")
    q=float(p.mean());ate=float(g.mean());value=float(np.mean(p*g));d=float(np.mean((p-q)*g))
    infl=(p-q)*(g-ate)-d;se=float(np.std(infl,ddof=1)/np.sqrt(len(g)))
    z=float(norm.ppf(1-alpha/2));zj=float(norm.ppf(1-alpha/(2*family)))
    return dict(n=len(g),n_selected=int(p.sum()),actual_fraction=q,ate=ate,policy_value=value,random_value=q*ate,gain=d,se=se,
                ci_low=d-z*se,ci_high=d+z*se,joint_ci_low=d-zj*se,joint_ci_high=d+zj*se,
                nominal_alpha=alpha,family_size=family,critical=zj)

def direct_gain(g,mask_a,mask_b):
    """Direct difference allowing distinct realized fractions, centered cost-matched gains."""
    a=np.asarray(mask_a,float);b=np.asarray(mask_b,float);g=np.asarray(g,float)
    p=a-b;q=float(p.mean());d=float(np.mean((p-q)*g))
    infl=(p-q)*(g-g.mean())-d;se=float(np.std(infl,ddof=1)/np.sqrt(len(g)))
    return dict(estimate=d,se=se,ci_low=d-1.95996398454*se,ci_high=d+1.95996398454*se,
                caveat="difference in random-matched excess gains; actual budgets can differ slightly, not exact equal-budget raw-policy difference")
