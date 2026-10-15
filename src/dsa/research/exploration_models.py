"""Frozen candidate family for an explicitly exploratory search. No product imports are mutated."""
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from dsa.models.features import build_preprocessor,FEATURE_COLUMNS
from dsa.models.tlearner import fit_t_learner
SEED=20261019

def interaction_design(z,t,scale):
    z=np.asarray(z,float);t=np.asarray(t,float)-.5
    if z.ndim!=2 or t.shape!=(len(z),):raise ValueError("invalid interaction design")
    if scale==0:return np.column_stack([z,t])
    return np.column_stack([z,t,float(scale)*t[:,None]*z])

def preprocessor():
    p=build_preprocessor()
    p.set_params(cat__sparse_output=False,sparse_threshold=0.)
    return p

def hparams(spec):
    return dict(max_iter=100,learning_rate=.05,max_leaf_nodes=7,max_depth=3,
                min_samples_leaf=spec.get("min_samples_leaf",200),l2_regularization=spec.get("l2_regularization",10),
                early_stopping=False,random_state=SEED)

@dataclass
class Candidate:
    kind:str
    pre:object
    learners:object
    scale:float=1.
    def predict(self,frame):
        if self.kind=="t_lr":return self.learners.predict_uplift(frame)
        z=self.pre.transform(frame.loc[:,list(FEATURE_COLUMNS)])
        n=len(z)
        if self.kind in ["t_hgb","t_rf"]:
            return self.learners[1].predict_proba(z)[:,1]-self.learners[0].predict_proba(z)[:,1]
        if self.kind=="s_lr":
            a=self.learners.predict_proba(interaction_design(z,np.zeros(n),self.scale))[:,1]
            b=self.learners.predict_proba(interaction_design(z,np.ones(n),self.scale))[:,1]
            return b-a
        if self.kind=="s_hgb":
            return self.learners.predict_proba(np.column_stack([z,np.ones(n)]))[:,1]-self.learners.predict_proba(np.column_stack([z,np.zeros(n)]))[:,1]
        if self.kind=="x_hgb":
            return .5*(self.learners[0].predict(z)+self.learners[1].predict(z))
        return self.learners.predict(z)

def fit_candidate(frame,spec,dr_target=None):
    kind=spec["kind"]
    if kind=="t_lr":return Candidate(kind,None,fit_t_learner(frame,seed=SEED))
    pre=preprocessor();z=pre.fit_transform(frame.loc[:,list(FEATURE_COLUMNS)])
    t=frame.treatment.to_numpy(bool);y=frame.y.to_numpy(int)
    if kind in ["t_hgb","t_rf"]:
        fits=[]
        for arm in [False,True]:
            model=HistGradientBoostingClassifier(**hparams(spec)) if kind=="t_hgb" else RandomForestClassifier(n_estimators=200,min_samples_leaf=spec["min_samples_leaf"],max_depth=6,max_features=1.,n_jobs=2,random_state=SEED)
            model.fit(z[t==arm],y[t==arm]);fits.append(model)
        return Candidate(kind,pre,fits)
    if kind=="s_lr":
        scale=spec["interaction_scale"]
        model=LogisticRegression(max_iter=2000,C=1.,random_state=SEED).fit(interaction_design(z,t,scale),y)
        return Candidate(kind,pre,model,scale)
    if kind=="s_hgb":
        model=HistGradientBoostingClassifier(**hparams(spec)).fit(np.column_stack([z,t]),y)
        return Candidate(kind,pre,model)
    if kind=="x_hgb":
        nuisance=[]
        for arm in [False,True]:
            m=HistGradientBoostingClassifier(**hparams(spec)).fit(z[t==arm],y[t==arm]);nuisance.append(m)
        d0=nuisance[1].predict_proba(z[~t])[:,1]-y[~t]
        d1=y[t]-nuisance[0].predict_proba(z[t])[:,1]
        fits=[HistGradientBoostingRegressor(**hparams(spec)).fit(z[~t],d0),
              HistGradientBoostingRegressor(**hparams(spec)).fit(z[t],d1)]
        return Candidate(kind,pre,fits)
    if dr_target is None or len(dr_target)!=len(frame):raise ValueError("DR requires inner-cross-fitted training targets")
    if kind=="dr_ridge":model=Ridge(alpha=spec["alpha"])
    elif kind=="dr_hgb":model=HistGradientBoostingRegressor(**hparams(spec))
    elif kind=="dr_rf":model=RandomForestRegressor(n_estimators=200,min_samples_leaf=spec["min_samples_leaf"],max_depth=5,max_features=1.,n_jobs=2,random_state=SEED)
    elif kind=="dr_tree":model=DecisionTreeRegressor(max_depth=spec["max_depth"],min_samples_leaf=spec["min_samples_leaf"],random_state=SEED)
    else:raise ValueError("unregistered candidate "+kind)
    model.fit(z,np.asarray(dr_target,float))
    return Candidate(kind,pre,model)
