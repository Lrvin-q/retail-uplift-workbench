"""已登记模型的唯一解析入口；最新登记不等于已批准默认使用。"""
import os

def resolve_model(plan, models):
    scoring_column={"uplift":"uplift", "purchase_probability":"purchase_probability",
                    "control_probability":"mu0", "random":None}.get(plan.get("ranking"))
    candidates=[]
    for m in models or []:
        if m.get("comparison_key") != plan.get("comparison_key"):
            continue
        if (plan.get("dataset_version") or m.get("experiment_key") is not None) and m.get("experiment_key") != plan.get("experiment_key"):
            continue
        if plan.get("dataset_version") and m.get("dataset_version") != plan["dataset_version"]:
            continue
        if plan.get("task_type")=="select_customers" and (plan.get("dataset_version") or "scoring" in m):
            score=m.get("scoring") or {}
            n=score.get("population_n",0)
            if n<=0 or (scoring_column and (score.get("scored_columns") or {}).get(scoring_column)!=n):
                continue
        candidates.append(m)
    candidates.sort(key=lambda m: (str(m.get("trained_at") or ""),str(m.get("model_key") or "")),reverse=True)
    explicit=plan.get("model_key") if plan.get("model_source") in {"explicit","config","registry_unique"} else None
    default=os.environ.get("DSA_DEFAULT_MODEL_KEY") or None
    if not default:
        from dsa.config import get_settings
        default=get_settings().default_model_key
    requested=explicit or default
    if requested:
        if any(m.get("model_key")==requested for m in candidates):
            return requested, plan.get("model_source","explicit") if explicit else "config", []
        return None,None,[{"field":"model_key","code":"E_CAP_001",
                           "message":f"指定模型 {requested} 不匹配实验、数据版本、比较或分数完整性"}]
    if len(candidates)==1:
        return candidates[0]["model_key"],"registry_unique",[]
    if not candidates:
        return None,None,[{"field":"model_key","code":"E_CAP_001",
                           "message":"没有实验、数据版本、比较与分数完整性均匹配的已登记模型"}]
    return None,None,[{"field":"model_key","code":"E_PLAN_001",
                       "message":"有多个合格模型，请指定：" + "、".join(m["model_key"] for m in candidates),
                       "candidates":[m["model_key"] for m in candidates]}]
