"""Frozen aggregate evidence and conditional economics; no database or model access.

Values are effects per eligible customer, not per contacted customer.
The supplied intervals condition on the original frozen strategy and assumptions.
"""
from __future__ import annotations

import argparse
import json
import math
from importlib.resources import files


def _resource(name):
    return json.loads(files("dsa.resources").joinpath(name).read_text(encoding="utf-8"))


def load_catalog():
    return _resource("research_catalog.json")


def load_exploration_ledger():
    return _resource("exploration_ledger.json")


def _number(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return value


def _interval(value):
    out = {k: _number(value[k], k, -1) for k in ("estimate", "ci_low", "ci_high")}
    if not -1 <= out["ci_low"] <= out["estimate"] <= out["ci_high"] <= 1:
        raise ValueError("invalid effect interval")
    return out


def _scale(value, factor, offset=0):
    return {k: value[k] * factor - offset for k in ("estimate", "ci_low", "ci_high")}


def decision_scenario(policy, *, margin=3., contact_cost=.15,
                      maintenance_cost=0., budget_mode="cap", population=10000):
    """Transform frozen effects, without selecting models or changing the budget.

    Margin is contribution per additional buying customer; all monetary inputs
    must use the same currency. Maintenance is an extra total cost for this
    population and activity, compared with random contact or no promotion.
    Constant costs and margins are assumptions, not fitted estimates.
    """
    margin = _number(margin, "margin")
    contact_cost = _number(contact_cost, "contact_cost")
    maintenance_cost = _number(maintenance_cost, "maintenance_cost")
    population = _number(population, "population", 1)
    if int(population) != population:
        raise ValueError("population must be an integer")
    if budget_mode not in {"cap", "fixed"}:
        raise ValueError("budget_mode must be cap or fixed")
    q = _number(policy["fraction"], "fraction")
    if not 0 < q <= 1:
        raise ValueError("fraction must be in (0, 1]")
    absolute = _interval(policy["value_vs_none"])
    relative = _interval(policy["gain_vs_random"])
    contacts = population * q
    fee = contacts * contact_cost
    net = _scale(absolute, population * margin, fee + maintenance_cost)
    relative_net = _scale(relative, population * margin, maintenance_cost)
    if net["ci_high"] <= 0:
        action = "do_not_contact" if budget_mode == "cap" else "escalate_mandatory_budget"
    elif net["ci_low"] > 0:
        action = "candidate_for_policy_trial"
    else:
        action = "collect_economic_evidence"
    added = population * absolute["estimate"]
    return {
        "policy_id": policy["id"], "role": policy["role"],
        "population": int(population), "fraction": q, "contacts": contacts,
        "budget_mode": budget_mode, "margin": margin,
        "contact_cost": contact_cost, "maintenance_cost": maintenance_cost,
        "contact_fee": fee,
        "added_buyers_vs_none": _scale(absolute, population),
        "added_buyers_vs_random": _scale(relative, population),
        "net_vs_none": net, "net_vs_random": relative_net,
        "break_even_margin_point": (fee + maintenance_cost) / added if added > 0 else None,
        "action": action,
        "scope": "固定策略区间的成本情景变换；非真实利润，不包含成本估计、训练和跨项目探索的不确定性。",
    }


def render_decision(result):
    messages = {
        "do_not_contact": "触达是上限时：该情景建议不投放；先降低成本或补充毛利证据。",
        "escalate_mandatory_budget": "触达是强制配额时：升级评审、争取调整活动约束；不能擅自取消触达，也不能宣称盈利。",
        "collect_economic_evidence": "经济证据不足：先核实毛利、送达和成本，保留简单方案；不能据此扩大投放。",
        "candidate_for_policy_trial": "该情景支持进入下一轮策略随机实验；仍需检验简单方案、送达与长期风险。",
    }
    return messages[result["action"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="starbucks")
    parser.add_argument("--list", action="store_true", help="list frozen studies")
    parser.add_argument("--ledger", action="store_true", help="show exploration index")
    parser.add_argument("--margin", type=float, default=3.)
    parser.add_argument("--contact-cost", type=float, default=.15)
    parser.add_argument("--maintenance-cost", type=float, default=0.)
    parser.add_argument("--population", type=int, default=10000)
    parser.add_argument("--budget-mode", choices=("cap", "fixed"), default="cap")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    catalog = load_catalog()
    if args.ledger:
        print(json.dumps(load_exploration_ledger(), ensure_ascii=False, indent=2))
        return 0
    if args.list:
        print(json.dumps([{"id": s["id"], "summary": s["summary"]} for s in catalog["studies"]],
                         ensure_ascii=False, indent=2))
        return 0
    study = next((s for s in catalog["studies"] if s["id"] == args.study), None)
    if study is None:
        parser.error("unknown study")
    if not study["policies"]:
        print(json.dumps(study, ensure_ascii=False, indent=2))
        return 0
    policy = next(p for p in study["policies"] if p["id"] == study["primary_policy"])
    try:
        decision = decision_scenario(policy, margin=args.margin, contact_cost=args.contact_cost,
                                     maintenance_cost=args.maintenance_cost,
                                     budget_mode=args.budget_mode, population=args.population)
    except ValueError as exc:
        parser.error(str(exc))
    payload = {"study": study["id"], "policy": policy, "decision": decision,
               "recommendation": render_decision(decision), "source": study["source"]}
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else
          f"{study['label']}\n{render_decision(decision)}\n"
          f"相对不促销的净贡献情景：{decision['net_vs_none']}\n"
          f"相对随机的额外净贡献情景：{decision['net_vs_random']}\n{decision['scope']}")
    return 0


if __name__ == "__main__":
    # Emit a consistent encoding when Windows redirects output through cp1252.
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
