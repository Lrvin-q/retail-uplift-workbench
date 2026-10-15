"""确定性核对可识别的报告事实；不宣称能证明任意自然语言。"""
import re

NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?\s*(?:%|％)?")
ALIASES = {
    "conversion_rate": ("购买率", "转化率", "conversion_rate"),
    "visit_rate": ("访问率", "visit_rate"),
    "spend_per_customer": ("人均消费", "人均", "spend_per_customer"),
}

def numeric_tokens(text):
    for m in NUMBER.finditer(text or ""):
        token = m.group().strip()
        core = token.rstrip("%％").replace(",", "").strip()
        try:
            value = float(core)
        except ValueError:
            continue
        decimals = len(core.split(".")[1].split("e")[0].split("E")[0]) if "." in core else 0
        exponent = int(re.split("[eE]", core)[1]) if re.search("[eE]", core) else 0
        yield m, token, value, token.endswith(("%","％")), decimals, exponent

def rounded_match(value, expected, decimals, exponent=0, minimum=2):
    if exponent == 0 and decimals < minimum:
        return False
    tolerance = 0.5 * 10.0 ** (exponent-decimals) * (1+1e-6)
    return abs(value-expected) <= tolerance


def _metric_in(text):
    hits = [m for m, aliases in ALIASES.items() if any(a in text for a in aliases)]
    return hits[0] if len(hits) == 1 else None

def _table_facts(report):
    """Bind supported Markdown columns before numeric checking; never infer by position alone."""
    out, headers, context = [], None, None
    for line in (report or "").splitlines():
        if line.startswith("#"):
            context = _metric_in(line)
            out.append(line)
            headers = None
            continue
        if not line.strip().startswith("|"):
            out.append(line)
            headers = None
            continue
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r"[:\s-]+", c) for c in cells):
            continue
        if any(c in {"组别", "组", "指标", "指标名", "项"} for c in cells):
            headers = cells
            continue
        if not headers or len(headers) != len(cells):
            # Unknown tables stay unverified rather than guessing their column roles.
            continue
        if headers[0] == "项":
            out.append(" ".join(cells))
            continue
        row_metric = _metric_in(cells[0]) or context
        group_column = next((i for i,h in enumerate(headers) if h in {"组别","组"}), None)
        if group_column is not None:
            if _metric_in(" ".join(headers)) is None:
                # Multi-metric descriptive aggregates have independent tool evidence;
                # they cannot be compared against a single effect payload.
                continue
            arm = cells[group_column]
            for i,h in enumerate(headers):
                metric = _metric_in(h)
                if metric is None:
                    continue
                cell = cells[i]
                n = next((cells[j] for j,hj in enumerate(headers)
                          if re.search(r"分配人数|样本量|\bn\b",hj)), None)
                k = next((cells[j] for j,hj in enumerate(headers)
                          if re.search(r"购买人数|成功人数|\bk\b",hj)), None)
                counts = f"（{k} / {n}）" if k is not None and n is not None else f" n={n}" if n else ""
                out.append(f"{arm}{ALIASES[metric][0]} {cell}{counts}")
            continue
        if row_metric is None:
            continue
        label = ALIASES[row_metric][0]
        for h,cell in zip(headers,cells):
            if h in {"指标","指标名"}:
                continue
            if re.search(r"RR.*CI|风险比.*区间",h,re.I):
                out.append(f"{label}风险比置信区间 {cell}")
                continue
            if re.search(r"差值|差异",h) and re.search(r"置信|\bCI\b",h,re.I):
                parts=re.split(r"[（(\[]",cell,maxsplit=1)
                if len(parts)==2 and parts[0].strip():
                    point=parts[0].strip()
                    if re.search(r"百分点|\bpp\b",cell,re.I) and not re.search(r"百分点|\bpp\b",point,re.I):
                        point+=" 个百分点"
                    out.append(f"{label}绝对差异 {point}")
                    out.append(f"{label}置信区间 {cell[len(parts[0]):]}")
                else:
                    out.append(f"{label}置信区间 {cell}")
                continue
            if re.search(r"置信|\bCI\b",h,re.I):
                if not re.search(r"百分点|\bpp\b",cell,re.I) and "百分点" in h:
                    cell += "个百分点"
                out.append(f"{label}置信区间 {cell}")
            elif re.search(r"差值|差异",h):
                if "百分点" in h and not re.search(r"百分点|\bpp\b",cell,re.I):
                    cell += "个百分点"
                out.append(f"{label}绝对差异 {cell}")
            elif re.search(r"相对提升",h):
                out.append(f"{label}相对提升 {cell}")
            elif re.search(r"风险比|\bRR\b",h,re.I):
                out.append(f"{label}风险比 {cell}")
            elif re.fullmatch(r"p\s*值?",h,re.I):
                out.append(f"{label} p={cell}")
            elif re.search(r"无邮件|对照|control|No E-Mail",h,re.I):
                out.append(f"无邮件{label} {cell}")
            elif re.search(r"男装|女装|mens|womens|干预",h,re.I):
                out.append(f"{h}{label} {cell}")
    return "\n".join(out)

def check_report_facts(report, payloads):
    errors, observed = [], set()
    verified = {}
    effects = {}
    for payload in payloads:
        for e in payload.get("effects") or []:
            effects[e.get("metric")] = (payload, e)
    for match in re.finditer(r"(?:结果窗口|实验后)[：:\s]*\**\s*(\d+)\s*\**\s*天",report or ""):
        if any(p.get("window_days") is not None and int(match[1])!=p["window_days"] for p in payloads):
            errors.append({"kind":"window","text":match.group()})
    context = None
    for line in _table_facts(report).splitlines():
        metrics = [metric for metric, aliases in ALIASES.items()
                   if any(alias in line for alias in aliases)]
        if line.startswith("#"):
            context = metrics[0] if len(metrics)==1 else None
            continue
        # Multiple metrics on a prose line must be bound by local clauses.
        # Unknown table columns are still filtered by _table_facts.
        # Converted values and distinct statistics in parentheses have their own units.
        semantic_line = line.replace("**", "").replace("相对量，非百分点","相对量/非百分点")
        semantic_line = re.sub(r"消费为\s*0\s*的占比", "零消费占比", semantic_line)
        semantic_line = re.sub(r"[（(]\s*(?=即|相对提升|相对量|风险比|RR\b)", "；", semantic_line)
        semantic_line = re.sub(
            r"[）)]\s*(?:与|和|及)\s*(?=购买率|转化率|访问率|人均消费|conversion_rate|visit_rate|spend_per_customer)",
            "；", semantic_line)
        zero_scope = False
        last_value_field = None
        interval_context = False
        ratio_scope = False
        clause_context = metrics[0] if len(metrics)==1 else context
        last_metric = None
        for clause in re.split(r"[，。；;、]", semantic_line):
            mentions = [metric for metric, aliases in ALIASES.items()
                        if any(alias in clause for alias in aliases)]
            metric = mentions[0] if len(mentions)==1 else clause_context
            if len(mentions)==1:
                clause_context = metric
            tokens = [t for t in numeric_tokens(clause)
                      if not re.match(r"\s*(?:天|周)(?:\b|购买|访问|消费|窗口|（|$)",clause[t[0].end():])]
            if metric is None and len(effects)==1 and re.search(
                    r"置信区间|\bCI\b|差值|绝对差异|风险比|\bRR\b|相对提升|(?<![A-Za-z])p\s*(?:值|[=<≈≤])",
                    clause,re.I):
                # A statistic-only section has a unique metric only when one was computed.
                metric = next(iter(effects))
            if not tokens or metric is None:
                continue
            if metric != last_metric:
                zero_scope = False
                last_value_field = None
                interval_context = False
                ratio_scope = False
                last_metric = metric
            if re.search(r"风险比|\bRR\b",clause,re.I):
                ratio_scope=True
            elif re.search(r"差值|差异",clause):
                ratio_scope=False
            if re.search(r"零消费占比|零值占比", clause):
                zero_scope = True
            if metric not in effects:
                errors.append({"kind":"metric","text":clause,"metric":metric})
                continue
            payload,e = effects[metric]
            observed.add(metric)
            if re.search(r"下降|降低|减少",clause) and (e.get("difference") or 0)>0:
                errors.append({"kind":"direction","text":clause})
            if re.search(r"提高|增加|上升",clause) and (e.get("difference") or 0)<0:
                errors.append({"kind":"direction","text":clause})
            # Confidence intervals bind to their statistic, not always to the difference.
            interval = re.search(r"置信区间|区间|\bCI\b",clause,re.I)
            ratio_bounds = ratio_scope and re.search(r"\[[^]]*[,~～][^]]*\]",clause)
            if interval or ratio_bounds or interval_context and clause.strip().startswith("即"):
                interval_context = True
                bounds=[t for t in tokens if not (t[3] and t[2] in {90,95,99})]
                fields=("ratio_ci_low","ratio_ci_high") if ratio_scope else ("ci_low","ci_high")
                if len(bounds)>=2 and all(e.get(k) is not None for k in fields):
                    scale=100 if not ratio_scope and re.search(r"百分点|\bpp\b",clause,re.I) else 1
                    for token,field in zip(bounds[-2:],fields):
                        if not rounded_match(token[2],float(e[field])*scale,token[4],token[5]):
                            errors.append({"kind":"interval","statistic":field,"text":clause})
                        else:
                            verified.setdefault(metric,set()).add(field)
                elif len(bounds)>=2:
                    errors.append({"kind":"missing_reference","statistics":fields,"text":clause})
                # A combined point estimate + CI must verify both statistics.
                if not (re.search(r"风险比|\bRR\b|差值|差异",clause,re.I)
                        and len(bounds)>=3):
                    continue
            field = None
            if zero_scope:
                if re.search(r"无邮件|对照|control|No[ -]E[ -]Mail",clause,re.I):
                    field="control_zero_share"
                elif re.search(r"男装|女装|mens|womens|干预",clause,re.I):
                    field="treatment_zero_share"
                elif last_value_field:
                    field=last_value_field.replace("_value","_zero_share")
            elif re.search(r"(?<![A-Za-z])p\s*(?:值|[=<≈≤])",clause,re.I):
                field="p_value"
            elif re.search(r"风险比|\bRR\b|倍",clause,re.I):
                field="ratio"
            elif re.search(r"相对提升|相对量",clause):
                field="relative_lift"
            elif re.search(r"百分点|\bpp\b|绝对差异|差值",clause):
                field="difference"
            elif re.search(r"提升|提高|增加|下降|降低|减少|相对",clause):
                field="relative_lift"
            elif re.search(r"无邮件|对照|control|No[ -]E[ -]Mail",clause,re.I):
                field="control_value"
            elif re.search(r"男装|女装|mens|womens|干预组",clause,re.I):
                arm = "womens" if re.search(r"女装|womens",clause,re.I) else "mens"
                if re.search(r"男装|女装|mens|womens",clause,re.I) and arm != payload.get("treatment_arm"):
                    errors.append({"kind":"arm","text":clause})
                field="treatment_value"
            if field is None or e.get(field) is None:
                continue
            if field in {"treatment_value","control_value"}:
                last_value_field=field
            # Statistical rows contain sample counts after the primary value.
            if field=="p_value":
                p_marker=re.search(r"(?<![A-Za-z])p\s*(?:值)?\s*[=<≤≈]",clause,re.I)
                tokens=[t for t in tokens if not p_marker or t[0].start()>=p_marker.end()]
                if not tokens:continue
            m, literal, value, is_pct, decimals, exponent = tokens[0]
            expected=float(e[field])
            if field=="ratio" and re.search(
                    r"(?:无邮件|对照).*?(?:是|为).*?(?:男装|女装|干预).*?倍",clause):
                expected=1/expected if expected else float("inf")
            after_number=clause[m.end():]
            before_number=clause[:m.start()]
            point_unit=(re.match(r"\s*(?:个?百分点|pp\b)",after_number,re.I)
                        or re.search(r"(?:百分点|\bpp\b)[^\d]*$",before_number,re.I))
            if field=="difference" and point_unit:
                expected*=100
            elif is_pct:
                if (field not in {"treatment_value","control_value","relative_lift",
                                  "treatment_zero_share","control_zero_share"}
                        or field in {"treatment_value","control_value"} and e.get("kind")=="continuous"):
                    errors.append({"kind":"unit","text":clause})
                    continue
                expected*=100
            elif field in {"treatment_value","control_value"} and e.get("kind")=="proportion":
                # Proportions can be stated in raw scale with sufficient precision.
                pass
            if field=="p_value" and "<" in clause[:m.start()]:
                good=expected<value
            else:
                good=rounded_match(value,abs(expected) if re.search(r"下降|降低|减少",clause) else expected,
                                   decimals,exponent,minimum=2)
            if good:
                verified.setdefault(metric,set()).add(field)
            if field in {"treatment_value","control_value"}:
                arm_prefix="treatment" if field=="treatment_value" else "control"
                count_pair=re.search(r"[（(]\s*([\d,]+)\s*/\s*([\d,]+)",clause)
                n_count=re.search(r"\bn\s*=\s*([\d,]+)",clause,re.I)
                if count_pair:
                    for token,key in zip(count_pair.groups(),(arm_prefix+"_k",arm_prefix+"_n")):
                        if e.get(key) is not None and int(token.replace(",",""))!=e[key]:
                            errors.append({"kind":"sample_count","statistic":key,"text":clause})
                elif n_count and e.get(arm_prefix+"_n") is not None:
                    if int(n_count[1].replace(",",""))!=e[arm_prefix+"_n"]:
                        errors.append({"kind":"sample_count","statistic":arm_prefix+"_n","text":clause})
            if not good:
                errors.append({"kind":"number","metric":metric,"statistic":field,
                               "text":clause,"expected":expected,"display":literal})
    return {"passed":not errors,"errors":errors,"observed_metrics":sorted(observed),
            "verified_statistics":{k:sorted(v) for k,v in verified.items()}}
