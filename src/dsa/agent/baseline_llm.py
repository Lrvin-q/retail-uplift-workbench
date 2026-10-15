"""LLM工具选择基线：共享数据、能力范围、工具schema与评分契约。
允许回答、澄清和拒绝；完整系统采用规则计划及程序报告。
规则基线baseline.py仅作消融。比较结果必须按保存的终态独立判分。
"""

from __future__ import annotations

import json
import re
from typing import Any

from dsa.agent import llm

#: 最多允许的工具选择轮数。给基线足够的预算，避免"因为轮数不够而失败"
#: 被误读成"基线能力差"。
MAX_TOOL_ROUNDS = 4

PROTOCOL = """你是通过JSON协议调用工具的问数助手。工具不会自动运行：
当前没有结果时，请输出一个实际工具调用JSON，由运行器执行。
调用格式示例：{"tool":"get_experiment_config","args":{"experiment_key":"hillstrom_email_v1"}}
这只是协议格式示例，实际工具和参数按用户问题及schema自行选择。
拿到相关工具证据后才可answer，不得返回待工具获取的占位报告或虚构零值。
最终格式：{"action":"answer","report":"Markdown字符串"}
澄清格式：{"action":"clarify","missing":["缺失字段"],"report":"缺失说明字符串"}
拒绝格式：{"action":"refuse","reason_code":"unsupported","report":"限制说明字符串"}
只输出一个JSON对象；report必须是非空字符串。人数写整数，比例与百分点至少两位小数，人均金额至少两位小数；p值可科学计数或正确阈值。语义缺失才澄清，
内部实验和版本信息可通过工具查，不把每条能力限制都要求用户确认。"""



def _safe_call_tool(call_tool: Any, tool: str, engine: Any, args: dict) -> Any:
    """调用工具，把**参数错误**变成结构化的 `E_TOOL_001`，不让它抛出。

    为什么需要：基线的工具参数由 LLM 生成，因此它**会**编出不存在的参数。
    实测在冻结验收集上，它给 `get_customer_scores` 传了
    `comparison_key`（那个工具没有这个参数），`TypeError` 直接冒到
    `run_question` 之外，把整轮评测打断。

    这里**不是**在替基线兜底、让它显得更好：
    参数错误的正确处置本来就是"记一条结构化错误并明确失败" ——
    Agent 侧的 `call_tool` 对未知工具就是这么做的，只是它对
    **已知工具 + 错误参数** 会直接抛 `TypeError`。两边应当在同一条规则下比较。
    """
    from dsa.tools.schemas import ToolErrorCode, error_result

    try:
        return call_tool(tool, engine, **args)
    except TypeError as exc:
        return error_result(tool, ToolErrorCode.INVALID_PARAM,
                            f"工具参数不合法：{exc}",
                            detail=f"收到的参数：{sorted(args)}")
    except Exception as exc:  # noqa: BLE001 —— 基线也要能"明确失败"
        return error_result(tool, ToolErrorCode.UNKNOWN,
                            f"{type(exc).__name__}: {exc}")


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def tool_hint() -> str:
    from dsa.tools.registry import list_tools
    from dsa.config import get_settings
    return ("未指定数据集时默认 Hillstrom，与完整系统一致；版本信息可由工具查询，"
            "不要求用户提供内部键值。仅检查当前任务所需字段：效果分析不要求模型或名单大小，"
            "名单任务不要求消费指标或客单价分母。男装/女装单独提及时默认与无邮件比较，"
            "未给窗口用已登记14天，人均消费已定义为全部分配客户消费；"
            "仅客单价等未定义分母才澄清。明确不支持的请求直接 refuse，"
            "说明限制，不以 clarify 要求用户确认换成另一个问题。"
            f"配置默认模型：{get_settings().default_model_key or '无，多个候选需澄清'}。"
            "能力范围：Hillstrom实验hillstrom_email_v1；primary=男装与无邮件，"
            "secondary_womens=女装与无邮件；只支持固定14天；指标conversion_rate/"
            "visit_rate/spend_per_customer，分母全部随机分配客户；渠道、历史消费等"
            "人群过滤和男装女装直接比较不支持。客单价分母不明需澄清。"
            "无成本数据，不支持ROI。名单按已登记模型分数，需人数或比例；"
            "比例应先用get_scoring_population查询规模，不可假定。"
            "模型需匹配实验/数据版本/比较；无唯一模型且无指定时应澄清。"
            "提供完整工具结果证据；最终报告应包含每个请求指标的组值、差值、区间与p值。"
            "工具及完整参数 schema：\n" + json.dumps(list_tools(), ensure_ascii=False)
            + "\n每次输出一个 JSON：调用工具用 tool/args；最终回答用"
              " action=answer/clarify/refuse, report。clarify 必须给 missing 字段列表；"
              "refuse 必须给 reason_code=blocked/unsupported。"
              "可以自行请求实验、指标、数据与模型能力。"
              "不要替换用户要求的比较、窗口、人群和指标。")

def answer(question: str, *, answers: dict | None = None) -> dict:
    """同等工具访问与合法输出接口；不替 LLM 做业务规划或报告回填。"""
    from dsa.config import get_engine
    from dsa.tools.registry import call_tool
    engine=get_engine(readonly=True)
    model=llm.get_llm()
    results,errors,history=[],[],[]
    terminal=None
    for round_index in range(MAX_TOOL_ROUNDS+1):
        prompt=(f"用户问题：{question}\n补充回答：{json.dumps(answers or {},ensure_ascii=False)}\n"
                + tool_hint() + "\n已经执行的完整小型结果或大型产物引用：\n"
                + json.dumps(results,ensure_ascii=False,default=str))
        if round_index == MAX_TOOL_ROUNDS:
            prompt+="\n工具预算已耗尽，请输出最终 action/report/missing/reason_code JSON。"
        try:
            reply=model.invoke([{"role":"system","content":PROTOCOL},
                                {"role":"user","content":prompt}])
            raw=str(getattr(reply,"content","") or "")
            choice=_extract_json(raw)
            history.append(raw)
        except Exception as exc:
            errors.append({"code":"E_TOOL_999","node":"baseline",
                           "message":f"模型调用失败：{type(exc).__name__}"})
            break
        if choice and choice.get("action") in {"answer","clarify","refuse"}:
            if not isinstance(choice.get("report"),str) or not choice["report"].strip():
                errors.append({"code":"E_TOOL_001","node":"baseline",
                               "message":"最终report必须是非空字符串"})
                break
            terminal=choice
            break
        if choice and isinstance(choice.get("tool"),str) and round_index<MAX_TOOL_ROUNDS:
            args=choice.get("args") or {}
            if not isinstance(args,dict):
                errors.append({"code":"E_TOOL_001","node":"baseline","message":"args必须为对象"})
                break
            result=_safe_call_tool(call_tool,choice["tool"],engine,args).to_dict()
            result["params"]=args
            results.append(result)
            continue
        # done=true is a legacy request to generate the final structured response.
        if choice and choice.get("done") and round_index<MAX_TOOL_ROUNDS:
            continue
        errors.append({"code":"E_TOOL_001","node":"baseline",
                       "message":"模型输出不符合工具调用或最终状态schema"})
        break
    terminal=terminal or {}
    action=terminal.get("action")
    category={"answer":"executable","clarify":"clarification",
              "refuse":"blocked" if terminal.get("reason_code")=="blocked" else "unsupported"}.get(action,"")
    missing=terminal.get("missing") or []
    if action=="clarify" and (not isinstance(missing,list) or not missing):
        category=""
        errors.append({"code":"E_TOOL_001","node":"baseline","message":"澄清缺少missing字段"})
    if action=="refuse" and terminal.get("reason_code") not in {"blocked","unsupported"}:
        category=""
        errors.append({"code":"E_TOOL_001","node":"baseline","message":"拒绝缺少合法reason_code"})
    return {"category":category,"question":question,"original_question":question,
            "report":str(terminal.get("report") or ""),"tool_results":results,
            "blocked":category=="blocked","clarifications":[{"missing_key":x} for x in missing],
            "answers":[{"key":k,"value":v} for k,v in (answers or {}).items()],
            "issues":[],"errors":errors,"plan":None,"last_execute_ok":bool(terminal),
            "_baseline":"llm","model_responses":history}
