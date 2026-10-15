"""统一业务终态；历史错误仅用于审计。"""
def execution_outcome(state: dict) -> tuple[str, str]:
    cat = str(state.get("category") or "")
    if state.get("blocked") or cat == "blocked":
        return "refused", "数据质量或口径阻断"
    if cat == "unsupported":
        return "refused", "明确说明能力限制"
    if cat == "clarification":
        if state.get("clarifications") and str(state.get("report") or "").strip():
            return "awaiting_user", "等待补充条件"
        return "failed", "没有可回答的澄清请求"
    if cat != "executable":
        return "failed", "未形成合法业务状态"
    issues = state.get("issues") or []
    if any(i.get("code") == "E_CAP_001" for i in issues):
        return "capability_limited", "缺少适配能力"
    if issues or not state.get("last_execute_ok", True):
        return "failed", "当前校验或工具执行失败"
    if not str(state.get("report") or "").strip():
        return "failed", "缺少最终回答"
    return "succeeded", "已产出回答；正确性仍须检查任务契约"
