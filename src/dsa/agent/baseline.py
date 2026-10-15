"""规则基线仅作编排消融：固定读取原始聚合，不承担语义规划。
使用相同工具注册表及故障注入路径；LLM能力对照由baseline_llm提供。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsa.config import get_engine  # noqa: E402


def answer(question: str) -> dict:
    """基线回答：直接取数并原样返回，**不做任何判断**。

    返回结构与 Agent 的 state 保持同形（`category` / `report` / `tool_results`
    / `blocked`），这样同一个 `judge_one` 能判两边的结果 —— 否则评测方式不同，
    比较就不公平。
    """
    from dsa.tools.registry import call_tool

    engine = get_engine(readonly=True)
    results: list[dict[str, Any]] = []

    # 基线不做意图理解，一律按「要整体效果」处理：取三个组的聚合。
    for tool, kwargs in (
        ("query_arm_aggregates", {"experiment_key": "hillstrom_email_v1"}),
        ("run_quality_checks", {"experiment_key": "hillstrom_email_v1",
                                "dataset_version": "hillstrom@2008-03-20"}),
    ):
        r = call_tool(tool, engine, **kwargs)
        results.append(r.to_dict())

    # 基线把数字原样列出来，不管问题问的是什么、也不检查是否该拒绝
    lines = ["（简化基线输出：原始聚合结果）"]
    for tr in results:
        if tr.get("tool") == "query_arm_aggregates" and tr.get("ok"):
            for a in (tr.get("data") or {}).get("arms", []):
                rate = a.get("conversion_rate")
                lines.append(
                    f"{a.get('arm_key')}: n={a.get('n_customers')} "
                    f"converted={a.get('n_converted')} "
                    f"rate={rate:.4%}" if isinstance(rate, (int, float))
                    else f"{a.get('arm_key')}: n={a.get('n_customers')}")

    return {
        # 基线**没有分类能力**，一律报 executable —— 这正是要考察的差距
        "category": "executable",
        "question": question,
        "report": "\n".join(lines),
        "tool_results": results,
        "blocked": False,
        "clarifications": [],
        "errors": [],
        "_baseline": True,
    }
