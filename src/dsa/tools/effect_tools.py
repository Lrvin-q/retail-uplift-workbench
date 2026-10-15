"""实验效果分析工具（阶段六）。

**这个工具是 F01 的修复本体。**

缺陷回顾（外部审查 F01）
------------------------
> Agent 没有接入核心实验效果计算，报告忽略用户指标。
> 三次返回均无结构化错误，表现为正常完成。……核心 `analyze_experiment()`
> 没有接到报告主链路；工具清单也没有对应的实验效果分析工具。
> **影响：**用户最核心的问题没有被回答。

也就是说：项目**算得出**效果差、置信区间和 p 值（命令行脚本能跑），
但 Agent **取不到**它们，于是报告只能用描述性聚合凑数。
根因不是"报告写得不好"，而是**工具层没有把这个能力暴露出来**。

因此本模块做的事只有一件：把 `dsa.stats.experiment.analyze_experiment`
（**已经独立验证过的那一个函数**）包成受控工具，
使 Agent / API / 工作台走**同一条计算路径**。

为什么不在这里另写一份计算
--------------------------
如果这里重算一遍效果，就出现了第二个计算点：
它可能和命令行脚本的结果不一致，而**没有任何测试会发现**，
因为两边各自都"自洽"。所以本模块**只调用、不计算**。
数值一致性由此成为**结构保证**，而不是靠人去比对。
"""

from __future__ import annotations

from typing import Any

from dsa.stats.effect import (
    CAUSAL_METRICS,
    effect_payload,
    metric_name,
    refuse_reason,
)
from dsa.tools.schemas import ToolErrorCode, ToolResult, error_result

#: 工具名（registry 与文档必须一致）
TOOL_ANALYZE_EFFECT = "analyze_experiment_effect"

#: 默认 bootstrap 次数。
#:
#: 独立脚本默认 10,000 次（离线跑一次，不介意几十秒）。
#: Agent 路径上每一次对话都会触发，必须在**秒级**返回，
#: 所以默认取 2,000 —— 这也是项目登记在案的已验证配置
#: （docs/13 里 `spend_per_customer` 的 bootstrap 区间就是 2,000 次的结果），
#: 因此不是"为了跑得快而临时降低标准"。
DEFAULT_N_BOOT = 2_000

#: 默认随机种子。**固定**，与命令行脚本、登记产物一致，保证可复现。
DEFAULT_SEED = 20261011

#: `n_boot` 上限：超过会拖垮交互响应，且统计收益递减。
MAX_N_BOOT = 20_000


def _bad(tool: str, msg: str, detail: str = "") -> ToolResult:
    return error_result(tool, ToolErrorCode.INVALID_PARAM, msg, detail)


def _no_capability(tool: str, msg: str, detail: str = "") -> ToolResult:
    return error_result(tool, ToolErrorCode.CAPABILITY, msg, detail)


def tool_analyze_experiment_effect(
    engine: Any,
    *,
    experiment_key: str,
    comparison_key: str,
    metrics: list[str] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    bootstrap_seed: int = DEFAULT_SEED,
    run_id: str | None = None,
) -> ToolResult:
    """按指定比较与指标计算实验效果（差、区间、检验）。

    这是**唯一**把统计结果送给 Agent 的入口。返回的 `data` 结构见
    :func:`dsa.stats.effect.effect_payload`：每个指标一条 `effects` 记录，
    含样本量、两组数值、绝对差异（原尺度 **与** 百分点）、置信区间、
    风险比、p 值与所用方法。

    Args:
        engine: **只读**引擎。
        experiment_key: 实验标识（白名单）。
        comparison_key: 预注册比较标识（如 ``primary`` / ``secondary_womens``）。
        metrics: 需要的指标；None 表示全部可做因果比较的指标。
            传入不可做因果比较的指标会被**拒绝**（不是静默忽略），
            理由见 `dsa.stats.effect.REFUSED_AS_CAUSAL`。
        n_boot: bootstrap 次数（仅连续型指标用到）。
        bootstrap_seed: bootstrap 种子，固定以保证可复现。
        run_id: 运行标识。提供时把完整效果载荷写成 JSON 产物并放进 `refs` ——
            **这是运行重放能"比较关键结果"的前提**：重放需要一个
            可读的、与本次运行绑定的历史结果。没有它，重放只能重新算一遍
            而**无从比对**（审查 F06 指出旧重放"没有实际重跑和结果比对"）。

    Returns:
        `ToolResult`。**注意**：分析被阻断（数据质量问题）仍是 `ok=True` ——
        阻断是业务结果，不是工具失败（docs/05 §4.1）。
    """
    tool = TOOL_ANALYZE_EFFECT

    # ---------------------------------------------------------- 1. 参数校验
    # 协议 §4.1：第一行校验参数，不合法立刻返回且**不碰数据库**。
    if not isinstance(experiment_key, str) or not experiment_key.strip():
        return _bad(tool, "experiment_key 必须是非空字符串", f"收到 {experiment_key!r}")
    if not isinstance(comparison_key, str) or not comparison_key.strip():
        return _bad(tool, "comparison_key 必须是非空字符串", f"收到 {comparison_key!r}")
    if metrics is not None and not isinstance(metrics, list):
        return _bad(tool, "metrics 必须是字符串列表或 null", f"收到 {type(metrics).__name__}")
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 100:
        return _bad(tool, "n_boot 必须是不小于 100 的整数", f"收到 {n_boot!r}")
    if n_boot > MAX_N_BOOT:
        return _bad(tool, f"n_boot 超出上限 {MAX_N_BOOT}",
                    "交互路径上过大 bootstrap 次数会让响应不可用")
    if not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool):
        return _bad(tool, "bootstrap_seed 必须是整数", f"收到 {bootstrap_seed!r}")

    from dsa.tools.analysis_tools import KNOWN_EXPERIMENTS, KNOWN_COMPARISONS

    if experiment_key not in KNOWN_EXPERIMENTS:
        return _no_capability(tool, f"未登记的实验：{experiment_key}",
                              f"已登记：{list(KNOWN_EXPERIMENTS)}")
    if comparison_key not in KNOWN_COMPARISONS:
        return _no_capability(tool, f"未登记的预注册比较：{comparison_key}",
                              f"已登记：{list(KNOWN_COMPARISONS)}")

    # 不可做因果比较的指标：**明确拒绝**并说明理由。
    #
    # 为什么不是"过滤掉就算了"：用户问「每个购买者平均花多少，两组差多少」时，
    # 静默丢掉该指标会让他以为"没有这个指标"，而真实原因是
    # **这个问题本身按干预后行为定义了分母**（AGENTS.md §2.2）。
    # 必须把这个区分告诉用户。
    refused: list[dict[str, str]] = []
    wanted: list[str] | None = None
    if metrics:
        wanted = []
        for m in metrics:
            reason = refuse_reason(m)
            if reason:
                refused.append({"metric": m, "reason": reason})
            else:
                wanted.append(m)
        if not wanted:
            return _no_capability(
                tool,
                "所请求的指标都不能做因果比较：" + "；".join(
                    f"{r['metric']}（{r['reason']}）" for r in refused),
                f"可用指标：{list(CAUSAL_METRICS)}",
            )

    # ---------------------------------------------------------- 2. 计算
    import time

    started = time.perf_counter()
    try:
        from dsa.stats.experiment import analyze_experiment

        analysis = analyze_experiment(
            engine,
            experiment_key=experiment_key,
            comparison_key=comparison_key,
            bootstrap_seed=bootstrap_seed,
            n_boot=n_boot,
        )
    except Exception as exc:  # noqa: BLE001 —— 必须分类后返回，不向上抛
        from dsa.tools.analysis_tools import _classify

        code = _classify(exc)
        return error_result(
            tool, code, f"{type(exc).__name__}: {exc}",
            detail=f"experiment_key={experiment_key} comparison_key={comparison_key}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    payload = effect_payload(analysis, metrics=wanted)
    payload["tool"] = tool
    payload["n_boot"] = n_boot
    payload["bootstrap_seed"] = bootstrap_seed
    payload["refused_metrics"] = refused

    names = "、".join(e.get("metric_name_zh") or e["metric"] for e in payload["effects"])
    summary = (
        f"{analysis.comparison_key}：{analysis.treatment_arm} vs {analysis.control_arm}"
        f"（{analysis.window_days} 天窗口）→ {len(payload['effects'])} 项效果量：{names}"
    )
    if refused:
        summary += f"；拒绝 {len(refused)} 项不可做因果比较的指标"

    data = {k: v for k, v in payload.items() if k != "tool"}

    # 把结构化效果写成产物：运行重放需要它作为"历史结果"的可读来源。
    refs: list[str] = []
    if run_id:
        try:
            refs = _write_effect_artifact(data, run_id=run_id,
                                          comparison_key=comparison_key)
        except Exception as exc:  # noqa: BLE001 —— 写产物失败不应让分析整体失败
            print(f"  ⚠️ 效果产物写入失败（不影响分析结果）: {type(exc).__name__}: {exc}")

    return ToolResult(
        ok=True,
        tool=tool,
        data=data,
        refs=refs,
        summary=summary,
        duration_ms=int((time.perf_counter() - started) * 1000),
        run_id=run_id,
    )


def _write_effect_artifact(data: dict[str, Any], *, run_id: str,
                           comparison_key: str) -> list[str]:
    """把效果载荷写成 ``artifacts/<run_id>/effect_<comparison>.json``。

    与 `write_table_artifact` 的约定一致：返回**相对 `artifacts/` 的路径**。
    文件名带比较标识 —— 一次运行可能同时有多个比较，不带就会互相覆盖。
    """
    import json

    from dsa.config import ARTIFACTS_DIR

    target_dir = ARTIFACTS_DIR / str(run_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"effect_{comparison_key}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str),
                    encoding="utf-8")
    return [str(path.relative_to(ARTIFACTS_DIR)).replace("\\", "/")]


#: 可打分人群工具名。
TOOL_SCORING_POPULATION = "get_scoring_population"


def tool_get_scoring_population(engine: Any, *, model_key: str) -> ToolResult:
    """某模型**可以打分的人群**规模与可用打分口径。

    **为什么需要这个工具**：用户按比例要名单（「前 10% 客户」）时，
    比例必须先乘上一个**人数**才能变成 `limit_n`。这个人数是数据事实，
    不能由 LLM 估、也不能在提示词里写死。

    原实现直接固定 `limit_n=100`，于是「前 10%」和「前 20%」
    得到的是同一份名单 —— 审查把这条列为 F02 的一部分。

    本工具把「10% 是 10% 的什么」变成**可核对的两个数字**：
    分母（`population_n`）与由此算出的 `limit_n` 都由程序给出，
    报告只需回显。

    Args:
        model_key: 已登记的模型标识。

    Returns:
        `data` 含 ``population_n``（该模型有分数的客户数）、``scored_columns``
        （实际非空的分数列）、以及各列的均值（供人核对名单来源）。
    """
    tool = TOOL_SCORING_POPULATION
    if not isinstance(model_key, str) or not model_key.strip():
        return _bad(tool, "model_key 必须是非空字符串", f"收到 {model_key!r}")

    from dsa.tools.analysis_tools import _classify, _run

    # 模型是否存在：在 _run 之外判定，避免把「未登记」误报成数据库错误
    try:
        from dsa.models.registry import load_model_record

        if load_model_record(engine, model_key) is None:
            return _no_capability(tool, f"未登记的模型：{model_key}", "")
    except Exception as exc:  # noqa: BLE001
        return error_result(tool, _classify(exc), f"{type(exc).__name__}: {exc}")

    def _q() -> dict:
        from sqlalchemy import text

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    select count(*)                    as population_n,
                           count(cs.uplift)            as n_uplift,
                           count(cs.mu0)               as n_mu0,
                           count(cs.mu1)               as n_mu1,
                           count(cs.propensity_score)  as n_purchase_probability,
                           avg(cs.uplift)              as avg_uplift,
                           avg(cs.propensity_score)    as avg_purchase_probability
                    from customer_scores cs
                    join models m on m.model_id = cs.model_id
                    where m.model_key = :mk
                    """
                ),
                {"mk": model_key},
            ).mappings().one()
        return {
            "model_key": model_key,
            "population_n": int(row["population_n"]),
            "scored_columns": {
                "uplift": int(row["n_uplift"]),
                "mu0": int(row["n_mu0"]),
                "mu1": int(row["n_mu1"]),
                "purchase_probability": int(row["n_purchase_probability"]),
            },
            "averages": {
                "uplift": None if row["avg_uplift"] is None else float(row["avg_uplift"]),
                "purchase_probability": (
                    None if row["avg_purchase_probability"] is None
                    else float(row["avg_purchase_probability"])
                ),
            },
            "summary": (
                f"模型 {model_key} 可打分人群 {int(row['population_n']):,} 人；"
                "按比例取名单时，比例乘的就是这个人数"
            ),
        }

    return _run(tool, engine, _q)


#: 工具名 → 函数（registry.py 合并进 TOOLS）
EFFECT_TOOLS = {
    TOOL_ANALYZE_EFFECT: tool_analyze_experiment_effect,
    TOOL_SCORING_POPULATION: tool_get_scoring_population,
}


__all__ = [
    "DEFAULT_N_BOOT",
    "DEFAULT_SEED",
    "EFFECT_TOOLS",
    "MAX_N_BOOT",
    "TOOL_ANALYZE_EFFECT",
    "TOOL_SCORING_POPULATION",
    "metric_name",
    "tool_analyze_experiment_effect",
    "tool_get_scoring_population",
]
