"""策略比较报告。

依据：docs/06-建模与策略评估协议.md §5 §6；docs/05-报告结构与异常分类.md §1 §2。

铁律
----
1. **报告中所有数字必须取自传入的结构化结果对象**，本模块不做任何计算、不写死任何数值。
2. **禁止措辞**：净收益、ROI、预算、投资回报、一定有效、证明有效、所以应当投放。
3. 留出集上的评估是**回顾性**的，报告必须显式声明这一点。
4. 没有成本数据，**不得**出现任何经济性结论。
5. **策略之间有没有差异，只能看差值区间**（``contrasts``）。
   "两条边际区间是否重叠"不是差异检验，本模块**不**再据此下任何结论
   （F09：两个策略跑在同一份留出客户上，估计相关）。

用法::

    from dsa.report.policy_report import render_policy_report

    render_policy_report(
        run_id=run_id,
        model_record=record,
        policy_scores=scores,          # list[PolicyScore]
        contrasts=contrasts,           # list[PolicyContrast]（差值区间，判断差异的唯一依据）
        out_path=Path("artifacts/policy_report.md"),
        chart_paths={"comparison": png_path},
    )
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

#: 触达比例集合。与 docs/06-建模与策略评估协议.md §6 一致。
#: 刻意**不从 `dsa.models.policy` 导入**——报告层不应依赖评估层的实现细节。
#: `tests/test_stage4_policy_report.py` 中有一条测试断言两者一致，防止漂移。
DEFAULT_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50)

#: 面向用户的文本中禁止出现的词（协议 §5.4 与 docs/05 措辞禁令）
BANNED_WORDS: tuple[str, ...] = (
    "净收益",
    "ROI",
    "ROAS",
    "投资回报",
    "预算",
    "一定有效",
    "证明有效",
    "所以应当投放",
    "应当立即投放",
    "保证提升",
    "确定有效",
)


class BannedWordingError(ValueError):
    """报告文本中出现禁止措辞。"""


def assert_no_banned_wording(text: str) -> None:
    """扫描文本，出现禁止措辞即抛错。

    这是**程序化防线**：措辞纪律不能只靠人工检查。
    """
    hits = [w for w in BANNED_WORDS if w in text]
    if hits:
        raise BannedWordingError(f"报告文本出现禁止措辞: {hits}")


# ---------------------------------------------------------------- 格式化


def _pct(x: Any, digits: int = 4) -> str:
    """比例 → 百分数。**不臆造缺失值**。"""
    if x is None:
        return "—（未提供）"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—（未提供）"
    if v != v:  # NaN
        return "—（无定义）"
    return f"{v * 100:.{digits}f}%"


def _num(x: Any, digits: int = 6) -> str:
    if x is None:
        return "—（未提供）"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—（未提供）"
    if v != v:
        return "—（无定义）"
    return f"{v:.{digits}f}"


def _ci(low: Any, high: Any, alpha: Any) -> str:
    if low is None or high is None:
        return "—（未提供）"
    try:
        lo, hi, a = float(low), float(high), float(alpha)
    except (TypeError, ValueError):
        return "—（未提供）"
    if lo != lo or hi != hi:
        return "—（无定义）"
    return f"[{lo * 100:+.4f}, {hi * 100:+.4f}] 个百分点（{(1 - a) * 100:.0f}%）"


_POLICY_ZH = {
    "random": "随机选择",
    "purchase_probability": "按合并购买预测选择",
    "control_probability": "按无干预购买概率选择",
    "uplift": "按预测增量选择",
    "all_treat": "全部触达（补充参照）",
    "none": "全部不触达（补充参照）",
}

#: 同一比例下的展示顺序：先固定比例策略，再补充参照。
_POLICY_ORDER = {
    "random": 0,
    "purchase_probability": 1,
    "control_probability": 2,
    "uplift": 3,
    "all_treat": 8,
    "none": 9,
}


def policy_label(p: str) -> str:
    """策略名的中文标签（未知名字原样返回，不臆造）。"""
    return _POLICY_ZH.get(p, p)


def _policy_label(p: str) -> str:
    return policy_label(p)


# ---------------------------------------------------------------- 报告


def render_policy_report(
    *,
    run_id: str,
    model_record: Any,
    policy_scores: Sequence[Any],
    out_path: Path,
    checks: Sequence[Any] = (),
    comparisons: Sequence[Any] = (),
    contrasts: Sequence[Any] = (),
    chart_paths: dict[str, Path] | None = None,
    generated_at: str | None = None,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> Path:
    """生成策略比较报告（Markdown）。

    Args:
        run_id: 运行标识，用于追溯。
        model_record: 模型登记对象（需具备 model_key / model_type / dataset_version /
            experiment_key / comparison_key / feature_columns / base_learner /
            params / split_config / code_version / trained_at 属性）。
        policy_scores: 策略评估结果序列（`PolicyScore`）。
        out_path: 输出路径。
        checks: 数据质量检查结果（可选）。
        comparisons: 统计比较结果（可选，用于「实验效果回顾」章节）。
        contrasts: 策略差值结果序列（`PolicyContrast`）。**判断策略差异的唯一依据**；
            不传时报告会明确写出"本次没有差值区间，因此不给任何策略间比较结论"。
        chart_paths: 图表路径字典。
        generated_at: 生成时间；缺省取当前时间。
        fractions: 触达比例集合（仅用于展示，不参与计算）。

    Returns:
        写入的文件路径。

    Raises:
        BannedWordingError: 生成的文本含禁止措辞。
    """
    ts = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    mr = model_record
    scores = list(policy_scores)
    contrast_rows = list(contrasts)

    L: list[str] = []
    add = L.append

    # ---------------------------------------------------------- 标题
    add(f"# 策略比较报告 · {getattr(mr, 'model_key', '—')}")
    add("")
    add("> 本报告由程序生成，所有数值取自结构化评估结果，未由语言模型生成。")
    add("> **本报告是回顾性评估**：名单与效果均为对已结束实验数据的离线评估，")
    add("> 不能表述为这些历史客户未来实际会发生的购买变化。")
    add("")

    # ---------------------------------------------------------- 1 分析计划
    add("## 1. 分析计划")
    add("")
    add("| 项目 | 内容 |")
    add("| --- | --- |")
    add(f"| 实验 | `{getattr(mr, 'experiment_key', '—')}` |")
    add(f"| 比较 | `{getattr(mr, 'comparison_key', '—')}` |")
    add("| 干预 / 对照 | Mens E-Mail / No E-Mail |")
    add("| 结果 | `conversion`（实验后两周内是否购买） |")
    add("| 估计目标 | 条件平均干预效果 CATE：τ(x) = E[Y(1) − Y(0) | X = x] |")
    add(f"| 触达比例集合 | {', '.join(f'{f:.0%}' for f in fractions)}（**全部报告，不只报最优**） |")
    add("| 评估方法 | 留出集上的自归一化逆概率加权（SNIPS） |")
    add("| 条件化分配概率 | e(x) = 0.5（由三组等概率条件化推导，与 x 无关；**不是**策略打分） |")
    add("| 差异判据 | **同一次共同 bootstrap 的配对差值区间**（不是边际区间是否重叠） |")
    add("| 主要比较 | 触达比例与对照策略在评估代码里预先固定；其余比较为探索性 |")
    add("")
    add("### 1.1 三类概率的分工（策略名必须如实）")
    add("")
    add("| 量 | 含义 | 来源 |")
    add("| --- | --- | --- |")
    add("| 分配倾向概率 | P(被分到干预组 \\| x)，真值 0.5 且不随 x 变化 | **不来自模型**（实验设计） |")
    add("| 无干预购买概率 | P(购买 \\| 对照, x) | T-Learner 的 mu0 → 策略 `control_probability` |")
    add("| 合并购买预测分数 | P(购买 \\| x)，跨组合并训练 | 合并结果模型 → 策略 `purchase_probability` |")
    add("")
    add("### 1.2 明确不做的事")
    add("")
    add("- **不估计个体真实增量**：每名客户只观测到一个干预结果，个体增量标签原理上不可得。")
    add("- **不给客户级预测分数附置信区间**：灵活模型不自动提供有效的条件效果区间。")
    add("- **不做经济性结论**：本项目没有成本与毛利数据。")
    add("")

    # ---------------------------------------------------------- 2 版本
    add("## 2. 数据与模型版本")
    add("")
    add("| 项目 | 值 |")
    add("| --- | --- |")
    add(f"| 运行标识 `run_id` | `{run_id}` |")
    add(f"| 数据版本 | `{getattr(mr, 'dataset_version', '—')}` |")
    add(f"| 模型标识 | `{getattr(mr, 'model_key', '—')}` |")
    add(f"| 模型类型 | `{getattr(mr, 'model_type', '—')}` |")
    add(f"| 基础学习器 | `{getattr(mr, 'base_learner', '—')}` |")
    add(f"| 代码版本 | `{getattr(mr, 'code_version', '—')}` |")
    fp = getattr(mr, "identity_fingerprint", None)
    if fp:
        add(f"| 身份指纹 | `{fp}` |")
    add(f"| 训练时间 | {getattr(mr, 'trained_at', '—')} |")
    feats = getattr(mr, "feature_columns", ()) or ()
    add(f"| 特征列（{len(feats)} 个） | {', '.join(f'`{f}`' for f in feats)} |")
    add(f"| 报告生成时间 | {ts} |")
    add("")

    split = getattr(mr, "split_config", None) or {}
    if split:
        add("### 2.1 数据划分")
        add("")
        add("| 集合 | 用途 |")
        add("| --- | --- |")
        add("| 训练集 | 拟合预处理与模型 |")
        add("| 验证集 | 选择模型与策略参数 |")
        add("| **最终留出集** | **只用于最终评估，不参与任何选择** |")
        add("")
        add("划分配置：")
        add("")
        add("```json")
        import json as _json

        add(_json.dumps(split, ensure_ascii=False, indent=2, default=str))
        add("```")
        add("")

    # ---------------------------------------------------------- 3 质量检查
    add("## 3. 数据质量检查与阻断项")
    add("")
    if checks:
        add("| 异常码 | 名称 | 级别 | 结果 | 说明 |")
        add("| --- | --- | --- | --- | --- |")
        for c in checks:
            mark = "通过" if getattr(c, "passed", True) else "**未通过**"
            add(
                f"| `{getattr(c, 'code', '—')}` | {getattr(c, 'name', '—')} | "
                f"{getattr(c, 'severity', '—')} | {mark} | {getattr(c, 'detail', '')} |"
            )
        add("")
    else:
        add("本次未传入质量检查结果。")
        add("")

    # ---------------------------------------------------------- 4 策略比较
    add("## 4. 策略比较（相同触达比例下）")
    add("")
    add("**增量估计为总体增量（绝对百分点）**，区间由留出集 bootstrap 给出。")
    add("`all_treat` 与 `none` 的资源消耗与其他策略不同，仅作补充参照，不参与同比例排名。")
    add("")
    add("> ⚠️ 判断策略之间**有没有差异**，只看第 4.x 节的**差值区间**。")
    add("> 两条边际区间是否重叠**不是**差异检验：两个策略跑在同一份留出客户上，")
    add("> 两个估计是相关的，差值的不确定性必须直接算（见 docs/06 §5.6）。")
    add("")

    by_fraction: dict[float, list[Any]] = {}
    for s in scores:
        by_fraction.setdefault(float(getattr(s, "fraction", 0.0)), []).append(s)

    contrasts_by_fraction: dict[float, list[Any]] = {}
    for c in contrast_rows:
        contrasts_by_fraction.setdefault(float(getattr(c, "fraction", 0.0)), []).append(c)

    for frac in sorted(by_fraction, reverse=True):
        add(f"### 4.{len(by_fraction) - list(sorted(by_fraction, reverse=True)).index(frac)} 触达比例 {frac:.0%}")
        add("")
        add("| 策略 | 触达人数 | 增量估计 | 区间 | 选取人群实际购买率（描述性，有偏） |")
        add("| --- | --- | --- | --- | --- |")
        for s in sorted(by_fraction[frac], key=lambda x: _POLICY_ORDER.get(getattr(x, "policy", ""), 5)):
            pol = getattr(s, "policy", "—")
            add(
                f"| {_policy_label(pol)} | {getattr(s, 'n_selected', '—'):,} | "
                f"{_pct(getattr(s, 'ips_estimate', None))} | "
                f"{_ci(getattr(s, 'ci_low', None), getattr(s, 'ci_high', None), getattr(s, 'alpha', 0.05))} | "
                f"{_pct(getattr(s, 'observed_rate_selected', None))} |"
            )
        add("")
        notes = {getattr(s, "note", "") for s in by_fraction[frac] if getattr(s, "note", "")}
        for n in sorted(notes):
            add(f"> {n}")
        add("")

        add(f"#### 4.{len(by_fraction) - list(sorted(by_fraction, reverse=True)).index(frac)}.1 "
            f"差值区间（{frac:.0%} 触达；差异的唯一判据）")
        add("")
        rows_here = contrasts_by_fraction.get(frac, [])
        if not rows_here:
            add("本次未提供该比例的差值区间，因此**不对该比例做任何策略间比较**。")
            add("")
            continue
        add("| 比较（差值 = 前者 − 后者） | 差值 | 差值区间 | 定性 | 判定 |")
        add("| --- | --- | --- | --- | --- |")
        for c in sorted(rows_here, key=lambda x: (not bool(getattr(x, "is_primary", False)),
                                                  getattr(x, "policy", ""),
                                                  getattr(x, "reference_policy", ""))):
            a = _policy_label(getattr(c, "policy", "—"))
            b = _policy_label(getattr(c, "reference_policy", "—"))
            kind = "**主要比较**" if getattr(c, "is_primary", False) else "探索性"
            diff = getattr(c, "difference", None)
            lo = getattr(c, "ci_low", None)
            hi = getattr(c, "ci_high", None)
            try:
                excludes = float(lo) > 0.0 or float(hi) < 0.0
            except (TypeError, ValueError):
                excludes = None
            verdict = (
                "差值区间排除 0：本次评估**有可检出的差别**"
                if excludes
                else (
                    "差值区间包含 0：**当前留出集无法区分**这两个策略"
                    if excludes is False
                    else "差值区间无定义：本次不做结论"
                )
            )
            add(
                f"| {a} − {b} | {_pct(diff)} | "
                f"{_ci(lo, hi, getattr(c, 'alpha', 0.05))} | {kind} | {verdict} |"
            )
        add("")
        add("> 差值来自**同一次共同 bootstrap 的配对抽样**"
            f"（{getattr(rows_here[0], 'n_bootstrap', '—')} 次重抽样，"
            "两个策略共用同一批索引）；独立重抽样会丢掉相关性、把区间做宽。")
        add("> 主要比较的比例与对照在评估代码里**预先固定**；其余为探索性，"
            "**未做多重比较校正**，不能当作已确认的策略差异。")
        add("> 两条边际区间重叠**不代表**没有差异；差值区间包含 0 也不代表两个策略等价。")
        add("")

    # ---------------------------------------------------------- 5 图表
    add("## 5. 图表")
    add("")
    if chart_paths:
        for name, p in chart_paths.items():
            add(f"- {name}：`{p}`")
    else:
        add("本次未提供图表。")
    add("")

    # ---------------------------------------------------------- 6 结论
    add("## 6. 结论")
    add("")

    primary = [c for c in contrast_rows if bool(getattr(c, "is_primary", False))]
    exploratory = [c for c in contrast_rows if not bool(getattr(c, "is_primary", False))]

    add("### 6.1 统计证据（只看差值区间）")
    add("")
    if not contrast_rows:
        add("本次**没有**传入差值区间。策略之间有没有差异**尚未完成正式推断**，")
        add("因此本节不给任何策略间比较结论——尤其不能因为两条边际区间重叠就说没有差异。")
        add("")
    else:
        if primary:
            c = primary[0]
            a = _policy_label(getattr(c, "policy", "—"))
            b = _policy_label(getattr(c, "reference_policy", "—"))
            add(
                f"**主要比较（预先固定）**：触达 {float(getattr(c, 'fraction', 0.0)):.0%}，"
                f"差值 = {a} − {b} = {_pct(getattr(c, 'difference', None))}，"
                f"差值区间 {_ci(getattr(c, 'ci_low', None), getattr(c, 'ci_high', None), getattr(c, 'alpha', 0.05))}。"
            )
            add("")
            add(f"> {getattr(c, 'note', '')}")
            add("")
        add(
            f"其余 {len(exploratory)} 条比较为**探索性**：未做多重比较校正，"
            "结论只能当作下一步的假设。"
        )
        add("")

    # 只陈述差值区间能支持的事实：有几条排除 0、有几条包含 0。**不给策略排名。**
    decided = [
        c for c in contrast_rows
        if None not in (getattr(c, "ci_low", None), getattr(c, "ci_high", None))
        and not (float(getattr(c, "ci_low")) != float(getattr(c, "ci_low")))
    ]
    excluding = [
        c for c in decided
        if float(getattr(c, "ci_low")) > 0.0 or float(getattr(c, "ci_high")) < 0.0
    ]
    if decided:
        add(
            f"本次共 {len(decided)} 条差值区间有定义：其中 {len(excluding)} 条排除 0，"
            f"{len(decided) - len(excluding)} 条包含 0（**包含 0 表示当前留出集无法区分**，"
            "既不表示两个策略相同，也不表示哪一个更优）。"
        )
        add("")
        add("以上都是**回顾性单次评估**的结果，不构成对下一轮效果的承诺；")
        add("本项目**不做**策略推荐，也不给经济性结论。")
        add("")

    add("### 6.2 效果大小")
    add("")
    add("增量估计以**绝对百分点**给出（见第 4 章表格），全部触达比例均报告，未做挑选。")
    add("")
    add("### 6.3 业务价值")
    add("")
    add("**本项目没有成本与毛利数据，因此不提供任何经济性结论或资源分配建议。**")
    add("")
    add("策略比较只回答「在相同触达比例下，各策略的总体增量估计**能否被区分**」。")
    add("若要讨论经济性，需要补充成本、毛利与触达约束等信息后另行建模。")
    add("")
    add("### 6.4 数据可信度")
    add("")
    add("- 数据来自**随机分配**的实验，条件化到二组后倾向得分仍为常数 0.5，重叠条件成立。")
    add("- 有效性依赖 SUTVA（无干扰、无多版本干预），这是**假设**，无法用数据验证。")
    add("- 策略评估在**固定留出集**上进行，留出集未参与模型与策略参数选择。")
    add("- 策略名与打分来源一一对应（分配倾向概率 / 无干预购买概率 / 合并购买预测分数"
        "三者分开，见 docs/06 §5.1）。")
    add("- 评估为**回顾性**；公开数据上的结果不能直接视为下一轮的真实效果。")

    # ---------------------------------------------------------- 7 复现
    add("## 7. 复现信息")
    add("")
    add("| 项目 | 值 |")
    add("| --- | --- |")
    add(f"| `run_id` | `{run_id}` |")
    add(f"| 数据版本 | `{getattr(mr, 'dataset_version', '—')}` |")
    add(f"| 模型 | `{getattr(mr, 'model_key', '—')}` |")
    add(f"| 代码版本 | `{getattr(mr, 'code_version', '—')}` |")
    seeds = {getattr(s, "bootstrap_seed", None) for s in scores}
    add(f"| bootstrap 种子 | {', '.join(str(x) for x in sorted(seeds, key=str))} |")
    add("")
    add("追溯路径：`run_id` → `analysis_runs` → 数据版本 / 统计参数 / 代码版本 / 产物路径。")
    add("")

    body = "\n".join(L) + "\n"
    assert_no_banned_wording(body)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    return out_path
