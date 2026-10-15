"""对**已登记**的模型版本重新做策略评估（F09 的口径迁移路径）。

为什么需要它
------------
策略评估是"训练版本的不可变分数 + 留出集 + 固定评估规则"的函数，
**不需要重新训练**。当评估口径本身变化时（例如本次：策略改名、
把"区间重叠"换成"配对差值区间"），正确做法是拿**库里已存的分数**重算，
而不是换一个模型版本重算——后者会把"口径变更"和"模型变更"两件事混在一起
（AGENTS.md §3.4：口径变更不得靠换数据/换模型实现）。

它同时承担一次**交叉核对**（AGENTS.md §2.6）：

* 从库里已存的 ``uplift`` 列重算 ``uplift`` 策略的每条评估，
  与库里原有的 ``uplift`` 行逐条对比，打印最大绝对差；
* 旧名 ``propensity`` 的行会被改名为 ``control_probability``（sql/006 迁移），
  本模块从 ``mu0`` 重算并核对：若两者一致，说明改名是"把名字改回真实语义"，
  而不是篡改证据。

用法::

    & "D:\\L\\miniconda\\envs\\ai_agent\\python.exe" -m dsa.models.reevaluate tlearner_lr_v1
    & "..." -m dsa.models.reevaluate tlearner_lr_v1 --no-write     # 只核对不落库
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from dsa.config import ARTIFACTS_DIR, get_engine
from dsa.models.policy import (
    CONTRAST_PRIMARY_FRACTION,
    CONTRAST_PRIMARY_POLICY,
    CONTRAST_PRIMARY_REFERENCE,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SEED,
    SCORED_POLICIES,
    TREATMENT_FRACTIONS,
    evaluate_policies_with_contrasts,
    scores_from_frame,
)
from dsa.models.registry import (
    load_model_record,
    load_policy_contrasts,
    load_policy_evaluations,
    save_policy_contrast,
    save_policy_evaluation,
)
from dsa.models.split import load_modeling_frame, make_split

#: 重算结果与库内旧值的最大可接受绝对差。
#:
#: **预先说明**（AGENTS.md §2.6，先定容差再看结果）：
#: 原始评估用的是**内存里的未取整分数**，而重算用的是库里已存的 ``numeric(9,8)``
#: 分数（四舍五入到 8 位小数）。选取规则是"按分数取前 k 名"，边界上 1e-8 的量化
#: 差异可能让个别客户进出名单：
#:
#: * 单名客户进出对 ``V`` 的影响约 ``share · y / k ≈ 1e-6`` 量级；
#: * 每次重抽样最多影响个位数客户，因此对区间端点的累积影响在 ``1e-5`` 量级；
#: * 区间半宽本身约 ``1e-3``（真实留出集），所以 ``1e-4`` 既留足量化余量，
#:   又足以发现任何**口径级**的变化（换错列、漏乘触达比例、换种子等的偏差远大于它）。
CROSS_CHECK_TOLERANCE: float = 1e-4


def _hdr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _load_stored_scores(engine, model_key: str) -> pd.DataFrame:
    """读取某个已登记模型版本的**全部已存分数**（按 customer_key）。"""
    sql = """
    select
        cu.customer_key,
        cs.arm_key,
        cs.mu1,
        cs.mu0,
        cs.uplift,
        cs.propensity_score
    from customer_scores cs
    join customers cu on cu.customer_id = cs.customer_id
    join models    m  on m.model_id      = cs.model_id
    where m.model_key = :model_key
    """
    with engine.connect() as conn:
        frame = pd.read_sql(text(sql), conn, params={"model_key": model_key})
    if frame.empty:
        raise ValueError(
            f"模型 {model_key!r} 没有任何已存分数；请先运行 scripts/train_and_evaluate.py"
        )
    for col in ("mu1", "mu0", "uplift", "propensity_score"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    return frame


def rebuild_holdout(engine, model_key: str):
    """重建该模型版本的建模人群与**最终留出集**（用库里登记的划分配置）。

    复现方式：``load_modeling_frame``（按 customer_key 排序）+ ``make_split(seed=...)``
    ——与训练时同一份代码路径。重建后**核对**人数与各组别人数是否等于
    ``models.split_config`` 里登记的 counts / arm_counts；不一致就报错，
    绝不"差不多就用"。
    """
    record = load_model_record(engine, model_key)
    if record is None:
        raise ValueError(f"模型未登记：{model_key!r}")
    split_cfg = record.split_config or {}
    seed = split_cfg.get("seed")
    if seed is None:
        raise ValueError(
            f"模型 {model_key!r} 的 split_config 没有 seed，无法复现划分（docs/06 §8）"
        )
    df = load_modeling_frame(engine)
    sp = make_split(df, seed=int(seed))

    expected_counts = split_cfg.get("counts")
    if expected_counts:
        actual = sp.sizes()
        if {k: int(v) for k, v in expected_counts.items()} != actual:
            raise ValueError(
                f"重建的划分与登记不一致：登记 {expected_counts}，重建 {actual}。"
                "数据版本或划分代码已经变了，不得用这份划分做评估。"
            )
    expected_arms = split_cfg.get("arm_counts")
    if expected_arms:
        for name, counts in expected_arms.items():
            actual = {k: int(v) for k, v in sp.arm_counts[name].items()}
            if {k: int(v) for k, v in counts.items()} != actual:
                raise ValueError(
                    f"{name} 的各组别人数与登记不一致：登记 {counts}，重建 {actual}"
                )
    return record, df, sp


def resolve_evaluation_config(old_rows: list[dict], n_bootstrap, bootstrap_seed) -> tuple[int, int]:
    """决定重算用哪套 bootstrap 配置。

    优先用**库里已登记**的那套（``policy_evaluations`` 每行都记着 ``bootstrap_seed``
    与 ``n_bootstrap``）：只有配置相同，重算值才能与旧值逐字段比较。
    库里存在多套配置而调用方没有显式指定时**报错**，不猜。
    """
    pairs = {(int(r["bootstrap_seed"]), int(r["n_bootstrap"])) for r in old_rows}
    if n_bootstrap is not None and bootstrap_seed is not None:
        return int(bootstrap_seed), int(n_bootstrap)
    if len(pairs) == 1:
        seed, n_boot = next(iter(pairs))
        return (int(bootstrap_seed) if bootstrap_seed is not None else seed,
                int(n_bootstrap) if n_bootstrap is not None else n_boot)
    if not pairs:
        return (int(bootstrap_seed or DEFAULT_SEED), int(n_bootstrap or DEFAULT_N_BOOTSTRAP))
    raise ValueError(
        f"库里的评估用了多套 bootstrap 配置 {sorted(pairs)}；"
        "请显式传 --seed / --n-bootstrap，避免拿不同配置的旧值做比较。"
    )


def reevaluate_registered_model(
    *,
    model_key: str,
    n_bootstrap: int | None = None,
    bootstrap_seed: int | None = None,
    write: bool = True,
    out_dir: Path | None = None,
) -> dict:
    """对已登记模型重算策略评估与差值，并可写回数据库 + 生成报告。

    ``n_bootstrap`` / ``bootstrap_seed`` 缺省时取库里已登记的那套评估配置
    （见 :func:`resolve_evaluation_config`），这样"重算"与"旧值"才可比。

    Returns:
        ``{"model_key", "model_id", "fractions", "n_scores", "n_contrasts",
        "cross_check_max_abs_diff", "report_path", "run_id"}``。
    """
    read_engine = get_engine(readonly=True)
    record, df, sp = rebuild_holdout(read_engine, model_key)

    _hdr(f"重算已登记版本 {model_key} 的策略评估（model_id={record.model_id}）")
    print(f"建模人群 {len(df):,} 行；划分 {sp.describe()}")

    old_rows = load_policy_evaluations(read_engine, model_key)
    bootstrap_seed, n_bootstrap = resolve_evaluation_config(old_rows, n_bootstrap, bootstrap_seed)
    print(
        f"bootstrap 配置取自"
        f"{'调用方' if n_bootstrap else '库内登记'}："
        f"seed={bootstrap_seed}, n_bootstrap={n_bootstrap}"
    )

    stored = _load_stored_scores(read_engine, model_key)
    print(f"已存分数 {len(stored):,} 行（列：mu1/mu0/uplift/propensity_score）")

    # 把分数对齐到建模人群的行序（load_modeling_frame 按 customer_key 排序）
    by_key = stored.set_index("customer_key")
    missing = sorted(set(df["customer_key"]) - set(by_key.index))
    if missing:
        raise ValueError(
            f"有 {len(missing)} 名建模客户没有分数（示例 {missing[:3]}）；"
            "覆盖率不足时评估不可信，不继续。"
        )
    scores_all = by_key.loc[df["customer_key"]].reset_index(drop=True)
    # 未参与评估的列（mu1）也解析一次，保证四列都在同一张表里被检查过
    print(
        "分数分布："
        f"uplift 均值={scores_all['uplift'].mean():+.8f}，"
        f"mu0 均值={scores_all['mu0'].mean():.8f}，"
        f"propensity_score 均值={scores_all['propensity_score'].mean():.8f}"
    )

    test_df = df.iloc[sp.test_idx].reset_index(drop=True)
    scores_test = scores_all.iloc[sp.test_idx].reset_index(drop=True)
    y_test = test_df["y"].to_numpy().astype(float)
    w_test = test_df["treatment"].to_numpy().astype(bool)
    score_arrays = scores_from_frame(scores_test, list(SCORED_POLICIES))
    print(
        f"留出集 n={len(test_df):,}；实际购买率 干预组={y_test[w_test].mean():.4%} "
        f"对照组={y_test[~w_test].mean():.4%}"
    )

    evaluation = evaluate_policies_with_contrasts(
        y=y_test,
        treatment=w_test,
        scores=score_arrays,
        fractions=TREATMENT_FRACTIONS,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    print(
        f"重算得到 {len(evaluation.scores)} 条策略结果、"
        f"{len(evaluation.contrasts)} 条差值（比例 {list(evaluation.fractions)}）"
    )

    # ---------------------------------------------------------------- 交叉核对
    _hdr("交叉核对：重算值 vs 库里旧值")
    old_by_key = {
        (r["policy"], round(float(r["fraction"]), 5)): r for r in old_rows
    }
    max_diff = 0.0
    compared = 0
    renamed_hits = 0
    for s in evaluation.scores:
        key = (s.policy, round(float(s.fraction), 5))
        old = old_by_key.get(key)
        if old is None and s.policy == "control_probability":
            # 旧名 propensity 的行（sql/006 迁移前）用来核对"改名是否名副实归"
            old = old_by_key.get(("propensity", round(float(s.fraction), 5)))
            if old is not None:
                renamed_hits += 1
        if old is None:
            continue
        for field, new_value in (
            ("ips_estimate", s.ips_estimate),
            ("ci_low", s.ci_low),
            ("ci_high", s.ci_high),
            ("n_selected", s.n_selected),
        ):
            old_value = old[field]
            if old_value is None or (isinstance(new_value, float) and math.isnan(new_value)):
                continue
            diff = abs(float(old_value) - float(new_value))
            max_diff = max(max_diff, diff)
            compared += 1
            if diff > CROSS_CHECK_TOLERANCE:
                print(
                    f"  ⚠️ {s.policy}@{s.fraction:.2f} {field}: "
                    f"库内={old_value} 重算={new_value} 差={diff:.3e}"
                )
    print(
        f"逐字段比较 {compared} 项；最大绝对差 = {max_diff:.3e}"
        f"（预先规定的容差 {CROSS_CHECK_TOLERANCE:.0e}）"
    )
    if renamed_hits:
        print(
            f"其中 {renamed_hits} 条对照的是**旧名 propensity**的行——"
            "它们与从 mu0 重算的 control_probability 一致，说明改名是名副其实的"
            "（旧行当时的数值确实由 mu0 算出，不是合并购买预测）。"
        )
    if max_diff > CROSS_CHECK_TOLERANCE:
        raise AssertionError(
            f"重算值与库内旧值最大差 {max_diff:.3e} 超过容差 {CROSS_CHECK_TOLERANCE:.0e}："
            "口径或分数已经变了，不得静默覆盖（AGENTS.md §2.6）。"
        )

    # ---------------------------------------------------------------- 落库与报告
    result: dict = {
        "model_key": model_key,
        "model_id": record.model_id,
        "fractions": list(evaluation.fractions),
        "n_scores": len(evaluation.scores),
        "n_contrasts": len(evaluation.contrasts),
        "cross_check_max_abs_diff": max_diff,
        "cross_check_compared_fields": compared,
        "report_path": None,
        "run_id": None,
    }
    if not write:
        print("\n（--no-write：未落库、未生成报告）")
        return result

    write_engine = get_engine(readonly=False)
    run_id = str(uuid.uuid4())
    stats_config = {
        "stage": 4,
        "model_key": model_key,
        "mode": "reevaluate_registered_scores",
        "split_seed": int((record.split_config or {}).get("seed", bootstrap_seed)),
        "bootstrap_seed": bootstrap_seed,
        "n_bootstrap": n_bootstrap,
        "fractions": list(evaluation.fractions),
        "allocation_propensity": 0.5,  # e(x)：分配倾向概率，真值恒为 0.5（不是策略打分）
        "policy_estimator": "snips_constant_propensity",
        "contrast_estimator": "paired_percentile_bootstrap",
        "primary_contrast": {
            "fraction": CONTRAST_PRIMARY_FRACTION,
            "policy": CONTRAST_PRIMARY_POLICY,
            "reference_policy": CONTRAST_PRIMARY_REFERENCE,
        },
        "score_source": "customer_scores（已登记版本的不可变分数，未重新训练）",
    }
    with write_engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into analysis_runs (
                    run_id, experiment_id, dataset_id, comparison_key,
                    question, status, code_version, stats_config, finished_at
                )
                select :rid, m.experiment_id, m.dataset_id, m.comparison_key,
                       '阶段四：按已登记分数重算策略评估与差值', 'succeeded', m.code_version,
                       cast(:cfg as jsonb), now()
                from models m where m.model_key = :mk
                on conflict (run_id) do nothing
                """
            ),
            {
                "rid": run_id,
                "mk": model_key,
                "cfg": json.dumps(stats_config, ensure_ascii=False),
            },
        )
    for s in evaluation.scores:
        save_policy_evaluation(write_engine, model_id=record.model_id, run_id=run_id, result=s)
    for c in evaluation.contrasts:
        save_policy_contrast(write_engine, model_id=record.model_id, run_id=run_id, contrast=c)
    print(
        f"\n已写入 {len(evaluation.scores)} 条策略评估、"
        f"{len(evaluation.contrasts)} 条差值（run_id={run_id}）"
    )

    from dsa.report.policy_report import render_policy_report

    out_dir = out_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rp = out_dir / f"policy_report_{model_key}_{run_id[:8]}.md"
    render_policy_report(
        run_id=run_id,
        model_record=record,
        policy_scores=evaluation.scores,
        contrasts=evaluation.contrasts,
        out_path=rp,
    )
    print(f"报告：{rp}")
    result["report_path"] = str(rp)
    result["run_id"] = run_id

    # 回读一次，证明"登记 -> 读回"闭环（新版本的登记、分数、策略评估能相互核对）
    back = load_policy_evaluations(read_engine, model_key)
    back_contrasts = load_policy_contrasts(read_engine, model_key)
    print(
        f"回读：policy_evaluations {len(back)} 行；policy_contrasts {len(back_contrasts)} 行"
    )
    result["rows_after_write"] = len(back)
    result["contrasts_after_write"] = len(back_contrasts)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="对已登记模型重算策略评估与差值")
    ap.add_argument("model_key", help="已登记的模型业务键，如 tlearner_lr_v1")
    ap.add_argument("--n-bootstrap", type=int, default=None,
                    help="缺省取库里已登记的那套配置（保证与旧值可比）")
    ap.add_argument("--seed", type=int, default=None,
                    help="缺省取库里已登记的那套配置")
    ap.add_argument("--no-write", action="store_true", help="只核对，不写库、不生成报告")
    args = ap.parse_args(argv)
    result = reevaluate_registered_model(
        model_key=args.model_key,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.seed,
        write=not args.no_write,
    )
    _hdr("完成")
    for k, v in result.items():
        print(f"  {k:28s} = {v}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
