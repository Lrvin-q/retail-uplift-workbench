"""阶段四端到端：训练 → 登记 → 打分 → 策略评估 → 报告。

用法：
    python scripts/train_and_evaluate.py
    & "..." scripts/train_and_evaluate.py --no-write     # 只跑不落库
    & "..." scripts/train_and_evaluate.py --model-key tlearner_lr_v2   # 配置变了就换 key

**训练是明确的离线任务**（协议 §8.1），不由 Agent 触发。
Agent 只能读取已登记的模型与分数。

⚠️ 配置变更是**显式**的（缺陷 F10）：同一个 ``model_key`` 只接受**完全相同**的训练配置
（身份指纹相同）。改了种子 / 超参 / 特征 / 代码版本之后再跑同一个 key 会**报错中止**，
并打印逐字段对照；脚本**不会**自动改 key，也不会用新配置覆盖旧登记。
要登记新配置，请显式传 ``--model-key`` 指定一个新 key。

⚠️ 只想重算策略评估（口径变更，模型不变）时**不要**重跑本脚本，用：
    & "..." -m dsa.models.reevaluate tlearner_lr_v1
它按已登记版本的**不可变分数**重算评估与差值，不会把"口径变更"混成"模型变更"。
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsa.config import ARTIFACTS_DIR, get_engine  # noqa: E402
from dsa.models.features import preprocessing_config  # noqa: E402
from dsa.models.policy import (  # noqa: E402
    CONTRAST_PRIMARY_FRACTION,
    CONTRAST_PRIMARY_POLICY,
    CONTRAST_PRIMARY_REFERENCE,
    TREATMENT_FRACTIONS,
    evaluate_policies_with_contrasts,
    policy_score_columns,
)
from dsa.models.registry import (  # noqa: E402
    IdentityMismatchError,
    register_model,
    save_customer_scores,
    save_policy_contrast,
    save_policy_evaluation,
)
from dsa.models.split import load_modeling_frame, make_split  # noqa: E402
from dsa.models.tlearner import fit_outcome_model, fit_t_learner  # noqa: E402
from dsa.report.policy_report import render_policy_report  # noqa: E402
from dsa.stats.experiment import code_version  # noqa: E402

MODEL_KEY = "tlearner_lr_v1"
SEED = 20261011
DATASET_VERSION = "hillstrom@2008-03-20"
EXPERIMENT_KEY = "hillstrom_email_v1"
COMPARISON_KEY = "primary"


def _hdr(t: str) -> None:
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段四端到端")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--no-write", action="store_true", help="不写数据库、不生成报告")
    ap.add_argument("--model-key", default=MODEL_KEY)
    args = ap.parse_args()

    read_engine = get_engine(readonly=True)

    # ---------------------------------------------------------- 1 取数与划分
    _hdr("1. 取数与划分")
    df = load_modeling_frame(read_engine)
    print(f"建模人群: {len(df):,} 行  arm={df['arm'].value_counts().to_dict()}")

    sp = make_split(df, seed=args.seed)
    print(f"划分(seed={args.seed}): train={len(sp.train_idx):,} "
          f"valid={len(sp.valid_idx):,} test={len(sp.test_idx):,}")
    print(f"各集合组别: {sp.arm_counts}")

    train_df = df.iloc[sp.train_idx]
    test_df = df.iloc[sp.test_idx]

    # ---------------------------------------------------------- 2 训练
    _hdr("2. 训练（离线任务）")
    tlearn = fit_t_learner(train_df, seed=args.seed)
    outcome = fit_outcome_model(train_df, seed=args.seed)
    print(f"T-Learner      : {tlearn.base_learner}  n_train={tlearn.n_train_treatment:,}/{tlearn.n_train_control:,}")
    print(f"购买预测基线   : {outcome.base_learner}  n_train={outcome.n_train:,}")

    # ---------------------------------------------------------- 3 打分
    _hdr("3. 打分")
    # 全量打分用于名单；**策略评估只用留出集**（协议 §3、§6）
    scores_all = pd.DataFrame({
        "customer_key": df["customer_key"].to_numpy(),
        "mu1": tlearn.predict_mu1(df),
        "mu0": tlearn.predict_mu0(df),
        "uplift": tlearn.predict_uplift(df),
        "propensity_score": outcome.predict_proba(df),
    })
    print(f"全量打分 {len(scores_all):,} 行（用于名单）")
    print(f"  uplift          mean={scores_all['uplift'].mean():+.8f} std={scores_all['uplift'].std():.8f}")
    print(f"  mu0             mean={scores_all['mu0'].mean():.8f}")
    print(f"  propensity_score mean={scores_all['propensity_score'].mean():.8f}")

    # 留出集：(y, treatment, 各种打分)
    # ⚠️ 策略名与列一一对应（缺陷 5.2）：
    #   uplift               -> customer_scores.uplift
    #   purchase_probability -> customer_scores.propensity_score（**合并**训练的结果模型）
    #   control_probability  -> customer_scores.mu0（T-Learner 的对照组模型）
    # 旧名 `propensity` 已废弃：它在离线评估里指 mu0、在名单接口里指合并预测，语义不明。
    y_test = test_df["y"].to_numpy().astype(float)
    w_test = test_df["treatment"].to_numpy().astype(bool)
    up_test = tlearn.predict_uplift(test_df)
    mu0_test = tlearn.predict_mu0(test_df)
    prop_test = outcome.predict_proba(test_df)

    print(f"\n留出集 n={len(test_df):,}")
    print(f"  实际购买率: 干预组={y_test[w_test].mean():.4%}  对照组={y_test[~w_test].mean():.4%}")
    print(f"  差值(描述性): {y_test[w_test].mean() - y_test[~w_test].mean():+.4%}")
    print(f"  uplift 与 mu0 相关系数: {np.corrcoef(up_test, mu0_test)[0, 1]:+.4f}")
    print(f"  合并购买预测与 mu0 相关系数: {np.corrcoef(prop_test, mu0_test)[0, 1]:+.4f}")

    # ---------------------------------------------------------- 4 策略评估
    _hdr("4. 策略评估（留出集，固定触达比例；差异只看配对差值区间）")
    evaluation = evaluate_policies_with_contrasts(
        y=y_test,
        treatment=w_test,
        scores={
            "uplift": up_test,
            "purchase_probability": prop_test,
            "control_probability": mu0_test,
        },
        fractions=TREATMENT_FRACTIONS,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.seed,
    )
    policy_scores = list(evaluation.scores)
    print(f"共 {len(policy_scores)} 条边际结果、{len(evaluation.contrasts)} 条差值")
    print()
    print(f"{'策略':<22}{'触达比例':>10}{'触达人数':>10}{'增量估计':>14}{'95%区间':>26}")
    for s in sorted(policy_scores, key=lambda x: (x.fraction, x.policy)):
        ci = f"[{s.ci_low:+.6f},{s.ci_high:+.6f}]" if s.ci_low == s.ci_low else "[—]"
        print(f"{s.policy:<22}{s.fraction:>10.0%}{s.n_selected:>10,}{s.ips_estimate:>+14.6f}{ci:>26}")

    print()
    print("差值区间（同一次共同 bootstrap 的配对抽样；主要比较在前）：")
    print(f"{'差值':<46}{'点估计':>14}{'95%区间':>26}{'排除0':>8}")
    for c in evaluation.contrasts:
        label = f"{c.policy} - {c.reference_policy} @ {c.fraction:.0%}"
        ci = f"[{c.ci_low:+.6f},{c.ci_high:+.6f}]" if c.ci_defined else "[—]"
        mark = "主要" if c.is_primary else ""
        print(f"{label + (' (' + mark + ')' if mark else ''):<46}"
              f"{c.difference:>+14.6f}{ci:>26}{('是' if c.excludes_zero else '否'):>8}")

    # ---------------------------------------------------------- 5 落库与报告
    if args.no_write:
        _hdr("完成（--no-write，未落库）")
        return 0

    _hdr("5. 登记模型与分数")
    write_engine = get_engine(readonly=False)
    code_ver = code_version()
    pre_cfg = preprocessing_config()
    try:
        model_id = register_model(
            write_engine,
            model_key=args.model_key,
            model_type=tlearn.to_registry_dict()["model_type"],
            dataset_version=DATASET_VERSION,
            experiment_key=EXPERIMENT_KEY,
            comparison_key=COMPARISON_KEY,
            feature_columns=tlearn.feature_columns,
            base_learner=tlearn.base_learner,
            params=tlearn.params,
            split_config=sp.to_registry_dict(),
            code_version=code_ver,
            preprocessing=pre_cfg,
            notes="第一轮固定 LogisticRegression，未做超参数搜索（协议 §4.3）",
        )
    except IdentityMismatchError as exc:
        _hdr("[中止] 训练配置与已登记版本不同：没有写任何东西")
        print(str(exc))
        print()
        suggestion = (
            args.model_key[:-3] + "_v2" if args.model_key.endswith("_v1") else args.model_key + "_v2"
        )
        print(f"本次配置：code_version={code_ver}、划分种子={args.seed}、"
              f"特征={list(tlearn.feature_columns)}")
        print(f"处置：换一个 model_key 重新运行，例如 "
              f"--model-key {suggestion}（脚本不会自动改 key）。")
        return 2
    print(f"model_id = {model_id}")

    n_scored = save_customer_scores(write_engine, model_id=model_id, scores_df=scores_all)
    print(f"客户分数写入 {n_scored:,} 行（同一版本重复写为 0 行，冲突不更新）")

    # 运行登记：阶段四的产物不走 `analyze_experiment` 那条路径，直接写 analysis_runs
    run_id = str(uuid.uuid4())
    stats_config = {
        "stage": 4,
        "model_key": args.model_key,
        "split_seed": args.seed,
        "bootstrap_seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "fractions": list(TREATMENT_FRACTIONS),
        "allocation_propensity": 0.5,  # e(x)：分配倾向概率，真值恒为 0.5（不是策略打分）
        "policy_estimator": "snips_constant_propensity",
        "contrast_estimator": "paired_percentile_bootstrap",
        "primary_contrast": {
            "fraction": CONTRAST_PRIMARY_FRACTION,
            "policy": CONTRAST_PRIMARY_POLICY,
            "reference_policy": CONTRAST_PRIMARY_REFERENCE,
        },
        "policy_score_columns": policy_score_columns(),
    }
    with write_engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into analysis_runs (
                    run_id, experiment_id, dataset_id, comparison_key,
                    question, status, code_version, stats_config, finished_at
                )
                select :rid, e.experiment_id, e.dataset_id, 'primary',
                       '阶段四：增量模型与策略评估', 'succeeded', :cv,
                       cast(:cfg as jsonb), now()
                from experiments e where e.experiment_key = 'hillstrom_email_v1'
                on conflict (run_id) do nothing
                """
            ),
            {
                "rid": run_id,
                "cv": code_ver,
                "cfg": json.dumps(stats_config, ensure_ascii=False),
            },
        )
    print(f"run_id = {run_id}")

    for s in policy_scores:
        save_policy_evaluation(write_engine, model_id=model_id, run_id=run_id, result=s)
    print(f"策略评估写入 {len(policy_scores)} 行")
    for c in evaluation.contrasts:
        save_policy_contrast(write_engine, model_id=model_id, run_id=run_id, contrast=c)
    print(f"策略差值写入 {len(evaluation.contrasts)} 行")

    _hdr("6. 生成策略报告")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # 报告里的版本信息一律取自**库里登记行**，而不是内存里的临时对象：
    # 这样报告展示的代码版本/指纹与数据库完全一致（缺陷 F10 的验收要求）。
    from dsa.models.registry import load_model_record

    rec = load_model_record(read_engine, args.model_key)
    if rec is None:  # pragma: no cover - 刚写入，必然能读到
        raise AssertionError(f"刚登记的模型读不回来：{args.model_key!r}")
    rp = ARTIFACTS_DIR / f"policy_report_{args.model_key}_{run_id[:8]}.md"
    render_policy_report(
        run_id=run_id,
        model_record=rec,
        policy_scores=policy_scores,
        contrasts=evaluation.contrasts,
        out_path=rp,
    )
    print(f"报告: {rp}")

    _hdr("完成")
    print(f"  模型       : {args.model_key} (model_id={model_id})")
    print(f"  身份指纹   : {rec.identity_fingerprint}")
    print(f"  运行       : {run_id}")
    print(f"  报告       : {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
