"""阶段三命令行入口：跑一次实验分析并产出报告。

用法：
    # 主要比较（默认）
    python scripts/analyze_experiment.py

    # 次要（探索性）比较
    & "..." scripts/analyze_experiment.py --comparison secondary_womens

    # 不生成报告，只打印结构化结果
    & "..." scripts/analyze_experiment.py --no-report

    # 调整 bootstrap 次数（会影响耗时）
    & "..." scripts/analyze_experiment.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsa.config import ARTIFACTS_DIR, get_engine  # noqa: E402
from dsa.stats.experiment import analyze_experiment, record_run  # noqa: E402


def _hdr(t: str) -> None:
    print()
    print("=" * 76)
    print(t)
    print("=" * 76)


def main() -> int:
    ap = argparse.ArgumentParser(description="实验分析与报告")
    ap.add_argument("--experiment", default="hillstrom_email_v1")
    ap.add_argument("--comparison", default="primary")
    ap.add_argument("--alpha", type=float, default=None, help="覆盖实验登记的 α（仅调试用）")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261011)
    ap.add_argument("--no-report", action="store_true", help="不生成报告与图表")
    ap.add_argument("--no-record", action="store_true", help="不写 analysis_runs 登记")
    args = ap.parse_args()

    read_engine = get_engine(readonly=True)

    # ---------------------------------------------------------------- 分析
    _hdr("实验分析")
    analysis = analyze_experiment(
        read_engine,
        experiment_key=args.experiment,
        comparison_key=args.comparison,
        bootstrap_seed=args.bootstrap_seed,
        n_boot=args.n_boot,
    )
    print(analysis.summary())

    # ---------------------------------------------------------------- 质量检查
    if analysis.checks:
        _hdr("数据质量检查")
        for c in analysis.checks:
            mark = "OK  " if c.passed else "FAIL"
            print(f"  [{mark}] {c.code}  {c.name}  ({c.severity})")
            print(f"         {c.detail}")

    # ---------------------------------------------------------------- 阻断判定
    _hdr("阻断判定")
    if analysis.blocked:
        print(f"  ⛔ 存在阻断级异常：{', '.join(analysis.blocking_codes)}")
        print("  → 不提供因果结论；描述性结果仍保留供排查。")
    else:
        print("  ✔ 无阻断级异常，允许输出因果结论。")

    # ---------------------------------------------------------------- 报告
    report_path = None
    if not args.no_report:
        _hdr("生成报告")
        try:
            from dsa.report.charts import effect_interval_chart
            from dsa.report.stats_report import render_stats_report

            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            charts: dict[str, Path] = {}
            for metric_id, st in analysis.metric_comparisons.items():
                p = ARTIFACTS_DIR / f"effect_interval_{metric_id}_{analysis.comparison_key}.png"
                chart = effect_interval_chart(st, p)
                charts[metric_id] = Path(chart)
                print(f"  图表: {chart}")

            report_path = ARTIFACTS_DIR / f"stats_report_{analysis.comparison_key}_{analysis.run_id[:8]}.md"
            render_stats_report(
                run_id=analysis.run_id,
                experiment_key=analysis.experiment_key,
                dataset_version=analysis.dataset_version,
                comparison_key=analysis.comparison_key,
                stats_by_metric=analysis.metric_comparisons,
                # 连续型指标（人均消费）必须一并传入，否则会从报告里漏掉
                continuous_by_metric=analysis.continuous_comparisons,
                checks=analysis.checks,
                out_path=report_path,
                window_days=analysis.window_days,
                generated_at=analysis.generated_at,
                chart_paths=charts,
            )
            print(f"  报告: {report_path}")
        except ImportError as exc:
            print(f"  ⚠ 报告模块不可用（{exc}），跳过。结构化结果已在上方打印。")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ 生成报告失败：{type(exc).__name__}: {exc}")
            return 1

    # ---------------------------------------------------------------- 登记
    if not args.no_record:
        _hdr("登记运行")
        write_engine = get_engine(readonly=False)
        run_id = record_run(
            write_engine,
            analysis,
            report_ref=str(report_path) if report_path else None,
        )
        print(f"  已写入 analysis_runs: run_id = {run_id}")

    _hdr("完成")
    print(f"  分析对象    : {analysis.comparison_key} ({analysis.comparison_kind})")
    print(f"  阻断        : {'是' if analysis.blocked else '否'}")
    print(f"  代码版本    : {analysis.code_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
