"""SQL 与 Python 交叉核对。

**目的**：SQL 实现与独立的 Python 实现在相同口径下必须给出相同结果。
两者不一致，说明至少一边的口径实现有误——这是发现静默错误的关键手段。

依据：AGENTS.md §2.6；实现蓝图.md 阶段二「用 Python 对照计算或人工小样本核对关键 SQL」。

用法：
    python scripts/cross_check.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import text  # noqa: E402

from dsa.config import get_engine, get_settings  # noqa: E402
from dsa.data.specs import HILLSTROM, HILLSTROM_EMAIL_V1  # noqa: E402
from dsa.stats.aggregates import arm_aggregates_from_raw  # noqa: E402

SQL_FILE = ROOT / "sql" / "002_core_queries.sql"
CSV_PATH = ROOT / HILLSTROM.relative_path()

#: 相对误差容忍度。浮点求和顺序不同会带来极小差异，故设 1e-9 而非 0。
#: （AGENTS.md 要求容忍度必须**预先说明**，不得事后放宽。）
RTOL = 1e-9

_failures: list[str] = []
_checks = 0


def load_named_queries(path: Path) -> dict[str, str]:
    """从 .sql 文件里按 `-- name: xxx` 标记切出命名查询。"""
    raw = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^--\s*name:\s*(\w+)\s*$", re.MULTILINE)
    marks = list(pattern.finditer(raw))
    out: dict[str, str] = {}
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        sql = raw[start:end]
        # 去掉块注释与行注释，保留可执行部分
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        sql = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
        out[m.group(1)] = sql.strip().rstrip(";")
    return out


def check(name: str, condition: bool, detail: str = "") -> None:
    global _checks
    _checks += 1
    status = "OK  " if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        _failures.append(name)


def close(a: float, b: float, rtol: float = RTOL) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    return abs(a - b) <= rtol * denom if denom else True


def main() -> int:
    if not CSV_PATH.exists():
        print(f"[FAIL] 源文件不存在: {CSV_PATH}")
        return 1

    queries = load_named_queries(SQL_FILE)
    print("=" * 76)
    print(f"交叉核对  ——  SQL ({SQL_FILE.name}) vs 独立 Python 实现")
    print(f"相对误差容忍度 RTOL = {RTOL:g}（预先设定，不事后放宽）")
    print("=" * 76)
    print(f"载入命名查询: {', '.join(queries)}")

    engine = get_engine(readonly=True)
    params = {
        "experiment_key": HILLSTROM_EMAIL_V1.experiment_key,
        "dataset_version": HILLSTROM.version_label,
    }

    # ================================================================
    print("\n---- 1. 各组聚合：SQL vs Python ----")
    sql_df = pd.read_sql(text(queries["arm_aggregates"]), engine.connect(), params=params)
    py = arm_aggregates_from_raw(CSV_PATH)

    check("SQL 返回组数 = 3", len(sql_df) == 3, f"实际 {len(sql_df)}")
    check("Python 返回组数 = 3", len(py) == 3, f"实际 {len(py)}")

    for _, r in sql_df.iterrows():
        arm = r["arm_key"]
        p = py.get(arm)
        if p is None:
            check(f"{arm} 在 Python 结果中存在", False)
            continue
        check(f"{arm} 样本量", int(r["n_customers"]) == p.n_customers,
              f"SQL={r['n_customers']} PY={p.n_customers}")
        check(f"{arm} 购买人数", int(r["n_converted"]) == p.n_converted,
              f"SQL={r['n_converted']} PY={p.n_converted}")
        check(f"{arm} 访问人数", int(r["n_visited"]) == p.n_visited,
              f"SQL={r['n_visited']} PY={p.n_visited}")
        check(f"{arm} 购买率", close(float(r["conversion_rate"]), p.conversion_rate),
              f"SQL={float(r['conversion_rate']):.10f} PY={p.conversion_rate:.10f}")
        check(f"{arm} 访问率", close(float(r["visit_rate"]), p.visit_rate),
              f"SQL={float(r['visit_rate']):.10f} PY={p.visit_rate:.10f}")
        check(f"{arm} 人均消费", close(float(r["spend_per_customer"]), p.spend_per_customer),
              f"SQL={float(r['spend_per_customer']):.6f} PY={p.spend_per_customer:.6f}")
        check(f"{arm} 结果无缺失", int(r["n_outcome_missing"]) == 0,
              f"SQL={r['n_outcome_missing']}")

    # ================================================================
    print("\n---- 2. 总体汇总 ----")
    ov = pd.read_sql(text(queries["overall_summary"]), engine.connect(), params=params).iloc[0]
    total_py = sum(p.n_customers for p in py.values())
    conv_py = sum(p.n_converted for p in py.values())
    check("总客户数", int(ov["n_assigned"]) == total_py, f"SQL={ov['n_assigned']} PY={total_py}")
    check("总购买人数", int(ov["n_converted"]) == conv_py, f"SQL={ov['n_converted']} PY={conv_py}")
    check("总客户数 = 64000", int(ov["n_assigned"]) == HILLSTROM.row_count)
    check("无结果缺失", int(ov["n_outcome_missing"]) == 0)

    # ================================================================
    print("\n---- 3. 关联膨胀 / 唯一性 / 覆盖率 ----")
    jr = pd.read_sql(text(queries["join_row_ratio"]), engine.connect(), params=params).iloc[0]
    check("关联后行数 = 分组数（无膨胀）", bool(jr["no_inflation"]),
          f"assigned={jr['n_assigned']} joined={jr['n_rows']} ratio={jr['join_row_ratio']}")

    ii = pd.read_sql(text(queries["identity_integrity"]), engine.connect(),
                     params={"dataset_version": HILLSTROM.version_label}).iloc[0]
    check("客户 ID / 源行号一一对应", bool(ii["identity_ok"]),
          f"n={ii['n_customers']} distinct_id={ii['n_distinct_id']} distinct_row={ii['n_distinct_row']}")
    check("源行号 1..64000 连续",
          int(ii["min_row"]) == 1 and int(ii["max_row"]) == HILLSTROM.row_count,
          f"范围 {ii['min_row']}..{ii['max_row']}")

    cov = pd.read_sql(text(queries["outcome_coverage"]), engine.connect(), params=params).iloc[0]
    check("结果覆盖率 = 1.0", close(float(cov["outcome_coverage"]), 1.0),
          f"coverage={cov['outcome_coverage']} missing={cov['n_missing']}")

    card = pd.read_sql(text(queries["customer_cardinality"]), engine.connect(), params=params)
    for _, r in card.iterrows():
        check(f"{r['check_name']} 每客户恰好 1 条", bool(r["ok"]),
              f"min={r['min_per_customer']} max={r['max_per_customer']}")

    # ================================================================
    print("\n---- 4. SRM 检查（预期比例来自原始说明，非观测反推）----")
    srm = pd.read_sql(text(queries["srm_check"]), engine.connect(), params=params)
    ratios = {a.arm_key: float(a.expected_ratio) for a in HILLSTROM_EMAIL_V1.arms}
    for _, r in srm.iterrows():
        check(f"{r['arm_key']} 预期计数 = 观测比例推算",
              close(float(r["expected"]), ratios[r["arm_key"]] * float(r["n_total"]), 1e-6),
              f"expected={r['expected']} ratio={ratios[r['arm_key']]:.10f}")
    chi2 = float(srm["chi_square"].iloc[0])
    df_ = int(srm["df"].iloc[0])
    check("卡方统计量可计算", chi2 >= 0, f"chi2={chi2:.4f} df={df_}")
    print(f"         （SRM 的 p 值由 stage 3 的统计模块统一计算，此处只核对分子分母）")

    # ================================================================
    print("\n" + "=" * 76)
    if _failures:
        print(f"结论：{len(_failures)}/{_checks} 项失败 ✘")
        for f in _failures:
            print(f"   - {f}")
        return 1
    print(f"结论：{_checks} 项全部通过 ✔   SQL 与独立 Python 实现结果一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
