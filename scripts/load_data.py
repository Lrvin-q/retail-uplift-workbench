"""把 Hillstrom 源文件导入 PostgreSQL。

用法：
    python scripts/load_data.py
    & "..." scripts/load_data.py --allow-hash-mismatch
    & "..." scripts/load_data.py --verify-only

脚本是幂等的：重复执行不会新增记录。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import text  # noqa: E402

from dsa.config import get_engine, get_settings  # noqa: E402
from dsa.data.loader import (  # noqa: E402
    ImportError_,
    compute_sha256,
    load_hillstrom,
    resolve_csv_path,
)
from dsa.data.specs import HILLSTROM  # noqa: E402


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def verify_only() -> int:
    """只校验文件，不写数据库。"""
    path = resolve_csv_path(HILLSTROM, None)
    _print_header("文件校验")
    if not path.exists():
        print(f"[FAIL] 源文件不存在: {path}")
        return 1
    actual = compute_sha256(path)
    ok = actual == HILLSTROM.file_sha256
    print(f"路径      : {path}")
    print(f"字节数    : {path.stat().st_size:,}")
    print(f"规格哈希  : {HILLSTROM.file_sha256}")
    print(f"实际哈希  : {actual}")
    print(f"结论      : {'一致 ✔' if ok else '不一致 ✘'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="导入 Hillstrom 数据")
    ap.add_argument("--csv", type=Path, default=None, help="源文件路径")
    ap.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="允许文件哈希与规格不符（会记入警告）",
    )
    ap.add_argument("--verify-only", action="store_true", help="只校验文件，不导入")
    args = ap.parse_args()

    if args.verify_only:
        return verify_only()

    settings = get_settings()
    _print_header("环境")
    print(f"数据库    : {settings.describe()}")

    _print_header("导入")
    engine = get_engine(readonly=False)  # 导入需要可写账号
    try:
        report = load_hillstrom(
            engine,
            csv_path=args.csv,
            allow_hash_mismatch=args.allow_hash_mismatch,
        )
    except ImportError_ as exc:
        print(f"[FAIL] {exc}")
        return 1

    print(report.summary())

    # ---- 导入后一致性校验 ----
    _print_header("导入后校验")
    with engine.connect() as conn:
        checks = {
            "客户数 = 源文件行数": conn.execute(
                text(
                    "select count(*) = :n from customers c "
                    "join datasets d on d.dataset_id = c.dataset_id "
                    "where d.version_label = :v"
                ),
                {"n": HILLSTROM.row_count, "v": HILLSTROM.version_label},
            ).scalar_one(),
            "内部客户 ID 无重复": conn.execute(
                text(
                    "select count(*) = count(distinct customer_id) from customers c "
                    "join datasets d on d.dataset_id = c.dataset_id where d.version_label = :v"
                ),
                {"v": HILLSTROM.version_label},
            ).scalar_one(),
            "源行号连续无空洞": conn.execute(
                text(
                    "select min(source_row_number) = 1 and max(source_row_number) = count(*) "
                    "from customers c join datasets d on d.dataset_id = c.dataset_id "
                    "where d.version_label = :v"
                ),
                {"v": HILLSTROM.version_label},
            ).scalar_one(),
            "分组数 = 客户数": conn.execute(
                text(
                    "select count(*) = (select count(*) from customers c2 join datasets d2 "
                    "on d2.dataset_id = c2.dataset_id where d2.version_label = :v) "
                    "from assignments s join experiments e on e.experiment_id = s.experiment_id "
                    "where e.experiment_key = 'hillstrom_email_v1'"
                ),
                {"v": HILLSTROM.version_label},
            ).scalar_one(),
            "结果数 = 客户数（无缺失）": conn.execute(
                text(
                    "select count(*) = (select count(*) from customers c2 join datasets d2 "
                    "on d2.dataset_id = c2.dataset_id where d2.version_label = :v) "
                    "from outcomes o join experiments e on e.experiment_id = o.experiment_id "
                    "where e.experiment_key = 'hillstrom_email_v1'"
                ),
                {"v": HILLSTROM.version_label},
            ).scalar_one(),
            "视图无结果缺失": conn.execute(
                text("select count(*) = 0 from v_experiment_customer where outcome_missing")
            ).scalar_one(),
        }

    all_ok = True
    for name, passed in checks.items():
        print(f"  [{'OK' if passed else 'FAIL'}] {name}")
        all_ok = all_ok and bool(passed)

    _print_header("结论")
    print("全部校验通过 ✔" if all_ok else "存在失败项 ✘")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
