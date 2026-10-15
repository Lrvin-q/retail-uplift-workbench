"""RetailHero 多表导入。

**分批 COPY 导入**，不把 4.46 GB 读进 pandas（AGENTS.md §5 与蓝图阶段五要求）。

幂等：重复执行不产生重复行（导入前按数据集清空对应表）。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import text

from dsa.config import PROJECT_ROOT, get_engine

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "retailhero" / "data"
from dsa.config import SQL_DIR
DDL_PATH = SQL_DIR.joinpath("005_retailhero_schema.sql")

#: 分批大小。purchases 有 4,578 万行，必须分批否则内存与事务日志都吃不消。
BATCH_ROWS = 300_000

#: 计算 SHA256 的读取块大小。purchases.csv 有 4.46 GB，不能一次读进内存。
HASH_CHUNK_BYTES = 8 * 1024 * 1024

TABLES = {
    "rh_clients": ("clients.csv", ["client_id", "first_issue_date", "first_redeem_date",
                                   "age", "gender"]),
    "rh_products": ("products.csv", ["product_id", "level_1", "level_2", "level_3",
                                     "level_4", "segment_id", "brand_id", "vendor_id",
                                     "netto", "is_own_trademark", "is_alcohol"]),
    "rh_uplift_train": ("uplift_train.csv", ["client_id", "treatment_flg", "target"]),
    "rh_uplift_test": ("uplift_test.csv", ["client_id"]),
}

PURCHASE_COLUMNS = [
    "client_id", "transaction_id", "transaction_datetime",
    "regular_points_received", "express_points_received",
    "regular_points_spent", "express_points_spent",
    "purchase_sum", "store_id", "product_id", "product_quantity",
    "trn_sum_from_iss", "trn_sum_from_red",
]


def apply_ddl(engine: Any) -> None:
    """执行建表 DDL。**注意**：必须用原始游标且不传参数，
    否则 SQL 里的 ``%`` / ``:`` 会被当成占位符（本项目踩过，见 sql/003 的说明）。
    """
    sql = DDL_PATH.read_text(encoding="utf-8")
    with engine.begin() as conn:
        raw = conn.connection.driver_connection
        with raw.cursor() as cur:
            cur.execute(sql)


def _copy_batches(engine: Any, table: str, columns: list[str],
                  rows: Iterator[str], *, batch_rows: int = BATCH_ROWS,
                  label: str = "") -> tuple[int, float]:
    """分批 COPY FROM STDIN，**每批一个事务**。返回 (总行数, 耗时秒)。

    ⚠️ **必须每批提交**，不能用一个事务包住全部行：
    ``purchases`` 有 4,578 万行 / 4.46 GB，单事务会导致
      1. 提交前数据完全不可见（无法观察进度，中断也留不下已完成部分）
      2. WAL 与事务快照随行数线性膨胀，内存与磁盘都会出问题
      3. 任何一批失败 → **全部回滚**，几小时白跑
    本项目第一次跑就踩了这条：导入启动很久后 ``count(*)`` 仍是 0。
    """
    started = time.perf_counter()
    total = 0
    batch: list[str] = []
    col_list = ", ".join(columns)

    def _flush(chunk: list[str]) -> int:
        buf = io.StringIO("".join(chunk))
        with engine.begin() as conn:  # ← 每批独立事务
            raw = conn.connection.driver_connection
            with raw.cursor() as cur:
                with cur.copy(
                    f"copy {table} ({col_list}) from stdin with (format csv)"
                ) as cp:
                    cp.write(buf.getvalue())
        return len(chunk)

    for line in rows:
        batch.append(line)
        if len(batch) >= batch_rows:
            total += _flush(batch)
            batch.clear()
            print(f"    {label or table}: {total:,} 行  "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    if batch:
        total += _flush(batch)
    return total, time.perf_counter() - started


def _csv_lines(path: Path, *, expect_columns: int | None = None) -> Iterator[str]:
    """逐行产出 **CSV 文本行**。

    **先探测再决定走哪条路**：

    - **快路径（原始行透传）**：文件里没有引号、没有 ``\\r`` 时，
      说明不存在跨行字段与转义问题，直接把原始行交给 ``COPY``。
      实测 ``purchases.csv`` 40 万行全部恰好 12 个逗号、0 处引号、0 处 ``\\r``。
    - **慢路径（csv 往返）**：只要发现引号，就退回 ``csv.reader`` + ``csv.writer``
      规整一遍。这一步是必要的：字段内含换行时直接透传，
      ``COPY`` 会把「字段里的换行」当成「一行结束」，**静默错位**。

    为什么分路径：csv 往返是逐行的 Python 开销，把 ``COPY`` 拖慢到约 2 万行/秒
    （4,578 万行要 38 分钟）。**但快路径必须建立在探测结果之上，不能靠假设** ——
    所以 ``expect_columns`` 会给每一行做逗号数校验，不符立即报错而不是导入错位数据。
    """
    with path.open(encoding="utf-8", newline="") as fh:
        head = fh.read(8192)
        fh.seek(0)
        simple = ('"' not in head) and ("\r" not in head)

        if simple:
            next(fh, None)  # 表头
            if expect_columns is None:
                for line in fh:
                    yield line
                return
            for lineno, line in enumerate(fh, start=2):
                if line.count(",") != expect_columns - 1:
                    raise ValueError(
                        f"{path.name} 第 {lineno} 行逗号个数与表头不符，"
                        "快路径不安全，请检查数据"
                    )
                yield line
            return

        # 慢路径：规整为标准 CSV
        fh.seek(0)
        reader = csv.reader(fh)
        next(reader, None)
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        for row in reader:
            writer.writerow(row)
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # 表头
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        for row in reader:
            writer.writerow(row)
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)


def load_small_tables(engine: Any) -> dict[str, int]:
    """导入 4 张小表（合计约 690 MB）。"""
    counts: dict[str, int] = {}
    for table, (fname, cols) in TABLES.items():
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"缺少数据文件: {path}")
        n, secs = _copy_batches(engine, table, cols,
                                _csv_lines(path, expect_columns=len(cols)), label=table)
        counts[table] = n
        print(f"  {table}: {n:,} 行 / {secs:.1f}s", flush=True)
    return counts


def load_purchases(engine: Any, *, resume: bool = False) -> tuple[int, float]:
    """导入 purchases.csv（4,578 万行 / 4.46 GB），分批。

    ``resume=True`` 时**不截断**，而是跳过库中已有的行数继续导入。

    为什么可以「按行数跳过」：本导入是**顺序**读取 CSV 的顺序导入，
    已导入的 N 行必然对应 CSV 的前 N 个数据行。因此跳过前 N 行即可续跑。
    ⚠️ 这个前提依赖「不并发、不重排」。若将来改成多进程并行导入，
    必须改用「按 client_id 分片」之类的可判定方式，不能再按行数跳过。
    """
    path = DATA_DIR / "purchases.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少数据文件: {path}")

    skip = 0
    if resume:
        with engine.connect() as conn:
            skip = int(conn.execute(text("select count(*) from rh_purchases")).scalar_one())
        print(f"  续跑：跳过已导入的 {skip:,} 行", flush=True)

    rows = _csv_lines(path, expect_columns=len(PURCHASE_COLUMNS))
    if skip:
        for _ in range(skip):
            next(rows, None)

    n, secs = _copy_batches(engine, "rh_purchases", PURCHASE_COLUMNS, rows,
                            label="rh_purchases")
    print(f"  rh_purchases: 本次 {n:,} 行 / {secs:.1f}s"
          + (f"（累计 {skip + n:,}）" if skip else ""), flush=True)
    return n, secs


def file_sha256(path: Path, *, chunk: int = HASH_CHUNK_BYTES) -> str:
    """流式算文件 SHA256。**必须分块**：purchases.csv 有 4.46 GB。

    为什么要在 loader 里真算：``rh_source_files.file_sha256`` 原实现**从不写入**
    （建表时也没在 INSERT 列里），于是「每个源文件都有 SHA256」这句话在库里是空的。
    特征快照（``rh_feature_snapshots.source_hash_set``）要复用这一列的哈希，
    留空会让整条证据链断在第一环。见 docs/日志.md L047/L048。
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def truncate_retailhero(engine: Any) -> None:
    """清空 RetailHero 相关表，保证幂等。**只动 rh_ 前缀的表。**

    ``rh_feature_snapshots`` 与 ``rh_feature_active`` 也在清空之列：重新导入就是
    换了一份数据，旧的快照身份不再成立，留着只会让「当前快照」指向
    一份早已不存在的数据。

    ⚠️ 快照表上有 `before update/delete` 的**行级**不可变触发器，
    而 ``TRUNCATE`` 不触发行级 DELETE 触发器 —— 这里的清空是「换数据源」
    这个合法重置路径，与「防止事后篡改单条快照」并不冲突。
    """
    with engine.begin() as conn:
        conn.execute(text(
            "truncate table rh_source_files, rh_feature_active, rh_feature_snapshots, "
            "rh_client_features, rh_uplift_test, "
            "rh_uplift_train, rh_purchases, rh_products, rh_clients restart identity cascade"
        ))


def register_source_files(engine: Any) -> None:
    """登记数据来源与校验信息，供追溯。

    列名以 ``sql/005_retailhero_schema.sql`` 的实际定义为准
    （``file_key`` / ``file_bytes`` / ``row_count`` …），
    不要凭记忆写 —— 本项目已因列名不匹配失败过一次。

    ``file_sha256`` **在这里实算**（流式分块）。原实现把它排除在 INSERT 列之外，
    于是这一列永远是 NULL，而 ``datasets`` 表里躺着一个占位哈希 ``'0'*64`` ——
    「可追溯」只停留在注释里。特征快照要复用这个哈希，所以必须先真的算出来。
    """
    specs = [
        ("clients", "clients.csv", "rh_clients", ["client_id", "first_issue_date",
                                                  "first_redeem_date", "age", "gender"]),
        ("products", "products.csv", "rh_products", ["product_id", "level_1", "level_2",
                                                     "level_3", "level_4", "segment_id",
                                                     "brand_id", "vendor_id", "netto",
                                                     "is_own_trademark", "is_alcohol"]),
        ("purchases", "purchases.csv", "rh_purchases", PURCHASE_COLUMNS),
        ("uplift_train", "uplift_train.csv", "rh_uplift_train",
         ["client_id", "treatment_flg", "target"]),
        ("uplift_test", "uplift_test.csv", "rh_uplift_test", ["client_id"]),
    ]
    note = ("ods.ai/x5-retailhero-uplift-modeling；经 8.8.8.8 解析 + curl --resolve 取得，"
            "见 docs/日志.md L025")
    with engine.begin() as conn:
        for key, fname, table, cols in specs:
            p = DATA_DIR / fname
            n = conn.execute(text(f"select count(*) from {table}")).scalar_one()
            if not n:
                # rh_source_files 有 row_count > 0 的约束：还没导入的表不登记，
                # 否则会插进一个"0 行"的假记录（比不登记更糟 —— 它看起来像已核实）。
                print(f"  （跳过 {key}：表为空，尚未导入）")
                continue
            if not p.exists():
                raise FileNotFoundError(f"表 {table} 有 {n} 行，但源文件缺失: {p}")
            sha = file_sha256(p)
            conn.execute(text(
                """
                insert into rh_source_files
                    (file_key, file_name, file_bytes, file_sha256, row_count,
                     column_names, notes, verified_at)
                values (:k, :f, :b, :sha, :n, :c, :note, now())
                on conflict (file_key) do update set
                    file_name = excluded.file_name,
                    file_bytes = excluded.file_bytes,
                    file_sha256 = excluded.file_sha256,
                    row_count = excluded.row_count,
                    column_names = excluded.column_names,
                    notes = excluded.notes,
                    verified_at = now()
                """
            ), {"k": key, "f": fname,
                "b": p.stat().st_size,
                "sha": sha,
                "n": int(n), "c": cols, "note": note})
            print(f"  已登记 {key}: {n:,} 行, sha256={sha[:12]}…", flush=True)


def aggregate_source_hash(engine: Any) -> str | None:
    """把所有已登记的源文件哈希折叠成一个摘要。取不到时返回 None。

    返回 None 表示**没有可用的哈希**，调用方不得用占位值冒充。
    """
    with engine.connect() as conn:
        rows = conn.execute(text(
            "select file_key, file_sha256 from rh_source_files "
            "where file_sha256 is not null order by file_key"
        )).all()
    if not rows:
        return None
    payload = json.dumps({str(k): str(v) for k, v in rows},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def update_dataset_hash(engine: Any) -> str | None:
    """把 ``datasets`` 里 retailhero 那行的占位哈希换成真实的汇总哈希。

    为什么保留 ``datasets.file_sha256`` 的「汇总哈希」语义而不是改成单文件哈希：
    本数据集是 5 个文件，一列存不下 5 个哈希；把某一个文件的哈希填进去
    会让「datasets.file_sha256 = 该数据版本的哈希」这条既有约定变成假话。
    单文件哈希放在 ``rh_source_files.file_sha256``，汇总哈希放在这里。
    """
    agg = aggregate_source_hash(engine)
    if agg is None:
        return None
    with engine.begin() as conn:
        conn.execute(text("""
            update datasets set file_sha256 = :sha
            where version_label = 'retailhero@x5-uplift-v1'
        """), {"sha": agg})
    return agg


def _index_statements() -> list[str]:
    """从 DDL 里抽出 ``create index`` 语句（用于导入后重建）。"""
    sql = DDL_PATH.read_text(encoding="utf-8")
    return [s.strip().rstrip(";") + ";"
            for s in sql.split(";")
            if "create index" in s.lower()]


#: 这些索引的目标表在**导入阶段还不存在**（由 ``build_features()`` 之后才用到
#: ``rh_feature_snapshots``）。``recreate_purchase_indexes()`` 是在 purchases
#: 导入完成后跑的，那时快照表可能还没建 —— 直接执行会
#: ``relation "rh_feature_snapshots" does not exist`` 而让整个重建中断。
#:
#: ⚠️ 这是 ``_index_statements()`` 用「字符串里有没有 create index」这种粗粒度
#: 过滤带来的既有隐患（它本来就会把**所有**表的索引都抓进来，只是以前 DDL 里
#: 的 create index 恰好都指向已存在的表）。修法是加一张显式跳过表，
#: 而不是改过滤规则 —— 后者会影响 purchases 索引重建这条已经验证过的路径。
_INDEX_SKIP_TABLES: frozenset[str] = frozenset({"rh_feature_snapshots"})


def recreate_purchase_indexes(engine: Any) -> int:
    """重建 ``rh_purchases`` 的索引（来自 DDL，不凭记忆写）。

    返回**实际执行**的语句数（跳过的不计入），便于调用方发现数量变化。
    """
    stmts = [s for s in _index_statements()
             if not any(t in s for t in _INDEX_SKIP_TABLES)]
    with engine.begin() as conn:
        for s in stmts:
            conn.execute(text(s.rstrip(";")))
    return len(stmts)


def drop_purchase_indexes(engine: Any) -> list[str]:
    """删除 ``rh_purchases`` 上除主键外的索引，返回被删的名字。

    **为什么**：``rh_purchases`` 有 6 个索引 + 主键。逐行插入时 PostgreSQL 要为
    每一行维护 7 套索引结构 —— 实测这把这个表的 COPY 拖到约 2 万行/秒
    （4,578 万行要 38 分钟）。先删索引再导入、导入完重建，
    是批量装载的标准做法，通常快数倍。

    ⚠️ 主键**不删**：它承担唯一性约束，删掉会让错误数据静默进来。
    """
    with engine.begin() as conn:
        rows = conn.execute(text(
            "select indexname from pg_indexes "
            "where tablename = 'rh_purchases' and indexname <> 'rh_purchases_pk'"
        )).scalars().all()
        for name in rows:
            conn.execute(text(f'drop index if exists "{name}"'))
    return list(rows)


def main(*, skip_purchases: bool = False, fast_load: bool = True,
         resume: bool = False) -> int:
    engine = get_engine(readonly=False)
    if resume:
        # 续跑模式：**不建表、不清空**，只接着导入未完成的部分。
        print("续跑模式：跳过建表与清空")
        print("5. 继续导入 purchases")
        n, secs = load_purchases(engine, resume=True)
        print("6. 重建索引")
        k = recreate_purchase_indexes(engine)
        print(f"   已重建 {k} 个索引")
        register_source_files(engine)
        agg = update_dataset_hash(engine)
        print(f"   datasets 汇总哈希 = {(agg or '（无）')[:12]}…")
        print(f"\n完成：rh_purchases 累计 {n:,} 行（本次）")
        return 0

    print("1. 建表")
    apply_ddl(engine)
    print("2. 清空（幂等）")
    truncate_retailhero(engine)
    print("3. 导入小表")
    counts = load_small_tables(engine)
    print("4. 登记来源（含每个文件的 SHA256）")
    register_source_files(engine)
    agg = update_dataset_hash(engine)
    print(f"   datasets 汇总哈希 = {(agg or '（无）')[:12]}…")
    if not skip_purchases:
        dropped: list[str] = []
        if fast_load:
            dropped = drop_purchase_indexes(engine)
            print(f"5. 已删除 {len(dropped)} 个索引以加速导入：{', '.join(dropped)}")
        print("6. 导入 purchases（4,578 万行，耗时较长）")
        n, secs = load_purchases(engine)
        counts["rh_purchases"] = n
        if dropped:
            print("7. 重建索引")
            k = recreate_purchase_indexes(engine)
            print(f"   已重建 {k} 个索引")
    print("\n完成：")
    for k, v in counts.items():
        print(f"  {k:22} {v:>12,} 行")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(
        skip_purchases="--skip-purchases" in sys.argv,
        fast_load="--no-fast" not in sys.argv,
        resume="--resume" in sys.argv,
    ))
