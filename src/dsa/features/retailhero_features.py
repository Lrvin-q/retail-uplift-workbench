"""RetailHero 客户级特征构建。

**两条铁律**（见 docs/日志.md L026 与 AGENTS.md §2.2）：

1. 交易级字段（``purchase_sum`` 等）被冗余到每一行商品明细上，
   **必须先按 ``transaction_id`` 收敛**（用 ``max``）再按客户聚合。
   直接 ``sum(purchase_sum)`` 会虚高约 5.81 倍且**不报错**。
2. 没有交易记录的客户**必须保留**（计数为 0、时间为 NULL），
   不得因 INNER JOIN 丢失。用 LEFT JOIN 从 ``rh_clients`` 出发。

**第三条（L048 新增）：每次构建都要登记快照身份。**

``FEATURE_PIPELINE_VERSION`` 只是一个代码常量，``built_at`` 只是一个时间戳 ——
它们都无法回答「这次特征是用哪份源数据算出来的」。
``build_features()`` 因此会写一行 ``rh_feature_snapshots``，
把源文件哈希集、流水线版本、可确认的历史边界、质量检查结果与行数封存起来，
并让本次写入的特征行通过 ``snapshot_id`` 引用它。

⚠️ **诚实说明**：``rh_client_features`` 按 ``client_id`` 主键 TRUNCATE 后重写，
所以快照**只保留元数据，不保留历史特征值**。详见 ``rh_feature_snapshots`` 的表注释。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

#: 特征计算使用的**固定时区**。
#:
#: ⚠️ 这个常量是必需的，不是装饰。``date_trunc('day', timestamptz)`` 会先在
#: **会话时区**里换算、再截断 —— 于是同一份数据、同一段代码，在不同
#: ``TimeZone`` 的会话里会算出不同的 ``days_active``。
#:
#: 本项目实测（fixture 里一笔 ``2018-01-06 12:00+03``、一笔 ``2018-01-06 20:00+03``，
#: 两笔在 UTC 下是同一天）：
#:
#: ```text
#: 时区                    原表达式   显式 UTC
#: UTC                         2         2
#: Asia/Shanghai               3         2      ← 本机默认，两笔被分到两天
#: America/New_York            2         2
#: Pacific/Kiritimati          3         2
#: ```
#:
#: 也就是说：**本机 Asia/Shanghai 会话下算出来的特征值，与 CI / UTC 容器里的不同。**
#: 这直接动摇 AGENTS.md §2.6「SQL 与 Python 必须交叉核对」的前提 ——
#: 两边根本不在同一口径上，核对也就无从谈起。
#:
#: 修法：所有日期截断都显式 ``at time zone`` 到本常量，与服务器/会话时区解耦。
#: 选 UTC 而不是门店当地时间：UTC 没有夏令时跳变，
#: 是唯一不需要额外假设就能全球一致的基准。
FEATURE_TIMEZONE = "UTC"

#: 特征 SQL。分三段：
#:   tx      —— 把「交易-商品」明细行收敛成**交易行**（关键去重步骤）
#:   cagg    —— 按客户聚合交易级指标
#:   pagg    —— 按客户聚合明细级指标（这些可以直接求和）
#:
#: ⚠️ ``days_active`` 的口径见 :data:`FEATURE_PIPELINE_VERSION` 下方的说明：
#: 这里算的是**有过交易的不同日历日的个数**（按 :data:`FEATURE_TIMEZONE` 截断），
#: 不是首末交易之间的间隔天数。
FEATURE_SQL = """
with tx as (
    -- ⚠️ 关键：交易级字段在这里取 max 收敛到一行。
    -- 用 max 而不是 sum：组内若不一致，max<>min 会暴露问题；
    -- 而 sum 对「不一致」和「重复」给出同样的错误结果，无法区分。
    select client_id,
           transaction_id,
           max(purchase_sum)          as purchase_sum,
           max(transaction_datetime)  as transaction_datetime,
           max(store_id)              as store_id,
           count(*)                   as n_lines,
           sum(product_quantity)      as n_items
    from rh_purchases
    group by client_id, transaction_id
),
cagg as (
    select client_id,
           count(*)                             as n_transactions,
           sum(purchase_sum)                    as total_spend,
           sum(purchase_sum) / nullif(count(*), 0) as avg_basket,
           sum(n_items)                         as n_items,
           sum(n_lines)                         as n_purchase_lines,
           count(distinct store_id)             as n_stores,
           min(transaction_datetime)            as first_txn_at,
           max(transaction_datetime)            as last_txn_at,
           -- days_active = 有过交易的**不同日历日**个数（不是首末间隔！）
           -- ⚠️ `at time zone 'UTC'` 不可省：缺了它，结果取决于会话时区
           -- （本机 Asia/Shanghai 与 CI 的 UTC 会给出不同的值）。见 FEATURE_TIMEZONE。
           count(distinct date_trunc('day',
                          transaction_datetime at time zone 'UTC')) as days_active
    from tx
    group by client_id
),
pagg as (
    -- 明细级指标：product_quantity 是明细级字段，可以直接求和；
    -- 商品种类数必须 count(distinct)，否则同一商品买两次会算成两种。
    select p.client_id,
           count(distinct p.product_id) as n_distinct_products,
           count(distinct pr.level_1)   as n_top_level1,
           -- brand_id 为 NULL 的明细不计入（count(distinct) 忽略 NULL）——
           -- 这是有意的：把「品牌未知」当成一个品牌会凭空造出一类
           count(distinct pr.brand_id)  as n_brands
    from rh_purchases p
    left join rh_products pr on pr.product_id = p.product_id
    group by p.client_id
)
select c.client_id,
       coalesce(ca.n_transactions, 0)      as n_transactions,
       coalesce(ca.total_spend, 0)         as total_spend,
       ca.avg_basket                       as avg_basket,
       coalesce(ca.n_items, 0)             as n_items,
       coalesce(ca.n_purchase_lines, 0)    as n_purchase_lines,
       coalesce(ca.n_stores, 0)            as n_stores,
       coalesce(pa.n_distinct_products, 0) as n_distinct_products,
       coalesce(pa.n_top_level1, 0)        as n_top_level1,
       coalesce(pa.n_brands, 0)            as n_brands,
       ca.first_txn_at                     as first_txn_date,
       ca.last_txn_at                      as last_txn_date,
       -- ⚠️ **不能**写 coalesce(ca.days_active, 0)。
       -- rh_cf_no_history_ck 明确要求无历史客户的 days_active 为 NULL ——
       -- 「活跃 0 天」与「没有交易记录」是两件事，用 0 冒充会让下游
       -- 分不清「来了但只活跃 0 天」和「根本没来过」。
       -- 现有开发库 400,162 个客户**全部有购买历史**，所以这条从来没被触发过；
       -- 小型 fixture 里第一个无历史客户就把它暴露出来了（见 L047）。
       ca.days_active                      as days_active,
       (ca.n_transactions is not null)     as has_purchase_history,
       c.age, c.gender,
       -- age 不合理时**显式标识**而不是丢弃该客户（L026 的同类处置）。
       -- ⚠️ 用 CASE 而不是 `c.age is not null and c.age between 14 and 100`：
       -- 后者在 age 为 NULL 时返回 **false**（三值逻辑里 NULL and ... 的结果
       -- 取决于实现，SQLite 直接给 0），于是「未判定」被记成「判定为不合理」。
       -- 语义上 age 缺失 ⇒ age_plausible 必须是 NULL（未判定），
       -- 由 rh_cf_age_ck 约束「age 非空 ⇒ age_plausible 非空」兜住另一半。
       case when c.age is null then null
            when c.age between 14 and 100 then true
            else false end                as age_plausible
from rh_clients c
left join cagg ca on ca.client_id = c.client_id
left join pagg pa on pa.client_id = c.client_id
"""

#: ``build_features()`` **实际写出的列**。
#:
#: ⚠️ 这个元组必须与 ``sql/005_retailhero_schema.sql`` 里 ``rh_client_features``
#: 的列定义一致。原实现把这份清单写成函数内的一个字符串字面量，
#: 结果 DDL 少 5 列而没人发现（现有开发库是手工 ALTER 补的，见 L047）。
#: 现在它升格为模块级常量，并有一条**离线**测试直接比对两边
#: （``tests/test_stage5_retailhero_scratch_db.py::test_ddl_covers_every_insert_column``），
#: 不需要连数据库就能抓到漏列。
FEATURE_COLUMNS: tuple[str, ...] = (
    "client_id",
    "n_transactions",
    "total_spend",
    "avg_basket",
    "n_items",
    "n_purchase_lines",
    "n_stores",
    "n_distinct_products",
    "n_top_level1",
    "n_brands",
    "first_txn_date",
    "last_txn_date",
    "days_active",
    "has_purchase_history",
    "age",
    "gender",
    "age_plausible",
)

#: ``snapshot_id`` 由 ``build_features()`` 在插入后回填，不属于上面的清单。
SNAPSHOT_COLUMN = "snapshot_id"


#: 特征流水线版本。**任何会改变特征数值的改动都必须递增它。**
#:
#: 版本历史：
#:   v1 —— 初版：交易级去重（max 收敛）+ 客户级聚合 + LEFT JOIN 保留全部客户
#:   v2 —— 把 ``days_active`` 的日期截断**显式钉到 UTC**（:data:`FEATURE_TIMEZONE`）。
#:         这是一次**真的会改变数值**的修正：在 Asia/Shanghai 会话下，
#:         一笔 ``2018-01-06 20:00+03`` 会被截到 01-07，与 ``12:00+03`` 那笔
#:         分到两天，``days_active`` 因此多算 1。UTC 容器里算出来的则是 2。
#:         也就是说 v1 的值**取决于跑构建的机器时区**，不可复现（见 L047）。
#:         按「改了数值就升版本」的规则，必须升到 v2。
#:
#: 为什么需要有版本号：特征一旦被用于训练，模型就与「当时那版特征」绑定。
#: 没有版本号，事后无法判断某个模型是用哪套特征训出来的，
#: 也无法判断「重跑为什么和上次不一样」。
#:
#: ⚠️ **L047 里其余几项修复（DDL 补列、days_active 注释口径、age_plausible 的 CASE）
#: 都没有单独升版本**，因为它们不改变既有开发库已经算出来的数值；
#: v2 是由时区修正这一项带来的，`docs/日志.md` L047 有完整说明。
FEATURE_PIPELINE_VERSION = "v2"

#: 质量检查里的「虚高倍数」预期区间，预先声明，不得事后放宽。
INFLATION_RATIO_MIN = 1.5
INFLATION_RATIO_MAX = 30.0


@dataclass(frozen=True)
class SnapshotIdentity:
    """一次构建的**数据身份**：它是「这次特征算的是什么数据」的答案。

    这里刻意**不包含** ``built_at`` —— 时间戳不是身份，
    同一份数据重跑两次应当得到同一个身份摘要。
    """

    pipeline_version: str
    source_hash_set: dict[str, str] | None
    source_hash_digest: str | None
    data_period_start: Any
    data_period_end: Any
    n_clients: int

    def describe(self) -> str:
        digest = self.source_hash_digest or "（无源哈希）"
        return (f"{self.pipeline_version} / 源摘要 {digest[:12]} / "
                f"{self.data_period_start} ~ {self.data_period_end} / "
                f"{self.n_clients:,} 客户")


def _fold_hash(source_hash_set: dict[str, str] | None) -> str | None:
    """把源哈希集折叠成一个摘要。

    ``sort_keys=True`` 是必需的：字典序不影响内容，但会影响字节序列，
    不确定的折叠结果会让「同一份数据」被算成两个不同的身份。
    空集与 None 一律返回 None —— 不能用哈希值冒充「已记录」。
    """
    if not source_hash_set:
        return None
    payload = json.dumps(source_hash_set, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_snapshot_identity(conn: Any, *, pipeline_version: str) -> SnapshotIdentity:
    """读取**数据侧**的快照身份：源文件哈希 + 交易时间边界 + 客户数。

    源哈希**复用** ``rh_source_files.file_sha256``，不另造一套哈希 ——
    同一份文件在两处存两个不同的哈希，等于没有哈希。

    取不到哈希（例如源文件尚未登记）时 ``source_hash_set`` 为 ``None``。
    **绝不**用占位值冒充，那会让快照看起来比实际可信。
    """
    rows = conn.execute(text(
        "select file_key, file_sha256 from rh_source_files where file_sha256 is not null"
    )).all()
    hashes = {str(k): str(v) for k, v in rows}

    period = conn.execute(text(
        "select min(transaction_datetime), max(transaction_datetime) from rh_purchases"
    )).one()
    n_clients = int(conn.execute(text("select count(*) from rh_clients")).scalar_one())

    hs = hashes or None
    return SnapshotIdentity(
        pipeline_version=pipeline_version,
        source_hash_set=hs,
        source_hash_digest=_fold_hash(hs),
        data_period_start=period[0],
        data_period_end=period[1],
        n_clients=n_clients,
    )


def quality_check(engine: Any, *, relation: str = "rh_client_features") -> dict:
    """特征质量检查。返回结构化结果，**不抛异常**（调用方据此决定是否放行）。

    Args:
        relation: 被检查的特征关系名。默认检查落地的 ``rh_client_features``；
            ``build_features()`` 传入本次构建的临时表，从而在**写入正式表之前**
            就拿到质量结果（否则 TRUNCATE 会让检查读到一张空表而全项失败）。

    ⚠️ ``relation`` 会被拼进 SQL。它**不是**用户输入，只接受本模块内部传入的
    固定字符串；不要把它暴露成 API 参数。

    检查项与判据：

    | 检查 | 判据 | 为什么 |
    |---|---|---|
    | 客户覆盖 | 特征行数 >= 客户表行数 | 少一行就是丢了一个客户，分母就错了 |
    | 无历史客户计数 | 计数为 0 而非 NULL | NULL 参与运算会静默变成 NULL 结果 |
    | 金额非负 | `total_spend >= 0` | 负消费多半是符号/聚合错误 |
    | 交易数自洽 | 特征总交易数 == 去重后的交易数 | 聚合链路是否完整 |
    | 金额虚高倍数 | 落在 [1.5, 30] 之间 | 见下 |
    | 年龄标识 | `age_plausible` 无 NULL | 标识必须显式，不能缺 |

    **「虚高倍数」这一项值得解释**：它是「直接求和 / 去重求和」的比值。
    下界 1.5 —— 若接近 1.0，说明数据里几乎没有多行交易，
    那么 L026 记录的去重逻辑已成无用开销，应当复核前提是否还成立；
    上界 30 —— 超出说明明细行数异常膨胀。
    **这条检查的价值不在于「通过」，而在于「不通过时提示前提变了」。**
    """
    table = relation
    out: dict = {"version": FEATURE_PIPELINE_VERSION, "relation": table,
                 "checks": [], "ok": True}

    def add(name: str, passed: bool, detail: str) -> None:
        out["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            out["ok"] = False

    with engine.connect() as conn:
        row = conn.execute(text(f"""
            select (select count(*) from {table})                          as n_feat,
                   (select count(*) from rh_clients)                        as n_clients,
                   (select count(*) from {table}
                     where not has_purchase_history
                       and (n_transactions <> 0 or total_spend <> 0))       as n_bad_zero,
                   (select count(*) from {table}
                     where total_spend < 0)                                 as n_negative,
                   (select count(*) from {table}
                     where age_plausible is null
                       and age is not null)                                 as n_age_null,
                   (select coalesce(sum(n_transactions), 0)
                      from {table})                                         as feat_tx,
                   (select count(distinct (client_id, transaction_id))
                      from rh_purchases)                                    as real_tx
        """)).mappings().one()
        infl = conn.execute(text("""
            select sum(purchase_sum) as naive, sum(mx) as deduped
            from (select transaction_id, sum(purchase_sum) as purchase_sum,
                         max(purchase_sum) as mx
                  from rh_purchases group by transaction_id) t
        """)).mappings().one()

    n_feat, n_clients = int(row["n_feat"]), int(row["n_clients"])
    add("customer_coverage", n_feat >= n_clients,
        f"特征 {n_feat:,} 行 vs 客户 {n_clients:,} 行")
    add("no_history_zeroed", int(row["n_bad_zero"]) == 0,
        f"无历史客户计数非零的行数 = {int(row['n_bad_zero'])}")
    add("spend_non_negative", int(row["n_negative"]) == 0,
        f"负消费客户数 = {int(row['n_negative'])}")
    add("transaction_count_consistent", int(row["feat_tx"]) == int(row["real_tx"]),
        f"特征总交易 {int(row['feat_tx']):,} vs 去重交易 {int(row['real_tx']):,}")
    # ⚠️ 判据是「age 非 NULL 时 age_plausible 必须非 NULL」。
    # 早先写成「age_plausible 整体不得为 NULL」，那会把 age 缺失的客户
    # 误判成缺陷 —— age 为 NULL 时 age_plausible 理应为 NULL（更诚实）。
    # 必须显式写 age is not null 这一半，否则该检查在 age 有缺失的真实数据上会假阳性。
    add("age_flag_explicit", int(row["n_age_null"]) == 0,
        f"age 有值但 age_plausible 为 NULL 的行数 = {int(row['n_age_null'])}")

    naive, deduped = float(infl["naive"] or 0), float(infl["deduped"] or 0)
    ratio = (naive / deduped) if deduped else None
    out["inflation_ratio"] = ratio
    add("redundancy_premise_holds",
        ratio is not None and INFLATION_RATIO_MIN <= ratio <= INFLATION_RATIO_MAX,
        f"虚高倍数 = {ratio:.4f}（预期 {INFLATION_RATIO_MIN}~{INFLATION_RATIO_MAX}；"
        f"接近 1 说明去重前提可能已失效）"
        if ratio is not None else "无法计算（没有交易数据）")
    return out


def build_features(engine: Any, *, pipeline_version: str | None = None) -> int:
    """重建 ``rh_client_features``，并登记一条不可变快照。返回写入行数。

    **执行顺序**（顺序本身是有原因的，不要调换）：

    1. 读数据身份（源哈希 / 历史边界）—— 必须在写入前读，否则读到的是半成品。
    2. 在**临时表**里算出全部特征 —— 用临时表是因为质量检查要在写入正式表
       **之前**拿到结果，而正式表会被 TRUNCATE。
    3. 对临时表跑质量检查。
    4. 写快照行（``is_active=true``）。
    5. 同一事务内：把旧快照置为 ``is_active=false`` → TRUNCATE → INSERT →
       回填 ``snapshot_id``。任何一步失败都会整体回滚，不会留下「新特征配旧快照」。

    **必须显式写出列名**：``insert into t select ...`` 依赖表的物理列序，
    而本表由 DDL 定义、后来又补过列，物理序与 SELECT 的书写顺序并不一致 ——
    本项目实测会报错在某个具体列上，且错误信息只指向位置、不说明是顺序问题。
    显式列名把这个隐患彻底消除。（漏列的隐患则由
    :data:`FEATURE_COLUMNS` 与 DDL 的结构测试兜住，见 L047。）
    """
    version = pipeline_version or FEATURE_PIPELINE_VERSION
    columns = ", ".join(FEATURE_COLUMNS)

    with engine.begin() as conn:
        # 1. 数据身份：源哈希 + 交易时间边界。必须在写入前读。
        ident = read_snapshot_identity(conn, pipeline_version=version)

        # 2. 临时表：ON COMMIT DROP 保证即使中途异常也不会留下残骸。
        #    用临时表而不是直接写正式表，是为了让质量检查在**写入前**拿到结果 ——
        #    正式表会被 TRUNCATE，先写后查只会读到一张空表。
        conn.execute(text(
            f"create temp table rh_cf_build on commit drop as {FEATURE_SQL}"))

        # 3. 质量检查（同事务，看得到临时表）
        qc = _quality_check_on(conn, ident)

        # 4. 新快照（**只增不改** —— 快照表有不可变触发器）
        snap_id = conn.execute(text("""
            insert into rh_feature_snapshots
                (pipeline_version, source_hash_set, source_hash_digest,
                 data_period_start, data_period_end,
                 n_clients, n_feature_rows, quality_ok, quality_checks)
            values
                (:version, cast(:hash_set as jsonb), :digest,
                 :pstart, :pend,
                 :n_clients, :n_rows, :ok, cast(:checks as jsonb))
            returning snapshot_id
        """), {
            "version": version,
            "hash_set": (json.dumps(ident.source_hash_set)
                         if ident.source_hash_set else None),
            "digest": ident.source_hash_digest,
            "pstart": ident.data_period_start,
            "pend": ident.data_period_end,
            "n_clients": ident.n_clients,
            "n_rows": qc["n_feature_rows"],
            "ok": qc["ok"],
            "checks": json.dumps(qc, ensure_ascii=False, default=str),
        }).scalar_one()

        # 5. 把「当前快照」指针移到新行。
        #
        #    ⚠️ 指针**不在** rh_feature_snapshots 上，而在单行表 rh_feature_active。
        #    本项目第一版用的是快照表上的 ``is_active`` 布尔列，结果与
        #    「快照不可变」的触发器**直接互斥** —— 第二次构建必然
        #    ``RestrictViolation: 不允许 UPDATE``。
        #    根因是把两种东西挤在一张表里：「要翻转的当前指针」vs「不可变的历史行」。
        #    拆开后，历史只增不改，指针随便改。见 L048。
        #
        #    用 INSERT ... ON CONFLICT 而不是「先查再决定 insert/update」：
        #    后者在并发下会互相撞主键。
        conn.execute(text("""
            insert into rh_feature_active (only_one, snapshot_id, updated_at)
            values (true, :sid, now())
            on conflict (only_one) do update
                set snapshot_id = excluded.snapshot_id,
                    updated_at  = excluded.updated_at
        """), {"sid": snap_id})

        conn.execute(text("truncate table rh_client_features"))
        conn.execute(text(
            f"insert into rh_client_features ({columns}) {FEATURE_SQL}"))
        # 回填 snapshot_id：INSERT 的 SELECT 里没有这一列（它是本次构建的产物，
        # 不是从数据算出来的），所以只能插入后统一回填。
        n = conn.execute(text(
            "update rh_client_features set snapshot_id = :sid"
        ), {"sid": snap_id}).rowcount
    return int(n)


def _quality_check_on(conn: Any, ident: SnapshotIdentity) -> dict:
    """:func:`quality_check` 的**同事务**版本：在给定连接上检查临时表。

    为什么需要它：:func:`quality_check` 自己 ``connect()``，只能看到已提交的数据。
    ``build_features()`` 需要「临时表算好 → 检查 → 再写正式表」，必须共用一个事务。
    检查项与判据和 :func:`quality_check` 完全一致，只是数据源换成
    ``rh_cf_build``（本事务内的临时表）。
    """
    out: dict = {"version": ident.pipeline_version, "relation": "rh_cf_build",
                 "checks": [], "ok": True}

    def add(name: str, passed: bool, detail: str) -> None:
        out["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            out["ok"] = False

    row = conn.execute(text("""
        select (select count(*) from rh_cf_build)                          as n_feat,
               (select count(*) from rh_clients)                            as n_clients,
               (select count(*) from rh_cf_build
                 where not has_purchase_history
                   and (n_transactions <> 0 or total_spend <> 0))           as n_bad_zero,
               (select count(*) from rh_cf_build
                 where total_spend < 0)                                     as n_negative,
               (select count(*) from rh_cf_build
                 where age_plausible is null and age is not null)           as n_age_null,
               (select coalesce(sum(n_transactions), 0) from rh_cf_build)   as feat_tx,
               (select count(distinct (client_id, transaction_id))
                  from rh_purchases)                                        as real_tx
    """)).mappings().one()
    infl = conn.execute(text("""
        select sum(purchase_sum) as naive, sum(mx) as deduped
        from (select transaction_id, sum(purchase_sum) as purchase_sum,
                     max(purchase_sum) as mx
              from rh_purchases group by transaction_id) t
    """)).mappings().one()

    n_feat, n_clients = int(row["n_feat"]), int(row["n_clients"])
    add("customer_coverage", n_feat >= n_clients,
        f"特征 {n_feat:,} 行 vs 客户 {n_clients:,} 行")
    add("no_history_zeroed", int(row["n_bad_zero"]) == 0,
        f"无历史客户计数非零的行数 = {int(row['n_bad_zero'])}")
    add("spend_non_negative", int(row["n_negative"]) == 0,
        f"负消费客户数 = {int(row['n_negative'])}")
    add("transaction_count_consistent", int(row["feat_tx"]) == int(row["real_tx"]),
        f"特征总交易 {int(row['feat_tx']):,} vs 去重交易 {int(row['real_tx']):,}")
    add("age_flag_explicit", int(row["n_age_null"]) == 0,
        f"age 有值但 age_plausible 为 NULL 的行数 = {int(row['n_age_null'])}")

    naive, deduped = float(infl["naive"] or 0), float(infl["deduped"] or 0)
    ratio = (naive / deduped) if deduped else None
    out["inflation_ratio"] = ratio
    add("redundancy_premise_holds",
        ratio is not None and INFLATION_RATIO_MIN <= ratio <= INFLATION_RATIO_MAX,
        f"虚高倍数 = {ratio:.4f}（预期 {INFLATION_RATIO_MIN}~{INFLATION_RATIO_MAX}；"
        f"接近 1 说明去重前提可能已失效）"
        if ratio is not None else "无法计算（没有交易数据）")

    out["n_feature_rows"] = n_feat
    out["identity"] = {
        "pipeline_version": ident.pipeline_version,
        "source_hash_digest": ident.source_hash_digest,
        "data_period_start": str(ident.data_period_start),
        "data_period_end": str(ident.data_period_end),
    }
    return out


def active_snapshot(engine: Any) -> dict | None:
    """返回当前生效的快照（含质量结果）。没有快照时返回 None。

    「当前快照」由单行表 ``rh_feature_active`` 指出 —— **不要**改成
    「``snapshot_id`` 最大的一条」或「``built_at`` 最新的一条」：
    那两种推断在并发构建或事务回滚后会指向错误的行。
    """
    with engine.connect() as conn:
        row = conn.execute(text("""
            select s.snapshot_id, s.pipeline_version, s.source_hash_set,
                   s.source_hash_digest, s.data_period_start, s.data_period_end,
                   s.n_clients, s.n_feature_rows, s.quality_ok, s.quality_checks,
                   s.built_at, a.updated_at as activated_at
            from rh_feature_active a
            join rh_feature_snapshots s on s.snapshot_id = a.snapshot_id
            where a.only_one
        """)).mappings().first()
    return dict(row) if row else None


def feature_summary(engine: Any) -> dict:
    """特征概览，用于核对与报告。"""
    with engine.begin() as conn:
        row = conn.execute(text("""
            select count(*)                                        as n_clients,
                   count(*) filter (where has_purchase_history)     as n_with_history,
                   count(*) filter (where not has_purchase_history) as n_without_history,
                   count(*) filter (where not age_plausible
                                      and age is not null)          as n_age_implausible,
                   sum(total_spend)                                 as total_spend,
                   sum(n_transactions)                              as total_transactions
            from rh_client_features
        """)).mappings().one()
        infl = conn.execute(text("""
            select sum(purchase_sum) as naive, sum(mx) as correct
            from (
                select transaction_id, sum(purchase_sum) as purchase_sum,
                       max(purchase_sum) as mx
                from rh_purchases group by transaction_id
            ) t
        """)).mappings().one()
        snap = conn.execute(text("""
            select s.snapshot_id, s.pipeline_version, s.source_hash_digest, s.built_at
            from rh_feature_active a
            join rh_feature_snapshots s on s.snapshot_id = a.snapshot_id
            where a.only_one
        """)).mappings().first()
    d = dict(row)
    naive, correct = float(infl["naive"] or 0), float(infl["correct"] or 0)
    d["naive_sum"] = naive
    d["deduped_sum"] = correct
    d["inflation_ratio"] = (naive / correct) if correct else None
    d["snapshot"] = dict(snap) if snap else None
    return d


if __name__ == "__main__":  # pragma: no cover
    import json as _json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from dsa.config import get_batch_engine

    # ⚠️ **批处理档引擎**：构建特征要对 `rh_purchases` 的 4,578 万行做分组聚合，
    # 单次远超交互档的 15 秒语句超时。用 `get_batch_engine()` 是**显式申请**
    # 更长的预算 —— 看到这一行就知道"这里会跑很久，而且是有意的"。
    eng = get_batch_engine(readonly=False)
    print(f"写入 {build_features(eng):,} 行")
    print(_json.dumps(feature_summary(eng), ensure_ascii=False, default=str, indent=2))
