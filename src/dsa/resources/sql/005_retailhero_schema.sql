-- ============================================================================
--  005_retailhero_schema.sql —— X5 RetailHero 多表关系模型
--
--  项目：零售营销增量决策助手
--  数据集：X5 RetailHero uplift 竞赛数据（与 Hillstrom 是**两个独立数据集**）
--  设计规范：.dsh/skills/supabase-postgres-best-practices
--      · 业务键用 text（源数据 client_id/product_id 是 10 位十六进制串，不是整数）
--      · 时间用 timestamptz；金额与点数用 numeric（不用 float）
--      · 所有外键列显式建索引（PostgreSQL 不会自动建）
--      · 大表先建表 → COPY → 后建索引/外键（避免逐行维护索引）
--
--  执行身份：dsa_app（可写账号）
--  本文件可重复执行（幂等）。
--
--  ⚠️ 铁律 1 —— purchase_sum / *_points_* / trn_sum_* 是**交易级**字段，
--     被冗余到该交易的每一行商品明细上（实测平均 5.81 行/交易，最高 19 行，
--     组内 100% 一致）。因此：
--         sum(purchase_sum)                    → 虚高约 5.81 倍（静默错误，不报错不为空）
--         先按 transaction_id 收敛再求和        → 正确
--     本文件的 rh_txn_fact（交易级收敛视图/物化）与
--     src/dsa/features/retailhero_features.py 都强制走收敛路径。
--     rh_purchases 保持**明细粒度，不做去重**；去重发生在聚合查询里。
--
--  ⚠️ 铁律 3 —— 数据集命名空间：
--     本文件所有表以 rh_ 前缀命名，客户标识只在本数据集内唯一。
--     禁止与 customers / outcomes（Hillstrom）做任何 JOIN 或拼接。
--     dataset_id 外键把本组表钉死在 datasets 表的 'retailhero' 版本上。
-- ============================================================================

-- ============================================================================
--  0. 数据版本登记
--     RetailHero 是多文件数据集，登记为一行 dataset（grain='purchase_line'），
--     每个源文件的实际行数与 SHA256 记在 rh_source_files 里。
-- ============================================================================
insert into datasets (
    dataset_key, version_label, source_url, file_name,
    file_sha256, file_bytes, row_count, grain, source_verified, notes
)
values (
    'retailhero',
    'retailhero@x5-uplift-v1',
    'https://ods.ai/tracks/retailhero-uplift',
    'retailhero/data/{clients,products,purchases,uplift_train,uplift_test}.csv',
    -- 多文件数据集无单一文件哈希：用 'multi' 占位会违反 ^[0-9a-f]{64}$ 的 CHECK，
    -- 因此这里存全文件清单的汇总哈希（由 005 导入程序写入前先算好，见 LOADER）；
    -- 首次建表时先写入占位哈希 '0'*64，loader 导入时会以真实汇总值更新。
    repeat('0', 64),
    null,
    45786568,          -- 明细粒度行数（purchases.csv），见 rh_source_files
    'purchase_line',
    true,
    'X5 RetailHero 多表数据集。明细粒度 = purchases.csv 的一行（一个商品明细）。'
    'purchase_sum 为交易级字段冗余在明细行上，聚合前必须按 transaction_id 收敛。'
)
on conflict (version_label) do nothing;

-- ============================================================================
--  0b. rh_source_files —— 每个源文件的行数与哈希（可追溯 + 幂等校验用）
-- ============================================================================
create table if not exists rh_source_files (
    file_key        text        primary key,     -- 'clients' / 'products' / ...
    file_name       text        not null,
    file_bytes      bigint,
    file_sha256     char(64),
    row_count       integer     not null,
    column_names    text[]      not null,
    notes           text,
    verified_at     timestamptz not null default now(),

    constraint rh_source_files_rows_ck   check (row_count > 0),
    constraint rh_source_files_sha_ck    check (file_sha256 is null or file_sha256 ~ '^[0-9a-f]{64}$'),
    constraint rh_source_files_key_ck    check (file_key ~ '^[a-z][a-z0-9_]*$')
);

comment on table  rh_source_files is
    'RetailHero 源文件清单。一行 = 一个源文件，记录行数、字节数与 SHA256，'
    '用于证明「库里行数 = CSV 行数」以及复现。';
comment on column rh_source_files.column_names is '源文件表头，按出现顺序；导入按位置读取，顺序变化必须被发现';
comment on column rh_source_files.file_sha256 is
    '【本文件】该源文件的 SHA256。由 loader 在导入时实算（原实现留空 —— 见 L047）：'
    '留空会让「可追溯」变成一句空话，快照里的 source_hash_set 也就无从取值。';

-- ============================================================================
--  1. rh_clients —— 客户维度表
-- ============================================================================
create table if not exists rh_clients (
    client_id          text        primary key,
    first_issue_date   timestamptz,          -- 首张卡发放时间
    first_redeem_date  timestamptz,          -- 首次核销时间
    age                smallint,
    gender             text,

    constraint rh_clients_id_ck   check (client_id ~ '^[0-9a-f]{10}$'),
    constraint rh_clients_gender_ck check (gender is null or gender in ('M','F','U')),
    -- ⚠️ age **不按合理性过滤**，只做存储层面的范围保护。
    --
    -- 实测 clients.csv 的 age 分布（400,162 行）：
    --   <=0       138 (0.03%)   最小 -7491
    --   1-17    5,414 (1.35%)
    --   18-100 393,561 (98.35%)
    --   >100    1,049 (0.26%)   最大 1901
    -- 共 1,187 个值超出 (0,100]，占 0.30%；常见异常值 115/119/0/944/949…
    --
    -- 若用 `age between 0 and 120` 约束，COPY 会**直接失败**（本项目已实测到）。
    -- 而"静默丢弃这些行"更糟 —— 那会无声地改变人群构成，
    -- 违反 AGENTS.md §2.2「不得按条件过滤分母」的精神。
    --
    -- 正确处置：**原样保存**，由特征层用 age_plausible 标识，
    -- 需要时在分析中显式排除，而不是让它悄悄消失。
    constraint rh_clients_age_ck check (age is null or (age > -100000 and age < 100000))
);

comment on table  rh_clients is
    'RetailHero 客户维度表。一行 = 一名客户。主键 client_id 是源数据的 10 位十六进制串。';
comment on column rh_clients.gender is 'M/F/U；U = 未知。保留 U 不做插补，插补属建模阶段的事';
comment on column rh_clients.age    is '年龄。源数据有缺失（NULL），不得用 0 填充——0 不是有效年龄';

-- ============================================================================
--  2. rh_products —— 商品维度表
-- ============================================================================
create table if not exists rh_products (
    product_id        text        primary key,
    level_1           text,
    level_2           text,
    level_3           text,
    level_4           text,
    segment_id        numeric(12,2),
    brand_id          text,
    vendor_id         text,
    netto             numeric(12,3),          -- 净重/净含量，源数据小数位不固定
    is_own_trademark  boolean,
    is_alcohol        boolean,

    constraint rh_products_id_ck check (product_id ~ '^[0-9a-f]{10}$')
);

comment on table  rh_products is
    'RetailHero 商品维度表。一行 = 一个 SKU。level_1..4 是四级商品品类层级。';
comment on column rh_products.level_1 is '一级品类（最粗）。特征 n_top_level1 = 客户购买涉及的一级品类数';
comment on column rh_products.netto   is '净含量。源数据小数位不固定，numeric 精确保存';

create index if not exists rh_products_level1_idx  on rh_products (level_1);
create index if not exists rh_products_vendor_idx  on rh_products (vendor_id);
create index if not exists rh_products_brand_idx   on rh_products (brand_id);

-- ============================================================================
--  3. rh_purchases —— 购买明细（**保持明细粒度，不做去重**）
--
--  主键说明：源数据有 45,786,568 行、平均 5.81 行/交易。
--  (client_id, transaction_id, product_id) 是否唯一由 scripts/preflight_retailhero.py 实测决定；
--  若唯一 → 用它作主键（天然防重复导入）；若不唯一 → 由 loader 抛错并改设计。
--  这里**不**加身份列：45.8M 行上加 identity + 索引会显著拖慢 COPY，
--  而业务键已能保证幂等。
-- ============================================================================
create table if not exists rh_purchases (
    client_id                text        not null,
    transaction_id           text        not null,
    transaction_datetime     timestamptz not null,

    -- 交易级字段（冗余到每一行明细上，聚合前必须按 transaction_id 收敛）
    regular_points_received  numeric(12,2),
    express_points_received  numeric(12,2),
    regular_points_spent     numeric(12,2),
    express_points_spent     numeric(12,2),
    purchase_sum             numeric(14,2),      -- 交易级：整笔交易金额
    store_id                 text        not null,

    -- 明细级字段（可直接按行求和）
    product_id               text        not null,
    product_quantity         numeric(12,3),
    trn_sum_from_iss         numeric(14,2),
    trn_sum_from_red         numeric(14,2),      -- 源数据大量为空，保留 NULL

    -- 唯一约束 = 幂等保证：重复导入同一文件不产生重复行
    constraint rh_purchases_pk unique (client_id, transaction_id, product_id)
);

comment on table  rh_purchases is
    'RetailHero 购买明细。一行 = 一笔交易中的一个商品明细（**不是一行交易**）。
     ⚠️ purchase_sum / *_points_* 是交易级字段，被冗余到每一行明细上；
     直接 sum(purchase_sum) 会虚高约 5.81 倍且不报错。'
    '正确做法：先 group by (client_id, transaction_id) 取 max 收敛，再求和。';
comment on column rh_purchases.purchase_sum is
    '【交易级】整笔交易的金额。同一 transaction_id 的所有明细行取值相同（实测 100% 一致）。'
    '聚合必须先收敛再求和，不能直接 sum。';
comment on column rh_purchases.product_quantity is
    '【明细级】本行商品件数。可按行直接求和得到 n_items。';
comment on column rh_purchases.trn_sum_from_iss is
    '【本行】该明细行来自签发卡（issued）的金额部分。';
comment on column rh_purchases.trn_sum_from_red is
    '【本行】该明细行来自核销（redeemed）的金额部分。源数据大量为空，NULL 表示不适用，'
    '不得当成 0 参与金额校验。';

-- ⚠️ 索引：两个去重聚合的关键索引
--    (client_id, transaction_id) 让「按客户收敛交易」走 Index Only Scan
--    transaction_id 单独索引让「按交易收敛」跨客户也能走索引
create index if not exists rh_purchases_client_idx    on rh_purchases (client_id);
create index if not exists rh_purchases_txn_idx       on rh_purchases (transaction_id);
create index if not exists rh_purchases_client_txn_idx on rh_purchases (client_id, transaction_id);
create index if not exists rh_purchases_product_idx   on rh_purchases (product_id);
create index if not exists rh_purchases_datetime_idx  on rh_purchases (transaction_datetime);

-- 外键：明细必须挂在已存在的客户与商品上（覆盖率由预检实测，见 scripts/preflight_retailhero.py）
alter table rh_purchases
    drop constraint if exists rh_purchases_client_fk;
alter table rh_purchases
    add constraint rh_purchases_client_fk
    foreign key (client_id) references rh_clients (client_id) on delete restrict;

alter table rh_purchases
    drop constraint if exists rh_purchases_product_fk;
alter table rh_purchases
    add constraint rh_purchases_product_fk
    foreign key (product_id) references rh_products (product_id) on delete restrict;

-- ============================================================================
--  4. rh_uplift_train —— 训练集（含干预标记与结果）
--     ⚠️ treatment_flg / target 是**干预后**信息：
--        treatment_flg 只在建模/评估时用，绝不能进特征表。
-- ============================================================================
create table if not exists rh_uplift_train (
    client_id      text     primary key,
    treatment_flg  smallint not null,
    target         smallint not null,

    constraint rh_uplift_train_client_fk foreign key (client_id)
        references rh_clients (client_id) on delete restrict,
    constraint rh_uplift_train_flg_ck check (treatment_flg in (0,1)),
    constraint rh_uplift_train_tgt_ck check (target in (0,1))
);

comment on table  rh_uplift_train is
    'RetailHero 训练集。一行 = 一名客户。treatment_flg=干预分配，target=结果。'
    '二者均为干预后信息，禁止作为特征（AGENTS.md §2.3）。';
comment on column rh_uplift_train.treatment_flg is '干预标记 0/1。【干预后信息，禁止进特征表】';
comment on column rh_uplift_train.target        is '结果（购买）。【干预后信息，禁止进特征表】';

-- 外键列索引（PostgreSQL 不会自动建）
create index if not exists rh_uplift_train_flg_idx on rh_uplift_train (treatment_flg);

-- ============================================================================
--  5. rh_uplift_test —— 测试集（只有客户标识）
-- ============================================================================
create table if not exists rh_uplift_test (
    client_id      text     primary key,

    constraint rh_uplift_test_client_fk foreign key (client_id)
        references rh_clients (client_id) on delete restrict
);

comment on table rh_uplift_test is 'RetailHero 测试集客户清单。一行 = 一名客户，无标签。';

-- ============================================================================
--  5b. rh_feature_snapshots —— 特征快照身份（数据版本 × 流水线版本）
--
--  【为什么需要】``FEATURE_PIPELINE_VERSION = 'v1'`` 只是代码里的一个常量，
--  库里没有任何东西把它和**具体一次构建**绑定。于是：
--      · 模型说「我用 v1 特征训的」，无法证明当时读的是哪份数据；
--      · 源文件换了（哪怕同名同大小），特征表看起来毫无变化；
--      · 质量检查结果只存在于当时的控制台输出里，事后无从查证。
--  ``built_at`` 只是时间戳，它**不能**替代源哈希与质量结果（审查 5.3 原文）。
--
--  本表一行 = 一次 ``build_features()`` 构建的**不可变身份**。
--
--  ⚠️ **诚实的边界**：``rh_client_features`` 按 client_id 主键 TRUNCATE 后重写，
--  所以本表**只保留元数据**，不保留历史特征值。它能回答
--  「上一次构建用的是哪份数据、哪版流水线、结果质量如何」，
--  **不能**回答「三个月前客户 X 的 total_spend 是多少」。
--  要回答后者必须改成按快照分区存储 —— 那是另一个设计，不在本次范围。
-- ============================================================================
create table if not exists rh_feature_snapshots (
    snapshot_id       bigint      generated always as identity primary key,

    pipeline_version  text        not null,     -- FEATURE_PIPELINE_VERSION，如 'v1'

    -- ---- 数据版本：复用 rh_source_files.file_sha256，不另造哈希 ----
    -- JSONB 形如 {"clients":"<sha256>","purchases":"<sha256>",...}；
    -- 以 NULL 取代 {}，让「没有哈希」这件事无法被误读成「哈希为空集」。
    source_hash_set   jsonb,
    -- 便于肉眼比对的折叠哈希（对 source_hash_set 的规范化 JSON 取 SHA256）
    source_hash_digest char(64),

    -- ---- 可确认的历史边界 ----
    data_period_start timestamptz,              -- min(transaction_datetime)
    data_period_end   timestamptz,              -- max(transaction_datetime)

    -- ---- 规模 ----
    n_clients         integer     not null,
    n_feature_rows    integer     not null,

    -- ---- 质量检查结果（原样存 quality_check() 的返回，不裁剪）----
    quality_ok        boolean     not null,
    quality_checks    jsonb       not null,

    -- ⚠️ 这里**刻意没有** is_active 之类的「当前」标记 ——
    -- 「当前是哪一条」由单独的单行表 rh_feature_active 回答。
    -- 原因见下方 §5c 与 L048：本表是不可变的（触发器禁止 UPDATE），
    -- 而翻转「当前」标记需要 UPDATE，两者在同一张表上互斥。
    built_at          timestamptz not null default now(),

    constraint rh_fs_pipeline_ck check (pipeline_version ~ '^v[0-9]+$'),
    constraint rh_fs_digest_ck   check (source_hash_digest is null
                                        or source_hash_digest ~ '^[0-9a-f]{64}$'),
    constraint rh_fs_rows_ck     check (n_feature_rows >= 0 and n_clients >= 0),
    constraint rh_fs_period_ck   check (data_period_start is null
                                        or data_period_end is null
                                        or data_period_start <= data_period_end)
    -- ⚠️ **刻意没有** (pipeline_version, source_hash_digest, n_feature_rows) 唯一约束。
    -- 本项目第一版加了它，结果 build_features() 第二次跑同一份数据时直接
    -- UniqueViolation —— 与「每次构建写一条新快照」的目的**正好相反**
    -- （见 L048）。同一份数据构建两次是**两个事件**，各自要有自己的
    -- built_at/quality_checks；要合并它们，唯一约束做不到「保留两次记录」。
);

comment on table  rh_feature_snapshots is
    'RetailHero 特征构建的不可变元数据快照。一行 = **一次** build_features() 的身份：'
    '源文件哈希集 + 流水线版本 + 可确认的历史边界 + 质量结果 + 行数 + 生成时间。'
    '同一份数据重复构建会产生**多条**记录 —— 每次构建是一个独立事件，'
    '各有自己的 built_at 与 quality_checks。'
    '⚠️ 本表**只增不改**（见下面的触发器）；「当前快照是哪一条」记在 rh_feature_active。'
    '⚠️ 只保留**元数据**；特征值本身因 TRUNCATE 重写而不留历史。';
comment on column rh_feature_snapshots.source_hash_set is
    '各源文件的 SHA256（取自 rh_source_files.file_sha256，不另造哈希）。'
    '取不到时为 NULL —— 不得用 {} 或占位值冒充「已记录」。';
comment on column rh_feature_snapshots.source_hash_digest is
    'source_hash_set 的折叠摘要，便于索引与肉眼比对。取不到哈希时为 NULL。'
    '**非唯一**：摘要相同表示「数据版本相同」，不代表「同一次构建」。';
comment on column rh_feature_snapshots.data_period_start is
    '本快照覆盖的**实际交易数据**最早时间。为空表示库里没有交易数据（不是「未知」）。';
comment on column rh_feature_snapshots.data_period_end is
    '本快照覆盖的**实际交易数据**最晚时间。与 start 一起构成「可确认的历史边界」。';
comment on column rh_feature_snapshots.quality_checks is
    'quality_check() 的完整返回（checks 数组 + inflation_ratio）。存全量而不是只存 ok，'
    '因为「哪一项没过」才是事后排查的入口。';

create index if not exists rh_feature_snapshots_built_idx
    on rh_feature_snapshots (built_at desc);
-- 「同一数据版本构建过几次」是可查的，但**不唯一**
create index if not exists rh_feature_snapshots_identity_idx
    on rh_feature_snapshots (pipeline_version, source_hash_digest);

-- ---- 不可变性：快照一旦写入，任何人（含 dsa_app）都不得改写 ----
-- 为什么用触发器而不是「靠自觉」：快照的全部价值就在于它是**当时事实的封存**。
-- 一个能被 UPDATE 的「不可变快照」在下游只会制造假的证据链。
create or replace function rh_feature_snapshots_immutable()
returns trigger
language plpgsql
as $$
begin
    raise exception
        'rh_feature_snapshots 是不可变快照：不允许 %（这是设计约束，不是操作失误）',
        tg_op
        using errcode = 'restrict_violation';
end
$$;

comment on function rh_feature_snapshots_immutable() is
    '禁止 UPDATE/DELETE rh_feature_snapshots。快照必须不可变，否则证据链是假的。';

drop trigger if exists rh_feature_snapshots_no_update on rh_feature_snapshots;
create trigger rh_feature_snapshots_no_update
    before update on rh_feature_snapshots
    for each row execute function rh_feature_snapshots_immutable();

drop trigger if exists rh_feature_snapshots_no_delete on rh_feature_snapshots;
create trigger rh_feature_snapshots_no_delete
    before delete on rh_feature_snapshots
    for each row execute function rh_feature_snapshots_immutable();

-- ============================================================================
--  5c. rh_feature_active —— 「当前快照是哪一条」
--
--  【为什么单独一张表，而不是快照表上的 is_active 布尔列】
--  这是本项目第一版的设计错误，值得记下来（见 L048）：
--
--  第一版在 rh_feature_snapshots 上加了 ``is_active`` 与部分唯一索引
--  ``on (is_active) where is_active``，构建时把旧行 UPDATE 成 false、新行 true。
--  但同一张表上还有「不许 UPDATE」的不可变触发器 ——
--  **两者直接互斥**，第二次构建必然
--  ``RestrictViolation: 不允许 UPDATE``。
--  更根本地说：**「要翻转的当前指针」与「不可变的历史行」是两种东西，
--  不该挤在同一张表里。**
--
--  现在拆开：历史行只增不改（触发器守住），当前指针单独一行、可以随便改写。
--  「谁是当前」永远由这一张单行表回答，不需要扫历史表、也不依赖
--  「snapshot_id 最大」这类在并发/回滚后会出错的推断。
-- ============================================================================
create table if not exists rh_feature_active (
    -- 单行表：用常量主键把「只能有一行」变成数据库层面的约束
    only_one   boolean     primary key default true,
    snapshot_id bigint,
    updated_at timestamptz not null default now(),

    constraint rh_feature_active_one_ck check (only_one),
    constraint rh_feature_active_snapshot_fk foreign key (snapshot_id)
        references rh_feature_snapshots (snapshot_id) on delete set null
);

comment on table rh_feature_active is
    '单行表：指出 rh_client_features 当前对应哪一条快照。'
    '与「不可变的 rh_feature_snapshots」分开，因为当前指针必须可改写。';
comment on column rh_feature_active.snapshot_id is
    '当前快照。为 NULL 表示「还没有任何快照」（不要用「表里没有行」表示，'
    '那与「查询失败」无法区分）。';

create index if not exists rh_feature_active_snapshot_idx
    on rh_feature_active (snapshot_id);

-- ============================================================================
--  6. rh_client_features —— 客户特征快照
--
--  ⚠️ 客户集合的权威来源是 rh_clients（400,162 行）。
--     特征表用 **LEFT JOIN** 从 rh_clients 出发，保证：
--        · 没有交易记录的客户不丢（n_transactions=0, total_spend=0, 时间为 NULL）
--        · 用 has_purchase_history 显式标识，而不是靠 NULL 隐含
--     **绝不**用 INNER JOIN —— 那会静默丢客户，让下游分母口径整体偏移。
--
--  ⚠️⚠️ **本表的列集合必须与 src/dsa/features/retailhero_features.py 里
--     build_features() 的插入列清单逐列一致。** 见 L047：
--     原 DDL 缺 n_purchase_lines / n_brands / age / gender / age_plausible
--     5 列，而开发库是**历史上手工 ALTER 出来的**，于是「现有库上测试全绿」
--     与「从零部署直接 UndefinedColumn 报错」同时成立。
--     tests/test_stage5_retailhero_scratch_db.py 用空库 fixture 钉死这条：
--     任何一边漏列都会失败，不会再被「本机库恰好是补过的」掩盖。
-- ============================================================================
create table if not exists rh_client_features (
    client_id            text     primary key,

    -- ---- 交易级特征（源字段已按 transaction_id 收敛）----
    n_transactions       integer  not null default 0,
    total_spend          numeric(16,2) not null default 0,
    avg_basket           numeric(16,4),            -- total_spend / n_transactions；0 笔交易时为 NULL

    -- ---- 明细级特征（可按行直接求和/去重）----
    n_items              numeric(16,3) not null default 0,
    n_purchase_lines     integer  not null default 0,   -- 明细行数 = count(*) over 明细
    n_distinct_products  integer  not null default 0,
    n_stores             integer  not null default 0,

    -- ---- 时间特征 ----
    first_txn_date       timestamptz,
    last_txn_date        timestamptz,
    -- ⚠️ 语义见下方 comment on column：**不同交易日的个数**，不是首末间隔。
    days_active          integer,

    -- ---- 品类 / 品牌特征 ----
    n_top_level1         integer  not null default 0,
    n_brands             integer  not null default 0,   -- count(distinct brand_id)，忽略 NULL

    -- ---- 客户维度（决策前信息，AGENTS.md §2.3 允许）----
    age                  smallint,
    gender               text,
    -- age 不合理时**显式标识**而不是丢弃该客户（与 L026 的同类处置一致）。
    -- 必须是 boolean（可为 NULL 由质量检查检出），不能用 0 冒充 false。
    age_plausible        boolean,

    -- ---- 显式标识：没有交易记录时为 false，此时上面计数为 0、时间为 NULL ----
    has_purchase_history boolean  not null default false,

    built_at             timestamptz not null default now(),

    -- 本行属于哪一次构建的元数据快照。见 L048 / rh_feature_snapshots。
    -- 只登记**元数据**：本表按客户主键 TRUNCATE 重写，历史特征值并不保留。
    snapshot_id          bigint,

    constraint rh_client_features_client_fk foreign key (client_id)
        references rh_clients (client_id) on delete cascade,

    constraint rh_client_features_snapshot_fk foreign key (snapshot_id)
        references rh_feature_snapshots (snapshot_id) on delete set null,

    -- 自洽性约束：没有购买历史 ⇒ 计数为 0、时间为空（防止 INNER JOIN 丢客户后被掩盖）
    constraint rh_cf_no_history_ck check (
        has_purchase_history
        or (n_transactions = 0 and total_spend = 0 and n_items = 0
            and n_purchase_lines = 0 and n_distinct_products = 0
            and n_stores = 0 and n_top_level1 = 0 and n_brands = 0
            and first_txn_date is null and last_txn_date is null
            and days_active is null and avg_basket is null)
    ),
    -- 有购买历史 ⇒ 至少一笔交易
    constraint rh_cf_history_ck check (not has_purchase_history or n_transactions > 0),
    constraint rh_cf_spend_ck   check (total_spend >= 0),
    constraint rh_cf_items_ck   check (n_items >= 0),
    constraint rh_cf_days_ck    check (days_active is null or days_active >= 0),
    -- age 要么缺失，要么被显式判定过合理性（防止「忘了算 age_plausible」）
    constraint rh_cf_age_ck     check (age is null or age_plausible is not null)
);

comment on table  rh_client_features is
    'RetailHero 客户特征快照。一行 = 一名 rh_clients 里的客户（含无交易客户）。'
    'total_spend 由交易级 purchase_sum 按 transaction_id 收敛后求和得到，'
    '**不是** sum(rh_purchases.purchase_sum)（后者虚高约 5.81 倍）。'
    '⚠️ 本表按 client_id 主键 TRUNCATE 后重写，因此只保留**最近一次**特征值；'
    '历史只以 rh_feature_snapshots 的元数据形式保留（行数、哈希、流水线版本），'
    '**不保留**历史特征值本身。';
comment on column rh_client_features.total_spend is
    '总消费 = Σ over distinct transactions of max(purchase_sum)。已按 transaction_id 去重。';
comment on column rh_client_features.avg_basket is
    '客单价 = total_spend / n_transactions。无交易时为 NULL（不是 0——0 会污染下游均值）。';
comment on column rh_client_features.n_items is
    '商品件数 = sum(product_quantity)。明细级字段，可直接求和，**不需要**按交易收敛。';
comment on column rh_client_features.n_purchase_lines is
    '购买明细行数 = 明细行 count(*)。与 n_items 不同：同一商品买 3 件是 1 行 / 3 件。';
comment on column rh_client_features.n_brands is
    '购买涉及的不同 brand_id 个数。count(distinct) 会忽略 NULL —— 品牌缺失的明细不计入。';
comment on column rh_client_features.days_active is
    '活跃天数 = **有过交易的「不同日历日」的个数**'
    '（count(distinct date_trunc(''day'', transaction_datetime at time zone ''UTC''))）。'
    '同一天内多笔交易算 1 天；单笔交易客户为 1。'
    '⚠️ 这**不是** last_txn_date - first_txn_date 的间隔天数 —— 两者语义不同，'
    '原注释写的是后者（见 L047 的裁决）。选「不同交易日个数」的理由：'
    '稀疏的月购客户与集中一天的客户，间隔天数可能相同而活跃度完全不同；'
    '活跃度特征必须反映「有多少天真的来过」，而不是首末两端之差。'
    '⚠️ 日期截断**必须显式 at time zone ''UTC''**：timestamptz 的 date_trunc '
    '会先按**会话时区**换算再截断，缺了它同一份数据在 Asia/Shanghai 与 UTC '
    '会话下会算出不同的天数（本项目实测 3 vs 2），特征值因此不可复现。';
comment on column rh_client_features.age is
    '客户年龄，源数据原样透传（含 -7491 / 1901 这类异常值，**不过滤不插补**）。'
    '缺失为 NULL —— 0 不是有效年龄。是否可信看 age_plausible。';
comment on column rh_client_features.gender is 'M/F/U，U = 未知。保留 U 不做插补（插补属建模阶段的事）';
comment on column rh_client_features.age_plausible is
    'age 是否落在 [14,100] 的可信区间。age 为 NULL 时本列为 NULL（未判定）；'
    'age 非 NULL 时必须非 NULL（rh_cf_age_ck 保证）。**不因不合理就删客户**。';
comment on column rh_client_features.has_purchase_history is
    '是否存在交易记录。false 时上面的计数一律为 0、时间为 NULL（由 CHECK 约束保证）。';
comment on column rh_client_features.snapshot_id is
    '指向 rh_feature_snapshots，登记本次构建的源文件哈希 / 流水线版本 / 历史边界 / 质量结果。'
    '⚠️ 特征值本身不留历史（TRUNCATE 重写），快照只保留元数据。';

create index if not exists rh_cf_history_idx on rh_client_features (has_purchase_history);
create index if not exists rh_cf_spend_idx   on rh_client_features (total_spend desc);
-- 外键列索引（PostgreSQL 不会自动建）。snapshot_id 是本表新增外键，见 L048。
create index if not exists rh_cf_snapshot_idx on rh_client_features (snapshot_id);

-- ============================================================================
--  7. 交易级收敛视图
--
--  这是「铁律 1」在 SQL 层的**唯一正确入口**：任何涉及交易级金额/点数的聚合
--  都应基于本视图，而不是直接读 rh_purchases。
--  用 max 而非 sum：组内若不一致，max <> min 会立刻暴露问题。
-- ============================================================================
create or replace view rh_transactions as
select
    client_id,
    transaction_id,
    max(transaction_datetime)     as transaction_datetime,
    max(store_id)                 as store_id,
    max(purchase_sum)             as purchase_sum,          -- 交易级字段取一行
    max(regular_points_received)  as regular_points_received,
    max(express_points_received)  as express_points_received,
    max(regular_points_spent)     as regular_points_spent,
    max(express_points_spent)     as express_points_spent,
    count(*)                      as n_lines,               -- 明细行数
    sum(product_quantity)         as n_items,               -- 明细级，可求和
    count(distinct product_id)    as n_distinct_products,
    -- 一致性探针：>0 说明交易级字段在组内不一致，聚合结果不可信
    (max(purchase_sum) is distinct from min(purchase_sum))::int
        + (max(transaction_datetime) is distinct from min(transaction_datetime))::int
                                  as inconsistency_flags
from rh_purchases
group by client_id, transaction_id;

comment on view rh_transactions is
    '交易级收敛视图：一行 = 一笔交易。'
    '交易级字段（purchase_sum、*_points_*）用 max 取一行，明细级字段（n_items）求和。'
    'inconsistency_flags > 0 表示组内交易级字段不一致，此时聚合结果不可信。'
    '⚠️ 任何 sum(purchase_sum) 都必须基于本视图，不能直接对 rh_purchases 求和。';

-- ============================================================================
--  8. 供只读账号使用的授权
-- ============================================================================
grant select on all tables    in schema public to dsa_ro;
grant select on all sequences in schema public to dsa_ro;
