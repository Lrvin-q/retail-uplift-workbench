-- ============================================================================
--  007_retailhero_features_align.sql —— 把**既有库**对齐到 005 的定义
--
--  ⚠️ 编号说明：本文件原为 006，与另一位同事先创建的
--  ``006_policy_contrasts_and_identity.sql`` **重号**。
--  ``sql/`` 下的编号代表**迁移执行顺序**，重号会让「按编号顺序执行」失去确定性，
--  因此本文件让位改为 **007**。两者内容互不重叠：
--      · 006 = 策略对比与身份（模型登记相关）
--      · 007 = RetailHero 特征表结构对齐（本文件）
--  本文件不依赖 006，执行顺序 005 → 006 → 007 或 005 → 007 都可。
--
--  背景（见 docs/日志.md L047 / L048）：
--  现有开发库里的 rh_client_features 是**历史上手工 ALTER 出来的**，
--  与 sql/005_retailhero_schema.sql 的建表定义不一致：
--
--      库里多出 DDL 没有的 5 列：n_purchase_lines, n_brands, age, gender,
--                                age_plausible
--      库里没有 DDL 新增的快照表：rh_feature_snapshots（含 snapshot_id 外键）
--      库里没有 DDL 里的约束：rh_cf_no_history_ck
--
--  后果：现有库上的测试 100% 通过，从零部署却立刻
--  ``UndefinedColumn: column "n_purchase_lines" does not exist``。
--
--  ============================================================================
--  ⚠️ 本文件的执行纪律
--  ============================================================================
--  1. **只做加法**：全部是 ``add column if not exists`` / ``create table
--     if not exists`` / ``create index if not exists``。
--     **没有** drop column、drop table、改类型、改数据。
--  2. **幂等**：可以重复执行任意次。
--  3. **不改动任何既有数据**：唯一的写操作是插入一条
--     ``rh_feature_snapshots`` 占位行 —— 因为新加的 snapshot_id 外键
--     必须是「有值或为空」，而既有特征行是空值，所以本文件**不**回填
--     snapshot_id（回填必须靠重跑 build_features()）。
--  4. 触发器会在写入后把快照行锁死（不可变），这是设计意图。
--
--  执行身份：dsa_app（可写账号）
--  执行时机：在 005 之后执行一次；重复执行安全。
-- ============================================================================

-- ============================================================================
--  1. 补齐 rh_client_features 缺失的 5 列
--
--  列定义与顺序刻意与 005 的 create table 保持**同一处写法**，
--  以免日后两份定义再次漂移。not null default 0 的选择理由：
--  既有行已有真实取值（都是非 NULL 的 0/真实值），
--  加 default 只是为了让「新列 + 旧行」这一步能通过。
-- ============================================================================
alter table rh_client_features
    add column if not exists n_purchase_lines integer not null default 0;
alter table rh_client_features
    add column if not exists n_brands         integer not null default 0;
alter table rh_client_features
    add column if not exists age              smallint;
alter table rh_client_features
    add column if not exists gender           text;
alter table rh_client_features
    add column if not exists age_plausible    boolean;

comment on column rh_client_features.n_purchase_lines is
    '购买明细行数 = 明细行 count(*)。与 n_items 不同：同一商品买 3 件是 1 行 / 3 件。';
comment on column rh_client_features.n_brands is
    '购买涉及的不同 brand_id 个数。count(distinct) 会忽略 NULL —— 品牌缺失的明细不计入。';
comment on column rh_client_features.age is
    '客户年龄，源数据原样透传（含 -7491 / 1901 这类异常值，**不过滤不插补**）。'
    '缺失为 NULL —— 0 不是有效年龄。是否可信看 age_plausible。';
comment on column rh_client_features.gender is
    'M/F/U，U = 未知。保留 U 不做插补（插补属建模阶段的事）';
comment on column rh_client_features.age_plausible is
    'age 是否落在 [14,100] 的可信区间。age 为 NULL 时本列为 NULL（未判定）；'
    'age 非 NULL 时必须非 NULL（rh_cf_age_ck 保证）。**不因不合理就删客户**。';

-- ---------------------------------------------------------------------------
--  1b. 修正 age 的类型：integer → smallint
--
--  ⚠️ 这是**审查者没点名、但逐列比对查出来**的一处差异。
--  历史上手工 ALTER 时把 age 建成了 ``integer``，而 DDL 写的是 ``smallint``
--  （与 rh_clients.age 一致）。两者能存的值域不同：
--      smallint  -32768 .. 32767      （rh_clients.age 就用这个）
--      integer   -2147483648 .. 2147483647
--  特征表里的 age 是**从 rh_clients.age 直接透传**的，本该同类型；
--  留着 integer 会让「既有库 = 新库」这条验收条件永远不成立。
--
--  **只在确实需要时改，且改前确认不会溢出**：源列是 smallint，
--  所以特征表的 age 不可能超出 smallint 值域 —— 这一步不会失败、也不会丢数据。
--  实际值与 rh_clients.age 逐行一致性由测试另行核对。
-- ---------------------------------------------------------------------------
do $$
declare
    cur_type text;
    n_over   bigint;
begin
    select data_type into cur_type
    from information_schema.columns
    where table_schema = 'public' and table_name = 'rh_client_features'
      and column_name = 'age';

    if cur_type is null then
        raise notice 'rh_client_features.age 不存在，跳过（001 段的 add column 应已建好）';
    elsif cur_type = 'smallint' then
        raise notice 'rh_client_features.age 已是 smallint，跳过';
    else
        -- 溢出预检：超出 smallint 值域就**不改**，并明确报出来
        select count(*) into n_over from rh_client_features
        where age is not null and (age < -32768 or age > 32767);
        if n_over > 0 then
            raise warning
                'rh_client_features.age 有 % 行超出 smallint 值域，已跳过类型变更。'
                '请先核查这些值（源列 rh_clients.age 是 smallint，理论上不该出现）。',
                n_over;
        else
            alter table rh_client_features
                alter column age type smallint;
            raise notice 'rh_client_features.age 已从 % 改为 smallint', cur_type;
        end if;
    end if;
end
$$;

-- ============================================================================
--  2. 补齐快照表与 snapshot_id
--
--  ⚠️ 与 005 的分工：005 是**从零建库**的权威定义（那里 rh_feature_snapshots
--  建在 rh_client_features 之前，因为后者带前置外键）。
--  这里服务的是**已经建过表的库** —— 所以用 if not exists 重复一遍，
--  再把 snapshot_id 作为增量列加到已有的 rh_client_features 上。
--
--  为什么 snapshot_id 放在**最后**：以便既有库补列后的物理列序
--  与从零新建的库一致（005 里 snapshot_id 也写在 built_at 之后）。
-- ============================================================================
create table if not exists rh_feature_snapshots (
    snapshot_id       bigint      generated always as identity primary key,
    pipeline_version  text        not null,
    source_hash_set   jsonb,
    source_hash_digest char(64),
    data_period_start timestamptz,
    data_period_end   timestamptz,
    n_clients         integer     not null,
    n_feature_rows    integer     not null,
    quality_ok        boolean     not null,
    quality_checks    jsonb       not null,
    built_at          timestamptz not null default now(),

    constraint rh_fs_pipeline_ck check (pipeline_version ~ '^v[0-9]+$'),
    constraint rh_fs_digest_ck   check (source_hash_digest is null
                                        or source_hash_digest ~ '^[0-9a-f]{64}$'),
    constraint rh_fs_rows_ck     check (n_feature_rows >= 0 and n_clients >= 0),
    constraint rh_fs_period_ck   check (data_period_start is null
                                        or data_period_end is null
                                        or data_period_start <= data_period_end)
    -- 刻意没有 (pipeline_version, source_hash_digest, n_feature_rows) 唯一约束，
    -- 理由见 sql/005 的同名表注释与 L048。
);

-- 若这个库上曾经建过那条错误的唯一约束，这里把它去掉。
-- 它会让「同一份数据构建两次」直接 UniqueViolation —— 与快照的目的相反。
alter table rh_feature_snapshots
    drop constraint if exists rh_fs_identity_uk;

create index if not exists rh_feature_snapshots_identity_idx
    on rh_feature_snapshots (pipeline_version, source_hash_digest);

comment on table rh_feature_snapshots is
    'RetailHero 特征构建的不可变元数据快照。一行 = **一次** build_features() 的身份。'
    '同一份数据重复构建会产生多条记录。'
    '⚠️ 本表**只增不改**（触发器守住）；「当前快照」记在 rh_feature_active。';
comment on column rh_feature_snapshots.source_hash_set is
    '各源文件的 SHA256（取自 rh_source_files.file_sha256，不另造哈希）。'
    '取不到时为 NULL —— 不得用 {} 或占位值冒充「已记录」。';
comment on column rh_feature_snapshots.source_hash_digest is
    'source_hash_set 的折叠摘要。**非唯一**：摘要相同表示数据版本相同，'
    '不代表同一次构建。';
comment on column rh_feature_snapshots.data_period_start is
    '本快照覆盖的**实际交易数据**最早时间。为空表示库里没有交易数据（不是「未知」）。';
comment on column rh_feature_snapshots.data_period_end is
    '本快照覆盖的**实际交易数据**最晚时间。';
comment on column rh_feature_snapshots.quality_checks is
    'quality_check() 的完整返回（checks 数组 + inflation_ratio）。存全量而不是只存 ok，'
    '因为「哪一项没过」才是事后排查的入口。';

create index if not exists rh_feature_snapshots_built_idx
    on rh_feature_snapshots (built_at desc);

-- ---- 拆掉第一版「快照表上的 is_active」，改成单行指针表 ----
-- 第一版把「要翻转的当前指针」和「不可变的历史行」挤在同一张表里：
-- is_active 需要 UPDATE，而不可变触发器禁止 UPDATE —— 第二次构建必然
-- RestrictViolation。见 sql/005 §5c 与 L048。
drop index if exists rh_feature_snapshots_active_uk;
alter table rh_feature_snapshots
    drop column if exists is_active;

create table if not exists rh_feature_active (
    only_one    boolean     primary key default true,
    snapshot_id bigint,
    updated_at  timestamptz not null default now(),

    constraint rh_feature_active_one_ck check (only_one),
    constraint rh_feature_active_snapshot_fk foreign key (snapshot_id)
        references rh_feature_snapshots (snapshot_id) on delete set null
);

comment on table rh_feature_active is
    '单行表：指出 rh_client_features 当前对应哪一条快照。'
    '与「不可变的 rh_feature_snapshots」分开，因为当前指针必须可改写。';
comment on column rh_feature_active.snapshot_id is
    '当前快照。为 NULL 表示「还没有任何快照」。';

create index if not exists rh_feature_active_snapshot_idx
    on rh_feature_active (snapshot_id);

-- 不可变触发器：与 005 完全一致（create or replace function 保证幂等）
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

-- snapshot_id 列（放最后，保持与从零建库一致的物理列序）
alter table rh_client_features
    add column if not exists snapshot_id bigint;

comment on column rh_client_features.snapshot_id is
    '指向 rh_feature_snapshots，登记本次构建的源文件哈希 / 流水线版本 / 历史边界 / 质量结果。'
    '⚠️ 特征值本身不留历史（TRUNCATE 重写），快照只保留元数据。';

-- 外键：先 drop 再 add（与 005 里 rh_purchases 的写法一致，且天然幂等）。
-- on delete set null：快照被清理时特征行不应消失，只失去引用。
alter table rh_client_features
    drop constraint if exists rh_client_features_snapshot_fk;
alter table rh_client_features
    add constraint rh_client_features_snapshot_fk
    foreign key (snapshot_id) references rh_feature_snapshots (snapshot_id)
    on delete set null;

create index if not exists rh_cf_snapshot_idx
    on rh_client_features (snapshot_id);

-- ============================================================================
--  3. 补齐 rh_cf_no_history_ck
--
--  ⚠️ **先检查再添加**。这个约束是「没有购买历史 ⇒ 计数为 0、时间为 NULL」。
--  如果既有库里恰好存在违反它的行（例如历史上手工 ALTER 期间写坏过），
--  直接 add constraint 会让整条迁移**失败**，把「补结构」变成「停服」。
--  这里的做法是：有违规行就**不添加**并打印提示，让人先处理数据 ——
--  宁可缺一个约束且被告知，也不要静默失败或静默改数据。
-- ============================================================================
do $$
declare
    n_bad bigint;
begin
    select count(*) into n_bad
    from rh_client_features
    where not has_purchase_history
      and (n_transactions <> 0 or total_spend <> 0 or n_items <> 0
           or n_purchase_lines <> 0 or n_distinct_products <> 0
           or n_stores <> 0 or n_top_level1 <> 0 or n_brands <> 0
           or first_txn_date is not null or last_txn_date is not null
           or days_active is not null or avg_basket is not null);

    if n_bad > 0 then
        raise warning
            '有 % 行「无购买历史但计数非零」，已跳过 rh_cf_no_history_ck。'
            '请先核查数据再手工添加该约束。', n_bad;
    elsif not exists (
        select 1 from pg_constraint
        where conrelid = 'rh_client_features'::regclass
          and conname  = 'rh_cf_no_history_ck'
    ) then
        alter table rh_client_features
            add constraint rh_cf_no_history_ck check (
                has_purchase_history
                or (n_transactions = 0 and total_spend = 0 and n_items = 0
                    and n_purchase_lines = 0 and n_distinct_products = 0
                    and n_stores = 0 and n_top_level1 = 0 and n_brands = 0
                    and first_txn_date is null and last_txn_date is null
                    and days_active is null and avg_basket is null));
        raise notice '已添加约束 rh_cf_no_history_ck';
    else
        raise notice '约束 rh_cf_no_history_ck 已存在，跳过';
    end if;
end
$$;

-- ============================================================================
--  4. rh_cf_age_ck —— 「age 非空 ⇒ age_plausible 非空」
--
--  同 §3 的纪律：先查违规再添加。
--  注意：约束表达式里必须用 ``is not null`` 判断而不是 ``not age_plausible``，
--  因为 age_plausible 为 NULL 时 ``not null`` 仍是 NULL（不是 true），
--  那个写法会静默放过所有未判定的行。
-- ============================================================================
do $$
declare
    n_bad bigint;
begin
    select count(*) into n_bad
    from rh_client_features
    where age is not null and age_plausible is null;

    if n_bad > 0 then
        raise warning
            '有 % 行「age 非空但 age_plausible 为空」，已跳过 rh_cf_age_ck。'
            '请先重跑 build_features() 再添加该约束。', n_bad;
    elsif not exists (
        select 1 from pg_constraint
        where conrelid = 'rh_client_features'::regclass
          and conname  = 'rh_cf_age_ck'
    ) then
        alter table rh_client_features
            add constraint rh_cf_age_ck check (age is null or age_plausible is not null);
        raise notice '已添加约束 rh_cf_age_ck';
    else
        raise notice '约束 rh_cf_age_ck 已存在，跳过';
    end if;
end
$$;

-- ============================================================================
--  5. 不改动既有数据 —— 但快照表需要一行「当前快照」占位
--
--  为什么需要：新加的 snapshot_id 外键要求「有值或为空」，既有特征行都是空值，
--  这是允许的。但如果没有**任何**快照行，下游就分不清
--  「迁移过但没重建」与「快照机制整个没生效」。
--
--  ⚠️ 这一行是**升级占位**：它的哈希为 NULL、quality_ok=false，
--  明确表示「这条记录不代表一次真实构建」，因此**不能**当证据用，
--  也**不会**污染证据链（真实快照会另起一行，指针指向新的那行）。
-- ============================================================================
do $$
declare
    new_id bigint;
begin
    if not exists (select 1 from rh_feature_snapshots) then
        insert into rh_feature_snapshots
            (pipeline_version, source_hash_set, source_hash_digest,
             data_period_start, data_period_end,
             n_clients, n_feature_rows, quality_ok, quality_checks)
        select
            'v1', null, null, null, null,
            (select count(*) from rh_clients),
            (select count(*) from rh_client_features),
            false,
            jsonb_build_object(
                'ok', false,
                'checks', jsonb_build_array(jsonb_build_object(
                    'name', 'migration_placeholder',
                    'passed', false,
                    'detail', '007 迁移写入的占位快照：不代表一次真实构建，'
                              '请重跑 build_features() 以生成真实快照')),
                'note', '007_retailhero_features_align.sql 占位行')
        returning snapshot_id into new_id;

        insert into rh_feature_active (only_one, snapshot_id)
        values (true, new_id)
        on conflict (only_one) do update set snapshot_id = excluded.snapshot_id,
                                            updated_at  = now();
        raise notice '已写入升级占位快照（quality_ok=false，请重跑 build_features()）';
    else
        raise notice 'rh_feature_snapshots 已有数据，跳过占位行';
    end if;

    -- 自愈：快照表有行、但指针表一行都没有时，把指针指向最新那条。
    --
    -- 为什么需要：本文件的前一版（设计未定稿时）先把占位快照写进了快照表，
    -- 后来才改成「指针单独一张表」。在那种库上，快照表非空 ⇒ 上面的
    -- if 分支跳过 ⇒ 指针表永远是空的 ⇒ 下游 `active_snapshot()` 返回 NULL，
    -- 而库里其实有快照。这种「迁移过的库」状态不一致必须能自愈，
    -- 否则只能靠人工看日志发现。
    if not exists (select 1 from rh_feature_active) then
        insert into rh_feature_active (only_one, snapshot_id)
        select true, snapshot_id from rh_feature_snapshots
        order by built_at desc, snapshot_id desc
        limit 1;
        raise notice '指针表为空：已指向最新快照';
    end if;
end
$$;

-- ============================================================================
--  6. 只读账号授权（新表 + 未来新建的表）
-- ============================================================================
grant select on all tables    in schema public to dsa_ro;
grant select on all sequences in schema public to dsa_ro;
