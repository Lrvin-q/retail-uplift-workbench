-- ============================================================================
--  001_schema.sql —— 核心关系模型
--
--  项目：零售营销增量决策助手
--  依据：实现蓝图.md 阶段二；项目详细思路.md §9；docs/02-分析协议.md
--  设计规范：.dsh/skills/supabase-postgres-best-practices
--      · 主键用 bigint generated always as identity
--      · 字符串用 text（不用 varchar(n)）
--      · 时间用 timestamptz
--      · 金额用 numeric（不用 float）
--      · 所有外键列建索引（PostgreSQL 不会自动建）
--
--  执行身份：dsa_app（可写账号）
--  本文件可重复执行（幂等）。
--
--  ⚠️ 设计要点：本模型严格区分「实验前特征」与「干预后结果」。
--     干预后结果只存在于 outcomes 表，不得混入 customers 表。
-- ============================================================================

-- ============================================================================
--  1. datasets —— 一个数据版本
-- ============================================================================
create table if not exists datasets (
    dataset_id      bigint generated always as identity primary key,
    dataset_key     text        not null,               -- 数据集标识，如 'hillstrom'
    version_label   text        not null,               -- 版本，如 'hillstrom@2008-03-20'
    source_url      text,
    file_name       text        not null,
    file_sha256     char(64)    not null,
    file_bytes      bigint,
    row_count       integer     not null,
    grain           text        not null default 'customer',
    source_verified boolean     not null default false, -- 是否已核查来源
    notes           text,
    loaded_at       timestamptz not null default now(),

    constraint datasets_version_uk   unique (version_label),
    constraint datasets_sha_ck       check (file_sha256 ~ '^[0-9a-f]{64}$'),
    constraint datasets_rowcount_ck  check (row_count > 0),
    constraint datasets_key_ck       check (dataset_key ~ '^[a-z][a-z0-9_]*$')
);

comment on table  datasets is '数据版本登记表。一行代表一个不可变的数据版本。';
comment on column datasets.version_label is '数据版本标识，所有分析运行必须记录它';
comment on column datasets.file_sha256   is '原始文件 SHA256，用于复现与防篡改';
comment on column datasets.grain         is '一行代表什么实体；hillstrom 为 customer';

-- ============================================================================
--  2. customers —— 某数据版本的一名客户（仅含实验前信息）
-- ============================================================================
create table if not exists customers (
    customer_id      bigint generated always as identity primary key,
    dataset_id       bigint      not null references datasets(dataset_id) on delete restrict,

    -- 带数据集命名空间的业务键：禁止跨数据集拼接客户
    customer_key     text        not null,
    source_row_number integer    not null,   -- 源文件中的 1-based 行号

    -- ---- 实验前特征（docs/02-分析协议.md §16 允许的建模输入）----
    recency          smallint    not null,
    history_segment  text        not null,
    -- 派生列：从 history_segment 解析出的档位序数，由 PostgreSQL 保证与源字段一致
    history_band     smallint    generated always as
                       ((substring(history_segment from '^([0-9]+)'))::smallint) stored,
    history          numeric(12,2) not null,
    mens             boolean     not null,
    womens           boolean     not null,
    zip_code         text        not null,
    newbie           boolean     not null,
    channel          text        not null,

    -- ---- 约束 ----
    -- ⚠️ 唯一约束保证「内部客户 ID ↔ 源行」是一一映射，绝不按整行内容去重
    constraint customers_dataset_row_uk unique (dataset_id, source_row_number),
    constraint customers_key_uk         unique (customer_key),

    constraint customers_recency_ck  check (recency between 1 and 12),
    constraint customers_history_ck  check (history > 0),
    constraint customers_band_ck     check (history_band between 1 and 7),
    -- 原始数据拼写即为 Surburban（非 Suburban），此处如实保存，不做"修正"
    constraint customers_zip_ck      check (zip_code in ('Surburban','Urban','Rural')),
    constraint customers_channel_ck  check (channel in ('Web','Phone','Multichannel')),
    constraint customers_segment_ck  check (
        history_segment in (
            '1) $0 - $100','2) $100 - $200','3) $200 - $350','4) $350 - $500',
            '5) $500 - $750','6) $750 - $1,000','7) $1,000 +'
        )
    )
);

comment on table  customers is '客户维度表。一行代表「某数据版本内的一名客户」，只含实验前信息。';
comment on column customers.customer_key      is '带数据集命名空间的业务键，格式 <dataset_key>#<源行号>。禁止跨数据集拼接';
comment on column customers.source_row_number is '源文件行号，用于回溯到原始记录';
comment on column customers.history_band      is '派生列：history_segment 的档位序数 1-7，由数据库自动维护';
comment on column customers.recency           is '距上次购买的月数（单位待核实，见 docs/00）';

-- 防御性说明：本表刻意不含 visit / conversion / spend。
-- 干预后结果一律放在 outcomes 表，避免特征泄漏。

create index if not exists customers_dataset_idx      on customers (dataset_id);
create index if not exists customers_band_idx         on customers (history_band);

-- ============================================================================
--  3. experiments —— 一个实验配置
-- ============================================================================
create table if not exists experiments (
    experiment_id        bigint generated always as identity primary key,
    experiment_key       text        not null,
    dataset_id           bigint      not null references datasets(dataset_id) on delete restrict,
    name                 text        not null,

    randomization_unit   text        not null,
    analysis_unit        text        not null,
    outcome_window_days  smallint    not null,
    outcome_window_fixed boolean     not null default true,

    -- SRM 预期比例的来源。禁止从观测比例反推（docs/02 §11.1）
    expected_ratio_source text       not null,
    expected_ratio_note   text,

    notes                text,
    created_at           timestamptz not null default now(),

    constraint experiments_key_uk      unique (experiment_key),
    constraint experiments_rand_ck     check (randomization_unit = 'customer'),
    constraint experiments_analysis_ck check (analysis_unit = 'customer'),
    constraint experiments_window_ck   check (outcome_window_days > 0),
    constraint experiments_ratio_src_ck check (
        expected_ratio_source in ('original_description','design_document','derived')
    )
);

comment on table  experiments is '实验配置表。一行代表一个已结束、窗口固定的随机实验。';
comment on column experiments.randomization_unit   is '随机化单位。必须与分析单位一致（docs/02 §6）';
comment on column experiments.expected_ratio_source is 'SRM 预期比例的权威来源；=original_description 表示原始说明明确写出';
comment on column experiments.outcome_window_fixed is '窗口是否固定。true 表示不支持任意日期筛选';

-- PostgreSQL 不会自动为外键列建索引，必须显式创建
create index if not exists experiments_dataset_idx on experiments (dataset_id);

-- ============================================================================
--  4. experiment_arms —— 实验的一个组
-- ============================================================================
create table if not exists experiment_arms (
    arm_id         bigint generated always as identity primary key,
    experiment_id  bigint    not null references experiments(experiment_id) on delete cascade,
    arm_key        text      not null,      -- 'mens' / 'womens' / 'control'
    segment_value  text      not null,      -- 原始数据中的 segment 取值
    role           text      not null,
    expected_ratio numeric(9,8) not null,

    constraint experiment_arms_key_uk  unique (experiment_id, arm_key),
    constraint experiment_arms_seg_uk  unique (experiment_id, segment_value),
    constraint experiment_arms_role_ck check (role in ('treatment','control')),
    constraint experiment_arms_ratio_ck check (expected_ratio > 0 and expected_ratio <= 1)
);

comment on table  experiment_arms is '实验组定义。一行代表一个组（干预组或对照组）。';
comment on column experiment_arms.segment_value is '与源数据 segment 字段的取值一一对应';
comment on column experiment_arms.expected_ratio is '设计上的预期分配比例，SRM 检验用';

create index if not exists experiment_arms_exp_idx on experiment_arms (experiment_id);

-- ============================================================================
--  5. experiment_comparisons —— 预注册的比较（主要 / 次要）
-- ============================================================================
create table if not exists experiment_comparisons (
    comparison_id     bigint generated always as identity primary key,
    experiment_id     bigint    not null references experiments(experiment_id) on delete cascade,
    comparison_key    text      not null,   -- 'primary' / 'secondary_womens'
    treatment_arm_id  bigint    not null references experiment_arms(arm_id),
    control_arm_id    bigint    not null references experiment_arms(arm_id),
    kind              text      not null,
    alpha             numeric(6,5) not null,
    sided             smallint  not null,
    multiplicity_note text,

    constraint experiment_comparisons_key_uk unique (experiment_id, comparison_key),
    constraint experiment_comparisons_kind_ck check (kind in ('confirmatory','exploratory')),
    constraint experiment_comparisons_diff_ck check (treatment_arm_id <> control_arm_id),
    constraint experiment_comparisons_alpha_ck check (alpha > 0 and alpha < 1),
    constraint experiment_comparisons_sided_ck check (sided in (1,2))
);

comment on table  experiment_comparisons is '预注册的比较。主要比较用于确认性结论，次要比较为探索性。';
comment on column experiment_comparisons.kind is 'confirmatory=确认性（主要）；exploratory=探索性（需多重检验说明）';

create index if not exists exp_cmp_exp_idx  on experiment_comparisons (experiment_id);
create index if not exists exp_cmp_trt_idx  on experiment_comparisons (treatment_arm_id);
create index if not exists exp_cmp_ctrl_idx on experiment_comparisons (control_arm_id);

-- ============================================================================
--  6. assignments —— 一名客户的一次实验分组
-- ============================================================================
create table if not exists assignments (
    assignment_id bigint generated always as identity primary key,
    experiment_id bigint not null references experiments(experiment_id) on delete cascade,
    customer_id   bigint not null references customers(customer_id)   on delete cascade,
    arm_id        bigint not null references experiment_arms(arm_id),
    assigned_at   timestamptz not null default now(),

    -- 同一客户在同一实验里只能被分配一次
    constraint assignments_uk unique (experiment_id, customer_id)
);

comment on table  assignments is '实验分组表。一行代表「一名客户在一次实验中的分组」。';
comment on column assignments.assigned_at is '写入本表的时间，不是实验的实际分配时间（源数据无该信息）';

create index if not exists assignments_exp_idx      on assignments (experiment_id);
create index if not exists assignments_customer_idx on assignments (customer_id);
create index if not exists assignments_arm_idx      on assignments (arm_id);

-- ============================================================================
--  7. outcomes —— 一名客户在固定窗口下的结果
-- ============================================================================
create table if not exists outcomes (
    outcome_id     bigint generated always as identity primary key,
    experiment_id  bigint not null references experiments(experiment_id) on delete cascade,
    customer_id    bigint not null references customers(customer_id)   on delete cascade,

    window_days    smallint not null,       -- 观察窗口，本数据集为 14
    visited        boolean  not null,
    converted      boolean  not null,
    spend          numeric(12,2) not null,

    constraint outcomes_uk unique (experiment_id, customer_id),
    constraint outcomes_window_ck check (window_days > 0),
    constraint outcomes_spend_ck  check (spend >= 0),

    -- 以下两条 CHECK 已用全部 64,000 行实测，0 违反（见 docs/日志.md 与阶段二验证记录）
    -- 1) 未访问网站不可能产生购买
    constraint outcomes_visit_before_conversion_ck check (visited or not converted),
    -- 2) 购买与消费金额必须一致：买了就一定有消费，没买就一定没有
    constraint outcomes_conversion_spend_ck check (
        (converted and spend > 0) or (not converted and spend = 0)
    )
);

comment on table  outcomes is '结果表。一行代表「一名客户在一次实验固定窗口内的结果」。';
comment on column outcomes.window_days is '观察窗口天数。窗口由实验设计固定，不支持任意区间';
comment on column outcomes.converted   is '是否购买。NULL 表示结果缺失，绝不等同于 false';
comment on column outcomes.spend       is '消费金额。未购买必须为 0（由 CHECK 保证）';

create index if not exists outcomes_exp_idx      on outcomes (experiment_id);
create index if not exists outcomes_customer_idx on outcomes (customer_id);
create index if not exists outcomes_conv_idx     on outcomes (experiment_id, converted);

-- 覆盖索引（性能关键，见 docs/日志.md L009）：
-- 让「按实验聚合各组指标」这类核心查询能走 Index Only Scan + Hash Left Join。
-- 未加之前：64,000 次索引查找的嵌套循环、192,545 次缓冲区命中、170 ms。
-- 加上之后：Hash Left Join、1,011 次缓冲区命中、89 ms。两者结果完全一致。
create index if not exists outcomes_exp_cov_idx on outcomes (experiment_id, customer_id)
    include (converted, visited, spend);

-- ============================================================================
--  8. analysis_runs —— 一次分析运行
-- ============================================================================
create table if not exists analysis_runs (
    run_id         uuid primary key default gen_random_uuid(),
    experiment_id  bigint      references experiments(experiment_id),
    dataset_id     bigint      references datasets(dataset_id),
    comparison_key text,
    question       text,
    plan           jsonb,
    status         text        not null default 'pending',
    result_refs    jsonb,
    report_ref     text,
    code_version   text,
    stats_config   jsonb,
    error_code     text,
    error_detail   text,
    started_at     timestamptz not null default now(),
    finished_at    timestamptz,

    constraint analysis_runs_status_ck check (
        status in ('pending','running','succeeded','blocked','failed')
    )
);

comment on table  analysis_runs is '分析运行登记表。一行代表一次分析运行，是报告追溯的入口。';
comment on column analysis_runs.run_id       is '运行标识。报告、查询、产物都通过它关联';
comment on column analysis_runs.stats_config is '本次运行使用的统计参数（方法、α、种子等），复现所需';
comment on column analysis_runs.code_version is '代码版本（git commit），复现所需';

create index if not exists analysis_runs_exp_idx     on analysis_runs (experiment_id);
create index if not exists analysis_runs_dataset_idx on analysis_runs (dataset_id);
create index if not exists analysis_runs_status_idx  on analysis_runs (status);
create index if not exists analysis_runs_started_idx on analysis_runs (started_at desc);

-- ============================================================================
--  9. query_executions —— 一次查询执行
-- ============================================================================
create table if not exists query_executions (
    query_exec_id  bigint generated always as identity primary key,
    run_id         uuid        references analysis_runs(run_id) on delete cascade,
    seq            integer     not null,
    tool_name      text,
    sql_text       text        not null,
    params         jsonb,
    row_count      integer,
    duration_ms    integer,
    status         text        not null,
    error_code     text,
    error_detail   text,
    executed_at    timestamptz not null default now(),

    constraint query_executions_status_ck check (status in ('ok','error','timeout','rejected')),
    constraint query_executions_seq_uk    unique (run_id, seq)
);

comment on table  query_executions is '查询执行记录。一行代表一次 SQL 执行，用于追溯与诊断。';
comment on column query_executions.sql_text is '实际执行的 SQL 文本（已参数化的最终形态）';

create index if not exists query_executions_run_idx on query_executions (run_id);

-- ============================================================================
--  10. 分析基础视图
--
--  ⚠️ 关键设计：用 LEFT JOIN 关联 outcomes，使「结果缺失」保持为 NULL。
--     这样缺失结果既不会被静默丢弃，也不会被当成「未购买」。
--     质量检查通过统计 NULL 数量来发现覆盖率问题。
-- ============================================================================
create or replace view v_experiment_customer as
select
    e.experiment_key,
    e.experiment_id,
    d.version_label                      as dataset_version,
    cu.customer_id,
    cu.customer_key,
    a.arm_id,
    ar.arm_key,
    ar.role                              as arm_role,
    -- 实验前特征
    cu.recency,
    cu.history_segment,
    cu.history_band,
    cu.history,
    cu.mens,
    cu.womens,
    cu.zip_code,
    cu.newbie,
    cu.channel,
    -- 干预后结果（可能为 NULL，表示结果缺失）
    o.visited,
    o.converted,
    o.spend,
    o.window_days,
    -- 缺失标记：便于质量检查与分母核算
    (o.outcome_id is null)               as outcome_missing
from assignments a
join experiments      e  on e.experiment_id = a.experiment_id
join datasets         d  on d.dataset_id    = e.dataset_id
join customers        cu on cu.customer_id  = a.customer_id
join experiment_arms  ar on ar.arm_id       = a.arm_id
left join outcomes    o  on o.experiment_id = a.experiment_id
                        and o.customer_id   = a.customer_id;

comment on view v_experiment_customer is
    '分析基础视图：一行 = 一名客户在一次实验中的分组与结果（含实验前特征）。
     已购买率的分母必须是本视图的全部行（按 arm 分组计数），
     绝不允许按 visited 过滤——那会破坏随机化（docs/02 §8.1）。';

-- ============================================================================
--  11. 供只读账号使用的授权
--     dsa_app 以后新建的对象，dsa_ro 通过 ALTER DEFAULT PRIVILEGES 自动获得 SELECT；
--     此处为已存在的对象补授一次。
-- ============================================================================
grant select on all tables    in schema public to dsa_ro;
grant select on all sequences in schema public to dsa_ro;
