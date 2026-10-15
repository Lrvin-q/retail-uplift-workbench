-- ============================================================================
--  003_models.sql —— 模型登记、客户分数与策略评估结果
--
--  项目：零售营销增量决策助手
--  依据：docs/06-建模与策略评估协议.md §8（版本登记与产物）、§6（预先固定的评估规则）
--        docs/02-分析协议.md §16（允许的建模输入）
--  设计规范：.dsh/skills/supabase-postgres-best-practices
--      · 主键用 bigint generated always as identity
--      · 字符串用 text（不用 varchar(n)）
--      · 时间用 timestamptz，默认 now()
--      · 比例用 numeric（不用 float/real/double）
--      · 所有外键列显式建索引（PostgreSQL 不会自动建）
--      · 每张表与每个列都有 COMMENT
--
--  执行身份：dsa_app（可写账号）。本文件可重复执行（幂等）。
--
--  ⚠️ 三条与本模型直接相关的铁律（在数据库层用 CHECK 强制或显式记录）：
--     1. 干预后结果（visit / conversion / spend）**不得**进入 feature_columns
--        —— 由 models_features_no_post_treatment_ck 在数据库层拦截（docs/06 §1.2）。
--     2. 策略名单的选取变量只能是**干预前**分数（uplift / propensity_score）；
--        选取人群的观测购买率因选择偏差**仅作描述**，不得作为策略优劣依据
--        —— 见 sql/004_policy_queries.sql 的 top_decile_check。
--     3. 一名客户在一个模型版本下只能有一行分数
--        —— 由 customer_scores_model_customer_uk 保证，重复写入用 ON CONFLICT DO NOTHING。
-- ============================================================================


-- ============================================================================
--  1. models —— 一个已训练的模型版本
--
--  一行代表「一次离线训练产生的一个模型版本」。训练是离线任务，
--  Agent 只能读已登记的模型，不得触发训练（docs/06 §8.1）。
-- ============================================================================
create table if not exists models (
    model_id        bigint generated always as identity primary key,

    -- 业务键：形如 tlearner_lr_v1 / outcome_model_lr_v1
    model_key       text        not null,
    model_type      text        not null,

    -- 数据与实验来源：模型绝不能跨数据集使用（AGENTS.md §2.5）
    dataset_id      bigint      not null references datasets(dataset_id)       on delete restrict,
    experiment_id   bigint      not null references experiments(experiment_id) on delete restrict,
    comparison_key  text        not null default 'primary',

    -- 特征集合：明确的列名列表（docs/06 §8「特征集合」要求）
    feature_columns text[]      not null,

    base_learner    text        not null,          -- 如 logistic_regression
    params          jsonb       not null default '{}'::jsonb,   -- 全部超参数
    split_config    jsonb       not null default '{}'::jsonb,   -- 种子/比例/各集合人数
    code_version    text        not null,          -- git commit
    trained_at      timestamptz not null default now(),
    notes           text,

    -- ---- 约束 ----
    constraint models_key_uk          unique (model_key),
    constraint models_key_ck          check (model_key ~ '^[a-z][a-z0-9_]*$'),
    constraint models_type_ck         check (model_type in ('t_learner','outcome_model')),
    constraint models_base_learner_ck check (base_learner <> ''),
    constraint models_code_version_ck check (code_version <> ''),
    constraint models_features_ck     check (cardinality(feature_columns) > 0),

    -- ⚠️ 数据泄漏防线（docs/06 §1.2）：干预后字段绝不允许作为模型输入。
    --    用数组重叠（&&）做**精确元素**匹配，故 'history' 不会误伤 'history_band'。
    constraint models_features_no_post_treatment_ck check (
        not (feature_columns && array['visit','visited','conversion','converted',
                                      'spend','visit_rate','conversion_rate'])
    ),

    -- 两个 jsonb 必须是对象，防止把数组或标量写进来
    constraint models_params_obj_ck   check (jsonb_typeof(params) = 'object'),
    constraint models_split_obj_ck    check (jsonb_typeof(split_config) = 'object')
);

comment on table  models is
    '模型登记表。一行代表一个已训练的模型版本（离线训练产物），是分数与策略评估的归属主体。';
comment on column models.model_id       is '代理主键（identity）；对外引用模型一律用 model_key';
comment on column models.model_key      is '模型版本业务键，如 tlearner_lr_v1；全局唯一，重复登记返回已有行';
comment on column models.model_type     is 't_learner=增量模型（μ1-μ0）；outcome_model=购买概率模型';
comment on column models.dataset_id     is '训练所用数据版本；与 001 的 datasets 对应，禁止跨数据集复用';
comment on column models.experiment_id  is '训练所用实验；决定分数所属人群与 IPS 的分配概率来源';
comment on column models.comparison_key is '使用的预注册比较（默认 primary），决定建模人群的两个组';
comment on column models.feature_columns is
    '模型输入列名（含预处理后的特征名）。禁止出现 visit/conversion/spend（由 CHECK 强制）';
comment on column models.base_learner   is '基础学习器名称，如 logistic_regression';
comment on column models.params         is '全部超参数（jsonb），复现所需';
comment on column models.split_config   is
    '划分配置：期望含 seed / ratios / counts / stratify_by（docs/06 §8 要求种子、比例、各集合人数）';
comment on column models.code_version   is '代码版本（git commit），复现所需';
comment on column models.trained_at     is '训练完成时间；与行写入时间无关，登记时显式给出或取 now()';
comment on column models.notes          is '自由备注，如「第一轮固定 LogisticRegression，不做超参搜索」';

-- 外键列显式索引
create index if not exists models_dataset_idx    on models (dataset_id);
create index if not exists models_experiment_idx on models (experiment_id);


-- ============================================================================
--  2. customer_scores —— 一名客户在一个模型版本下的打分
--
--  一行代表「某模型版本对某客户的一次打分」。这是策略名单的唯一数据来源。
--  规模：单个模型约 42,613 行（docs/06 §2.3 的建模人群），必须批量写入（COPY）。
-- ============================================================================
create table if not exists customer_scores (
    score_id         bigint generated always as identity primary key,
    model_id         bigint      not null references models(model_id)         on delete cascade,
    customer_id      bigint      not null references customers(customer_id)   on delete cascade,

    -- 该客户在实验中的实际组别（IPS 权重与分层核对需要）
    arm_key          text        not null,

    -- 预测结果：μ1/μ0 为两个结果模型的预测，uplift = μ1 - μ0，propensity_score 为预测购买概率
    mu1              numeric(9,8),
    mu0              numeric(9,8),
    uplift           numeric(9,8),
    propensity_score numeric(9,8),
    scored_at        timestamptz not null default now(),

    -- ---- 约束 ----
    -- ⚠️ 幂等的基石：同一模型下同一客户只能有一行
    constraint customer_scores_model_customer_uk unique (model_id, customer_id),

    -- 预测的是概率，必须在 [0,1]；uplift 是差值，允许 [-1,1]
    constraint customer_scores_mu1_ck    check (mu1 between 0 and 1),
    constraint customer_scores_mu0_ck    check (mu0 between 0 and 1),
    constraint customer_scores_uplift_ck check (uplift between -1 and 1),
    constraint customer_scores_prop_ck   check (propensity_score between 0 and 1),

    constraint customer_scores_arm_ck    check (arm_key <> ''),

    -- 一行分数至少要有一种分数，否则这行没有意义
    constraint customer_scores_any_ck    check (
        mu1 is not null or mu0 is not null
        or uplift is not null or propensity_score is not null
    )
);

comment on table  customer_scores is
    '客户分数表。一行代表「一名客户在一个模型版本下的一组预测分数」，是名单查询的唯一来源。';
comment on column customer_scores.score_id  is '代理主键（identity）';
comment on column customer_scores.model_id  is '所属模型版本；级联删除，删模型即删其分数';
comment on column customer_scores.customer_id is '客户内部 ID，指向 001 的 customers';
comment on column customer_scores.arm_key   is '该客户在实验中的实际组别（mens/control），IPS 加权与核对用';
comment on column customer_scores.mu1       is '预测结果 μ̂1(x) ≈ E[Y|X=x, W=1]，取值 [0,1]';
comment on column customer_scores.mu0       is '预测结果 μ̂0(x) ≈ E[Y|X=x, W=0]，取值 [0,1]';
comment on column customer_scores.uplift    is '预测增量 τ̂(x) = μ̂1 - μ̂0，取值 [-1,1]；uplift 策略按它排序';
comment on column customer_scores.propensity_score is
    '预测购买概率（基线策略 propensity 的打分）；注意它不是实验的分配概率 e(x)=0.5';
comment on column customer_scores.scored_at is '打分写入时间（不是训练时间）';

-- 外键列显式索引
-- （unique(model_id, customer_id) 的前导列已能满足查找，此处仍按规范显式建，
--   代价在 4 万行量级可忽略，好处是任何单列查询与级联删除都有确定的索引可用）
create index if not exists customer_scores_model_idx    on customer_scores (model_id);
create index if not exists customer_scores_customer_idx on customer_scores (customer_id);


-- ============================================================================
--  3. policy_evaluations —— 一个策略在一个触达比例下的评估结果
--
--  一行代表「某模型版本 × 某策略 × 某触达比例」在最终留出集上的 IPS 评估。
--  触达比例集合在 docs/06 §6 中预先固定：{0.05, 0.10, 0.20, 0.30, 0.50}，全部报告。
-- ============================================================================
create table if not exists policy_evaluations (
    eval_id        bigint generated always as identity primary key,
    model_id       bigint      not null references models(model_id) on delete cascade,

    policy         text        not null,
    fraction       numeric(6,5) not null,       -- 触达比例 q
    n_selected     integer     not null,        -- 被选中的人数

    -- 主指标：SNIPS 估计的总体增量（绝对百分点），不确定性用 bootstrap 给定种子
    ips_estimate   numeric(12,8),
    ci_low         numeric(12,8),
    ci_high        numeric(12,8),
    alpha          numeric(6,5) not null default 0.05,
    bootstrap_seed bigint      not null,
    n_bootstrap    integer     not null,

    -- 辅助指标：选取人群的实际观测购买率（**有偏，仅描述**，见 docs/06 §6.3）
    observed_rate_selected numeric(9,8),

    -- 可追溯性：本次评估所属的分析运行
    run_id         uuid        references analysis_runs(run_id) on delete set null,
    created_at     timestamptz not null default now(),

    -- ---- 约束 ----
    -- 幂等的基石：同一模型 × 同一策略 × 同一比例只有一行
    constraint policy_evaluations_fraction_uk unique (model_id, policy, fraction),

    constraint policy_evaluations_policy_ck check (
        policy in ('random','propensity','uplift','all_treat','none')
    ),
    constraint policy_evaluations_fraction_ck  check (fraction between 0 and 1),
    constraint policy_evaluations_n_sel_ck     check (n_selected >= 0),
    constraint policy_evaluations_alpha_ck     check (alpha > 0 and alpha < 1),
    constraint policy_evaluations_n_boot_ck    check (n_bootstrap > 0),
    constraint policy_evaluations_obs_rate_ck  check (observed_rate_selected between 0 and 1),

    -- 区间必须包住点估计。任一端为 NULL 时比较结果为 NULL，
    -- CHECK 在 SQL 三值逻辑下视为通过——即允许「只有点估计、没有区间」的登记。
    constraint policy_evaluations_ci_ck check (
        ci_low <= ips_estimate and ips_estimate <= ci_high
    )
);

comment on table  policy_evaluations is
    '策略评估结果表。一行代表「一个模型版本 × 一个策略 × 一个触达比例」的离线（IPS/SNIPS）评估。';
comment on column policy_evaluations.eval_id   is '代理主键（identity）';
comment on column policy_evaluations.model_id  is '被评估的模型版本；级联删除，删模型即删其评估';
comment on column policy_evaluations.policy    is 'random/propensity/uplift/all_treat/none（docs/06 §5.1）';
comment on column policy_evaluations.fraction  is '触达比例 q，取值 [0,1]；预先固定为 {0.05,0.10,0.20,0.30,0.50}';
comment on column policy_evaluations.n_selected is '被该策略选中的客户数（选取变量只能是干预前分数）';
comment on column policy_evaluations.ips_estimate is 'SNIPS 估计的总体增量（绝对百分点），主指标';
comment on column policy_evaluations.ci_low    is 'bootstrap 区间下界；可为 NULL 表示未计算区间';
comment on column policy_evaluations.ci_high   is 'bootstrap 区间上界；可为 NULL 表示未计算区间';
-- 注：全文刻意不出现 ASCII 百分号，否则用 psycopg 直接执行整份 DDL 时会被当成占位符
comment on column policy_evaluations.alpha     is '区间名义显著性水平，默认 0.05（即名义覆盖率 0.95 的区间）';
comment on column policy_evaluations.bootstrap_seed is 'bootstrap 随机种子（固定 20261011），复现所需';
comment on column policy_evaluations.n_bootstrap is 'bootstrap 重抽样次数';
comment on column policy_evaluations.observed_rate_selected is
    '选取人群的实际观测购买率。**有选择偏差**：分母是被策略选中的客户而非随机化全组，仅作描述，不得用于策略排名';
comment on column policy_evaluations.run_id    is '产生该评估的分析运行；运行被删除时置空，不丢评估证据';
comment on column policy_evaluations.created_at is '登记时间';

-- 外键列显式索引
create index if not exists policy_evaluations_model_idx on policy_evaluations (model_id);
create index if not exists policy_evaluations_run_idx   on policy_evaluations (run_id);


-- ============================================================================
--  4. 供只读账号使用的授权
--     dsa_app 新建的对象已由 ALTER DEFAULT PRIVILEGES 自动授权给 dsa_ro（见 001 §11）；
--     此处再显式授一次，使本文件在默认权限缺失时也能自给自足（幂等）。
-- ============================================================================
grant select on models, customer_scores, policy_evaluations to dsa_ro;


-- ============================================================================
--  5. 建表后自检（用 .dsh/skills/supabase-postgres-best-practices 的
--     references/schema-foreign-key-indexes.md「Find missing FK indexes」原句）
--
--     期望结果：0 行（即全部外键列都有索引）。
--     tests/test_stage4_registry.py::test_all_foreign_keys_are_indexed 会断言这一点。
-- ============================================================================
-- select
--   conrelid::regclass as table_name,
--   a.attname          as fk_column
-- from pg_constraint c
-- join pg_attribute a on a.attrelid = c.conrelid and a.attnum = any(c.conkey)
-- where c.contype = 'f'
--   and not exists (
--     select 1 from pg_index i
--     where i.indrelid = c.conrelid and a.attnum = any(i.indkey)
--   );
