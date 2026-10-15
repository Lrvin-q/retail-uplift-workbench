-- ============================================================================
--  006_policy_contrasts_and_identity.sql —— 策略差值表 + 模型身份指纹
--
--  项目：零售营销增量决策助手
--  依据：docs/06-建模与策略评估协议.md §5.6（策略差异只能用差值区间）、
--        §8.2（模型身份指纹）、docs/日志.md L038
--
--  修复的两个缺陷：
--    · F09（P1）：用"两条边际区间是否重叠"判断策略差异不成立
--      → 新增 policy_contrasts 表，存放**同一触达比例下**的差值与其配对 bootstrap 区间。
--    · F10（P1）：同一个 model_key 可以接受不同训练配置，版本登记与新评估错配
--      → models 新增 identity_fingerprint（身份指纹）与 preprocessing_config（预处理配置）。
--
--  本文件**幂等**：全部 DDL 用 `if not exists` / 先 drop 后 add 的约束替换 / 条件 update，
--  可以重复执行。既有库与新库走同一条路径。
--
--  ⚠️ 执行顺序：必须在 003_models.sql 之后（本文件 alter 该文件建的表）。
--     `sql/003_models.sql` 与 `sql/004_policy_queries.sql` 的内容**没有**被本文件改写，
--     新列与新表统一放在这里，因此"从零建库"(001 → 003 → 006) 与"既有库迁移"等价。
--
--  执行身份：dsa_app（可写账号）。
-- ============================================================================


-- ============================================================================
--  1. 策略旧名迁移：propensity -> control_probability
--
--  背景（缺陷 5.2）：历史上 `propensity` 这个策略名在两处代表不同的模型输出——
--     · 离线评估（scripts/train_and_evaluate.py）实际用 mu0（无干预购买概率）；
--     · 名单接口（sql/004 / tools）按 propensity_score（跨组合并购买预测）排序。
--  因此旧行**不是**"按合并购买预测排序"的结果，把它改名成 control_probability 是
--  把名字改回它真正的语义（数值一字不改）。改名可由 Python 侧复核：
--  `dsa.models.reevaluate` 会从库里已存的 mu0 重算这些行，并打印最大绝对差。
--
--  幂等：改名后 `propensity` 不再存在，重复执行是空操作；
--  若同 (model, fraction) 已经有 control_probability 行，则跳过，避免撞唯一约束。
--
--  ⚠️ 顺序：**先删旧的 CHECK，再改名，最后加新的 CHECK**。
--     若先加新 CHECK（不含 'propensity'），加约束时的全表校验会因为历史行而失败；
--     若先改名，旧 CHECK（不含 'control_probability'）会拒绝新值。
-- ============================================================================
alter table policy_evaluations
    drop constraint if exists policy_evaluations_policy_ck;

update policy_evaluations pe
set policy = 'control_probability'
where pe.policy = 'propensity'
  and not exists (
      select 1
      from policy_evaluations other
      where other.model_id = pe.model_id
        and other.fraction = pe.fraction
        and other.policy   = 'control_probability'
  );

-- 策略名的 CHECK：换成权威定义里的取值（dsa.models.policy.POLICY_DEFINITIONS）。
alter table policy_evaluations
    add constraint policy_evaluations_policy_ck check (
        policy in ('random','purchase_probability','control_probability',
                   'uplift','all_treat','none')
    );

comment on column policy_evaluations.policy is
    'random/purchase_probability/control_probability/uplift/all_treat/none（docs/06 §5.1）。
     旧名 propensity 已废弃：它同时指过 mu0 与合并购买预测两个输出，语义不明；
     sql/006 已把历史行改名为 control_probability（它们当时的数值确实由 mu0 算出）。';


-- ============================================================================
--  2. policy_contrasts —— 两个策略在**同一触达比例**下的差值及其区间（F09）
--
--  一行代表「某模型版本 × 某触达比例 × 策略 a − 对照策略 b」的配对 bootstrap 结果。
--
--  ⚠️ 为什么不是"把两条边际区间相减"：两个策略跑在同一份留出客户上，估计**相关**。
--     差值必须来自**同一次共同 bootstrap 的配对抽样**：第 b 次重抽样里两个策略共用
--     同一批索引，差值就是 draws[a][b] − draws[b][b]。独立重抽样会丢掉相关性、
--     把差值区间做宽（见 tests/test_stage4_policy_contrasts.py 的对照测试）。
--
--  ⚠️ 判断策略差异**只能**看本表：边际区间重叠不是差异检验。
-- ============================================================================
create table if not exists policy_contrasts (
    contrast_id      bigint generated always as identity primary key,
    model_id         bigint      not null references models(model_id) on delete cascade,

    fraction         numeric(6,5) not null,     -- 触达比例 q
    policy           text        not null,      -- 被减的策略 a
    reference_policy text        not null,      -- 作为对照的策略 b；差值 = V(a) − V(b)

    -- 主指标：总体增量之差（绝对比例），区间由**配对** bootstrap 给出
    difference       numeric(12,8),
    ci_low           numeric(12,8),
    ci_high          numeric(12,8),
    alpha            numeric(6,5) not null default 0.05,
    bootstrap_seed   bigint      not null,
    n_bootstrap      integer     not null,
    n_pairs_defined  integer     not null default 0,  -- 两个策略同时有定义的配对次数

    -- 预先固定的主要比较只有一条；其余都是探索性（未做多重比较校正）
    is_primary       boolean     not null default false,

    run_id           uuid        references analysis_runs(run_id) on delete set null,
    created_at       timestamptz not null default now(),

    -- ---- 约束 ----
    constraint policy_contrasts_pair_uk unique (model_id, fraction, policy, reference_policy),

    constraint policy_contrasts_policy_ck check (
        policy in ('random','purchase_probability','control_probability',
                   'uplift','all_treat','none')
    ),
    constraint policy_contrasts_reference_ck check (
        reference_policy in ('random','purchase_probability','control_probability',
                             'uplift','all_treat','none')
    ),
    -- 自己减自己没有意义
    constraint policy_contrasts_distinct_ck check (policy <> reference_policy),
    constraint policy_contrasts_fraction_ck check (fraction between 0 and 1),
    constraint policy_contrasts_alpha_ck    check (alpha > 0 and alpha < 1),
    constraint policy_contrasts_n_boot_ck   check (n_bootstrap > 0),
    constraint policy_contrasts_n_pairs_ck  check (n_pairs_defined >= 0),

    -- 区间必须包住差值点估计。任一端为 NULL 时比较结果为 NULL，
    -- CHECK 在 SQL 三值逻辑下视为通过——即允许"只有点估计、没有区间"的登记。
    constraint policy_contrasts_ci_ck check (
        ci_low <= difference and difference <= ci_high
    )
);

comment on table  policy_contrasts is
    '策略差值表。一行代表「一个模型版本 × 一个触达比例 × 策略 a 减对照策略 b」的配对 '
    'bootstrap 结果。判断策略差异只能看本表：两条边际区间是否重叠不是差异检验。';
comment on column policy_contrasts.contrast_id      is '代理主键（identity）';
comment on column policy_contrasts.model_id         is '被评估的模型版本；级联删除，删模型即删其差值';
comment on column policy_contrasts.fraction         is '触达比例 q，预先固定为 {0.05,0.10,0.20,0.30,0.50}';
comment on column policy_contrasts.policy           is '被减的策略 a（权威定义见 dsa.models.policy.POLICY_DEFINITIONS）';
comment on column policy_contrasts.reference_policy is '作为对照的策略 b；差值 = V(a) − V(b)';
comment on column policy_contrasts.difference       is '总体增量之差（绝对比例），主指标';
comment on column policy_contrasts.ci_low           is '配对 bootstrap 区间下界；NULL 表示未计算';
comment on column policy_contrasts.ci_high          is '配对 bootstrap 区间上界；NULL 表示未计算';
comment on column policy_contrasts.alpha            is '区间名义显著性水平，默认 0.05（名义覆盖率 0.95）';
comment on column policy_contrasts.bootstrap_seed   is 'bootstrap 随机种子（固定 20261011），复现所需';
comment on column policy_contrasts.n_bootstrap      is 'bootstrap 重抽样次数';
comment on column policy_contrasts.n_pairs_defined  is
    '两个策略都算出有限值的配对重抽样次数；小于 n_bootstrap 说明有重抽样无法比较';
comment on column policy_contrasts.is_primary       is
    '是否为预先固定的主要比较（比例 0.10 × uplift 减 random，见 dsa.models.policy）。'
    '其余比较为探索性：未做多重比较校正，不能当作已确认的策略差异';
comment on column policy_contrasts.run_id           is '产生该差值的分析运行；运行被删除时置空，不丢证据';
comment on column policy_contrasts.created_at       is '登记时间';

-- 外键列显式索引（PostgreSQL 不会自动建）
create index if not exists policy_contrasts_model_idx on policy_contrasts (model_id);
create index if not exists policy_contrasts_run_idx   on policy_contrasts (run_id);


-- ============================================================================
--  3. models 新增两列：身份指纹与预处理配置（F10）
--
--  为什么放在新文件里而不是直接改 003：这样"从零建库"（001 → 003 → 006）与
--  "既有库迁移"走的是同一条 ALTER 路径，不会出现两份定义不一致。
--
--  identity_fingerprint：由 dsa.models.registry.compute_identity_fingerprint 计算，
--      覆盖 model_type / 数据集版本 / 实验 / 比较 / feature_columns（有序）/
--      base_learner / params（规范化 JSON）/ split_config / code_version / 预处理配置。
--      相同 -> 幂等返回旧 model_id；不同 -> 拒绝并列出差异字段。
--  preprocessing_config：进入指纹的那份预处理配置，写进库里之后
--      "指纹可以只从数据库重算"，回填因此是可复现的。
-- ============================================================================
alter table models
    add column if not exists identity_fingerprint text;

alter table models
    add column if not exists preprocessing_config jsonb not null default '{}'::jsonb;

-- 格式约束：要么为空（迁移前的旧行，需回填），要么是 sha256: 加 64 位小写十六进制。
alter table models
    drop constraint if exists models_identity_fingerprint_ck;

alter table models
    add constraint models_identity_fingerprint_ck check (
        identity_fingerprint is null
        or identity_fingerprint ~ '^sha256:[0-9a-f]{64}$'
    );

-- 预处理配置必须是 jsonb 对象（与 params / split_config 的约定一致）
alter table models
    drop constraint if exists models_preprocessing_obj_ck;

alter table models
    add constraint models_preprocessing_obj_ck check (
        jsonb_typeof(preprocessing_config) = 'object'
    );

comment on column models.identity_fingerprint is
    '模型身份指纹 sha256:<64 hex>。由已登记的全部身份字段规范化后哈希（见 '
    'dsa.models.registry.compute_identity_fingerprint）。相同 model_key 且指纹相同才幂等；'
    '指纹不同必须拒绝或换 key。NULL 表示迁移前的旧行，需用 '
    'backfill_identity_fingerprints 回填';
comment on column models.preprocessing_config is
    '预处理配置（jsonb）：类别列/数值列、编码器与设置、remainder、只在训练集 fit。'
    '进身份指纹；同时使指纹可以**只从数据库**重算，回填可复现';


-- ============================================================================
--  4. 供只读账号使用的授权（dsa_app 新建对象已由 ALTER DEFAULT PRIVILEGES 自动授权，
--     此处再显式授一次，使本文件在默认权限缺失时也能自给自足）
-- ============================================================================
grant select on policy_contrasts to dsa_ro;


-- ============================================================================
--  5. 建表后自检（期望：0 行 = 全部外键列都有索引）
--     与 tests/test_stage4_registry.py::test_all_foreign_keys_are_indexed 同一句 SQL。
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
