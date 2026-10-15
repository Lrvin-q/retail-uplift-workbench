-- ============================================================================
--  004_policy_queries.sql —— 名单查询与核对 SQL
--
--  依据：docs/06-建模与策略评估协议.md §5（策略定义）、§5.4（表述边界）、
--        §6（预先固定的评估规则）、AGENTS.md §2.2（分母口径）
--  执行身份：dsa_ro（只读）
--
--  命名查询用 `-- name: xxx` 标记，scripts/cross_check.py 的 load_named_queries
--  可解析（同样的解析器也用于 tests/test_stage4_registry.py）。
--
--  ⚠️ 三条红线（写在本文件顶部，任何修改都不得越过）：
--   1. 名单排序只能使用**干预前**分数（uplift / propensity_score / mu0），
--      customer_ranking 里不得出现 visit / conversion / spend 任何形式的筛选或排序。
--   2. 观察到的购买率（top_decile_check）分母是**被策略选中的客户**，不是随机化全组，
--      因此**有选择偏差**，只作描述，不得据此对策略排名（docs/06 §5.4）。
--   3. 名单是留出集上的**回顾性演示输出**，不代表这些历史客户未来会发生的购买变化；
--      也无成本数据，不得据此声称净收益或 ROI（docs/06 §5.4）。
--
--  ⚠️ **策略名必须与 `dsa.models.policy.POLICY_DEFINITIONS` 一致**（缺陷 5.2）。
--     本文件原先用旧名 `propensity`，而那个名字在离线评估里代表 `mu0`、
--     在名单里代表 `propensity_score` —— **同名两义**。
--     现在统一为三个如实命名的取值：
--       'uplift'                → cs.uplift              （预测增量）
--       'purchase_probability'  → cs.propensity_score    （合并购买预测分数）
--       'control_probability'   → cs.mu0                 （无干预购买概率）
--     旧名 `propensity` **不再接受**：它不匹配任何 `when` 分支，
--     因此 `policy_score` 为 **NULL**，排序退化为按 `customer_key`（确定但**任意**）。
--     也就是说它不会**静默地**给出某个策略的名单 —— 这是与旧版最关键的区别。
--     ⚠️ 但"不给出错误答案"仍不等于"报错"：直接跑本 SQL 的调用方拿到的是
--     一份按客户键排的名单，而不是一个错误。**Agent 侧的名单工具会显式拒绝
--     旧名并返回 `E_TOOL_001`**（见 `DEPRECATED_POLICY_KINDS`），
--     那才是首选路径；本文件是给人工核对用的。
-- ============================================================================


-- ============================================================================
--  Q1  customer_ranking —— 按模型分数取前 N 名客户（名单的唯一来源）
--
--  参数：:model_key    模型版本业务键，如 tlearner_lr_v1
--        :policy_kind  排序依据，取值见文件顶部说明：
--                      'uplift'（默认）/ 'purchase_probability' / 'control_probability'；
--                      'random' 用 md5(客户) 给出可复现的确定性顺序；
--                      其余取值（all_treat/none/已废弃的 propensity）退化为按 customer_id 排序
--        :limit_n      返回行数上限
--
--  返回：客户业务键 + 实验组别 + 四个分数 + **该客户实验前的关键特征**。
--
--  ⚠️ 排序变量只能是干预前信息。这里**没有**、也**不允许**出现
--     visit / conversion / spend —— 一旦出现即为数据泄漏与分母污染（AGENTS.md §2.2）。
--  ⚠️ 大结果通过引用传递：Agent 只取 limit_n 行，不把全量名单塞进上下文（docs/06 §9）。
-- ============================================================================
-- name: customer_ranking
with scored as (
    select
        cs.customer_id,
        cs.arm_key,
        cs.mu1,
        cs.mu0,
        cs.uplift,
        cs.propensity_score,
        -- 策略打分：只允许干预前的预测分数
        case coalesce(cast(:policy_kind as text), 'uplift')
            when 'uplift'               then cs.uplift
            when 'purchase_probability' then cs.propensity_score
            when 'control_probability'  then cs.mu0
        end as policy_score,
        -- random 策略：用模型 id + 客户 id 的散列做确定性排序，
        -- 使同一模型重复查询得到完全相同的名单（不同于评估用的固定随机种子）
        case when coalesce(cast(:policy_kind as text), 'uplift') = 'random'
             then md5(cs.model_id::text || '#' || cs.customer_id::text)
        end as random_key
    from customer_scores cs
    join models m on m.model_id = cs.model_id
    where m.model_key = cast(:model_key as text)
)
select
    cu.customer_key,
    s.arm_key,
    s.mu1,
    s.mu0,
    s.uplift,
    s.propensity_score,
    -- ---- 实验前特征（docs/06 §1.1 允许的全部字段）----
    cu.recency,
    cu.history_segment,
    cu.history_band,
    cu.history,
    cu.mens,
    cu.womens,
    cu.zip_code,
    cu.newbie,
    cu.channel
from scored s
join customers cu on cu.customer_id = s.customer_id
order by
    s.policy_score desc nulls last,
    s.random_key   asc  nulls last,
    s.customer_id  asc
limit cast(:limit_n as integer);


-- ============================================================================
--  Q2  top_decile_check —— 名单核对：选取人群的各组别人数与实际观测购买率
--
--  参数：:model_key、:policy_kind、:fraction（默认 0.1，即十分位）
--
--  取前 **round(fraction × n)** 名客户（四舍五入到整数人数，即 floor(q·n + 0.5)），
--  按实验组别统计人数与观测购买率。取法与评估器
--  `dsa.models.policy._selected_mask`（`k = int(fraction * n + 0.5)`）一致，
--  因此本查询的 n_selected 可以直接与 policy_evaluations.n_selected 对照。
--
--  ⚠️ 该购买率**有偏**：分母是被分数选中的客户，而不是被随机分配到该组的客户，
--     因此它既不是干预效果，也不能用来给策略排名。返回值里的 bias_note 列
--     会把这条边界随数据一起带出去，避免下游误用（docs/06 §5.4、§6.3）。
--  ⚠️ 购买率的分母永远是「被选中的全部客户」，不得按 visit / spend 再过滤。
-- ============================================================================
-- name: top_decile_check
with scored as (
    select
        cs.customer_id,
        cs.arm_key,
        m.experiment_id,
        row_number() over (
            order by
                case coalesce(cast(:policy_kind as text), 'uplift')
                    when 'uplift'               then cs.uplift
                    when 'purchase_probability' then cs.propensity_score
                    when 'control_probability'  then cs.mu0
                end desc nulls last,
                case when coalesce(cast(:policy_kind as text), 'uplift') = 'random'
                     then md5(cs.model_id::text || '#' || cs.customer_id::text)
                end asc nulls last,
                cs.customer_id asc
        ) as rn,
        count(*) over () as n_scored
    from customer_scores cs
    join models m on m.model_id = cs.model_id
    where m.model_key = cast(:model_key as text)
),
selected as (
    select *
    from scored
    where rn <= floor(n_scored * coalesce(cast(:fraction as numeric), 0.1) + 0.5)
)
select
    sel.arm_key,
    count(*)                                as n_selected,
    count(*) filter (where o.converted)     as n_converted,
    round(
        (count(*) filter (where o.converted))::numeric / nullif(count(*), 0), 6
    )                                       as observed_conversion_rate,
    min(sel.n_scored)                       as n_scored,
    round(count(*)::numeric / nullif(min(sel.n_scored), 0), 6) as share_of_scored,
    '选择偏差提示：分母是被策略选中的客户（选择变量为干预前分数），不是随机化全组；'
    || '该购买率仅作描述，不得用于策略优劣比较或因果结论（docs/06 §5.4）' as bias_note
from selected sel
left join outcomes o
       on o.experiment_id = sel.experiment_id
      and o.customer_id   = sel.customer_id
group by sel.arm_key
order by sel.arm_key;


-- ============================================================================
--  Q3  score_coverage —— 分数覆盖率：已打分客户 / 建模人群总数
--
--  参数：:model_key
--
--  建模人群 = 该模型所用预注册比较的两个组（treatment + control）的全部客户，
--  对 hillstrom 主比较即 21,307 + 21,306 = 42,613（docs/06 §2.3）。
--  覆盖率应为 1.0；小于 1 说明有客户没打上分，名单与 IPS 评估都不可信。
-- ============================================================================
-- name: score_coverage
with population as (
    select
        m.model_id,
        m.model_key,
        count(a.customer_id) as n_population
    from models m
    join experiment_comparisons ec
      on ec.experiment_id   = m.experiment_id
     and ec.comparison_key  = m.comparison_key
    left join assignments a
      on a.experiment_id = m.experiment_id
     and a.arm_id in (ec.treatment_arm_id, ec.control_arm_id)
    where m.model_key = cast(:model_key as text)
    group by m.model_id, m.model_key
),
scored as (
    select cs.model_id, count(*) as n_scored
    from customer_scores cs
    group by cs.model_id
)
select
    p.model_key,
    p.n_population,
    coalesce(s.n_scored, 0)                                  as n_scored,
    p.n_population - coalesce(s.n_scored, 0)                 as n_missing,
    round(coalesce(s.n_scored, 0)::numeric / nullif(p.n_population, 0), 6) as coverage,
    (coalesce(s.n_scored, 0) = p.n_population)               as coverage_ok
from population p
left join scored s on s.model_id = p.model_id;


-- ============================================================================
--  Q4  policy_summary —— 某模型各策略、各触达比例的评估汇总
--
--  参数：:model_key
--
--  返回 policy_evaluations 的全部登记行（策略 × 比例），**不做任何筛选与排名**：
--  docs/06 §6 要求 {0.05,0.10,0.20,0.30,0.50} 全部报告，不许只报最好的那个。
--  all_treat / none 是资源消耗不同的补充参照，不得与固定比例策略直接排名。
-- ============================================================================
-- name: policy_summary
select
    m.model_key,
    pe.policy,
    pe.fraction,
    pe.n_selected,
    pe.ips_estimate,
    pe.ci_low,
    pe.ci_high,
    (pe.ci_high - pe.ci_low)        as ci_width,
    pe.alpha,
    pe.bootstrap_seed,
    pe.n_bootstrap,
    pe.observed_rate_selected,
    pe.run_id,
    pe.created_at,
    -- 补充参照策略标记：资源消耗与固定比例策略不同，不能直接排名
    (pe.policy in ('all_treat','none')) as is_reference_only
from policy_evaluations pe
join models m on m.model_id = pe.model_id
where m.model_key = cast(:model_key as text)
order by pe.policy, pe.fraction;


-- ============================================================================
--  Q5  model_manifest —— 模型清单（数据版本、特征数、训练时间、分数行数）
--
--  参数：:model_key（可为 NULL，表示列出全部模型）
--
--  这是「报告里每一个数字属于哪个模型版本」的追溯入口（docs/06 §8）。
-- ============================================================================
-- name: model_manifest
select
    m.model_key,
    m.model_type,
    m.base_learner,
    d.version_label                          as dataset_version,
    e.experiment_key,
    m.comparison_key,
    cardinality(m.feature_columns)           as n_features,
    m.feature_columns,
    m.params,
    m.split_config,
    m.code_version,
    m.trained_at,
    m.notes,
    (select count(*) from customer_scores cs where cs.model_id = m.model_id)      as n_scores,
    (select count(*) from policy_evaluations pe where pe.model_id = m.model_id)   as n_policy_evals
from models m
join datasets    d on d.dataset_id    = m.dataset_id
join experiments e on e.experiment_id = m.experiment_id
where cast(:model_key as text) is null
   or m.model_key = cast(:model_key as text)
order by m.trained_at desc, m.model_key;
