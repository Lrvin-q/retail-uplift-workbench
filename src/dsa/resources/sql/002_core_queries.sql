-- ============================================================================
--  002_core_queries.sql —— 核心分析 SQL
--
--  依据：docs/02-分析协议.md §8（指标口径）、§9（缺失）、§11（SRM）
--  执行身份：dsa_ro（只读）
--
--  ⚠️ 全文铁律：**分母 = 全部被随机分配到该组的客户数**。
--     绝不允许出现 `where visited` 或任何干预后条件参与分母计算。
--     因此所有聚合都以 assignments 为基准表，用 LEFT JOIN 挂结果。
-- ============================================================================


-- ============================================================================
--  Q1  总体汇总：一行给出整个实验的规模
-- ============================================================================
-- name: overall_summary
select
    e.experiment_key,
    d.version_label                            as dataset_version,
    e.outcome_window_days                      as window_days,
    count(*)                                   as n_assigned,
    count(*) filter (where o.outcome_id is null) as n_outcome_missing,
    count(*) filter (where o.converted)        as n_converted,
    count(*) filter (where o.visited)          as n_visited,
    round(sum(o.spend), 2)                     as total_spend
from assignments a
join experiments     e on e.experiment_id = a.experiment_id
join datasets        d on d.dataset_id    = e.dataset_id
left join outcomes   o on o.experiment_id = a.experiment_id
                      and o.customer_id   = a.customer_id
where e.experiment_key = :experiment_key
group by e.experiment_key, d.version_label, e.outcome_window_days;


-- ============================================================================
--  Q2  分组聚合：主要与次要指标（每组的分子、分母都显式给出）
-- ============================================================================
-- name: arm_aggregates
select
    ar.arm_key,
    ar.role,
    ar.segment_value,
    -- 分母：全部被分配到该组的客户（绝不为任何干预后条件所改变）
    count(*)                                              as n_customers,
    -- 结果缺失：单独计数，绝不并入"未购买"
    count(*) filter (where o.outcome_id is null)          as n_outcome_missing,
    -- 购买率分子
    count(*) filter (where o.converted)                   as n_converted,
    -- 访问率分子
    count(*) filter (where o.visited)                     as n_visited,
    -- 消费：分母含未购买客户（spend=0）
    coalesce(sum(o.spend), 0)                             as total_spend,
    -- 比率（在完整母集上计算）
    (count(*) filter (where o.converted))::numeric
        / nullif(count(*), 0)                             as conversion_rate,
    (count(*) filter (where o.visited))::numeric
        / nullif(count(*), 0)                             as visit_rate,
    coalesce(sum(o.spend), 0)::numeric
        / nullif(count(*), 0)                             as spend_per_customer
from assignments a
join experiments      e  on e.experiment_id = a.experiment_id
join experiment_arms  ar on ar.arm_id       = a.arm_id
left join outcomes    o  on o.experiment_id = a.experiment_id
                        and o.customer_id   = a.customer_id
where e.experiment_key = :experiment_key
group by ar.arm_key, ar.role, ar.segment_value, ar.arm_id
order by ar.arm_id;


-- ============================================================================
--  Q3  主要比较的客户级明细（SQL 与 Python 对照的基准数据）
--     一行 = 一名客户，且必须恰好 64,000 行 / 或比较子集的确切行数
-- ============================================================================
-- name: comparison_customer_level
select
    cu.customer_key,
    cu.source_row_number,
    ar.arm_key,
    ar.role,
    cu.recency,
    cu.history_band,
    cu.history,
    cu.mens,
    cu.womens,
    cu.zip_code,
    cu.newbie,
    cu.channel,
    o.visited,
    o.converted,
    o.spend
from experiment_comparisons ec
join experiment_arms tr on tr.arm_id = ec.treatment_arm_id
join experiment_arms ct on ct.arm_id = ec.control_arm_id
join assignments a
  on a.experiment_id = ec.experiment_id
 and a.arm_id in (ec.treatment_arm_id, ec.control_arm_id)
join customers cu on cu.customer_id = a.customer_id
join experiment_arms ar on ar.arm_id = a.arm_id
left join outcomes o on o.experiment_id = a.experiment_id
                    and o.customer_id   = a.customer_id
where ec.experiment_id = :experiment_id
  and ec.comparison_key = :comparison_key;


-- ============================================================================
--  Q4  关联膨胀检查：客户级关联不得放大行数
-- ============================================================================
-- name: join_row_ratio
with base as (
    select count(*) as n_assigned
    from assignments a
    join experiments e on e.experiment_id = a.experiment_id
    where e.experiment_key = :experiment_key
),
joined as (
    select count(*) as n_rows
    from assignments a
    join experiments e on e.experiment_id = a.experiment_id
    join experiment_arms ar on ar.arm_id = a.arm_id
    join customers cu on cu.customer_id = a.customer_id
    left join outcomes o on o.experiment_id = a.experiment_id
                        and o.customer_id   = a.customer_id
    where e.experiment_key = :experiment_key
)
select
    base.n_assigned,
    joined.n_rows,
    joined.n_rows::numeric / nullif(base.n_assigned, 0) as join_row_ratio,
    (joined.n_rows = base.n_assigned)                   as no_inflation
from base, joined;


-- ============================================================================
--  Q5  结果覆盖率检查（缺失结果绝不能被当作未购买）
-- ============================================================================
-- name: outcome_coverage
select
    e.experiment_key,
    count(*)                                                  as n_expected,
    count(o.outcome_id)                                       as n_present,
    count(*) - count(o.outcome_id)                            as n_missing,
    round(count(o.outcome_id)::numeric / nullif(count(*), 0), 6) as outcome_coverage
from assignments a
join experiments e on e.experiment_id = a.experiment_id
left join outcomes o on o.experiment_id = a.experiment_id
                    and o.customer_id   = a.customer_id
where e.experiment_key = :experiment_key
group by e.experiment_key;


-- ============================================================================
--  Q6  唯一性检查：内部客户 ID 与源行映射必须一一对应
-- ============================================================================
-- name: identity_integrity
select
    d.version_label,
    count(*)                                     as n_customers,
    count(distinct cu.customer_id)               as n_distinct_id,
    count(distinct cu.source_row_number)         as n_distinct_row,
    count(distinct cu.customer_key)              as n_distinct_key,
    min(cu.source_row_number)                    as min_row,
    max(cu.source_row_number)                    as max_row,
    (count(*) = count(distinct cu.customer_id)
     and count(*) = count(distinct cu.source_row_number)) as identity_ok
from customers cu
join datasets d on d.dataset_id = cu.dataset_id
where d.version_label = :dataset_version
group by d.version_label;


-- ============================================================================
--  Q7  SRM 检查：观测计数 vs 预期比例（卡方拟合优度）
--     预期比例来自 original_description，禁止从观测反推（协议 §11.1）
-- ============================================================================
-- name: srm_check
with obs as (
    select
        ar.arm_key,
        ar.expected_ratio,
        count(*)::numeric as observed
    from assignments a
    join experiments     e  on e.experiment_id = a.experiment_id
    join experiment_arms ar on ar.arm_id       = a.arm_id
    where e.experiment_key = :experiment_key
    group by ar.arm_key, ar.expected_ratio, ar.arm_id
    order by ar.arm_id
),
tot as (
    select sum(observed) as n_total,
           sum(expected_ratio) as ratio_total
    from obs
),
calc as (
    select
        o.arm_key,
        o.observed,
        (o.expected_ratio / t.ratio_total) * t.n_total as expected,
        t.n_total,
        count(*) over () as n_arms
    from obs o cross join tot t
)
select
    arm_key,
    observed,
    round(expected, 2)                                  as expected,
    round(observed - expected, 2)                       as diff,
    n_total,
    -- 卡方统计量
    sum(power(observed - expected, 2) / expected) over () as chi_square,
    n_arms - 1                                          as df
from calc
order by arm_key;


-- ============================================================================
--  Q8  客户关联核对：确认每个客户恰好一个组、一条结果
--     用于发现"一对多"导致的静默重复计数
-- ============================================================================
-- name: customer_cardinality
select
    '每客户分组数' as check_name,
    count(*)                  as n_customers,
    min(n) as min_per_customer,
    max(n) as max_per_customer,
    (min(n) = 1 and max(n) = 1) as ok
from (
    select a.customer_id, count(*) as n
    from assignments a
    join experiments e on e.experiment_id = a.experiment_id
    where e.experiment_key = :experiment_key
    group by a.customer_id
) t
union all
select
    '每客户结果数',
    count(*), min(n), max(n), (min(n) = 1 and max(n) = 1)
from (
    select o.customer_id, count(*) as n
    from outcomes o
    join experiments e on e.experiment_id = o.experiment_id
    where e.experiment_key = :experiment_key
    group by o.customer_id
) t2;
