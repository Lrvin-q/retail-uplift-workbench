-- 兼容性新增：效果重算关联原运行，旧记录父运行为空。
alter table analysis_runs add column if not exists parent_run_id uuid
    references analysis_runs(run_id);
create index if not exists analysis_runs_parent_idx on analysis_runs(parent_run_id);
do $$ begin
    if not exists (select 1 from pg_constraint where conname='analysis_runs_not_self_parent_ck') then
        alter table analysis_runs add constraint analysis_runs_not_self_parent_ck
            check (parent_run_id is null or parent_run_id <> run_id);
    end if;
end $$;
