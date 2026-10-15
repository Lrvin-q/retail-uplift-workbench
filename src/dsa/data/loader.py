"""Hillstrom 数据导入。

设计要点
--------
1. **幂等**：重复导入同一文件不会新增客户、分组或结果记录。
   全部写入使用 `INSERT ... ON CONFLICT ... DO NOTHING`，
   冲突键分别是 `(dataset_id, source_row_number)` 与 `(experiment_id, customer_id)`。

2. **一一映射**：`customers` 每个内部 ID 恰好对应源文件的一行。
   `customer_key` = `<version_label>#<补零行号>`，带数据集命名空间。
   **绝不按整行内容去重**——源数据有 6562 行字段完全相同（docs/日志.md L008）。

3. **纯集合法**：用 `COPY` 灌入临时表，再以 `INSERT ... SELECT` 关联写入。
   没有 Python 逐行循环，64,000 行也在秒级完成。

4. **哈希校验**：导入前核对文件 SHA256 与规格一致；不一致则拒绝导入，
   除非显式传 `allow_hash_mismatch=True`（用于研究新文件，会写入 warning）。

用法
----
    from dsa.config import get_engine
    from dsa.data.loader import load_hillstrom

    report = load_hillstrom(get_engine(readonly=False))
    print(report.summary())
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from dsa.config import PROJECT_ROOT
from dsa.data.specs import (
    HILLSTROM,
    HILLSTROM_EMAIL_V1,
    SOURCE_COLUMNS,
    DatasetSpec,
    ExperimentSpec,
)

#: 导入用临时表名（事务结束自动消失）
_TMP = "tmp_hillstrom_load"

_TMP_DDL = f"""
create temp table {_TMP} (
    source_row_number integer      not null,
    recency           smallint     not null,
    history_segment   text         not null,
    history           numeric(12,2) not null,
    mens              boolean      not null,
    womens            boolean      not null,
    zip_code          text         not null,
    newbie            boolean      not null,
    channel           text         not null,
    segment           text         not null,
    visit             boolean      not null,
    conversion        boolean      not null,
    spend             numeric(12,2) not null
) on commit drop
"""

_COPY_SQL = f"""
copy {_TMP} (
    source_row_number, recency, history_segment, history, mens, womens,
    zip_code, newbie, channel, segment, visit, conversion, spend
) from stdin
"""


class ImportError_(RuntimeError):
    """导入失败。"""


@dataclass
class LoadReport:
    """一次导入的结果，用于校验与写日志。"""

    dataset_version: str
    file_sha256: str
    rows_in_file: int
    customers_before: int
    customers_after: int
    assignments_before: int
    assignments_after: int
    outcomes_before: int
    outcomes_after: int
    arm_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    dataset_id: int = 0
    experiment_id: int = 0

    @property
    def customers_inserted(self) -> int:
        return self.customers_after - self.customers_before

    @property
    def assignments_inserted(self) -> int:
        return self.assignments_after - self.assignments_before

    @property
    def outcomes_inserted(self) -> int:
        return self.outcomes_after - self.outcomes_before

    def summary(self) -> str:
        lines = [
            f"数据版本      : {self.dataset_version}",
            f"文件 SHA256   : {self.file_sha256}",
            f"源文件行数    : {self.rows_in_file:,}",
            f"客户          : {self.customers_after:,} 行（本次新增 {self.customers_inserted:,}）",
            f"分组          : {self.assignments_after:,} 行（本次新增 {self.assignments_inserted:,}）",
            f"结果          : {self.outcomes_after:,} 行（本次新增 {self.outcomes_inserted:,}）",
        ]
        if self.arm_counts:
            lines.append("各组人数      : " + ", ".join(f"{k}={v:,}" for k, v in sorted(self.arm_counts.items())))
        for w in self.warnings:
            lines.append(f"⚠ 警告        : {w}")
        return "\n".join(lines)


# ============================================================================
#  辅助
# ============================================================================


def compute_sha256(path: Path) -> str:
    """分块计算文件 SHA256。"""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_csv_path(spec: DatasetSpec, csv_path: Path | None) -> Path:
    if csv_path is not None:
        return Path(csv_path)
    return PROJECT_ROOT / spec.relative_path()


def read_source(path: Path, spec: DatasetSpec) -> pd.DataFrame:
    """读取并校验源文件结构。"""
    df = pd.read_csv(path)

    if tuple(df.columns) != SOURCE_COLUMNS:
        raise ImportError_(
            f"列不匹配。期望 {SOURCE_COLUMNS}，实际 {tuple(df.columns)}"
        )
    if len(df) != spec.row_count:
        raise ImportError_(f"行数不匹配。期望 {spec.row_count}，实际 {len(df)}")
    if df.isna().any().any():
        bad = df.isna().sum()
        bad = bad[bad > 0].to_dict()
        raise ImportError_(f"源文件出现缺失值，导入中止：{bad}")
    return df


def _rows_for_copy(df: pd.DataFrame):
    """把 DataFrame 转成 COPY 可写的基本类型元组，避免 numpy 标量。"""
    cols = list(df.columns)
    idx = {c: i for i, c in enumerate(cols)}
    for pos, row in enumerate(df.itertuples(index=False, name=None), start=1):
        yield (
            pos,                                        # 源行号，1-based
            int(row[idx["recency"]]),
            str(row[idx["history_segment"]]),
            str(row[idx["history"]]),
            bool(row[idx["mens"]]),
            bool(row[idx["womens"]]),
            str(row[idx["zip_code"]]),
            bool(row[idx["newbie"]]),
            str(row[idx["channel"]]),
            str(row[idx["segment"]]),
            bool(row[idx["visit"]]),
            bool(row[idx["conversion"]]),
            str(row[idx["spend"]]),
        )


# ============================================================================
#  主流程
# ============================================================================


def load_hillstrom(
    engine: Engine,
    *,
    csv_path: Path | None = None,
    spec: DatasetSpec = HILLSTROM,
    experiment: ExperimentSpec = HILLSTROM_EMAIL_V1,
    allow_hash_mismatch: bool = False,
) -> LoadReport:
    """把 Hillstrom 源文件导入数据库（幂等）。

    Args:
        engine: **可写**引擎（dsa_app）。只读账号无法导入。
        csv_path: 源文件路径，默认取 `data/raw/<spec.file_name>`。
        allow_hash_mismatch: 允许文件哈希与规格不符（会记入 warnings）。

    Returns:
        LoadReport

    Raises:
        ImportError_: 文件缺失、哈希不符、结构不符、或数据库中已有不一致的配置。
    """
    path = resolve_csv_path(spec, csv_path)
    if not path.exists():
        raise ImportError_(f"源文件不存在: {path}")

    actual_sha = compute_sha256(path)
    warnings: list[str] = []
    if actual_sha != spec.file_sha256:
        msg = (
            f"文件哈希与规格不符。\n  规格: {spec.file_sha256}\n  实际: {actual_sha}\n"
            f"  文件: {path}"
        )
        if not allow_hash_mismatch:
            raise ImportError_(msg + "\n如确认使用新文件，请传 allow_hash_mismatch=True。")
        warnings.append(msg)

    df = read_source(path, spec)

    # ---- 全部写入在同一个事务里；失败则整体回滚 ----
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()

        ds_id = _ensure_dataset(cur, spec, actual_sha, path, warnings)
        exp_id = _ensure_experiment(cur, experiment, ds_id, warnings)

        before = _counts(cur, ds_id, exp_id)

        # ---- COPY 到临时表 ----
        cur.execute(_TMP_DDL)
        with cur.copy(_COPY_SQL) as cp:
            for row in _rows_for_copy(df):
                cp.write_row(row)

        # ---- 客户：一一映射，冲突即已存在 ----
        cur.execute(
            """
            insert into customers (
                dataset_id, customer_key, source_row_number,
                recency, history_segment, history, mens, womens, zip_code, newbie, channel
            )
            select
                %(ds)s,
                %(ver)s || '#' || lpad(t.source_row_number::text, 6, '0'),
                t.source_row_number,
                t.recency, t.history_segment, t.history,
                t.mens, t.womens, t.zip_code, t.newbie, t.channel
            from {tmp} t
            on conflict (dataset_id, source_row_number) do nothing
            """.format(tmp=_TMP),
            {"ds": ds_id, "ver": spec.version_label},
        )

        # ---- 分组 ----
        cur.execute(
            """
            insert into assignments (experiment_id, customer_id, arm_id)
            select %(exp)s, c.customer_id, a.arm_id
            from {tmp} t
            join customers c
              on c.dataset_id = %(ds)s and c.source_row_number = t.source_row_number
            join experiment_arms a
              on a.experiment_id = %(exp)s and a.segment_value = t.segment
            on conflict (experiment_id, customer_id) do nothing
            """.format(tmp=_TMP),
            {"ds": ds_id, "exp": exp_id},
        )

        # ---- 结果 ----
        cur.execute(
            """
            insert into outcomes (
                experiment_id, customer_id, window_days, visited, converted, spend
            )
            select %(exp)s, c.customer_id, %(win)s, t.visit, t.conversion, t.spend
            from {tmp} t
            join customers c
              on c.dataset_id = %(ds)s and c.source_row_number = t.source_row_number
            on conflict (experiment_id, customer_id) do nothing
            """.format(tmp=_TMP),
            {"ds": ds_id, "exp": exp_id, "win": experiment.outcome_window_days},
        )

        after = _counts(cur, ds_id, exp_id)
        arm_counts = _arm_counts(cur, exp_id)

        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()

    return LoadReport(
        dataset_version=spec.version_label,
        file_sha256=actual_sha,
        rows_in_file=len(df),
        customers_before=before["customers"],
        customers_after=after["customers"],
        assignments_before=before["assignments"],
        assignments_after=after["assignments"],
        outcomes_before=before["outcomes"],
        outcomes_after=after["outcomes"],
        arm_counts=arm_counts,
        warnings=warnings,
        dataset_id=ds_id,
        experiment_id=exp_id,
    )


# ============================================================================
#  内部：登记数据集与实验配置
# ============================================================================


def _ensure_dataset(cur, spec: DatasetSpec, sha: str, path: Path, warnings: list[str]) -> int:
    cur.execute(
        """
        insert into datasets (
            dataset_key, version_label, source_url, file_name,
            file_sha256, file_bytes, row_count, grain, source_verified, notes
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        on conflict (version_label) do nothing
        """,
        (
            spec.dataset_key,
            spec.version_label,
            spec.source_url,
            spec.file_name,
            sha,
            path.stat().st_size,
            spec.row_count,
            spec.grain,
            True,
            spec.notes,
        ),
    )
    cur.execute(
        "select dataset_id, file_sha256 from datasets where version_label = %s",
        (spec.version_label,),
    )
    row = cur.fetchone()
    if row is None:  # pragma: no cover - 不可能发生
        raise ImportError_(f"无法登记数据版本 {spec.version_label}")
    ds_id, stored_sha = row
    if stored_sha != sha:
        warnings.append(
            f"数据库中已登记的数据版本 {spec.version_label} 的哈希为 {stored_sha}，"
            f"与本次文件 {sha} 不同。已沿用原登记，未覆盖。"
        )
    return ds_id


def _ensure_experiment(cur, exp: ExperimentSpec, ds_id: int, warnings: list[str]) -> int:
    cur.execute(
        """
        insert into experiments (
            experiment_key, dataset_id, name, randomization_unit, analysis_unit,
            outcome_window_days, outcome_window_fixed,
            expected_ratio_source, expected_ratio_note, notes
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        on conflict (experiment_key) do nothing
        """,
        (
            exp.experiment_key,
            ds_id,
            exp.name,
            exp.randomization_unit,
            exp.analysis_unit,
            exp.outcome_window_days,
            exp.outcome_window_fixed,
            exp.expected_ratio_source,
            exp.expected_ratio_note,
            exp.notes,
        ),
    )
    cur.execute(
        "select experiment_id, dataset_id from experiments where experiment_key = %s",
        (exp.experiment_key,),
    )
    row = cur.fetchone()
    if row is None:  # pragma: no cover
        raise ImportError_(f"无法登记实验 {exp.experiment_key}")
    exp_id, stored_ds = row
    if stored_ds != ds_id:
        raise ImportError_(
            f"实验 {exp.experiment_key} 已绑定到 dataset_id={stored_ds}，"
            f"与本次导入的 dataset_id={ds_id} 不一致。跨数据集混用被禁止。"
        )

    for arm in exp.arms:
        cur.execute(
            """
            insert into experiment_arms (
                experiment_id, arm_key, segment_value, role, expected_ratio
            )
            values (%s, %s, %s, %s, %s)
            on conflict (experiment_id, arm_key) do nothing
            """,
            (exp_id, arm.arm_key, arm.segment_value, arm.role, arm.expected_ratio),
        )

    for cmp_ in exp.comparisons:
        cur.execute(
            """
            insert into experiment_comparisons (
                experiment_id, comparison_key, treatment_arm_id, control_arm_id,
                kind, alpha, sided, multiplicity_note
            )
            select %(exp)s, %(key)s, tr.arm_id, ct.arm_id, %(kind)s, %(alpha)s, %(sided)s, %(note)s
            from experiment_arms tr, experiment_arms ct
            where tr.experiment_id = %(exp)s and tr.arm_key = %(tr)s
              and ct.experiment_id = %(exp)s and ct.arm_key = %(ct)s
            on conflict (experiment_id, comparison_key) do nothing
            """,
            {
                "exp": exp_id,
                "key": cmp_.comparison_key,
                "tr": cmp_.treatment_arm_key,
                "ct": cmp_.control_arm_key,
                "kind": cmp_.kind,
                "alpha": cmp_.alpha,
                "sided": cmp_.sided,
                "note": cmp_.multiplicity_note,
            },
        )

    # 一致性自检：arm 与 comparison 必须齐全
    cur.execute(
        "select count(*) from experiment_arms where experiment_id = %s", (exp_id,)
    )
    n_arms = cur.fetchone()[0]
    if n_arms != len(exp.arms):
        raise ImportError_(
            f"实验 {exp.experiment_key} 的组数不符：数据库 {n_arms}，规格 {len(exp.arms)}"
        )
    return exp_id


def _counts(cur, ds_id: int, exp_id: int) -> dict[str, int]:
    cur.execute("select count(*) from customers where dataset_id = %s", (ds_id,))
    n_cust = cur.fetchone()[0]
    cur.execute("select count(*) from assignments where experiment_id = %s", (exp_id,))
    n_asg = cur.fetchone()[0]
    cur.execute("select count(*) from outcomes where experiment_id = %s", (exp_id,))
    n_out = cur.fetchone()[0]
    return {"customers": n_cust, "assignments": n_asg, "outcomes": n_out}


def _arm_counts(cur, exp_id: int) -> dict[str, int]:
    cur.execute(
        """
        select a.arm_key, count(*)
        from assignments s
        join experiment_arms a on a.arm_id = s.arm_id
        where s.experiment_id = %s
        group by a.arm_key
        order by a.arm_key
        """,
        (exp_id,),
    )
    return {k: int(v) for k, v in cur.fetchall()}
