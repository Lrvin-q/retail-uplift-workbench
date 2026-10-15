"""产物存储：把大结果与报告落盘，并返回**相对 `artifacts/` 的引用**。

依据：
- [docs/08-工具协议.md](../../../docs/08-工具协议.md) §5「结果引用结构」
- [docs/05-报告结构与异常分类.md](../../../docs/05-报告结构与异常分类.md) §2.3 引用完整性与 `run_id` 追溯

## 为什么用引用而不是把数据塞进返回体

大结果进入 LLM 上下文会**同时**造成成本、延迟与精度问题。正确做法是
"有 42,613 行，前 10 行长这样，完整结果在 `<ref>`"。

## 安全（目录穿越防护）

所有 `ref` 必须是**相对 `ARTIFACTS_ROOT` 的路径**。解析后必须仍在
`ARTIFACTS_ROOT` 内，否则抛 `ValueError`：

```python
read_artifact("../../etc/passwd")      # ValueError
artifact_exists("..\\..\\windows\\x")  # ValueError
read_artifact("C:\\Windows\\win.ini")  # ValueError（绝对路径/盘符）
```

防护方式：先用 `Path.resolve()` 归一化（同时解开 `.` / `..` / 符号链接），
再判断结果是否等于根目录或位于根目录之下。**先归一化再判断**是关键——
只在字符串层面拦 `..` 会被 `a/../b`、符号链接、Windows 盘符等写法绕过。

## 布局

```text
artifacts/
  <run_id>/
    customer_scores.csv
    report.md
```

`ref` 形如 `<run_id>/customer_scores.csv`（统一用 `/` 分隔，便于迁移与审计）。
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dsa.config import PROJECT_ROOT

__all__ = [
    "ARTIFACTS_ROOT",
    "SUPPORTED_TABLE_FORMATS",
    "artifact_abs_path",
    "artifact_exists",
    "read_artifact",
    "resolve_artifact_path",
    "write_report_artifact",
    "write_table_artifact",
]

#: 产物根目录（已 gitignore）。与 `dsa.config.ARTIFACTS_DIR` 指向同一路径。
from dsa.config import ARTIFACTS_DIR
ARTIFACTS_ROOT = ARTIFACTS_DIR

#: 表格产物支持的格式 → 文件后缀
SUPPORTED_TABLE_FORMATS: dict[str, str] = {
    "csv": ".csv",
    "tsv": ".tsv",
    "json": ".json",
    "jsonl": ".jsonl",
}

_REPORT_SUFFIX = ".md"
_KNOWN_SUFFIXES = tuple(sorted(set(SUPPORTED_TABLE_FORMATS.values()) | {_REPORT_SUFFIX}, key=len,
                               reverse=True))


# ============================================================================
#  路径解析与目录穿越防护
# ============================================================================
def _artifacts_root() -> Path:
    """每次调用都重新读模块全局，便于测试用 monkeypatch 换根目录。"""
    return Path(ARTIFACTS_ROOT).resolve()


def resolve_artifact_path(ref: str, *, must_exist: bool = False) -> Path:
    """把相对引用解析为 `ARTIFACTS_ROOT` 内的绝对路径。

    Raises:
        ValueError: `ref` 不是字符串、为空、是绝对路径/含盘符，或解析后越界。
        FileNotFoundError: `must_exist=True` 且目标不存在。
    """
    if not isinstance(ref, str):
        raise ValueError(f"ref 必须是字符串，收到: {type(ref).__name__}")
    if not ref.strip():
        raise ValueError("ref 不能为空")
    if "\x00" in ref:
        raise ValueError("ref 不得包含空字符")

    candidate = Path(ref)
    # Windows 上 Path("C:/x").drive == "C:"；UNC 路径的 drive 是 \\\\host\\share
    if candidate.is_absolute() or candidate.drive or candidate.root:
        raise ValueError(f"ref 必须是相对 {ARTIFACTS_ROOT.name}/ 的路径，不得是绝对路径: {ref!r}")

    root = _artifacts_root()
    # resolve() 同时完成 规范化、解开 .. 与跟随符号链接——防护的关键一步
    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"ref 越界，解析后不在 {ARTIFACTS_ROOT.name}/ 内: {ref!r} -> {target}")

    if must_exist and not target.is_file():
        raise FileNotFoundError(f"产物不存在: {ref!r}")
    return target


def artifact_abs_path(ref: str) -> Path:
    """公开的路径解析入口（同样是防穿越的）。"""
    return resolve_artifact_path(ref)


# 保持旧名可用：文档与其它模块可能引用 `_resolve_ref`
_resolve_ref = resolve_artifact_path


def _check_run_id(run_id: str | uuid.UUID) -> str:
    """`run_id` 只作为产物目录名，必须是不含分隔符的单一路径段。"""
    text_id = str(run_id).strip()
    if not text_id:
        raise ValueError("run_id 不能为空")
    if any(sep in text_id for sep in ("/", "\\")) or text_id in (".", ".."):
        raise ValueError(f"run_id 不得包含路径分隔符: {run_id!r}")
    if Path(text_id).drive:
        raise ValueError(f"run_id 不得包含盘符: {run_id!r}")
    return text_id


def _check_name(name: str) -> Path:
    """`name` 是 `<run_id>/` 之下的相对路径，同样不得穿越。"""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"name 必须是非空字符串，收到: {name!r}")
    if "\x00" in name:
        raise ValueError("name 不得包含空字符")
    candidate = Path(name)
    if candidate.is_absolute() or candidate.drive or candidate.root:
        raise ValueError(f"name 不得是绝对路径: {name!r}")
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"name 不得包含 '..': {name!r}")
    return candidate


def _apply_suffix(rel: Path, suffix: str, original: str) -> Path:
    """补后缀；若已带**另一个**已知后缀则报错（避免 report.json 写成 csv）。"""
    if rel.suffix == suffix:
        return rel
    if rel.suffix and rel.suffix.lower() in {s.lower() for s in _KNOWN_SUFFIXES}:
        raise ValueError(f"name 的后缀 {rel.suffix!r} 与请求的格式后缀 {suffix!r} 不一致: {original!r}")
    return rel.with_name(rel.name + suffix)


def _prepare_target(run_id: str | uuid.UUID, name: str, suffix: str) -> tuple[Path, str]:
    """构造目标绝对路径与相对引用，并保证不会越界。"""
    safe_run_id = _check_run_id(run_id)
    rel = _apply_suffix(_check_name(name), suffix, name)
    ref = f"{safe_run_id}/{rel.as_posix()}"
    # 复用同一套防护：写路径与读路径走完全相同的校验
    target = resolve_artifact_path(ref)
    target.parent.mkdir(parents=True, exist_ok=True)
    # mkdir 之后再校验一次：防止"先建目录再跟随符号链接"的时序问题
    root = _artifacts_root()
    resolved = target.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"ref 越界，解析后不在 {ARTIFACTS_ROOT.name}/ 内: {ref!r}")
    return resolved, ref


def _atomic_write_text(target: Path, content: str) -> None:
    """先写临时文件再 `os.replace`，避免半截产物被当成完整产物引用。"""
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(content, encoding="utf-8", newline="\n")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:  # pragma: no cover - 清理失败不影响主流程
                pass


# ============================================================================
#  写产物
# ============================================================================
def write_table_artifact(
    rows_or_df: Any,
    *,
    run_id: str | uuid.UUID,
    name: str,
    fmt: str = "csv",
) -> str:
    """把表格结果落盘，返回相对 `ARTIFACTS_ROOT` 的引用。

    Args:
        rows_or_df: `pandas.DataFrame`，或 `Iterable[Mapping]`（每行一个 dict）。
        run_id: 关联的分析运行标识（产物目录名）。
        name: 相对 `<run_id>/` 的文件名，可含子目录；缺后缀时按 `fmt` 自动补。
        fmt: `csv` / `tsv` / `json` / `jsonl`。

    Returns:
        相对引用，如 `"<run_id>/customer_scores.csv"`。

    Raises:
        ValueError: 格式不支持、名字非法、后缀与格式不一致、输入类型不支持。
        TypeError: `rows_or_df` 既不是 DataFrame 也不是行序列。
    """
    if fmt not in SUPPORTED_TABLE_FORMATS:
        raise ValueError(
            f"不支持的表格格式: {fmt!r}；允许值: {', '.join(sorted(SUPPORTED_TABLE_FORMATS))}"
        )
    suffix = SUPPORTED_TABLE_FORMATS[fmt]
    # 先转换再落盘：输入类型不合法时不留下空目录
    df = _as_dataframe(rows_or_df)
    target, ref = _prepare_target(run_id, name, suffix)

    if fmt == "csv":
        df.to_csv(target, index=False, encoding="utf-8")
    elif fmt == "tsv":
        df.to_csv(target, index=False, encoding="utf-8", sep="\t")
    elif fmt == "json":
        df.to_json(target, orient="records", force_ascii=False, indent=2)
    else:  # jsonl
        df.to_json(target, orient="records", lines=True, force_ascii=False)
    return ref


def _as_dataframe(rows_or_df: Any):
    """把受支持的输入统一成 DataFrame；不做静默的类型兜底。"""
    import pandas as pd

    if isinstance(rows_or_df, pd.DataFrame):
        return rows_or_df
    if isinstance(rows_or_df, (str, bytes, Mapping)):
        raise TypeError(
            f"rows_or_df 必须是 DataFrame 或行的序列（每行是 dict），"
            f"收到: {type(rows_or_df).__name__}"
        )
    if isinstance(rows_or_df, Sequence) or isinstance(rows_or_df, Iterable):
        rows = list(rows_or_df)
        if not rows:
            return pd.DataFrame()
        if not all(isinstance(row, Mapping) for row in rows):
            bad = next(type(row).__name__ for row in rows if not isinstance(row, Mapping))
            raise TypeError(f"rows_or_df 的每一行必须是 dict，发现: {bad}")
        return pd.DataFrame([dict(row) for row in rows])
    raise TypeError(f"不支持的 rows_or_df 类型: {type(rows_or_df).__name__}")


def write_report_artifact(text: str, *, run_id: str | uuid.UUID, name: str) -> str:
    """把报告文本（Markdown）落盘，返回相对引用。缺后缀时补 `.md`。"""
    if not isinstance(text, str):
        raise ValueError(f"text 必须是字符串，收到: {type(text).__name__}")
    target, ref = _prepare_target(run_id, name, _REPORT_SUFFIX)
    _atomic_write_text(target, text)
    return ref


# ============================================================================
#  读产物
# ============================================================================
def read_artifact(ref: str, *, max_bytes: int = 1_000_000) -> str:
    """读取文本产物。

    Args:
        ref: 相对 `ARTIFACTS_ROOT` 的路径。
        max_bytes: 允许读取的最大字节数。

    Returns:
        文件内容（UTF-8 文本）。

    Raises:
        ValueError: 引用越界、文件超过 `max_bytes`、或内容不是 UTF-8 文本。
            超限**不静默截断**——截断的产物被当成完整结果引用，比报错危险得多。
        FileNotFoundError: 产物不存在。
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError(f"max_bytes 必须是正整数，收到: {max_bytes!r}")
    target = resolve_artifact_path(ref, must_exist=True)
    size = target.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"产物超过 max_bytes 上限，拒绝读取以免被当成完整结果: "
            f"{ref!r} 大小 {size} 字节 > {max_bytes} 字节"
        )
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"产物不是 UTF-8 文本，无法按文本读取: {ref!r}") from exc


def artifact_exists(ref: str) -> bool:
    """产物是否存在。

    与 `read_artifact` 使用**同一套**越界校验：越界同样抛 `ValueError`，
    绝不返回 `False`——否则越界探测会被误读成"文件不存在"。
    """
    target = resolve_artifact_path(ref)
    return target.is_file()


def list_artifacts(run_id: str | uuid.UUID) -> list[str]:
    """列出某次运行目录下的全部产物引用（按路径排序）。"""
    safe_run_id = _check_run_id(run_id)
    root = _artifacts_root()
    run_dir = resolve_artifact_path(safe_run_id)
    if not run_dir.is_dir():
        return []
    refs: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            resolved = path.resolve()
            if resolved == root or root in resolved.parents:
                refs.append(resolved.relative_to(root).as_posix())
    return refs
