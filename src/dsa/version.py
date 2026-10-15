"""代码版本标识（复现凭据的一部分）。

**为什么要有这个模块**：项目里原本有**两份** `code_version()`：

- `dsa.tools.run_records.code_version()` —— 返回 `hash` 或 `hash-dirty`
- `dsa.stats.experiment.code_version()` —— 只返回 `hash`

两份实现意味着同一个问题有两个答案，而它们会被写进不同的表
（`analysis_runs.code_version` 与 `models.code_version`），
于是"这次分析用的是哪个版本的代码"取决于**是谁写的**，而不是事实。

更糟的是 `run_records` 那一份在 Windows 上是坏的：
`subprocess.run(..., text=True)` 按本地代码页（GBK）解码
`git status --porcelain` 的 UTF-8 输出，解码失败后 `stdout` 为 None，
`.strip()` 抛 `AttributeError`。它长期没被发现，因为**没有生产路径调用它**
—— 直到运行登记真正接进 Agent 默认流程（审查 F06）才暴露。
这和本项目另外两次教训同族：**有函数、有测试 ≠ 被调用**。

因此：**一处实现，两处引用**。
"""

from __future__ import annotations

import subprocess

from dsa.config import PROJECT_ROOT

__all__ = ["code_version"]

#: 取不到 git 信息时的取值。**不伪装成正常版本号**（AGENTS.md §2.7）。
UNKNOWN_VERSION = "unknown"


def code_version() -> str:
    """当前代码版本：``<短哈希>`` 或 ``<短哈希>-dirty``。

    `-dirty` 很重要：工作区有未提交改动时，仅凭哈希**无法复现**，
    而"以为是干净版本"正是复现失败最常见的原因。
    取不到 git（没有仓库、没有 git 可执行文件）时返回 :data:`UNKNOWN_VERSION`，
    **不抛异常** —— `/health` 与运行登记必须在没有 git 的环境下也能工作。

    `encoding="utf-8"` 是**必须**的：Windows 下 `text=True` 会用本地代码页
    解码子进程输出，遇到中文路径就解码失败并把 `stdout` 置为 None。
    """
    def _git(*args: str) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None

    try:
        head = _git("rev-parse", "--short", "HEAD")
        if head is None or head.returncode != 0 or not (head.stdout or "").strip():
            return UNKNOWN_VERSION
        version = (head.stdout or "").strip()
        status = _git("status", "--porcelain")
        if status is not None and status.returncode == 0 and (status.stdout or "").strip():
            version = f"{version}-dirty"
        return version
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_VERSION
