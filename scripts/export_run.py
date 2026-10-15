"""运行导出与重放的**命令行入口**（薄包装）。

实现全部在 :mod:`dsa.tools.run_export` —— 这样工作台、服务与其他包内代码
可以 import 它，而不必依赖"项目根目录在 `sys.path` 上"这个前提。
（原先工作台写的是 `from scripts.export_run import ...`，
在 `pip install .` 之后会失败 —— 因为 `scripts/` 不在包里。）

用法：
    python scripts/export_run.py list
    python scripts/export_run.py export <run_id>
    python scripts/export_run.py replay <run_id>

退出码：

| 码 | 含义 |
|---|---|
| 0 | 成功（重放时表示**逐字段一致**）|
| 1 | 运行不存在 / 重跑失败 |
| 2 | 导出不完整，或**重放未执行**（缺可比对的历史产物）|
| 3 | 重放**有差异** |
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsa.tools.run_export import cmd_export, cmd_list, cmd_replay  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="运行导出与重放")
    ap.add_argument("cmd", choices=["list", "export", "replay"])
    ap.add_argument("run_id", nargs="?")
    args = ap.parse_args()
    if args.cmd == "list":
        return cmd_list()
    if not args.run_id:
        print("请提供 run_id")
        return 1
    return cmd_export(args.run_id) if args.cmd == "export" else cmd_replay(args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())
