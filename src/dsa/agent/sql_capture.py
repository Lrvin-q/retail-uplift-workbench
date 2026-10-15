"""捕获工具调用期间**实际执行**的 SQL 与绑定参数（阶段六追溯）。

缺陷回顾（外部审查 F06）
------------------------
> 即使传入运行标识，查询记录中写入的 SQL 也是 `-- tool: ...` 注释，
> 未保存实际参数化 SQL。部分行数从不存在的 `rows` 字段推导，
> 对于聚合返回并不可靠。

也就是说：`query_executions` 表里那一列叫 `sql_text`，
存进去的却是工具名注释。**表的形状承诺了一件事，写入的内容是另一件事。**
看表的人会以为"这里有 SQL 可查"，实际上没有。

修法是不要再手写一句话描述 SQL，而是**让数据库自己告诉我们它执行了什么**：
用 SQLAlchemy 的 `before_cursor_execute` / `after_cursor_execute` 事件，
把工具调用期间真正发到服务器的语句与参数记下来。

这样做的额外好处：参数化查询的**参数值**也一并留下，
而"参数值"正是"分母是不是被筛过"这类问题的直接证据。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

__all__ = ["SqlCapture", "capture_sql"]


class SqlCapture:
    """一次捕获的结果。

    Attributes:
        statements: 每次 `execute` 的 ``(sql, params, rowcount)``。
            `rowcount` 为 None 表示驱动没有给出（如某些 DDL）。
    """

    def __init__(self) -> None:
        self.statements: list[tuple[str, Any, int | None]] = []

    @property
    def last_sql(self) -> str | None:
        return self.statements[-1][0] if self.statements else None

    @property
    def last_params(self) -> Any:
        return self.statements[-1][1] if self.statements else None

    @property
    def last_rowcount(self) -> int | None:
        return self.statements[-1][2] if self.statements else None

    @property
    def sql_text(self) -> str | None:
        """全部语句按顺序拼接。

        多语句时**不截断**：少记一条就可能漏掉真正改变分母的那一条。
        语句之间用分号与换行分隔，便于人工阅读与 `sqlglot` 解析。
        """
        if not self.statements:
            return None
        return ";\n".join(s.strip() for s, _p, _r in self.statements)

    @property
    def params(self) -> dict[str, Any]:
        """合并全部语句的绑定参数（后出现的覆盖先出现的同名参数）。"""
        merged: dict[str, Any] = {}
        for _s, p, _r in self.statements:
            if isinstance(p, dict):
                merged.update(p)
            elif isinstance(p, (list, tuple)):
                merged[f"__positional_{len(merged)}"] = list(p)
        return merged

    @property
    def row_count(self) -> int | None:
        """最后一条语句返回的行数。

        为什么取**最后一条**而不是全部之和：一次工具调用往往先查配置
        再查数据，读者关心的是"结果有多少行"，也就是最后那条。
        全部为 None 时返回 None —— 缺失就是缺失，不填 0。
        """
        for _s, _p, r in reversed(self.statements):
            if isinstance(r, int) and r >= 0:
                return r
        return None


@contextmanager
def capture_sql(engine: Any) -> Iterator[SqlCapture]:
    """捕获 `engine` 上在本上下文内执行的 SQL。

    监听器在退出时**一定**被移除：engine 通常是缓存的单例，
    泄漏的监听器会让后续无关查询也被记进来，使追溯信息失真。
    """
    from sqlalchemy import event

    cap = SqlCapture()

    def _before(conn: Any, cursor: Any, statement: str, parameters: Any,
                context: Any, executemany: bool) -> None:  # noqa: ARG001
        cap.statements.append((statement, parameters, None))

    def _after(conn: Any, cursor: Any, statement: str, parameters: Any,
               context: Any, executemany: bool) -> None:  # noqa: ARG001
        if not cap.statements:
            return
        try:
            rowcount = int(cursor.rowcount)
        except Exception:  # noqa: BLE001 —— 驱动不给 rowcount 是正常的
            rowcount = -1
        sql, params, _old = cap.statements[-1]
        cap.statements[-1] = (sql, params, rowcount if rowcount >= 0 else None)

    event.listen(engine, "before_cursor_execute", _before)
    event.listen(engine, "after_cursor_execute", _after)
    try:
        yield cap
    finally:
        event.remove(engine, "before_cursor_execute", _before)
        event.remove(engine, "after_cursor_execute", _after)
