"""受控 NL2SQL —— SQL 校验与语义检查（本模块的核心）。

设计立场
--------
**自由生成的 SQL 与参数化工具是两种风险等级完全不同的东西。**
参数化工具的 SQL 是写死的，风险在参数；自由 SQL 的风险在**语句本身**。
因此这里不依赖"调用方会传入好 SQL"这一假设，而是把三道闸门做成可独立测试的函数：

1. :func:`parse_single_select` —— **语法层面**只放行单条 SELECT，判据是 AST。
2. :func:`check_dangerous` —— 危险语句与越界对象：多语句、DDL/DML、
   文件/网络/延时函数、``into outfile``、``pg_*`` 与 ``information_schema``。
3. :func:`check_semantics` —— **业务口径**检查。这一层才是本项目真正的价值所在：
   语法合法的 SQL 依然可能是错的（分母按 visit 过滤、金额没去重、
   两个数据集混用），而且这些错误**执行成功、不报错、数字还挺像样**。

为什么必须用 AST，而不是文本匹配
---------------------------------
蓝图对这一步的要求是「限制危险语句、多语句以及不允许的函数和对象；
**不只检查文本是否以 SELECT 开头**」。下面这组结果是 sqlglot 30.22.0
（``read='postgres'``）在本机**实测**出来的：

==========================================================  =========  =========
样例                                                        语句数      含写操作
==========================================================  =========  =========
``select count(*) from assignments``                        1          False
``select 1; drop table t``                                  **2**      True
``  select 1; drop table t``                                **2**      True
``with x as (insert into t values (1) returning *) select * from x``
                                                            **1**      **True**
``select * from pg_shadow``                                 1          False
==========================================================  =========  =========

最后那行是关键：**语句数是 1、文本以 ``with`` 开头、最外层确实是 SELECT**。
任何基于「首词是不是 select」「数了几个分号」「正则扫危险关键字」的检查
都会放行它，而它内部真的会执行 ``INSERT``。
只有把语句解析成树、再在**整棵树**上找写操作节点才能发现。

因此判定顺序是：**先 AST（权威），再用词法做快速预筛（非权威）**。
:func:`has_dml_anywhere` / :func:`parse_statements` 是权威判据；
分号计数、括号配对、危险关键字扫描只是预筛，它们**永远不会**被用来"放行"
（预筛只可能拒绝，不可能批准），所以它们的漏报不影响安全性，只影响报错速度。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence, Union

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from dsa.nl2sql import scope as sc

__all__ = [
    "ALLOWED_ROOT_NAMES",
    "CHECK_CODES",
    "DANGEROUS_FUNCTIONS",
    "DANGEROUS_KEYWORDS",
    "FORBIDDEN_NODE_NAMES",
    "Issue",
    "MAX_ROWS_DEFAULT",
    "STATEMENT_TIMEOUT_MS_DEFAULT",
    "SqlParseError",
    "Token",
    "check_dangerous",
    "check_semantics",
    "clauses",
    "dangerous_nodes",
    "enforce_limits",
    "find_top_level_clause",
    "has_dml_anywhere",
    "mask_literals",
    "parse_single_select",
    "parse_statements",
    "post_treatment_columns",
    "referenced_tables",
    "remove_trailing_semicolon",
    "statement_keyword",
    "statement_type",
    "tokenize",
]


# ============================================================================
#  常量
# ============================================================================

#: 默认内联行数上限：与 docs/08 §2 的 ``MAX_INLINE_ROWS`` 一致，
#: 保证 NL2SQL 的结果不会比参数化工具更容易撑爆 LLM 上下文。
MAX_ROWS_DEFAULT: int = 1000

#: 与 docs/08 §4.1 一致：语句超时 10 秒，超时映射为 ``E_DB_003``。
STATEMENT_TIMEOUT_MS_DEFAULT: int = 10_000

#: DDL / DML / 权限 / 文件类关键字。**出现在任何位置都拒绝**（含 CTE 内层）。
DANGEROUS_KEYWORDS: tuple[str, ...] = (
    "insert", "update", "delete", "drop", "truncate", "alter", "create",
    "grant", "revoke", "copy", "merge", "call", "do", "vacuum", "analyze",
    "cluster", "reindex", "refresh", "comment", "set", "reset", "show",
    "listen", "notify", "prepare", "execute", "deallocate", "lock",
    "into",          # select ... into new_table 会建表
    "outfile",       # ... into outfile（MySQL 习惯写法，一并挡掉）
    "dumpfile",
)

#: 危险函数：文件读写、延时、外部连接、大对象。
DANGEROUS_FUNCTIONS: tuple[str, ...] = (
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "pg_sleep", "pg_sleep_for", "pg_sleep_until", "pg_terminate_backend",
    "pg_cancel_backend", "pg_reload_conf", "pg_rotate_logfile",
    "lo_import", "lo_export", "lo_unlink", "lo_create", "lo_get", "lo_put",
    "dblink", "dblink_connect", "dblink_exec", "dblink_send_query",
    "postgres_fdw_disconnect", "query_to_xml", "database_to_xml",
    "current_setting", "set_config", "pg_read_server_files",
    "pg_write_file", "pg_file_write", "pg_file_unlink", "pg_logdir_ls",
    "xp_cmdshell", "load_extension", "readfile", "writefile",
)

#: 只允许出现在最外层的语句起始关键字。
SELECT_LEADING: frozenset[str] = frozenset({"select", "with", "table", "values"})

#: 子查询起始关键字（用于找 NOT EXISTS 之类的子查询表引用）。
SUBSELECT_LEADING: frozenset[str] = frozenset({"select", "with", "table", "values"})

#: 顶层子句关键字：用于切分「where 部分」与「group by 部分」等。
_CLAUSE_STARTERS: frozenset[str] = frozenset(
    {"where", "group", "having", "order", "limit", "offset", "union",
     "intersect", "except", "fetch", "window", "for", "returning"}
)

#: FROM 子句的终止关键字。
_FROM_STOP: frozenset[str] = frozenset(
    {"where", "group", "having", "order", "limit", "offset", "union",
     "intersect", "except", "fetch", "window", "for", "on", "using", "returning"}
)

#: 连接关键字。
_JOIN_WORDS: frozenset[str] = frozenset({"join", "inner", "left", "right", "full", "cross", "natural", "lateral"})

#: 在 FROM / JOIN 里出现但**不是表**的写法（表函数、仅索引扫描提示等）。
_TABLE_POSITION_IGNORE: frozenset[str] = frozenset({"only", "lateral"})

#: 干预后字段：**不得**进入 where / join ... on（会影响分母或人群）。
#: 统一小写；比对时取列名（忽略 schema 前缀）。
POST_TREATMENT_COLUMNS: frozenset[str] = frozenset(
    {
        # Hillstrom
        "visit", "visited", "conversion", "converted", "spend",
        # RetailHero
        "treatment_flg", "target",
    }
)

#: Hillstrom 结果字段的**正确**用法是放进 FILTER / CASE，而不是 where。
_FILTER_WORDS: frozenset[str] = frozenset({"filter"})
_CASE_WORDS: frozenset[str] = frozenset({"case", "when", "then", "else", "end", "if"})

#: 聚合函数（用于金额去重检查）。
_AGGREGATES: frozenset[str] = frozenset({"sum", "avg", "min", "max", "count", "stddev", "variance", "percentile_cont", "percentile_disc"})

#: 交易级字段：出现在**明细表**上的聚合必须先按 transaction_id 收敛。
TRANSACTION_LEVEL_COLUMNS: frozenset[str] = frozenset(
    {
        "purchase_sum",
        "regular_points_received", "express_points_received",
        "regular_points_spent", "express_points_spent",
    }
)

#: 客户标识列（唯一客户数检查用）。
_CUSTOMER_ID_COLUMNS: frozenset[str] = frozenset({"client_id", "customer_id"})

#: 通过「按交易收敛」的入口：这些对象上的 purchase_sum 已经是收敛后的。
CONVERGED_OBJECTS: frozenset[str] = frozenset({"rh_transactions", "rh_client_features"})


class CheckCode:
    """语义检查的条目编码。**每条 issue 都必须带 code**，便于测试与下游路由。"""

    DENOMINATOR_POST_TREATMENT = "SEM_DENOM_001"      # 分母红线
    JOIN_EXPANSION = "SEM_JOIN_001"                   # 关联扩张
    CUSTOMER_COUNT_NOT_DISTINCT = "SEM_COUNT_001"     # 唯一客户数
    DATASET_MIXED = "SEM_DATASET_001"                 # 数据集不混用
    AMOUNT_NOT_DEDUPED = "SEM_AMOUNT_001"             # 金额未按交易去重
    AGGREGATE_WITHOUT_GROUP_BY = "SEM_AGG_001"        # 混用聚合与非聚合列
    CROSS_DATASET_JOIN = "SEM_DATASET_002"            # 跨数据集 JOIN


#: 全部语义检查编码（自检用）。
CHECK_CODES: tuple[str, ...] = (
    CheckCode.DENOMINATOR_POST_TREATMENT,
    CheckCode.JOIN_EXPANSION,
    CheckCode.CUSTOMER_COUNT_NOT_DISTINCT,
    CheckCode.DATASET_MIXED,
    CheckCode.AMOUNT_NOT_DEDUPED,
    CheckCode.AGGREGATE_WITHOUT_GROUP_BY,
    CheckCode.CROSS_DATASET_JOIN,
)

#: 阻断级编码：出现即不得把结果当作正确结果返回。
BLOCKING_CODES: frozenset[str] = frozenset(
    {
        CheckCode.DENOMINATOR_POST_TREATMENT,
        CheckCode.DATASET_MIXED,
        CheckCode.CROSS_DATASET_JOIN,
        CheckCode.AMOUNT_NOT_DEDUPED,
    }
)


# ============================================================================
#  分词器
# ============================================================================


@dataclass(frozen=True, slots=True)
class Token:
    """一个词法单元。``start`` / ``end`` 是它在原 SQL 中的字符区间。"""

    kind: str  # word | quoted | number | string | punct | op | param
    text: str  # 原始文本（引号/美元符保留，便于回填）
    start: int
    end: int

    @property
    def low(self) -> str:
        """小写文本（未去引号）。"""
        return self.text.lower()

    @property
    def bare(self) -> str:
        """去引号后的标识符名。``"Customers"`` → ``Customers``，``customers`` → ``customers``。"""
        if self.kind == "quoted":
            return unquote_ident(self.text)
        return self.text

    @property
    def low_bare(self) -> str:
        return self.bare.lower()


def unquote_ident(raw: str) -> str:
    """去掉标识符外侧的双引号并按 SQL 规则还原内部的双引号。"""
    value = raw.strip()
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('""', '"')
    return value


_WS = " \t\r\n\f\v"
_IDENT_START = re.compile(r"[A-Za-z_\u0080-\uffff]")
_IDENT_BODY = re.compile(r"[A-Za-z0-9_$\u0080-\uffff]")
_OPERATOR_CHARS = set("+-*/%<>=!|&^~#@?:")


def tokenize(sql: str) -> list[Token]:
    """把 SQL 切成词法单元，**保留每个 token 在原串中的位置**。

    支持：单引号字符串（含 ``''`` 转义）、双引号与方括号标识符、
    行注释 ``--``、块注释（**可嵌套**，PostgreSQL 如此）、
    ``$$...$$`` / ``$tag$...$tag$`` 美元引用、``:name`` 与 ``%s`` 参数占位符。

    遇到无法闭合的字符串或注释时**抛 ValueError** ——
    残缺的输入不应该被"尽力解析"，那是绕过检查的常见手法。
    """
    tokens: list[Token] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]

        # ---- 空白 ----
        if ch in _WS:
            i += 1
            continue

        # ---- 行注释 ----
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            j = sql.find("\n", i)
            i = n if j < 0 else j + 1
            continue

        # ---- 块注释（PostgreSQL 允许嵌套）----
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            depth = 1
            j = i + 2
            while j < n and depth:
                if sql.startswith("/*", j):
                    depth += 1
                    j += 2
                elif sql.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            if depth:
                raise ValueError("SQL 中的块注释没有闭合")
            i = j
            continue

        # ---- 单引号字符串 ----
        if ch == "'":
            j = i + 1
            while True:
                k = sql.find("'", j)
                if k < 0:
                    raise ValueError("SQL 中的字符串字面量没有闭合")
                if k + 1 < n and sql[k + 1] == "'":  # '' 转义
                    j = k + 2
                    continue
                j = k + 1
                break
            tokens.append(Token("string", sql[i:j], i, j))
            i = j
            continue

        # ---- 双引号 / 方括号标识符 ----
        if ch == '"':
            j = i + 1
            while True:
                k = sql.find('"', j)
                if k < 0:
                    raise ValueError("SQL 中的双引号标识符没有闭合")
                if k + 1 < n and sql[k + 1] == '"':
                    j = k + 2
                    continue
                j = k + 1
                break
            tokens.append(Token("quoted", sql[i:j], i, j))
            i = j
            continue
        if ch == "[":
            k = sql.find("]", i + 1)
            if k < 0:
                raise ValueError("SQL 中的方括号标识符没有闭合")
            tokens.append(Token("quoted", sql[i : k + 1], i, k + 1))
            i = k + 1
            continue

        # ---- 美元引用字符串 / 位置参数 ----
        if ch == "$":
            m = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$", sql[i:])
            if m:
                tag = m.group(0)
                k = sql.find(tag, i + len(tag))
                if k < 0:
                    raise ValueError(f"SQL 中的美元引用 {tag} 没有闭合")
                j = k + len(tag)
                tokens.append(Token("string", sql[i:j], i, j))
                i = j
                continue
            m = re.match(r"\$\d+", sql[i:])
            if m:
                j = i + len(m.group(0))
                tokens.append(Token("param", sql[i:j], i, j))
                i = j
                continue

        # ---- 数字 ----
        if ch.isdigit() or (ch == "." and i + 1 < n and sql[i + 1].isdigit()):
            m = re.match(r"\d+(\.\d*)?([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?", sql[i:])
            assert m is not None
            j = i + len(m.group(0))
            tokens.append(Token("number", sql[i:j], i, j))
            i = j
            continue

        # ---- 参数占位符 :name / %(name)s / %s ----
        if ch == ":" and i + 1 < n and (_IDENT_START.match(sql[i + 1]) or sql[i + 1] == ":"):
            if sql[i + 1] != ":":  # :: 是类型转换，交给操作符处理
                m = re.match(r":[A-Za-z_][A-Za-z0-9_]*", sql[i:])
                assert m is not None
                j = i + len(m.group(0))
                tokens.append(Token("param", sql[i:j], i, j))
                i = j
                continue
        if ch == "%":
            m = re.match(r"%\([A-Za-z_][A-Za-z0-9_]*\)s|%s", sql[i:])
            if m:
                j = i + len(m.group(0))
                tokens.append(Token("param", sql[i:j], i, j))
                i = j
                continue

        # ---- 标识符 / 关键字 ----
        if _IDENT_START.match(ch):
            m = re.match(r"[A-Za-z_\u0080-\uffff][A-Za-z0-9_$\u0080-\uffff]*", sql[i:])
            assert m is not None
            j = i + len(m.group(0))
            tokens.append(Token("word", sql[i:j], i, j))
            i = j
            continue

        # ---- 标点 ----
        if ch in "(),.;":
            tokens.append(Token("punct", ch, i, i + 1))
            i += 1
            continue

        # ---- 操作符（连续的同字符连写算一个，:: 与 -> 之类的组合单独处理）----
        if ch in _OPERATOR_CHARS:
            m = re.match(r"::|->>|->|#>>|#>" + "|" + re.escape(ch) + r"+", sql[i:])
            assert m is not None
            j = i + len(m.group(0))
            tokens.append(Token("op", sql[i:j], i, j))
            i = j
            continue

        # ---- 其它单个字符 ----
        tokens.append(Token("punct", ch, i, i + 1))
        i += 1

    return tokens


def _pairs(tokens: Sequence[Token]) -> dict[int, int]:
    """括号配对：返回 ``{开括号下标: 闭括号下标}`` 与反向映射（缺配对的记 ``-1``）。"""
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for idx, tok in enumerate(tokens):
        if tok.kind == "punct" and tok.text == "(":
            stack.append(idx)
        elif tok.kind == "punct" and tok.text == ")":
            if stack:
                open_idx = stack.pop()
                pairs[open_idx] = idx
                pairs[idx] = open_idx
            else:
                pairs[idx] = -1
    for idx in stack:
        pairs[idx] = -1
    return pairs


def _matching(tokens: Sequence[Token], open_idx: int, pairs: dict[int, int]) -> int | None:
    """返回与 ``tokens[open_idx]`` 配对的闭括号下标；缺失或不匹配返回 ``None``。"""
    if not (0 <= open_idx < len(tokens)):
        return None
    if tokens[open_idx].text != "(":
        return None
    close = pairs.get(open_idx, -1)
    return None if close < 0 else close


def _depth_map(tokens: Sequence[Token]) -> list[int]:
    """每个 token 所处的括号深度（顶层 = 0）。"""
    depths: list[int] = []
    depth = 0
    for tok in tokens:
        if tok.kind == "punct" and tok.text == ")":
            depth -= 1
        depths.append(max(depth, 0))
        if tok.kind == "punct" and tok.text == "(":
            depth += 1
    return depths


# ============================================================================
#  结构解析
# ============================================================================


def remove_trailing_semicolon(sql: str) -> str:
    """去掉末尾的分号（只去最后一个；中间的分号是多语句，不允许出现）。"""
    stripped = sql.rstrip()
    if stripped.endswith(";"):
        return stripped[:-1].rstrip()
    return stripped


def statement_keyword(sql: str, *, tokens: Sequence[Token] | None = None) -> str:
    """返回最外层语句的起始关键字（小写）。

    ``WITH x AS (...) SELECT ...`` 返回 ``select`` —— 因为 ``WITH`` 只是前缀，
    真正决定语句类型的是 CTE 之后那个关键字。
    """
    toks = list(tokens) if tokens is not None else tokenize(sql)
    # 去掉开头的多余分号
    idx = 0
    while idx < len(toks) and toks[idx].kind == "punct" and toks[idx].text == ";":
        idx += 1
    if idx >= len(toks):
        return ""

    first = toks[idx].low_bare
    if first != "with":
        return first

    # 跳过 WITH [RECURSIVE] cte_name [(cols)] AS [NOT] MATERIALIZED ( ... ) [, ...]
    j = idx + 1
    if j < len(toks) and toks[j].low_bare == "recursive":
        j += 1
    pairs = _pairs(toks)
    while j < len(toks):
        # CTE 名
        if toks[j].kind in ("word", "quoted"):
            j += 1
        else:
            return ""
        # 可选列清单
        if j < len(toks) and toks[j].text == "(":
            close = _matching(toks, j, pairs)
            if close is None:
                return ""
            j = close + 1
        # AS
        if j < len(toks) and toks[j].low_bare == "as":
            j += 1
        else:
            return ""
        # [NOT] MATERIALIZED
        if j < len(toks) and toks[j].low_bare == "not":
            j += 1
        if j < len(toks) and toks[j].low_bare == "materialized":
            j += 1
        # CTE 主体
        if j < len(toks) and toks[j].text == "(":
            close = _matching(toks, j, pairs)
            if close is None:
                return ""
            j = close + 1
        else:
            return ""
        # 逗号则继续下一个 CTE
        if j < len(toks) and toks[j].text == ",":
            j += 1
            continue
        break

    if j >= len(toks):
        return ""
    return toks[j].low_bare


def _has_toplevel_dml(tokens: Sequence[Token]) -> str | None:
    """返回出现在**括号外**的 DML/DDL 关键字（用于拦截 ``with x as (...) delete``）。

    ⚠️ **这个函数只查括号外，因此不足以单独作为危险语句判据。**
    它漏掉「DML 藏在括号内」的情形，而 CTE 恰恰就是括号内的子查询：

    ```sql
    with x as (insert into t values (1) returning *) select * from x
    ```

    这条语句：文本以 ``with`` 开头、外层确实是 SELECT、只有一个分号（在末尾），
    而 ``insert`` 在 ``as ( ... )`` 里面 —— **上面的三重检查全部通过**。
    本项目实测它曾被放行。见 :func:`has_dml_anywhere`。
    """
    depths = _depth_map(tokens)
    for idx, tok in enumerate(tokens):
        if depths[idx] != 0:
            continue
        if tok.kind != "word":
            continue
        low = tok.low
        if low in DANGEROUS_KEYWORDS:
            return low
    return None


#: sqlglot 里代表**写操作 / 非查询**的节点名。出现任一即判定为「非只读」。
#:
#: 用**节点类型**而不是关键字文本 —— 后者会把列名/别名里的同名词误判，
#: 也会漏掉解析器归一化后的写法（例如 ``DELETE`` 的变体），
#: 更看不见 ``with x as (insert ...) select`` 这种「合法外壳 + 非法内核」。
FORBIDDEN_NODE_NAMES: tuple[str, ...] = (
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable",
    "TruncateTable", "Grant", "Revoke", "Merge", "Copy", "Command",
    "Into", "Set", "SetItem", "Use", "Transaction", "Commit", "Rollback",
    "LoadData", "Cache", "Uncache", "Refresh", "Describe", "Pragma",
    "Lock", "Analyze", "Attach", "Detach", "Export", "Put", "Get",
)

#: 允许作为**根节点**的查询类型名。
#:
#: 这里用**白名单**而不是黑名单：sqlglot 解析不了的语句会退化成一个 ``Command``
#: 节点（实测 ``vacuum`` / ``show`` / ``do`` 都是这样），白名单能一并挡住，
#: 不必为每种语句单独写规则。``Set`` 同样落在白名单外，因此
#: ``set statement_timeout = ...`` 不会被当成查询放行。
ALLOWED_ROOT_NAMES: tuple[str, ...] = (
    "Select", "Union", "Intersect", "Except", "Subquery", "Values", "Table",
)

#: 词法**预筛**用的关键字子集。
#:
#: ⚠️ 这**不是判据**。判据是 AST（:data:`ALLOWED_ROOT_NAMES` 与
#: :data:`FORBIDDEN_NODE_NAMES`）。这里只挑「在本项目 schema 里不可能作为标识符
#: 出现、且一定处于语句动词位置」的词，免得把 ``comment`` / ``set`` / ``show``
#: 这类可能被用作列名的词误伤。
#: **预筛只能拒绝、不能批准**：它漏报的任何东西，后面的 AST 都会兜住。
_PREFILTER_KEYWORDS: tuple[str, ...] = (
    "insert", "update", "delete", "drop", "truncate", "alter", "create",
    "grant", "revoke", "copy", "merge", "vacuum", "reindex", "refresh",
    "cluster", "listen", "notify", "deallocate", "prepare",
)

#: sqlglot 的报错里带 ANSI 颜色码，不能直接进 ToolError / 日志。
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class SqlParseError(ValueError):
    """SQL 无法解析。**fail-closed**：解析不了就拒绝，不做"尽力而为"。"""


def _clean_parse_message(exc: BaseException, *, limit: int = 200) -> str:
    """把 sqlglot 的报错压成一行。

    它带 ANSI 颜色码与多行定位（``Line 1, Col: 20.`` 加下划线标注），
    直接塞进 ``ToolError.message`` 会让响应体里出现转义字符。
    """
    text = _ANSI_RE.sub("", str(exc)).strip()
    first = text.splitlines()[0] if text else type(exc).__name__
    return first[:limit] + ("…" if len(first) > limit else "")


def parse_statements(sql: str) -> list[exp.Expression]:
    """把 SQL 解析成 AST 列表（**多语句会返回多个元素**）。

    用 ``read='postgres'`` —— 目标是 PostgreSQL 18，方言必须显式指定；
    不指定会按默认方言解析，可能把某些写法"理解成"别的意思。

    Raises:
        SqlParseError: 解析失败、不是字符串、或结果为空。
    """
    if not isinstance(sql, str):
        raise SqlParseError(f"SQL 必须是字符串，收到 {type(sql).__name__}")
    text = sql.strip()
    if not text:
        raise SqlParseError("SQL 为空")
    try:
        parsed = sqlglot.parse(text, read="postgres")
    except SqlglotError as exc:  # ParseError / TokenizeError 的公共父类
        raise SqlParseError(_clean_parse_message(exc)) from exc
    except Exception as exc:  # noqa: BLE001 —— 解析器内部异常同样 fail-closed
        raise SqlParseError(f"{type(exc).__name__}: {_clean_parse_message(exc)}") from exc

    exprs = [e for e in parsed if e is not None]
    if not exprs:
        raise SqlParseError("SQL 解析结果为空（可能只有注释）")
    return exprs


def _parse_or_none(sql: str) -> list[exp.Expression] | None:
    """解析失败返回 ``None``（供"解析不了就不放行"的调用方使用）。"""
    try:
        return parse_statements(sql)
    except SqlParseError:
        return None


def statement_type(sql: str) -> str:
    """最外层语句的节点类型名（``Select`` / ``Insert`` / ``Command`` …）。"""
    exprs = _parse_or_none(sql)
    return type(exprs[0]).__name__ if exprs else ""


def dangerous_nodes(exprs: Sequence[exp.Expression]) -> list[str]:
    """在**整棵树**上找出被禁止的节点类型（去重，按发现顺序）。

    含 CTE 内层与子查询内层 —— 这正是文本匹配做不到的部分。
    """
    found: list[str] = []
    for tree in exprs:
        for node in tree.walk():
            name = type(node).__name__
            if name in FORBIDDEN_NODE_NAMES and name not in found:
                found.append(name)
    return found


def mask_literals(sql: str) -> str:
    """把字符串字面量、引号标识符、注释、美元引用替换成**等长空格**。

    位置保持不变，所以调用方仍可按偏移量定位原文。

    为什么需要它：``select 'a;b' from t`` 与 ``select 1 -- ; drop table t``
    都含分号却不是多语句。词法预筛必须先清掉这些内容，否则会给合法查询报假警。

    Raises:
        ValueError: 引号 / 注释 / 美元引用没有闭合（残缺输入一律拒绝）。
    """
    out = list(sql)
    n = len(sql)
    i = 0

    def blank(lo: int, hi: int) -> None:
        for k in range(lo, min(hi, n)):
            if out[k] != "\n":  # 保留换行，便于按行报错
                out[k] = " "

    while i < n:
        ch = sql[i]

        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            j = sql.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
            continue

        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            depth = 1
            j = i + 2
            while j < n and depth:  # PostgreSQL 的块注释可以嵌套
                if sql.startswith("/*", j):
                    depth += 1
                    j += 2
                elif sql.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            if depth:
                raise ValueError("块注释没有闭合")
            blank(i, j)
            i = j
            continue

        if ch in ("'", '"'):
            quote = ch
            j = i + 1
            while True:
                k = sql.find(quote, j)
                if k < 0:
                    raise ValueError("字符串字面量或引号标识符没有闭合")
                if k + 1 < n and sql[k + 1] == quote:  # '' / "" 转义
                    j = k + 2
                    continue
                j = k + 1
                break
            blank(i, j)
            i = j
            continue

        if ch == "$":
            m = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$", sql[i:])
            if m:
                tag = m.group(0)
                k = sql.find(tag, i + len(tag))
                if k < 0:
                    raise ValueError(f"美元引用 {tag} 没有闭合")
                j = k + len(tag)
                blank(i, j)
                i = j
                continue

        i += 1

    return "".join(out)


def has_dml_anywhere(sql: str) -> str | None:
    """用 **AST** 检查整棵树里是否有写操作节点，返回节点名或 ``None``。

    **为什么必须用 AST 而不是扫关键字**：

    - 扫**括号外**（:func:`_has_toplevel_dml`）会漏掉 CTE 内的 DML；
    - 扫**全部关键字**（不分括号）会把标识符误伤 —— 例如列名恰好叫 ``delete``；
    - **只有解析成语法树，才能区分「这里是语句」还是「这里是名字」**。

    这正是蓝图那句「**不只检查文本是否以 SELECT 开头**」的落点：
    危险的不是文本长什么样，而是**解析后这棵树里有没有写操作**。
    """
    exprs = _parse_or_none(sql)
    if exprs is None:
        # 解析不了就不放行 —— 返回 None 由调用方按「无法解析」处理（fail-closed）
        return None
    found = dangerous_nodes(exprs)
    return found[0] if found else None


def parse_single_select(sql: str) -> tuple[bool, str]:
    """只允许**单条 SELECT**。

    **判据是 AST**（三层），词法检查只是后面的快速预筛：

    1. :func:`parse_statements` 的**元素个数必须为 1** —— 多语句直接拒绝。
       ``select 1; drop table t`` 会解析成 2 个表达式，与"数了几个分号"无关；
    2. **根节点类型**必须在 :data:`ALLOWED_ROOT_NAMES` 内
       （``Insert`` / ``Update`` / ``Delete`` / ``Drop`` / ``Command`` … 全部出局）；
    3. **整棵树**不得含写操作节点 —— 这一步专门拦
       ``with x as (insert into t values (1) returning *) select * from x``：
       它语句数为 1、根节点是 ``Select``，**任何"首词/正则/分号计数"式检查都会放行**，
       而树里有一个 ``Insert``。

    后面的词法预筛（分号计数、括号配对）**只能拒绝、不能批准**：
    即便它全部通过，放行与否仍由上面的 AST 三层决定。

    Examples:
        >>> parse_single_select("select 1")
        (True, '')
        >>> parse_single_select("  select 1; drop table t")[0]
        False
        >>> parse_single_select(
        ...     "with x as (insert into t values (1) returning *) select * from x")[0]
        False
        >>> parse_single_select("with x as (select 1) select * from x")[0]   # 合法 CTE 不误伤
        True
    """
    # ---- 判据 1~3：AST ----
    exprs = _parse_or_none(sql)
    if exprs is None:
        try:
            parse_statements(sql)  # 取具体报错原因
        except SqlParseError as exc:
            return False, f"SQL 无法解析（已拒绝）：{exc}"
        return False, "SQL 无法解析（已拒绝）"

    if len(exprs) != 1:
        return False, (
            f"解析出 {len(exprs)} 条语句（本模块只接受单条 SELECT）；"
            "分号分隔的多语句一律拒绝"
        )

    root_name = type(exprs[0]).__name__
    if root_name not in ALLOWED_ROOT_NAMES:
        return False, (
            f"最外层语句类型是 {root_name}，不是查询"
            f"（允许：{', '.join(ALLOWED_ROOT_NAMES)}）；DDL/DML 与权限语句一律拒绝"
        )

    bad_nodes = dangerous_nodes(exprs)
    if bad_nodes:
        return False, (
            f"SQL 语法树中存在写操作节点 {bad_nodes}（可能在 WITH 子句或子查询内部）；"
            "本模块只接受**只读**的单条 SELECT"
        )

    # ---- 以下只是预筛：只能拒绝、不能批准 ----
    try:
        tokens = tokenize(sql)
    except ValueError as exc:
        return False, f"SQL 无法解析：{exc}"

    if not tokens:
        return False, "SQL 为空"

    # 分号：只允许出现在末尾（字符串/注释里的分号已经被分词器排除）
    semis = [i for i, t in enumerate(tokens) if t.kind == "punct" and t.text == ";"]
    if semis:
        last = semis[-1]
        if last != len(tokens) - 1:
            return False, (
                f"检测到第 {last + 1} 个词法单元后有分号，说明存在多条语句；"
                "本模块只接受单条 SELECT"
            )
        if len(semis) > 1:
            return False, f"检测到 {len(semis)} 个分号，存在多条语句；本模块只接受单条 SELECT"

    # 括号必须配对（AST 能解析出来的通常已经配对，这里是给残缺输入留的明确报错）
    pairs = _pairs(tokens)
    for idx, tok in enumerate(tokens):
        if tok.kind == "punct" and tok.text == "(" and pairs.get(idx, -1) < 0:
            return False, "SQL 中的括号没有配对"

    return True, ""


def clauses(
    tokens: Sequence[Token],
    *,
    start: int = 0,
    end: int | None = None,
) -> dict[str, tuple[int, int]]:
    """找出**顶层**子句的 token 区间，返回 ``{子句名: (起, 止)}``（止为开区间）。

    只识别当前深度（相对 ``start``）的子句；子查询内部不递归。
    """
    stop = len(tokens) if end is None else end
    depths = _depth_map(tokens)
    base = depths[start] if start < len(tokens) else 0
    found: dict[str, tuple[int, int]] = {}

    idx = start
    while idx < stop:
        tok = tokens[idx]
        if depths[idx] != base:
            idx += 1
            continue
        low = tok.low_bare
        if tok.kind == "word" and low in _CLAUSE_STARTERS:
            name = low
            if low == "group" and idx + 1 < stop and tokens[idx + 1].low_bare == "by":
                name = "group by"
            elif low == "order" and idx + 1 < stop and tokens[idx + 1].low_bare == "by":
                name = "order by"
            elif low == "union" and idx + 1 < stop and tokens[idx + 1].low_bare == "all":
                name = "union"
            # 找本子句结束位置
            j = idx + 1
            while j < stop:
                if depths[j] == base and tokens[j].kind == "word" and tokens[j].low_bare in _CLAUSE_STARTERS:
                    break
                j += 1
            found.setdefault(name, (idx, j))
            idx = j
            continue
        idx += 1
    return found


def find_top_level_clause(tokens: Sequence[Token], name: str) -> tuple[int, int] | None:
    """取一个顶层子句的区间；不存在返回 ``None``。"""
    return clauses(tokens).get(name)


def _top_level_limit(tokens: Sequence[Token]) -> int | None:
    """返回顶层 ``LIMIT`` 的行数；没有或无法解析返回 ``None``。"""
    span = find_top_level_clause(tokens, "limit")
    if span is None:
        return None
    start, end = span
    # limit all / limit 10 / limit :n
    for tok in tokens[start + 1 : end]:
        if tok.kind == "number":
            digits = re.match(r"\d+", tok.text)
            if digits:
                return int(digits.group(0))
        if tok.kind == "word" and tok.low == "all":
            return None
        if tok.kind == "word" and tok.low == "nulls":
            continue
    return None


def _existing_limit(expr: exp.Expression) -> tuple[int | None, bool]:
    """返回 ``(已有 LIMIT 的行数, 是否存在 limit 子句)``。

    区分这两件事很重要：``limit all`` / ``limit :n`` **有** limit 子句但**看不出行数**，
    若按"没有 LIMIT"再追加一句，就会拼出 ``... limit all limit 1000`` 这种语法错误。
    """
    node = expr.args.get("limit")
    if node is None:
        return None, False
    target = node.expression if isinstance(node, exp.Limit) else None
    if isinstance(target, exp.Literal) and target.is_int:
        try:
            return int(target.this), True
        except (TypeError, ValueError):
            return None, True
    return None, True  # limit all / limit :param：有子句，但行数未知


def enforce_limits(
    sql: str,
    *,
    max_rows: int = MAX_ROWS_DEFAULT,
    timeout_ms: int = STATEMENT_TIMEOUT_MS_DEFAULT,
) -> str:
    """给 SQL 加上 ``LIMIT``；语句超时由执行层用 ``SET LOCAL statement_timeout`` 施加。

    - 没有 LIMIT → 追加 ``limit <max_rows>``；
    - 已有 LIMIT 且**更严格**（``<= max_rows``）→ **原样保留，一个字符都不改**
      （审计与复现要求 SQL 文本可追溯，不做无谓重排）；
    - 已有 LIMIT 但更宽松 / ``LIMIT ALL`` / ``LIMIT :param`` → 收紧成 ``max_rows``；
    - 末尾分号会被去掉（分号由执行层按需要处理）。

    ``timeout_ms`` 只做参数校验（正数），真正的超时设置在 :mod:`dsa.nl2sql.engine`
    里用 ``SET LOCAL statement_timeout`` 施加 —— 那是事务级设置，
    必须与查询在**同一个连接、同一个事务**里，写进 SQL 文本反而不生效。

    Raises:
        ValueError: ``max_rows`` 或 ``timeout_ms`` 不是正整数，或 SQL 为空。
    """
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0:
        raise ValueError(f"max_rows 必须是正整数，收到 {max_rows!r}")
    if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms <= 0:
        raise ValueError(f"timeout_ms 必须是正整数，收到 {timeout_ms!r}")

    body = remove_trailing_semicolon(sql)
    if not body:
        raise ValueError("SQL 为空，无法限制行数")

    # ---- 首选 AST：能区分「有 LIMIT 子句」与「LIMIT 行数是多少」----
    expr: exp.Expression | None = None
    try:
        expr = sqlglot.parse_one(body, read="postgres")
    except Exception:  # noqa: BLE001 —— 解析不了就走下面的词法兜底；反正闸门会拒绝它
        expr = None

    if expr is not None:
        existing, has_limit = _existing_limit(expr)
        if has_limit and existing is not None and existing <= max_rows:
            return body
        if not has_limit:
            return f"{body} limit {max_rows}"
        try:
            return expr.limit(max_rows).sql("postgres")
        except Exception:  # noqa: BLE001 —— 回写失败就退回文本追加，绝不返回未受限的 SQL
            return f"{body} limit {max_rows}"

    # ---- 词法兜底（AST 不可用时）----
    try:
        tokens = tokenize(body)
    except ValueError:
        return body

    existing = _top_level_limit(tokens)
    if existing is not None and existing <= max_rows:
        return body  # 已有更严格（或同等）的 LIMIT，保留

    if existing is not None:
        # 收紧：把顶层 limit 子句整段替换掉
        span = find_top_level_clause(tokens, "limit")
        assert span is not None
        start, end = span
        head = body[: tokens[start].start].rstrip()
        tail_start = tokens[end].start if end < len(tokens) else len(body)
        tail = body[tail_start:].strip()
        rebuilt = f"{head} limit {max_rows}" + (f" {tail}" if tail else "")
        return rebuilt

    return f"{body} limit {max_rows}"


# ============================================================================
#  表引用抽取
# ============================================================================


def _cte_names(tokens: Sequence[Token]) -> set[str]:
    """收集 ``WITH`` 里定义的 CTE 名（它们出现在 FROM 位置但不是真表）。"""
    names: set[str] = set()
    if not tokens or tokens[0].low_bare != "with":
        return names
    pairs = _pairs(tokens)
    j = 1
    if j < len(tokens) and tokens[j].low_bare == "recursive":
        j += 1
    while j < len(tokens):
        if tokens[j].kind not in ("word", "quoted"):
            break
        names.add(tokens[j].low_bare)
        j += 1
        if j < len(tokens) and tokens[j].text == "(":
            close = _matching(tokens, j, pairs)
            if close is None:
                break
            j = close + 1
        if j < len(tokens) and tokens[j].low_bare == "as":
            j += 1
        else:
            break
        if j < len(tokens) and tokens[j].low_bare == "not":
            j += 1
        if j < len(tokens) and tokens[j].low_bare == "materialized":
            j += 1
        if j < len(tokens) and tokens[j].text == "(":
            close = _matching(tokens, j, pairs)
            if close is None:
                break
            j = close + 1
        else:
            break
        if j < len(tokens) and tokens[j].text == ",":
            j += 1
            continue
        break
    return names


def _read_relation(tokens: Sequence[Token], idx: int, pairs: dict[int, int]) -> tuple[str | None, int]:
    """从 ``tokens[idx]`` 读一个关系名（支持 ``schema.table``）。返回 (名字, 下一个下标)。

    读到 ``(`` 说明是子查询或表函数 —— 返回 ``(None, 跳过后的下标)``。
    """
    parts: list[str] = []
    j = idx
    while j < len(tokens):
        tok = tokens[j]
        if tok.kind in ("word", "quoted"):
            if parts and tokens[j - 1].text != ".":
                break
            parts.append(tok.bare)
            j += 1
            if j < len(tokens) and tokens[j].text == ".":
                j += 1
                continue
            break
        break
    if j < len(tokens) and tokens[j].text == "(":
        close = _matching(tokens, j, pairs)
        return None, (len(tokens) if close is None else close + 1)
    if not parts:
        return None, idx + 1
    return ".".join(parts), j


def _tables_from_tokens(
    tokens: Sequence[Token],
    *,
    include_cte_names: bool = False,
) -> list[str]:
    """（词法路径）从 token 流里抽表引用。

    覆盖：``FROM a, b``、各类 ``JOIN``、``FROM (子查询)`` 之后的表、
    ``FROM ONLY t``、以及 ``INSERT/UPDATE/DELETE`` 的目标表。
    刻意保守：**宁可多报一个对象（然后被白名单拒绝），也不要漏报** ——
    漏报等于放行了越界查询。
    """
    if not tokens:
        return []
    pairs = _pairs(tokens)
    depths = _depth_map(tokens)
    base = depths[0] if depths else 0
    cte_names = _cte_names(tokens)
    found: list[str] = []

    def add(name: str | None) -> None:
        if not name:
            return
        bare = sc.normalize_object_name(name)
        if not include_cte_names and bare in cte_names:
            return
        if name not in found:
            found.append(name)

    words = {"from", "join", "into", "update", "table"}
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        # 只处理顶层（子查询会在下面的循环里被单独走到 —— 我们只跳过被 ( ) 包住的
        # **子查询主体**，因为 FROM (...) 之后那个别名的处理需要它）
        low = tok.low_bare
        if tok.kind == "word" and low in words:
            if low == "from":
                j = idx + 1
                expect_relation = True
                while j < len(tokens):
                    t = tokens[j]
                    if depths[j] == base and t.kind == "word" and t.low_bare == "select" :
                        break
                    if t.text == "(":
                        close = _matching(tokens, j, pairs)
                        j = len(tokens) if close is None else close + 1
                        expect_relation = False
                        continue
                    if depths[j] == base and t.kind == "word" and t.low_bare in (_FROM_STOP | {"join"}):
                        break
                    if depths[j] == base and t.text == ",":
                        expect_relation = True
                        j += 1
                        continue
                    if expect_relation:
                        if t.kind == "word" and t.low_bare in _TABLE_POSITION_IGNORE:
                            j += 1
                            continue
                        name, nxt = _read_relation(tokens, j, pairs)
                        if name is not None:
                            add(name)
                        expect_relation = False
                        j = nxt
                        continue
                    j += 1
                idx = j
                continue

            if low == "join":
                j = idx + 1
                if j < len(tokens) and tokens[j].kind == "word" and tokens[j].low_bare in _TABLE_POSITION_IGNORE:
                    j += 1
                name, nxt = _read_relation(tokens, j, pairs)
                add(name)
                idx = nxt
                continue

            if low in ("into", "update", "table"):
                j = idx + 1
                while j < len(tokens) and tokens[j].kind == "word" and tokens[j].low in ("only",):
                    j += 1
                name, nxt = _read_relation(tokens, j, pairs)
                add(name)
                idx = nxt
                continue

        # 进入子查询：不做特殊处理（token 全量遍历，子查询里的 from 也会被看到）
        idx += 1

    return found


def _as_exprs(source: Union[str, exp.Expression, Sequence[exp.Expression]]) -> list[exp.Expression]:
    """把 ``str`` / 单个 AST / AST 列表统一成 AST 列表。"""
    if isinstance(source, str):
        return parse_statements(source)
    if isinstance(source, exp.Expression):
        return [source]
    return [e for e in source if isinstance(e, exp.Expression)]


def _ast_tables(
    exprs: Sequence[exp.Expression],
    *,
    include_cte_names: bool = False,
) -> list[str]:
    """（AST 路径）抽出引用到的对象名，**保留 schema 前缀**。

    保留 schema 是必须的：``information_schema.tables`` 与 ``pg_catalog.pg_shadow``
    的裸表名（``tables`` / ``pg_shadow``）不足以判断「是不是系统对象」。

    CTE 名字默认**不作为表引用**返回：``with x as (select 1) select * from x``
    里的 ``x`` 是查询内部定义的别名，不是数据库对象，拿它去撞白名单会误伤合法 SQL。
    CTE **内部**引用的真表（``with x as (select * from pg_shadow) ...``）仍会被抓到。
    """
    cte_names = {
        cte.alias_or_name.lower()
        for e in exprs
        for cte in e.find_all(exp.CTE)
        if cte.alias_or_name
    }
    found: list[str] = []
    for e in exprs:
        for table in e.find_all(exp.Table):
            parts = [p for p in (table.catalog, table.db, table.name) if p]
            name = ".".join(parts)
            if not name:
                continue
            if not include_cte_names and name.lower() in cte_names:
                continue
            if name not in found:
                found.append(name)
    return found


def referenced_tables(
    source: Union[str, exp.Expression, Sequence[exp.Expression], Sequence[Token]],
    *,
    include_cte_names: bool = False,
) -> list[str]:
    """抽出 SQL 引用到的对象名（按出现顺序去重）。

    支持两种输入：

    - **SQL 文本 / AST**（推荐，走 AST 路径 :func:`_ast_tables`）——
      ``referenced_tables("select * from customers")``；
    - 词法 token 列表（旧接口，走 :func:`_tables_from_tokens`），
      供已经分好词的调用方复用，避免重复分词。
    """
    if isinstance(source, (str, exp.Expression)):
        return _ast_tables(_as_exprs(source), include_cte_names=include_cte_names)
    seq = list(source)
    if seq and isinstance(seq[0], Token):
        return _tables_from_tokens(seq, include_cte_names=include_cte_names)
    return _ast_tables(_as_exprs(seq), include_cte_names=include_cte_names)


# ============================================================================
#  危险语句检查
# ============================================================================


def check_dangerous(
    sql: str,
    *,
    tokens: Sequence[Token] | None = None,
    allow_objects: Iterable[str] | None = None,
) -> tuple[bool, str]:
    """拒绝：多语句、DDL/DML、不允许的函数与对象。

    **判定顺序：先 AST（唯一判据），后词法预筛（只能拒绝、不能批准）。**

    AST 层（判据）::

        1. 语句数 != 1        → 拒绝（``select 1; drop table t`` 解析出 2 条）
        2. 根节点不在白名单    → 拒绝（insert / update / delete / drop / truncate /
                                alter / create / grant / copy / merge / vacuum /
                                show / do / set … 全部落在白名单外）
        3. 整棵树含写操作节点  → 拒绝（**含 WITH 子句与子查询内部**）
        4. 函数节点名在黑名单  → 拒绝 pg_read_file / pg_sleep / lo_import / dblink …
        5. 引用的对象越界      → 拒绝（``pg_*`` / ``information_schema`` / 未登记表）

    词法层（预筛，跑在掩码后的文本上，字符串与注释已清空）::

        · 分号计数（``'a;b'`` 与 ``-- ; drop`` 不会误伤）
        · ``into outfile`` / ``dumpfile``、``copy ... program``
        · 危险函数名后面直接跟 ``(`` 的写法

    为什么函数检查做两遍：AST 能识别 ``pg_catalog.pg_sleep(1)`` 这类限定写法，
    词法则能在解析器"看不懂"某些写法时兜底。两遍都只可能拒绝。

    Args:
        sql: 待检查的 SQL。
        tokens: **仅为兼容旧调用方保留**。判定不看它；提供时用于词法预筛。
        allow_objects: 允许引用的对象名；缺省用
            :func:`dsa.nl2sql.scope.allowed_object_names`。

    Returns:
        ``(True, "")`` 表示通过；否则 ``(False, 原因)``。
    """
    # ==================== 判据：AST ====================
    exprs = _parse_or_none(sql)
    if exprs is None:
        try:
            parse_statements(sql)  # 取具体报错原因
        except SqlParseError as exc:
            return False, f"SQL 无法解析（已拒绝）：{exc}"
        return False, "SQL 无法解析（已拒绝）"

    if len(exprs) != 1:
        return False, (
            f"解析出 {len(exprs)} 条语句；只允许单条 SELECT（分号分隔的多语句一律拒绝）"
        )

    root_name = type(exprs[0]).__name__
    if root_name not in ALLOWED_ROOT_NAMES:
        return False, (
            f"最外层语句类型是 {root_name}，不是查询"
            f"（允许：{', '.join(ALLOWED_ROOT_NAMES)}）；DDL/DML 与权限语句一律拒绝"
        )

    bad_nodes = dangerous_nodes(exprs)
    if bad_nodes:
        return False, (
            f"语句树中存在被禁止的节点 {bad_nodes}：即使最外层是 SELECT，"
            "WITH 子句或子查询里也不得出现 DDL/DML"
            "（例：with x as (insert into t values (1) returning *) select * from x）"
        )

    # ---- 危险函数（AST 级）----
    for tree in exprs:
        for node in tree.find_all(exp.Func):
            name = (
                (node.name or "").lower()
                if isinstance(node, exp.Anonymous)
                else (node.sql_name() or "").lower()
            )
            if name in DANGEROUS_FUNCTIONS:
                return False, (
                    f"检测到被禁止的函数 {name} —— 文件/网络/延时类函数不在允许范围内"
                )

    # ---- 对象范围（AST 级；CTE 别名不算对象，避免误伤合法 WITH）----
    allowed = set(allow_objects) if allow_objects is not None else set(sc.allowed_object_names())
    for name in _ast_tables(exprs):
        bare = sc.normalize_object_name(name)
        if bare and bare in allowed:
            continue  # 白名单优先，避免名字巧合被系统表规则误伤
        if sc.is_system_object(name) or sc.is_system_object(bare):
            return False, (
                f"检测到系统表/系统模式 {name} —— pg_* 与 information_schema 不在查询范围内"
            )
        return False, (
            f"检测到范围外的对象 {name}。允许的对象：" + ", ".join(sorted(allowed))
        )

    # ==================== 预筛：词法（只能拒绝） ====================
    try:
        toks = list(tokens) if tokens is not None else tokenize(sql)
    except ValueError as exc:
        return False, f"SQL 无法解析：{exc}"

    if not toks:
        return False, "SQL 为空"

    semis = [i for i, t in enumerate(toks) if t.kind == "punct" and t.text == ";"]
    for pos in semis:
        if pos != len(toks) - 1:
            return False, (
                f"检测到分号分隔的多条语句（第 {pos + 1} 个词法单元）；只允许单条 SELECT"
            )
    if len(semis) > 1:
        return False, f"检测到 {len(semis)} 个分号，存在多条语句；只允许单条 SELECT"

    for tok in toks:
        if tok.kind != "word":
            continue
        low = tok.low
        if low in _PREFILTER_KEYWORDS:
            return False, f"检测到被禁止的语句关键字 {tok.text.upper()}"
        if low in ("outfile", "dumpfile"):
            return False, f"检测到被禁止的导出写法 {tok.text.upper()}（会往文件系统写数据）"
        if low == "into":
            return False, "检测到 INTO（select ... into 会建表；into outfile/dumpfile 会写文件）"

    for idx, tok in enumerate(toks):
        if tok.kind not in ("word", "quoted"):
            continue
        if tok.low_bare not in DANGEROUS_FUNCTIONS:
            continue
        nxt = toks[idx + 1] if idx + 1 < len(toks) else None
        if nxt is not None and nxt.text == "(":
            return False, (
                f"检测到被禁止的函数 {tok.text} —— 文件/网络/延时类函数不在允许范围内"
            )

    return True, ""


# ============================================================================
#  语义检查
# ============================================================================


@dataclass(frozen=True, slots=True)
class Issue:
    """一条语义问题。``code`` 用于测试断言与下游路由，``message`` 用于修复提示词。"""

    code: str
    message: str
    severity: str = "error"
    evidence: str = ""

    @property
    def blocking(self) -> bool:
        """是否阻断级（不得把结果当作正确结果返回）。"""
        return self.severity in ("error", "blocking") or self.code in BLOCKING_CODES

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "evidence": self.evidence,
        }


def _is_column(tok: Token, names: frozenset[str]) -> bool:
    """该 token 是否是目标列名（支持 ``表.列``，并跳过函数名位置）。"""
    return tok.kind in ("word", "quoted") and tok.low_bare in names


def _iter_call_spans(
    tokens: Sequence[Token], names: frozenset[str]
) -> Iterator[tuple[str, int, int]]:
    """遍历 ``name(...)`` 形式的调用，产出 ``(小写函数名, 开括号下标, 闭括号下标)``。"""
    pairs = _pairs(tokens)
    for idx, tok in enumerate(tokens):
        if tok.kind not in ("word", "quoted"):
            continue
        if tok.low_bare not in names:
            continue
        j = idx + 1
        while j < len(tokens) and tokens[j].kind == "punct" and tokens[j].text == "(":
            close = _matching(tokens, j, pairs)
            if close is None:
                break
            yield tok.low_bare, j, close
            break


def _filter_spans(tokens: Sequence[Token]) -> list[tuple[int, int]]:
    """``FILTER (WHERE ...)`` 的括号区间。

    这是聚合的**行内过滤**：它只改变分子，不改变分组，因此是分母红线的正确写法。
    这些区间里的干预后字段**不算违规**。
    """
    pairs = _pairs(tokens)
    spans: list[tuple[int, int]] = []
    for idx, tok in enumerate(tokens):
        if tok.kind != "word" or tok.low not in _FILTER_WORDS:
            continue
        if idx + 1 >= len(tokens) or tokens[idx + 1].text != "(":
            continue
        close = _matching(tokens, idx + 1, pairs)
        if close is None:
            continue
        inner = tokens[idx + 2 : close]
        if inner and inner[0].kind == "word" and inner[0].low == "where":
            spans.append((idx + 1, close))
    return spans


def _case_spans(tokens: Sequence[Token]) -> list[tuple[int, int]]:
    """``CASE ... END`` 的区间（同样只影响聚合值，不影响分组）。"""
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    for idx, tok in enumerate(tokens):
        if tok.kind != "word":
            continue
        if tok.low == "case":
            stack.append(idx)
        elif tok.low == "end" and stack:
            spans.append((stack.pop(), idx))
    return spans


def _span_hits(spans: Sequence[tuple[int, int]], lo: int, hi: int) -> bool:
    return any(lo >= a and hi <= b for a, b in spans)


def _collect_columns(
    tokens: Sequence[Token],
    *,
    lo: int,
    hi: int,
    names: frozenset[str],
    skip_spans: Sequence[tuple[int, int]] = (),
) -> list[Token]:
    """在 ``[lo, hi)`` 内收集目标列名 token（跳过 ``skip_spans`` 覆盖的位置）。"""
    hits: list[Token] = []
    for idx in range(lo, min(hi, len(tokens))):
        tok = tokens[idx]
        if not _is_column(tok, names):
            continue
        if skip_spans and _span_hits(skip_spans, idx, idx):
            continue
        # 跳过函数名位置：`sum` 被当成列名的情况不存在（列名不会紧跟 "("）
        if idx + 1 < len(tokens) and tokens[idx + 1].text == "(" and tok.kind == "word":
            continue
        hits.append(tok)
    return hits


def _post_treatment_violations(tokens: Sequence[Token]) -> list[Token]:
    """找出**会改变分组/人群**的干预后字段引用（分母红线的判定核心）。

    只检查顶层 ``WHERE`` 与 ``JOIN ... ON``；``FILTER (WHERE ...)`` 与
    ``CASE WHEN ...`` 内部是正确写法，跳过。
    """
    if not tokens:
        return []
    depths = _depth_map(tokens)
    base = depths[0]
    skips = _filter_spans(tokens) + _case_spans(tokens)
    hits: list[Token] = []

    # ---- 顶层 where ----
    span = find_top_level_clause(tokens, "where")
    if span is not None:
        start, end = span
        hits.extend(
            _collect_columns(tokens, lo=start + 1, hi=end,
                             names=POST_TREATMENT_COLUMNS, skip_spans=skips)
        )

    # ---- join ... on ----
    for idx, tok in enumerate(tokens):
        if tok.kind == "word" and tok.low == "on" and depths[idx] == base:
            j = idx + 1
            while j < len(tokens):
                if depths[j] == base and tokens[j].kind == "word" and (
                    tokens[j].low_bare in _FROM_STOP or tokens[j].low_bare in _JOIN_WORDS
                ):
                    break
                j += 1
            hits.extend(
                _collect_columns(tokens, lo=idx + 1, hi=j,
                                 names=POST_TREATMENT_COLUMNS, skip_spans=skips)
            )

    # ---- group by 里出现干预后字段也是错的（等于按结果分组）----
    gspan = find_top_level_clause(tokens, "group by")
    if gspan is not None:
        start, end = gspan
        hits.extend(
            _collect_columns(tokens, lo=start + 1, hi=end,
                             names=POST_TREATMENT_COLUMNS, skip_spans=skips)
        )

    return hits


def post_treatment_columns(sql: str, *, tokens: Sequence[Token] | None = None) -> list[str]:
    """返回 SQL 中**以违规方式**引用的干预后字段名（去重，按出现顺序）。"""
    toks = list(tokens) if tokens is not None else tokenize(sql)
    out: list[str] = []
    for tok in _post_treatment_violations(toks):
        if tok.low_bare not in out:
            out.append(tok.low_bare)
    return out


def _amount_not_deduped(tokens: Sequence[Token]) -> list[str]:
    """找出「直接对明细表聚合交易级字段」的字段名。"""
    if not tokens:
        return []
    tables = {sc.normalize_object_name(t) for t in referenced_tables(tokens)}
    if tables & CONVERGED_OBJECTS:
        # 已经走了收敛视图/收敛后的特征表：同一 SQL 里即使还引用明细表
        # （例如为了统计明细行数），金额聚合也在收敛入口上，不判违规。
        converged_amount = any(
            sc.normalize_object_name(t) in CONVERGED_OBJECTS for t in referenced_tables(tokens)
        )
        if converged_amount and not (
            {"rh_purchases"} <= tables and "rh_transactions" not in tables
        ):
            return []
    if "rh_purchases" not in tables:
        return []

    bad: list[str] = []
    for col in TRANSACTION_LEVEL_COLUMNS:
        for _name, open_idx, close_idx in _iter_call_spans(tokens, _AGGREGATES):
            inner = tokens[open_idx + 1 : close_idx]
            has_distinct = any(t.kind == "word" and t.low == "distinct" for t in inner)
            if has_distinct:
                continue  # 显式 distinct 是另一种（语义不同但非"忘记收敛"）写法
            for tok in inner:
                if tok.kind in ("word", "quoted") and tok.low_bare == col:
                    if col not in bad:
                        bad.append(col)
    return bad


def semantic_blockers(issues: Sequence[dict]) -> list[dict]:
    """从 :func:`check_semantics` 的结果里挑出**必须拒绝**的硬错误。

    `check_semantics` 返回的 issue 有轻重之分：

    - **硬错误（阻断）**：违反分母红线、金额未去重、数据集混用 ——
      它们会让结果**静默算错**（不报错、不为空、量级看似合理），必须拒绝；
    - **软提醒**：关联扩张/粒度提示、唯一客户数、聚合缺 group by ——
      这些要看上下文（客户级表上 ``count(client_id)`` 就是客户数），
      静态判不出来，交给人或交给执行后的结果核对。

    判据是 issue 自带的 ``severity``（缺省 ``error`` 视为阻断），
    再用 :data:`BLOCKING_CODES` 兜一层底 ——
    **不要在调用点上硬编码编码字符串**：那样 ``check_semantics`` 一旦新增
    一类硬错误，这里就会默默漏掉（数据集混用就差点这样漏过去）。

    ⚠️ **这个函数是补齐的**：`engine.py` 已三处调用 ``G.semantic_blockers(...)``，
    但本模块当时**没有这个函数** —— 与 ``ToolResult.ok_result`` 是同一类缺陷：
    **调用方与被调用方的契约没对齐**，且只有走到那条路径才会 ``AttributeError``。
    本项目这一族问题已出现四次，见 docs/日志.md L037。
    """
    out: list[dict] = []
    for issue in issues or []:
        code = issue.get("code")
        severity = issue.get("severity", "error")
        if severity == "warning" and code not in BLOCKING_CODES:
            continue
        out.append(issue)
    return out


def check_semantics(
    sql: str,
    *,
    tokens: Sequence[Token] | None = None,
) -> list[dict]:
    """语义检查。**返回 issue 列表**，每条含 ``code`` / ``message`` / ``severity``。

    检查项（每一项都对应一个曾经真实发生过的错法）：

    1. ``SEM_DENOM_001`` **分母红线**（AGENTS.md §2.2）：
       不得按干预后字段（``visit`` / ``visited`` / ``converted`` / ``conversion`` /
       ``spend``）过滤分母 —— 即出现在顶层 ``WHERE`` / ``JOIN ... ON`` /
       ``GROUP BY``。正确写法是 ``count(*) filter (where o.visited)``，
       该写法在本检查里被显式放行。
    2. ``SEM_JOIN_001`` **关联扩张**：客户级 join 到交易/明细级（``rh_purchases``、
       ``rh_transactions``）而没有 ``GROUP BY`` 时，行数被放大，客户数/均值全部失真。
    3. ``SEM_COUNT_001`` **唯一客户数**：``count(client_id)`` 这类写法在明细级表上
       统计的是**行数**不是客户数，必须 ``count(distinct client_id)``。
    4. ``SEM_DATASET_001`` / ``SEM_DATASET_002`` **数据集不混用**（§2.5）：
       Hillstrom 表与 ``rh_*`` 表不得出现在同一条 SQL 里，更不得 JOIN。
    5. ``SEM_AMOUNT_001`` **金额去重**：``sum(purchase_sum)`` 直接落在
       ``rh_purchases``（交易级字段冗余在明细行上）上会虚高约 5.81 ~ 11.41 倍且不报错
       （docs/日志.md L026）。必须先按 ``transaction_id`` 收敛
       （走 ``rh_transactions`` 视图或 ``group by (client_id, transaction_id)``）。
    6. ``SEM_AGG_001``：出现聚合函数却没有任何 ``GROUP BY``，
       同时 select 列表里还有非聚合列 —— 语法合法但几乎总是写错了。

    每条 issue 都是 :class:`Issue` 的 ``to_dict()`` 结果（纯 dict，便于 JSON 化与测试）。
    """
    try:
        toks = list(tokens) if tokens is not None else tokenize(sql)
    except ValueError as exc:
        return [Issue("SEM_PARSE_000", f"SQL 无法解析：{exc}").to_dict()]

    if not toks:
        return [Issue("SEM_PARSE_000", "SQL 为空").to_dict()]

    issues: list[Issue] = []
    tables = referenced_tables(toks)
    bare_tables = [sc.normalize_object_name(t) for t in tables]
    table_set = {t for t in bare_tables if t}

    # ---- 1. 分母红线 ----
    offenders = _post_treatment_violations(toks)
    if offenders:
        names = []
        for tok in offenders:
            if tok.low_bare not in names:
                names.append(tok.low_bare)
        issues.append(
            Issue(
                code=CheckCode.DENOMINATOR_POST_TREATMENT,
                message=(
                    "分母口径红线（AGENTS.md §2.2）：SQL 在 where/join on/group by 里引用了"
                    f"干预后字段 {names}。按干预后行为过滤分母 = 换了一个人群，破坏随机化。"
                    "正确写法：分子用 count(*) filter (where o.visited)，分母仍用 count(*)"
                ),
                evidence=", ".join(names),
            )
        )

    # ---- 2. 关联扩张 ----
    # 只有在**真的发生关联**（有 JOIN 或多表）时才谈"扩张"。
    # 单表读明细对象不是扩张问题，而是**粒度**问题 —— 降级为提示，
    # 否则 `select count(*) from rh_purchases` 会被扣上"关联扩张"的错名。
    expanding = [t for t in bare_tables if sc.is_multi_row_per_customer(t)]
    has_group_by = find_top_level_clause(toks, "group by") is not None
    has_join = (
        any(t.kind == "word" and t.low == "join" for t in toks)
        or len({t for t in bare_tables if t}) > 1
    )
    if expanding and has_join and not has_group_by:
        issues.append(
            Issue(
                code=CheckCode.JOIN_EXPANSION,
                message=(
                    f"关联扩张：SQL 关联了交易/明细级对象 {expanding}，但没有 group by。"
                    "一名客户对应多行，客户数、比率、均值都会失真到无法解释"
                ),
                evidence=", ".join(expanding),
            )
        )
    elif expanding:
        issues.append(
            Issue(
                code=CheckCode.JOIN_EXPANSION,
                message=(
                    f"涉及交易/明细级对象 {expanding}（一行是一笔交易或一个商品，**不是一名客户**）："
                    "请确认 group by 的粒度等于最终答案的粒度；需要客户数时必须写 "
                    "count(distinct client_id)，否则拿到的是明细行数"
                ),
                severity="warning",
                evidence=", ".join(expanding),
            )
        )

    # ---- 3. 唯一客户数 ----
    # ⚠️ 这条是**提示级**，不是阻断级：在客户级表上 ``count(client_id)`` 就是客户数
    # （``rh_clients`` 一行一名客户），只有明细级表上才是行数。
    # 静态判不出表粒度时宁可提示、不可错杀 —— 真正的兜底是执行后的结果核对。
    for _name, open_idx, close_idx in _iter_call_spans(toks, frozenset({"count"})):
        inner = toks[open_idx + 1 : close_idx]
        has_distinct = any(t.kind == "word" and t.low == "distinct" for t in inner)
        if has_distinct:
            continue
        for tok in inner:
            if tok.kind in ("word", "quoted") and tok.low_bare in _CUSTOMER_ID_COLUMNS:
                issues.append(
                    Issue(
                        code=CheckCode.CUSTOMER_COUNT_NOT_DISTINCT,
                        message=(
                            f"唯一客户数：count({tok.bare}) 统计的是行数而不是客户数。"
                            "在明细级表上（或其关联结果上）必须写 "
                            f"count(distinct {tok.bare})"
                        ),
                        severity="warning",
                        evidence=tok.bare,
                    )
                )
                break

    # ---- 4. 数据集不混用 ----
    kinds = {sc.dataset_of(t) for t in bare_tables if sc.dataset_of(t)}
    real_kinds = {k for k in kinds if k in ("hillstrom", "retailhero")}
    if len(real_kinds) > 1:
        hill = sorted(t for t in bare_tables if sc.dataset_of(t) == "hillstrom")
        rh = sorted(t for t in bare_tables if sc.dataset_of(t) == "retailhero")
        joined = len(tables) > 1 and any(
            t.kind == "word" and t.low == "join" for t in toks
        )
        issues.append(
            Issue(
                code=CheckCode.CROSS_DATASET_JOIN if joined else CheckCode.DATASET_MIXED,
                message=(
                    "数据集不混用（AGENTS.md §2.5）：同一条 SQL 里同时出现 Hillstrom 表 "
                    f"{hill} 与 RetailHero 表 {rh}。两个数据集的客户标识各自独立，"
                    "禁止 JOIN、拼接或互为对照"
                ),
                evidence=f"hillstrom={hill}; retailhero={rh}",
            )
        )

    # ---- 5. 金额去重 ----
    bad_amounts = _amount_not_deduped(toks)
    if bad_amounts:
        issues.append(
            Issue(
                code=CheckCode.AMOUNT_NOT_DEDUPED,
                message=(
                    f"金额未按交易去重：{'、'.join(bad_amounts)} 是**交易级**字段，"
                    "被冗余在 rh_purchases 的每一行商品明细上；直接聚合会虚高约 5.81 倍"
                    "（实测到 11.41 倍）且**不报错**（docs/日志.md L026）。"
                    "正确写法：from rh_transactions（已按 transaction_id 收敛），"
                    "或先 group by (client_id, transaction_id) 取 max 再聚合"
                ),
                evidence=", ".join(bad_amounts),
            )
        )

    # ---- 6. 聚合但无 group by ----
    select_span = _select_list_span(toks)
    if select_span is not None and has_group_by is False:
        lo, hi = select_span
        inner = toks[lo:hi]
        has_agg = any(
            t.kind == "word" and t.low_bare in _AGGREGATES and i + 1 < len(toks) and toks[i + 1].text == "("
            for i, t in enumerate(inner)
        )
        if has_agg:
            # 只有在同时出现「非聚合列」时才算问题；纯聚合是合法的整表汇总
            plain = [
                t.bare
                for i, t in enumerate(inner)
                if t.kind in ("word", "quoted")
                and t.low_bare not in _AGGREGATES
                and not (i + 1 < len(inner) and inner[i + 1].text == "(")
                and t.low_bare not in ("as", "distinct", "cast", "null", "true", "false")
                and not t.low_bare.startswith("o.")
            ]
            if plain:
                issues.append(
                    Issue(
                        code=CheckCode.AGGREGATE_WITHOUT_GROUP_BY,
                        message=(
                            f"select 列表里同时有聚合函数和普通列 {plain[:5]}，但没有 group by；"
                            "这在 PostgreSQL 里会直接报错，说明查询结构没写对"
                        ),
                        severity="warning",
                        evidence=", ".join(plain[:5]),
                    )
                )

    return [i.to_dict() for i in issues]


def _select_list_span(tokens: Sequence[Token]) -> tuple[int, int] | None:
    """``select`` 与 ``from`` 之间的区间（顶层）。"""
    depths = _depth_map(tokens)
    base = depths[0] if depths else 0
    sel: int | None = None
    for idx, tok in enumerate(tokens):
        if depths[idx] != base or tok.kind != "word":
            continue
        if tok.low == "select" and sel is None:
            sel = idx
            continue
        if sel is not None and tok.low == "from":
            return (sel + 1, idx)
        if sel is not None and tok.low in ("union", "intersect", "except"):
            return (sel + 1, idx)
    if sel is not None:
        return (sel + 1, len(tokens))
    return None
