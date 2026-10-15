"""工具层的输入 / 输出 / 错误结构（阶段六）。

依据
----
- `docs/08-工具协议.md` —— 统一 `ToolResult`、错误码表、五类错误可分辨、结果引用结构
- `docs/05-报告结构与异常分类.md` §3.3 —— 异常码注册表：
  `E_TOOL_001`（工具参数错误）、`E_PLAN_001`（业务条件不足）、`E_CAP_001`（能力范围）、
  `E_DB_001`（数据库连接/权限）、`E_DB_003`（执行超时）、`E_TOOL_999`（未知兜底）
- `AGENTS.md` §2 —— 数值由程序产生；日志与异常信息里不得出现数据库密码、API key

三条硬约束（由本模块的结构本身保证，而不是靠调用方自觉）
------------------------------------------------------
1. **可 JSON 序列化**：`ToolResult.to_dict()` 的结果必须能被 `json.dumps()` 直接编码。
   `Decimal` / `datetime` / `UUID` / `numpy` 标量 / `dataclass` 统一转成内建类型；
   `NaN` 与 `inf` 一律转 `None`（JSON 没有这两个值，"未定义"就是缺失，见 docs/05 §2.1）。
2. **大结果不入上下文**：`data` 里任何列表的行数不得超过 `MAX_INLINE_ROWS`（200）。
   超过时由工具写文件、把相对路径放进 `refs`。这条在 `__post_init__` 里强制，
   违反即抛 `ValueError`——宁可报错，也不悄悄把 42,613 行塞进 LLM 上下文（docs/08 §2）。
3. **错误信息脱敏**：`ToolError` 的 `message` / `detail` 必须过 `sanitize_text`。
   SQLAlchemy 的连接类异常里常常带完整 DSN（含密码），不过滤就会泄进日志与报告。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Iterable, Mapping
from uuid import UUID

#: `data` 内联行数上限。超过即必须写文件走 `refs`（docs/08 §2）。
MAX_INLINE_ROWS: int = 200

#: 大结果落盘时，内联返回的预览行数上限（供 LLM 判断"结果长什么样"）
PREVIEW_ROWS: int = 200

__all__ = [
    "MAX_INLINE_ROWS",
    "PREVIEW_ROWS",
    "ToolError",
    "ToolErrorCode",
    "ToolResult",
    "check_inline_rows",
    "jsonable",
    "sanitize_text",
]


# ============================================================================
#  错误码（取值字符串必须与 docs/05 §3.3 的注册表逐字一致）
# ============================================================================


class ToolErrorCode(str, Enum):
    """工具错误码。**五类必须可分辨**，混成"失败"下游就无法决定重试/澄清/放弃。

    取值字符串与 `docs/05-报告结构与异常分类.md` §3.3 的异常码注册表一致：

    - `E_TOOL_001` 参数错误（提示，不阻断，`FIX_AND_RERUN`）——修正参数后重试；
    - `E_PLAN_001` 业务条件不足（提示，不阻断，`REQUEST_INPUT`）——应向用户澄清；
    - `E_CAP_001` 超出能力范围（警告，阻断）——说明能力限制，重试无意义；
    - `E_DB_001` 数据库异常（连接失败 / 权限不足）；
    - `E_DB_003` 执行超时（超过 `statement_timeout`）；
    - `E_TOOL_999` 未知（程序 bug 兜底）。
    """

    #: 参数错误：类型/范围/枚举不合法（docs/05 §3.3 I 类）
    INVALID_PARAM = "E_TOOL_001"
    #: 业务条件不足：参数合法但当前数据不满足前提（docs/05 §3.3 I 类）
    PRECONDITION = "E_PLAN_001"
    #: 超出该数据适配的能力范围（docs/05 §3.3 E 类）
    CAPABILITY = "E_CAP_001"
    #: 数据库异常：连接失败、权限不足、SQL 报错（docs/05 §3.3 F 类）
    DB_ERROR = "E_DB_001"
    #: 超时：语句被 `statement_timeout` 取消（docs/05 §3.3 F 类）
    TIMEOUT = "E_DB_003"
    #: 未知异常（程序 bug 兜底）
    UNKNOWN = "E_TOOL_999"


#: 数据库异常里代表"语句被取消/超时"的 SQLSTATE
TIMEOUT_SQLSTATES: frozenset[str] = frozenset({"57014"})

#: 权限 / 认证类 SQLSTATE：**不重试**，属配置问题（docs/05 §3.3 F 类）
PERMISSION_SQLSTATES: frozenset[str] = frozenset({"42501", "28P01", "28000", "3D000"})


# ============================================================================
#  脱敏
# ============================================================================

#: `scheme://user:password@host` —— 连接串里的密码
_URL_CRED_RE = re.compile(
    r"(?P<scheme>[a-z0-9+.\-]+://)(?P<user>[^:/@\s]+):(?P<pw>[^@/\s]+)@", re.IGNORECASE
)

#: `password=...` / `api_key: ...` 这类键值对
_KV_SECRET_RE = re.compile(
    r"(?P<key>password|passwd|pwd|api[_-]?key|apikey|token|secret)"
    r"(?P<sep>\s*[=:]\s*)"
    r"(?P<val>[^\s,;'\")]+)",
    re.IGNORECASE,
)

_SECRETS_CACHE: tuple[str, ...] | None = None


def _known_secrets() -> tuple[str, ...]:
    """当前配置里的口令/密钥明文（用于把异常文本里的它们替换掉）。

    取不到配置时返回空元组：脱敏失败不能让工具本身失败。
    """
    global _SECRETS_CACHE
    if _SECRETS_CACHE is not None:
        return _SECRETS_CACHE

    secrets: list[str] = []
    try:
        from sqlalchemy.engine import make_url

        from dsa.config import get_settings

        settings = get_settings()
        for url in (settings.database_url, settings.database_url_ro):
            try:
                pw = make_url(url).password
            except Exception:  # noqa: BLE001 - 配置异常不影响脱敏
                pw = None
            if pw:
                secrets.append(str(pw))
        api_key = settings.deepseek_api_key.get_secret_value()
        if api_key:
            secrets.append(str(api_key))
    except Exception:  # noqa: BLE001 - 同上
        pass

    _SECRETS_CACHE = tuple(secrets)
    return _SECRETS_CACHE


def reset_secret_cache() -> None:
    """清空脱敏用的密钥缓存（测试或配置热更新时使用）。"""
    global _SECRETS_CACHE
    _SECRETS_CACHE = None


def sanitize_text(value: Any) -> str:
    """把文本里的凭据替换成 `***`。

    三层防线：连接串形式的 `user:password@`、`password=...` 形式的键值对、
    以及当前 `.env` 里真实存在的口令/密钥明文（防止被拼接进别的措辞里）。
    """
    out = str(value)
    out = _URL_CRED_RE.sub(lambda m: f"{m['scheme']}{m['user']}:***@", out)
    out = _KV_SECRET_RE.sub(lambda m: f"{m['key']}{m['sep']}***", out)
    for secret in _known_secrets():
        if secret and secret in out:
            out = out.replace(secret, "***")
    return out


# ============================================================================
#  JSON 化
# ============================================================================


def jsonable(value: Any) -> Any:
    """递归转成可 `json.dumps()` 的结构。

    规则（docs/05 §2.1「缺失与为零必须区分」的落地）：

    - `Decimal` → `float`（PostgreSQL `numeric` 的默认映射）；
    - `NaN` / `inf` → `None`（JSON 无此二者；"未定义"就是缺失，不写 0）；
    - `datetime` / `date` → ISO 字符串；`UUID` → 字符串；`Enum` → `.value`；
    - `dataclass`（含 `slots=True` 的冻结结构，如 `CheckResult` / `ComparisonStats`）→ dict；
    - `numpy` 标量与数组 → 内建标量 / list（不引入 numpy 依赖，按鸭子类型判断）；
    - 其余未知类型 → `str()`，**绝不**让序列化在工具出口处炸掉。
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item") and hasattr(value, "dtype"):  # numpy 标量
        return jsonable(value.item())
    if hasattr(value, "tolist"):  # numpy 数组
        return jsonable(value.tolist())
    return str(value)


def check_inline_rows(
    data: Any, *, limit: int = MAX_INLINE_ROWS, path: str = "data"
) -> None:
    """递归检查 `data` 里任何列表的行数是否超过 `limit`。

    Raises:
        ValueError: 存在超限列表。消息里带路径与行数，便于定位是哪个工具塞多了。
    """
    if isinstance(data, Mapping):
        for key, value in data.items():
            check_inline_rows(value, limit=limit, path=f"{path}.{key}")
    elif isinstance(data, (list, tuple)):
        if len(data) > limit:
            raise ValueError(
                f"{path} 有 {len(data)} 行，超过内联上限 {limit}；"
                "大结果必须写文件并把相对路径放进 ToolResult.refs（docs/08 §2）"
            )
        for index, value in enumerate(data):
            check_inline_rows(value, limit=limit, path=f"{path}[{index}]")


# ============================================================================
#  统一返回结构
# ============================================================================


@dataclass(frozen=True, slots=True)
class ToolError:
    """工具错误。`code` 决定下游动作（重试 / 澄清 / 放弃），不只是给人看的文字。"""

    code: ToolErrorCode
    message: str
    detail: str = ""

    def __post_init__(self) -> None:
        # 脱敏在构造点完成：任何走到这里的文本都已经过过滤，调用方无需记得再洗一遍
        object.__setattr__(self, "message", sanitize_text(self.message))
        object.__setattr__(self, "detail", sanitize_text(self.detail))

    @property
    def code_value(self) -> str:
        """异常码字符串（如 `"E_TOOL_001"`），写入运行记录与报告用。"""
        return self.code.value

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ToolResult:
    """工具统一返回。`ok=False` 时 `error` 非空。

    Attributes:
        ok: 是否成功。**注意**：分析被"阻断"（如 SRM 未通过）仍是 `ok=True`——
            阻断是业务结果，不是工具失败（docs/05 §4.1：「阻断不等于运行失败」）。
        tool: 工具名。
        data: 内联结构化数据；任何列表的行数不得超过 `MAX_INLINE_ROWS`。
        refs: 大结果的文件引用，**相对 `artifacts/`**（docs/08 §5）。
        summary: 一句话摘要，供人/LLM 阅读；数值仍以 `data` 为准。
        error: `ok=False` 时的错误对象。
        duration_ms: 工具耗时（毫秒）。
        run_id: 关联的分析运行标识（有则给出，供追溯）。
    """

    ok: bool
    tool: str
    data: dict | None = None
    refs: list[str] = field(default_factory=list)
    summary: str = ""
    error: ToolError | None = None
    duration_ms: int = 0
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.ok and self.error is None:
            raise ValueError(f"工具 {self.tool!r} 返回 ok=False 但 error 为空")
        if self.ok and self.error is not None:
            raise ValueError(f"工具 {self.tool!r} 返回 ok=True 却带 error")
        if self.data is not None:
            check_inline_rows(self.data)

    def to_dict(self) -> dict[str, Any]:
        """转成可直接 `json.dumps()` 的字典（唯一对外序列化出口）。"""
        return {
            "ok": bool(self.ok),
            "tool": self.tool,
            "data": jsonable(self.data),
            "refs": [str(r) for r in self.refs],
            "summary": self.summary,
            "error": None if self.error is None else self.error.to_dict(),
            "duration_ms": int(self.duration_ms),
            "run_id": self.run_id,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """`to_dict()` 的 JSON 文本形式（`ensure_ascii=False`，中文不转义）。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def with_duration(self, duration_ms: int) -> "ToolResult":
        """回填耗时（工具包装器统一计时）。"""
        return replace(self, duration_ms=int(duration_ms))

    def with_tool(self, tool: str) -> "ToolResult":
        """回填工具名（避免每个工具函数重复写自己的名字）。"""
        return replace(self, tool=tool)


def error_result(
    tool: str,
    code: ToolErrorCode,
    message: str,
    detail: str = "",
    *,
    data: dict | None = None,
    refs: Iterable[str] = (),
    duration_ms: int = 0,
    run_id: str | None = None,
) -> ToolResult:
    """构造失败结果。

    `data` 允许陪跑：例如"分析已完成但运行登记失败"时，产物引用与摘要仍要交回调用方，
    否则用户拿不回已经算出来的结果（异常不静默，但也不丢弃已完成的工作）。
    """
    return ToolResult(
        ok=False,
        tool=tool,
        data=data,
        refs=[str(r) for r in refs],
        summary="",
        error=ToolError(code=code, message=message, detail=detail),
        duration_ms=duration_ms,
        run_id=run_id,
    )
