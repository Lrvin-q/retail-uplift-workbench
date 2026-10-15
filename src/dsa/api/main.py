"""FastAPI 服务层：把受限工具层暴露为可检查的 HTTP 接口。

依据：
- [docs/08-工具协议.md](../../../docs/08-工具协议.md) §3（错误码）、§7（日志与敏感信息）、§8（服务接口）
- [docs/05-报告结构与异常分类.md](../../../docs/05-报告结构与异常分类.md) §3.3 I 类异常码
- `AGENTS.md` 铁律 1、2、3

## 接口

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/health` | 健康检查；**DB 不可用时 `db_ok=false` 但不返回 500** |
| GET | `/tools` | 列出可用工具 |
| POST | `/tools/{tool_name}` | 调用工具（请求体**直接就是**工具 kwargs，可选 `X-Run-Id` 头） |
| GET | `/runs` | 运行记录列表（分页） |
| GET | `/runs/{run_id}` | 运行详情（含 `query_executions`） |
| GET | `/capability/{dataset_key}` | 数据适配能力说明（转发 `describe_capability`） |
| GET | `/metrics/{metric_id}` | 指标定义（转发 `get_metric_definition`） |

## 错误码 → HTTP 状态码（不一律 500）

| 错误码 | 类别 | HTTP |
|---|---|---|
| `E_TOOL_001` | 参数错误（结构校验失败、枚举/取值域非法） | 400 |
| `E_PLAN_001` | 业务条件不足（参数合法但前提不满足） | 422 |
| `E_CAP_001` | 超出能力范围 | 422 |
| `E_DB_001` | 数据库异常（连接失败、权限不足） | 503 |
| `E_DB_003` | 执行超时 | 504 |
| `E_TOOL_999` | 未归类异常 | 500 |

> 422 同时对应 `E_PLAN_001` 与 `E_CAP_001`，**只看状态码分不出是哪一个**，
> 因此响应体必须带 `error.code`；调用方据它决定"改参数 / 澄清 / 放弃 / 重试"。
>
> `E_TOOL_002` 是已废弃的旧码（与 `E_PLAN_001` 语义重复），仍接受但会在
> 响应体里**规范化为 `E_PLAN_001`**，避免同一个语义出现两个码。

## 脱敏（铁律 1）

- 响应体与日志中的文本一律先过 `redact()`：连接串里的密码、`PGPASSWORD`
  等环境变量值、`password=...` / `api_key=...` 形式的键值对全部替换为 `***`。
- 中间件**只记录元数据**（方法、路径、状态、耗时、`run_id`、错误码），
  不记录请求体，也不记录工具参数值。

## 运行标识的归属（与工具层的边界）

本层负责**记录**请求的 `run_id`（日志、`X-Run-Id` 响应头、`/runs` 追溯），
但**不把它注入工具参数**：约定接口是 `call_tool(name, engine, **kwargs)`，
没有 `run_id` 形参，擅自传入会让不支持的实现抛 `TypeError`。
`query_executions.run_id` 由工具层自行决定；`POST /tools/{name}` 的响应
`run_id` 优先取 `ToolResult.run_id`，为空时回退到请求头。
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import time
from typing import Any, Callable, Mapping, NoReturn, TypeVar

from fastapi import Body, FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, RootModel
from sqlalchemy import Engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from dsa.config import get_engine, get_settings
from dsa.tools import run_records

__all__ = [
    "ERROR_CODE_TO_HTTP",
    "ApiError",
    "ToolLayerUnavailable",
    "app",
    "canonical_error_code",
    "create_app",
    "get_readonly_engine",
    "get_tool_registry",
    "get_writable_engine",
    "http_status_for",
    "redact",
]

logger = logging.getLogger("dsa.api")
# 库代码不配置全局日志：加 NullHandler 避免"无 handler"告警，由宿主（uvicorn/CLI）决定输出
logger.addHandler(logging.NullHandler())

#: 工具层的导入目标。子代理实现 `src/dsa/tools/registry.py`，本模块**惰性导入**。
TOOL_REGISTRY_MODULE = "dsa.tools.registry"

T = TypeVar("T")

# ============================================================================
#  错误码与 HTTP 状态码
# ============================================================================
#: 已废弃的旧码 → 现行规范码。语义相同，只保留一个对外码。
CODE_ALIASES: dict[str, str] = {"E_TOOL_002": "E_PLAN_001"}

#: 规范码 → HTTP 状态码
ERROR_CODE_TO_HTTP: dict[str, int] = {
    "E_TOOL_001": 400,   # 参数错误
    "E_PLAN_001": 422,   # 业务条件不足（澄清，不算故障）
    "E_CAP_001": 422,    # 超出能力范围
    "E_DB_001": 503,     # 数据库异常
    "E_DB_003": 504,     # 超时
    "E_TOOL_999": 500,   # 未归类
}
UNKNOWN_CODE = "E_TOOL_999"


def canonical_error_code(code: Any) -> str:
    """把枚举/字符串错误码规范化为注册表中的码（含别名折叠）。"""
    raw = getattr(code, "value", None)
    if raw is None:
        raw = "" if code is None else str(code)
    raw = str(raw).strip()
    if not raw:
        return UNKNOWN_CODE
    return CODE_ALIASES.get(raw, raw)


def http_status_for(code: Any) -> int:
    """错误码 → HTTP 状态码；未注册的码按 `E_TOOL_999`（500）处理。"""
    return ERROR_CODE_TO_HTTP.get(canonical_error_code(code), 500)


# ============================================================================
#  脱敏（铁律 1：不得泄露密码 / API key）
# ============================================================================
_SECRET_ENV_KEYS = (
    "PGPASSWORD",
    "POSTGRES_PASSWORD",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "DATABASE_URL",
    "DATABASE_URL_RO",
)

#: scheme://user:password@host  → scheme://***:***@host
_URL_CRED_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)([^:/@\s]+):([^@/\s]+)@")

#: password=xxx / api_key: xxx / token="xxx"
_KV_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|apikey|authorization)\b"
    r"(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;)\]}]+)"
)

_MASK = "***"

#: `/runs/{run_id}` 路径中的运行标识（中间件在路由之前记录它）
_RUNS_PATH_RE = re.compile(r"^/runs/(?P<run_id>[^/]+)/?$")


def _configured_secrets() -> list[str]:
    """收集需要脱敏的字面量密钥（环境变量 + `Settings`），按长度倒序。

    过滤掉长度 < 4 的值：把 `"a"` 之类短串做全局替换会破坏正常文本，
    而真实密码/密钥不会这么短。
    """
    values: set[str] = set()
    for key in _SECRET_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            values.add(value)
    try:
        settings = get_settings()
    except Exception:  # noqa: BLE001 - 配置读取失败时只依赖环境变量
        settings = None
    if settings is not None:
        for url in (settings.database_url, settings.database_url_ro):
            try:
                password = make_url(url).password
            except Exception:  # noqa: BLE001 - 连接串异常不影响脱敏其它来源
                password = None
            if password:
                values.add(password)
        api_key = settings.deepseek_api_key.get_secret_value()
        if api_key:
            values.add(api_key)
    return sorted((v for v in values if len(v) >= 4), key=len, reverse=True)


def redact(value: Any) -> str:
    """脱敏后的字符串。用于日志与错误响应体。"""
    text_value = value if isinstance(value, str) else str(value)
    text_value = _URL_CRED_RE.sub(r"\1" + _MASK + ":" + _MASK + "@", text_value)
    for secret in _configured_secrets():
        if secret in text_value:
            text_value = text_value.replace(secret, _MASK)
    text_value = _KV_SECRET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{_MASK}", text_value)
    return text_value


def redact_obj(value: Any) -> Any:
    """递归脱敏任意 JSON 结构中的字符串。"""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {str(k): redact_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_obj(v) for v in value]
    return value


def _safe_token(value: Any, *, max_length: int = 64) -> str | None:
    """把外部传入的标识符收敛成可安全写入日志的 token（防日志注入）。"""
    if value is None:
        return None
    token = re.sub(r"[^A-Za-z0-9._:\-]", "", str(value))
    if not token:
        return None
    return token[:max_length]


# ============================================================================
#  异常
# ============================================================================
class ApiError(Exception):
    """带上错误码的 API 异常；由统一异常处理器转成响应体。"""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        detail: Any = None,
        http_status: int | None = None,
    ) -> None:
        self.code = canonical_error_code(code)
        self.message = message
        self.detail = detail
        self.http_status = http_status if http_status is not None else http_status_for(self.code)
        super().__init__(f"{self.code}: {message}")


class ToolLayerUnavailable(RuntimeError):
    """工具层无法导入。**不静默降级**：显式报错，让调用方知道是代码依赖缺失。"""


# ============================================================================
#  工具层接入（约定接口，惰性导入）
# ============================================================================
_tool_registry: Any | None = None


def _import_tool_registry() -> Any:
    """导入工具层模块。失败时抛 `ToolLayerUnavailable`，绝不返回降级实现。"""
    try:
        return importlib.import_module(TOOL_REGISTRY_MODULE)
    except Exception as exc:  # noqa: BLE001 - 任何导入失败都必须显式暴露
        raise ToolLayerUnavailable(
            f"工具层不可用：无法导入 {TOOL_REGISTRY_MODULE}"
            f"（{type(exc).__name__}: {redact(exc)}）"
        ) from exc


def get_tool_registry() -> Any:
    """返回工具层模块（首次调用时导入并缓存）。

    约定接口：

    - `list_tools() -> list[...]`
    - `call_tool(name: str, engine, **kwargs) -> ToolResult`
    """
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = _import_tool_registry()
    return _tool_registry


def set_tool_registry(module: Any | None) -> None:
    """注入/清空工具层模块（测试与脚本用）。"""
    global _tool_registry
    _tool_registry = module


# ============================================================================
#  引擎接入
# ============================================================================
def get_readonly_engine() -> Engine:
    """只读引擎（`dsa_ro`）。分析查询一律用它。"""
    return get_engine(readonly=True)


def get_writable_engine() -> Engine:
    """可写引擎（`dsa_app`）。仅用于运行记录等写入操作。"""
    return get_engine(readonly=False)


def _readonly_engine_or_503() -> Engine:
    try:
        return get_readonly_engine()
    except Exception as exc:  # noqa: BLE001 - 配置缺失等价于数据库不可用
        raise ApiError(
            code="E_DB_001",
            message="数据库引擎不可用",
            detail=redact(f"{type(exc).__name__}: {exc}"),
        ) from exc


def _is_timeout_error(exc: BaseException) -> bool:
    lowered = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in lowered
        for marker in ("timeout", "timed out", "canceling statement", "querycanceled")
    )


def _db_api_error(exc: BaseException) -> ApiError:
    """数据库异常 → 错误码：超时 504，其余连接/权限/SQL 错误 503。"""
    detail = redact(f"{type(exc).__name__}: {exc}")
    if _is_timeout_error(exc):
        return ApiError(code="E_DB_003", message="数据库操作超时", detail=detail)
    return ApiError(code="E_DB_001", message="数据库不可用或查询失败", detail=detail)


def db_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """执行一次数据库操作并把异常映射为可区分的错误码。"""
    try:
        return func(*args, **kwargs)
    except ApiError:
        raise
    except (SQLAlchemyError, OSError) as exc:
        raise _db_api_error(exc) from exc
    except (ValueError, TypeError, LookupError):
        raise
    except Exception as exc:  # noqa: BLE001 - 未归类异常不得伪装成数据库故障
        raise ApiError(
            code=UNKNOWN_CODE,
            message="未归类的服务端异常",
            detail=redact(f"{type(exc).__name__}: {exc}"),
        ) from exc


# ============================================================================
#  Pydantic 模型
# ============================================================================
class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: Any = None


class ErrorResponse(BaseModel):
    ok: bool = False
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str = Field(description="ok=数据库可用；degraded=数据库不可用但服务仍在")
    db_ok: bool
    code_version: str
    db_error: str | None = None


class ToolInfo(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] | None = None


class ToolListResponse(BaseModel):
    ok: bool = True
    count: int
    tools: list[ToolInfo]


class ToolCallRequest(RootModel[dict[str, Any]]):
    """`POST /tools/{tool_name}` 的请求体：**直接就是工具 kwargs**。"""

    root: dict[str, Any] = Field(default_factory=dict)


class ToolCallResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    tool: str
    data: dict[str, Any] | None = None
    refs: list[str] = Field(default_factory=list)
    summary: str = ""
    duration_ms: int = 0
    run_id: str | None = None
    error: ErrorDetail | None = None


class LookupResponse(BaseModel):
    """`/capability/{key}` 与 `/metrics/{key}` 的统一成功响应。"""

    ok: bool = True
    key: str
    tool: str
    data: dict[str, Any] | None = None
    refs: list[str] = Field(default_factory=list)
    summary: str = ""
    duration_ms: int = 0
    run_id: str | None = None
    error: ErrorDetail | None = None


class RunListResponse(BaseModel):
    ok: bool = True
    count: int
    limit: int
    offset: int
    status: str | None = None
    runs: list[dict[str, Any]]


class RunDetailResponse(BaseModel):
    ok: bool = True
    run: dict[str, Any]


class AnalyzeRequest(BaseModel):
    """`POST /analyze` 的请求体。"""

    model_config = ConfigDict(extra="forbid")

    question: str


class AnalyzeResponse(BaseModel):
    """`POST /analyze` 的响应。

    ⚠️ **`run_id` 一定非空**（登记失败会返回 500，而不是静默给一个 null）。
    「从 API 发起的分析有非空运行 ID」是外部审查 F06 的验收条件之一：
    没有它，`/runs/{run_id}`、导出与重放都无从谈起。
    """

    ok: bool = True
    run_id: str
    thread_id: str
    category: str
    outcome: str
    plan: dict[str, Any] | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)
    blocking_codes: list[str] = Field(default_factory=list)
    report: str = ""
    tools_called: list[dict[str, Any]] = Field(default_factory=list)
    refs: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


# ============================================================================
#  响应构造
# ============================================================================
def error_response(
    code: str,
    message: str,
    detail: Any = None,
    *,
    http_status: int | None = None,
) -> JSONResponse:
    """构造统一错误响应体：`{"ok": false, "error": {code, message, detail}}`。"""
    canonical = canonical_error_code(code)
    body = {
        "ok": False,
        "error": {
            "code": canonical,
            "message": redact(message),
            "detail": redact_obj(detail),
        },
    }
    return JSONResponse(
        status_code=http_status if http_status is not None else http_status_for(canonical),
        content=jsonable_encoder(body),
    )


def _tool_error_parts(tool_name: str, error: Any) -> tuple[str, str, Any]:
    """从工具层返回的错误对象里取出 (code, message, detail)，兼容 Mapping 与对象。"""
    if error is None:
        return (
            UNKNOWN_CODE,
            f"工具 {tool_name} 返回 ok=False 但未提供错误信息",
            None,
        )
    if isinstance(error, Mapping):
        code = error.get("code")
        message = error.get("message")
        detail = error.get("detail")
    else:
        code = getattr(error, "code", None)
        message = getattr(error, "message", None)
        detail = getattr(error, "detail", None)
    canonical = canonical_error_code(code)
    message = str(message) if message else f"工具 {tool_name} 调用失败（{canonical}）"
    return canonical, message, {"tool": tool_name, "detail": detail}


def _raise_for_tool_error(tool_name: str, error: Any) -> NoReturn:
    code, message, detail = _tool_error_parts(tool_name, error)
    raise ApiError(code=code, message=message, detail=detail)


def _normalize_tool_info(entry: Any) -> ToolInfo:
    """把工具层的工具描述规范化为响应模型（兼容 Mapping 与对象两种形态）。"""
    if isinstance(entry, Mapping):
        name = entry.get("name")
        description = entry.get("description") or ""
        schema = entry.get("input_schema") or entry.get("parameters") or entry.get("schema")
    else:
        name = getattr(entry, "name", None)
        description = getattr(entry, "description", "") or ""
        schema = (
            getattr(entry, "input_schema", None)
            or getattr(entry, "parameters", None)
            or getattr(entry, "schema", None)
        )
    if not name or not isinstance(name, str):
        raise ApiError(
            code=UNKNOWN_CODE,
            message="工具层返回的工具描述缺少 name 字段",
            detail=redact(repr(entry))[:500],
        )
    if schema is not None and not isinstance(schema, dict):
        schema = jsonable_encoder(schema)
    return ToolInfo(name=name, description=str(description), input_schema=schema)


def _extract_run_id(request: Request) -> str | None:
    """从 `X-Run-Id` 头 / `run_id` 查询参数 / `/runs/{run_id}` 路径中取运行标识。

    路径参数必须**在路由匹配之前**自行解析：中间件先于路由执行，
    此时 `scope["path_params"]` 还没有被填充。
    """
    token = _safe_token(request.headers.get("X-Run-Id"))
    if token:
        return token
    token = _safe_token(request.query_params.get("run_id"))
    if token:
        return token
    match = _RUNS_PATH_RE.match(request.url.path)
    if match:
        return _safe_token(match.group("run_id"))
    return None


# ============================================================================
#  应用
# ============================================================================
def create_app() -> FastAPI:
    """构造 FastAPI 应用（测试可多次调用，互不共享状态）。"""
    application = FastAPI(
        title="零售营销增量决策助手 · 数据科学 Agent 服务层",
        version="0.1.0",
        description=(
            "受限工具层的 HTTP 接口。数值由 SQL/Python 产生；"
            "错误码到状态码的映射见 docs/08-工具协议.md。"
        ),
    )

    # ---------------------------------------------------------------- 中间件
    @application.middleware("http")
    async def request_context_middleware(request: Request, call_next: Callable[[Request], Any]):
        """记录每个请求的 run_id / 耗时 / 状态，并回写响应头。

        日志只含元数据：**不记录请求体、不记录工具参数值**（可能含敏感项）。
        """
        started = time.perf_counter()
        run_id = _extract_run_id(request)
        request.state.run_id = run_id
        request.state.error_code = None
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - 记录后原样抛出，交给上层
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.error(
                "request failed method=%s path=%s status=500 duration_ms=%d run_id=%s error=%s",
                request.method,
                request.url.path,
                duration_ms,
                run_id or "-",
                redact(f"{type(exc).__name__}: {exc}"),
            )
            raise
        duration_ms = int((time.perf_counter() - started) * 1000)
        error_code = getattr(request.state, "error_code", None) or "-"
        if run_id:
            response.headers["X-Run-Id"] = run_id
        response.headers["X-Duration-Ms"] = str(duration_ms)
        logger.info(
            "request method=%s path=%s status=%d duration_ms=%d run_id=%s error_code=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            run_id or "-",
            error_code,
        )
        return response

    # ------------------------------------------------------------ 异常处理
    @application.exception_handler(ApiError)
    async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        request.state.error_code = exc.code
        return error_response(exc.code, exc.message, exc.detail, http_status=exc.http_status)

    @application.exception_handler(ToolLayerUnavailable)
    async def _tool_layer_handler(request: Request, exc: ToolLayerUnavailable) -> JSONResponse:
        request.state.error_code = "E_DB_001"
        return error_response(
            "E_DB_001",
            "工具层不可用（依赖缺失，非静默降级）",
            redact(exc),
            http_status=503,
        )

    @application.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # 结构校验失败属于 E_TOOL_001（参数错误）→ 400，而不是框架默认的 422：
        # 422 在本服务里表示"参数合法但前提不满足/超出能力"，必须与"写错了"区分。
        request.state.error_code = "E_TOOL_001"
        return error_response(
            "E_TOOL_001",
            "请求参数结构校验失败",
            redact_obj(exc.errors()),
            http_status=400,
        )

    @application.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = f"E_HTTP_{exc.status_code}"
        request.state.error_code = code
        return error_response(code, str(exc.detail), None, http_status=exc.status_code)

    @application.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        request.state.error_code = UNKNOWN_CODE
        return error_response(
            UNKNOWN_CODE,
            "未归类的服务端异常",
            redact(f"{type(exc).__name__}: {exc}"),
            http_status=500,
        )

    # ---------------------------------------------------------------- /health
    @application.get("/health", response_model=HealthResponse, tags=["系统"])
    def health() -> HealthResponse:
        """健康检查。**数据库不可用时返回 200 + `db_ok=false`，不返回 500。**"""
        db_ok = False
        db_error: str | None = None
        try:
            engine = get_readonly_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_ok = True
        except Exception as exc:  # noqa: BLE001 - 健康检查必须永不 500
            db_error = redact(f"{type(exc).__name__}: {exc}")
        return HealthResponse(
            status="ok" if db_ok else "degraded",
            db_ok=db_ok,
            code_version=run_records.code_version(),
            db_error=db_error,
        )

    # ---------------------------------------------------------------- /tools
    @application.get("/tools", response_model=ToolListResponse, tags=["工具"])
    def list_tools_endpoint() -> ToolListResponse:
        registry = get_tool_registry()
        raw = registry.list_tools()
        if raw is None:
            raise ApiError(
                code=UNKNOWN_CODE, message="工具层 list_tools() 返回 None，接口约定被破坏"
            )
        tools = [_normalize_tool_info(entry) for entry in raw]
        return ToolListResponse(count=len(tools), tools=tools)

    @application.post("/tools/{tool_name}", response_model=ToolCallResponse, tags=["工具"])
    def call_tool_endpoint(
        tool_name: str,
        request: Request,
        payload: ToolCallRequest | None = Body(default=None),
    ) -> ToolCallResponse:
        """调用工具。请求体**直接就是** kwargs；`run_id` 由 `X-Run-Id` 头提供。"""
        if not tool_name or "/" in tool_name:
            raise ApiError(code="E_TOOL_001", message=f"tool_name 非法: {tool_name!r}")
        kwargs = dict(payload.root) if payload is not None else {}
        registry = get_tool_registry()
        engine = _readonly_engine_or_503()
        started = time.perf_counter()
        try:
            result = registry.call_tool(tool_name, engine, **kwargs)
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 - 工具层异常统一映射为可区分错误码
            raise ApiError(
                code=UNKNOWN_CODE,
                message=f"工具 {tool_name} 调用抛出未归类异常",
                detail=redact(f"{type(exc).__name__}: {exc}"),
            ) from exc
        duration_ms = int((time.perf_counter() - started) * 1000)
        return _tool_result_response(tool_name, result, request=request, duration_ms=duration_ms)

    def _tool_result_response(
        tool_name: str,
        result: Any,
        *,
        request: Request,
        duration_ms: int,
    ) -> ToolCallResponse:
        """把 `ToolResult` 转成响应；`ok=False` 时抛 `ApiError` 走统一错误体。"""
        if result is None:
            raise ApiError(
                code=UNKNOWN_CODE,
                message=f"工具 {tool_name} 返回 None，接口约定被破坏",
            )
        if isinstance(result, Mapping):
            ok = bool(result.get("ok", False))
            error = result.get("error")
            data = result.get("data")
            refs = result.get("refs") or []
            summary = result.get("summary") or ""
            result_run_id = result.get("run_id")
            result_duration = result.get("duration_ms")
        elif hasattr(result, "ok"):
            ok = bool(result.ok)
            error = result.error
            data = result.data
            refs = result.refs or []
            summary = result.summary or ""
            result_run_id = getattr(result, "run_id", None)
            result_duration = getattr(result, "duration_ms", None)
        else:
            raise ApiError(
                code=UNKNOWN_CODE,
                message=f"工具 {tool_name} 返回结构无法识别: {type(result).__name__}",
            )

        header_run_id = getattr(request.state, "run_id", None)
        run_id = result_run_id or header_run_id
        if not ok:
            _raise_for_tool_error(tool_name, error)
        return ToolCallResponse(
            ok=True,
            tool=tool_name,
            data=data if isinstance(data, dict) else (None if data is None else {"value": data}),
            refs=[str(ref) for ref in refs],
            summary=str(summary),
            duration_ms=int(result_duration) if isinstance(result_duration, int) else duration_ms,
            run_id=str(run_id) if run_id else None,
            error=None,
        )

    # ------------------------------------------------------------ 能力 / 指标
    def _lookup(tool_name: str, key: str, kwargs: dict[str, Any], request: Request) -> LookupResponse:
        """转发到只读工具（`describe_capability` / `get_metric_definition`）。"""
        registry = get_tool_registry()
        engine = _readonly_engine_or_503()
        started = time.perf_counter()
        try:
            result = registry.call_tool(tool_name, engine, **kwargs)
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ApiError(
                code=UNKNOWN_CODE,
                message=f"工具 {tool_name} 调用抛出未归类异常",
                detail=redact(f"{type(exc).__name__}: {exc}"),
            ) from exc
        duration_ms = int((time.perf_counter() - started) * 1000)
        call = _tool_result_response(tool_name, result, request=request, duration_ms=duration_ms)
        return LookupResponse(
            ok=True,
            key=key,
            tool=tool_name,
            data=call.data,
            refs=call.refs,
            summary=call.summary,
            duration_ms=call.duration_ms,
            run_id=call.run_id,
            error=None,
        )

    @application.get("/capability/{dataset_key}", response_model=LookupResponse, tags=["能力"])
    def capability_endpoint(dataset_key: str, request: Request) -> LookupResponse:
        """数据适配能力说明（该数据集支持/不支持哪些分析）。"""
        return _lookup("describe_capability", dataset_key, {"dataset_key": dataset_key}, request)

    @application.get("/metrics/{metric_id}", response_model=LookupResponse, tags=["能力"])
    def metric_endpoint(metric_id: str, request: Request) -> LookupResponse:
        """指标定义（分子、分母、单位、约束）。"""
        return _lookup("get_metric_definition", metric_id, {"metric_id": metric_id}, request)

    # ---------------------------------------------------------------- /runs
    @application.get("/runs", response_model=RunListResponse, tags=["运行记录"])
    def list_runs_endpoint(
        limit: int = Query(default=run_records.DEFAULT_LIST_LIMIT, ge=1,
                           le=run_records.MAX_LIST_LIMIT),
        offset: int = Query(default=0, ge=0),
        status: str | None = Query(default=None),
    ) -> RunListResponse:
        """运行记录列表（分页，按 `started_at` 倒序）。"""
        if status is not None and status not in run_records.RUN_STATUSES:
            raise ApiError(
                code="E_TOOL_001",
                message=f"status 取值不合法: {status!r}",
                detail=f"允许值: {', '.join(run_records.RUN_STATUSES)}",
            )
        engine = _readonly_engine_or_503()
        runs = db_call(run_records.list_runs, engine, limit=limit, offset=offset, status=status)
        return RunListResponse(
            count=len(runs), limit=limit, offset=offset, status=status, runs=runs
        )

    @application.get("/runs/{run_id}", response_model=RunDetailResponse, tags=["运行记录"])
    def run_detail_endpoint(run_id: str) -> RunDetailResponse:
        """运行详情，含该运行的 `query_executions`（报告第 4 章追溯用）。"""
        try:
            normalized = run_records.as_run_id(run_id)
        except ValueError as exc:
            raise ApiError(code="E_TOOL_001", message="run_id 不是合法 UUID", detail=str(exc)) from exc
        engine = _readonly_engine_or_503()
        run = db_call(run_records.get_run, engine, normalized)
        if run is None:
            raise ApiError(
                code="E_HTTP_404",
                message=f"运行记录不存在: {normalized}",
                detail=None,
                http_status=404,
            )
        return RunDetailResponse(run=run)

    # ---------------------------------------------------------------- /analyze
    @application.post("/analyze", response_model=AnalyzeResponse, tags=["分析"])
    def analyze_endpoint(payload: AnalyzeRequest) -> AnalyzeResponse:
        """用自然语言提问，跑一次完整分析并**登记运行**。

        外部审查 F06 的验收条件之一是「**在 API／工作台统一创建运行**」。
        本端点与工作台、命令行调用的是**同一个** `run_once`，
        因此三处的 `run_id` 语义、产物引用与终态判据完全一致 ——
        不会出现"页面能导出、API 导不出"这类各自演化的问题。

        **登记失败即 500**：不能返回一个 `run_id=None` 的成功响应，
        否则调用方会以为这次分析可追溯，而实际上 `/runs/{run_id}` 查不到。
        """
        question = (payload.question or "").strip()
        if not question:
            raise ApiError(code="E_PLAN_001", message="question 不能为空", detail=None)

        try:
            from dsa.agent.graph import _register_run, build_graph
            from dsa.agent.graph import run_status_for
            from dsa.agent.persistence import make_checkpointer, new_thread_config
            from dsa.agent.state import initial_state
        except Exception as exc:  # noqa: BLE001
            raise ApiError(code=UNKNOWN_CODE, message="Agent 模块不可用",
                           detail=redact(f"{type(exc).__name__}: {exc}")) from exc

        run_id = _register_run(question, run_id=None)
        if not run_id:
            raise ApiError(
                code="E_DB_001",
                message="运行登记创建失败，本次分析无法追溯，已中止",
                detail="请检查可写数据库连接（analysis_runs 表）",
            )

        import uuid

        thread_id = f"api-{uuid.uuid4().hex[:12]}"
        try:
            app_graph = build_graph(checkpointer=make_checkpointer(), use_llm=True)
            state = app_graph.invoke(
                initial_state(question, run_id=run_id, thread_id=thread_id),
                config=new_thread_config(thread_id),
            )
        except Exception as exc:  # noqa: BLE001
            raise ApiError(code=UNKNOWN_CODE, message="分析执行失败",
                           detail=redact(f"{type(exc).__name__}: {exc}")) from exc

        from dsa.agent.graph import _finish_run

        _finish_run(run_id, state)

        refs: list[str] = []
        tools_called: list[dict[str, Any]] = []
        for tr in state.get("tool_results") or []:
            refs.extend(tr.get("refs") or [])
            tools_called.append({
                "tool": tr.get("tool"),
                "ok": bool(tr.get("ok")),
                "error_code": (tr.get("error") or {}).get("code"),
            })

        return AnalyzeResponse(
            run_id=run_id,
            thread_id=thread_id,
            category=str(state.get("category") or ""),
            outcome=run_status_for(state),
            plan=state.get("plan"),
            issues=list(state.get("issues") or []),
            blocking_codes=list(state.get("blocking_codes") or []),
            report=str(state.get("report") or ""),
            tools_called=tools_called,
            refs=refs,
            errors=list(state.get("errors") or []),
        )

    return application


app = create_app()
