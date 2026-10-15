"""模型适配层：提供商适配 + 明确标识的 mock 模式。

**纪律**（docs/09-Agent协议.md §7）：
- ``provider_info()['mode']`` 必须能被上层读到，用于标注「本次为 mock 演示」。
- mock 输出带 ``[MOCK]`` 前缀，**绝不能被当成真实模型评测结果**。
- 任何返回值、日志、异常信息**都不得包含密钥内容**。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

#: 环境变量名。只读它的**存在性**，绝不把值放进返回值。
API_KEY_ENV = "DEEPSEEK_API_KEY"
MODE_ENV = "DSA_LLM_MODE"
DEFAULT_MODEL = "deepseek-chat"
PROVIDER = "deepseek"

MOCK_PREFIX = "[MOCK] "


def _load_dotenv_once() -> None:
    """把项目根 ``.env`` 载入 ``os.environ``（只做一次）。

    ⚠️ 这一步**不可省略**：``dsa.config`` 用 pydantic-settings 的 ``env_file``
    读取 ``.env``，但那只填充 Settings 对象，**不会写回 ``os.environ``**。
    而 ``ChatDeepSeek`` 自己是从 ``os.environ`` 找密钥的。
    若不显式载入，即使 ``.env`` 里配好了密钥，
    本模块也会永远判定为 mock 模式 —— 这是一个真实踩过的坑
    （见 docs/日志.md L021）。
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv

        from dsa.config import ENV_FILE

        load_dotenv(ENV_FILE, override=False)
    except Exception:  # noqa: BLE001 —— 缺 dotenv 时退化为只看进程环境
        pass


_DOTENV_LOADED = False


def _env(name: str) -> str:
    _load_dotenv_once()
    return os.environ.get(name, "").strip()


def key_configured() -> bool:
    """密钥是否已配置。**只返回布尔值，不返回密钥本身。**"""
    return bool(_env(API_KEY_ENV))


def is_mock() -> bool:
    """未配置密钥、或显式设置 ``DSA_LLM_MODE=mock`` 时为 True。"""
    if _env(MODE_ENV).lower() == "mock":
        return True
    return not key_configured()


def provider_info() -> dict:
    """返回提供商信息。**绝不含密钥**。"""
    return {
        "provider": PROVIDER,
        "model": _env("DSA_LLM_MODEL") or DEFAULT_MODEL,
        "mode": "mock" if is_mock() else "real",
        "key_configured": key_configured(),
        "mock_warning": (
            "结果由确定性 mock 生成，不代表真实模型能力，不得用于模型评测结论"
            if is_mock() else None
        ),
    }


# ============================================================ 用量


@dataclass(frozen=True, slots=True)
class LlmUsage:
    """一次模型调用的用量。

    ⚠️ **`usage_known` 是必需的**（外部审查 5.4 的修复点）。
    供应商不一定回传用量：网络中断、流式响应提前结束、
    或 SDK 版本变化都可能让 `usage_metadata` 缺失。
    早期实现把它写成 `meta.get("input_tokens", 0) or 0` ——
    于是"没拿到用量"和"确实用了 0 个 token"变成同一个值，
    费用汇总再把它们加总，**把「没测到」伪装成「没花钱」**，
    平均费用因此系统性偏低。

    因此：拿不到用量时 `usage_known=False`，token 数**不代表真实用量**，
    费用汇总必须把它单独计数并声明总费用未知。
    """

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: int
    is_mock: bool
    #: 用量是否来自供应商的**真实回传**。False 时 token 数不可用于计费。
    usage_known: bool = True

    def to_dict(self) -> dict:
        return {
            "provider": self.provider, "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms, "is_mock": self.is_mock,
            "usage_known": self.usage_known,
        }


_USAGE: list[LlmUsage] = []


def get_usage_sink() -> list[LlmUsage]:
    return _USAGE


def record_usage(u: LlmUsage) -> None:
    _USAGE.append(u)


def reset_usage() -> None:
    _USAGE.clear()


# ============================================================ 返回对象


@dataclass(frozen=True, slots=True)
class LlmReply:
    content: str
    is_mock: bool = False
    usage: LlmUsage | None = None


def _approx_tokens(s: str) -> int:
    """粗估 token 数（mock 用；真实用量以供应商返回为准）。"""
    return max(1, len(s) // 3)


class MockLlm:
    """确定性 mock：同输入同输出，不联网。

    输出**必须**带 ``[MOCK]`` 前缀，使 mock 结果在任何下游呈现中都可辨识。
    """

    def __init__(self, *, temperature: float = 0.0, **kw: Any) -> None:
        self.temperature = temperature
        self.is_mock = True

    def invoke(self, messages: Sequence[Any], **kw: Any) -> LlmReply:
        started = time.perf_counter()
        text = _flatten(messages)
        content = MOCK_PREFIX + _canned(text)
        usage = LlmUsage(
            provider=PROVIDER, model="mock", prompt_tokens=_approx_tokens(text),
            completion_tokens=_approx_tokens(content),
            total_tokens=_approx_tokens(text) + _approx_tokens(content),
            duration_ms=int((time.perf_counter() - started) * 1000), is_mock=True,
        )
        record_usage(usage)
        return LlmReply(content=content, is_mock=True, usage=usage)


def _flatten(messages: Sequence[Any]) -> str:
    parts = []
    for m in messages or ():
        if isinstance(m, str):
            parts.append(m)
        elif isinstance(m, dict):
            parts.append(str(m.get("content", "")))
        else:
            parts.append(str(getattr(m, "content", m)))
    return "\n".join(parts)


def _canned(text: str) -> str:
    """确定性、可辨识的占位回答。

    **不试图伪装成真实的意图识别结果**——mock 的职责是让链路跑通，
    不是让结果看起来合理。上层据此判断路径即可。
    """
    low = text.lower()
    if "plan" in low or "计划" in text:
        return '{"experiment_key":"hillstrom_email_v1","comparison_key":"primary",' \
               '"metrics":["conversion_rate"],"question":"(mock plan)"}'
    if "澄清" in text or "clarify" in low or "缺失" in text:
        return "请补充：本次要分析哪个实验、关注哪个指标？"
    return "（mock 模式占位回答，未调用真实模型）"


# ============================================================================
#  受控注入（**测试与故障注入专用**）
# ============================================================================
#
# 为什么需要它：本项目的两条核心可靠性断言，都要求"让模型给出坏输出，
# 看程序怎么办"，而这两件事**不能靠真实模型碰运气**：
#
# 1. **数值守卫**（审查 F03）：必须能注入「邮件带来 99.9% 的提升，
#    已经证明有效」这样的受控回复，验证它**进不了报告**。
# 2. **对称故障注入**（审查 F08）：评测要给两套系统注入**同一个**故障。
#
# 因此在模型适配层开一个显式的注入口，而不是在节点里加
# `if testing:` 分支 —— 后者会让"测试路径"与"生产路径"分叉，
# 于是测试通过不代表生产可用。
_INJECTED: list[str] | None = None


def inject_replies(replies: Sequence[str] | None) -> None:
    """注入一组按顺序返回的回复；``None`` 表示取消注入。

    注入生效时**不联网**：无论是否配置了真实密钥，都返回注入内容。
    这是刻意的 —— 注入的目的是考察**程序对模型输出的处置**，
    与模型本身的能力无关。
    """
    global _INJECTED
    _INJECTED = None if replies is None else [str(r) for r in replies]


def injected_replies() -> list[str] | None:
    """当前注入的回复（副本）；未注入时为 None。"""
    return None if _INJECTED is None else list(_INJECTED)


class _InjectedLlm:
    """注入模式下的模型替身：按顺序吐出注入的回复，用完后返回空串。"""

    is_mock = False  # 刻意不是 mock：要检验的是"真实模型说错话"时的行为

    def __init__(self, *, temperature: float = 0.0, **kw: Any) -> None:
        self.temperature = temperature

    def invoke(self, messages: Sequence[Any], **kw: Any) -> LlmReply:
        started = time.perf_counter()
        pool = _INJECTED or []
        content = pool.pop(0) if pool else ""
        if _INJECTED is not None:
            _INJECTED[:] = pool
        usage = LlmUsage(
            provider=PROVIDER, model="injected", prompt_tokens=0,
            completion_tokens=_approx_tokens(content), total_tokens=_approx_tokens(content),
            duration_ms=int((time.perf_counter() - started) * 1000), is_mock=True,
        )
        # 刻意**不记 usage**：注入内容不是真实计费调用，
        # 记进去会让费用统计把"测试"算成"生产开销"。
        return LlmReply(content=content, is_mock=True, usage=usage)


_REQUEST_ATTEMPTS: list[dict] = []

def get_request_attempts() -> list[dict]:
    return [dict(x) for x in _REQUEST_ATTEMPTS]

class RealLlm:
    """真实模型适配。**惰性创建底层客户端**，不在 import 时联网。"""

    def __init__(self, *, temperature: float = 0.0, **kw: Any) -> None:
        self.temperature = temperature
        self.is_mock = False
        self._client: Any = None
        self._kw = kw

    def _ensure(self) -> Any:
        if self._client is None:
            from langchain_deepseek import ChatDeepSeek

            opts: dict[str, Any] = {
                "model": _env("DSA_LLM_MODEL") or DEFAULT_MODEL,
                "temperature": self.temperature,
                "max_retries": 0,
                "timeout": 45,
            }
            client = _http_client_bypassing_dead_proxy()
            if client is not None:
                opts["http_client"] = client
            opts.update(self._kw)
            self._client = ChatDeepSeek(**opts)
        return self._client

    def invoke(self, messages: Sequence[Any], **kw: Any) -> LlmReply:
        started = time.perf_counter()
        limit=int(_env("DSA_MODEL_REQUEST_LIMIT") or "0")
        if limit and len(_REQUEST_ATTEMPTS)>=limit:
            raise RuntimeError("模型请求预算已耗尽")
        client=self._ensure()
        attempt={"request":len(_REQUEST_ATTEMPTS)+1,"status":"started"}
        _REQUEST_ATTEMPTS.append(attempt)
        try:
            resp = client.invoke(list(messages), **kw)
            attempt["status"]="succeeded"
        except Exception as exc:
            attempt.update(status="failed",error_type=type(exc).__name__,
                           duration_ms=int((time.perf_counter()-started)*1000))
            record_usage(LlmUsage(provider=PROVIDER,model=_env("DSA_LLM_MODEL") or DEFAULT_MODEL,
                         prompt_tokens=0,completion_tokens=0,total_tokens=0,
                         duration_ms=attempt["duration_ms"],is_mock=False,usage_known=False))
            raise
        content = getattr(resp, "content", str(resp))
        meta = getattr(resp, "usage_metadata", None) or {}
        # ⚠️ 拿不到用量必须**显式标记**，不能填 0 了事：
        # 0 与"未知"在下游汇总里长得一模一样，而它们的含义相反。
        known = bool(meta) and (
            meta.get("input_tokens") is not None
            or meta.get("output_tokens") is not None
        )
        pt = int(meta.get("input_tokens") or 0)
        ct = int(meta.get("output_tokens") or 0)
        usage = LlmUsage(
            provider=PROVIDER, model=_env("DSA_LLM_MODEL") or DEFAULT_MODEL,
            prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct,
            duration_ms=int((time.perf_counter() - started) * 1000), is_mock=False,
            usage_known=known,
        )
        attempt.update(duration_ms=usage.duration_ms,usage=usage.to_dict())
        record_usage(usage)
        return LlmReply(content=content, is_mock=False, usage=usage)


# ============================================================ 死代理绕行


def _proxy_candidates() -> list[tuple[str, int]]:
    """收集可能生效的代理地址（环境变量 + Windows 系统代理）。"""
    out: list[tuple[str, int]] = []
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY"):
        v = os.environ.get(name, "").strip()
        if v:
            out.append(_parse_proxy(v))
    try:  # Windows 系统代理（WinINET）—— httpx/openai 在 Windows 上会读它
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if enabled:
                server, _ = winreg.QueryValueEx(key, "ProxyServer")
                out.append(_parse_proxy(str(server)))
    except Exception:  # noqa: BLE001 —— 非 Windows 或读不到就跳过
        pass
    return [c for c in out if c]


def _parse_proxy(value: str) -> tuple[str, int] | None:
    v = value.strip()
    if not v:
        return None
    # 形如 "http=host:port;https=host:port" 时取 http 段
    if "=" in v:
        for part in v.split(";"):
            if part.lower().startswith("http="):
                v = part.split("=", 1)[1]
                break
    v = v.split("://", 1)[-1].rstrip("/")
    if ":" not in v:
        return None
    host, _, port = v.rpartition(":")
    try:
        return (host or "127.0.0.1", int(port))
    except ValueError:
        return None


def _port_alive(host: str, port: int, timeout: float = 0.35) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def proxy_status() -> dict:
    """报告代理状态。**这是可观测性的一部分**：代理是否生效会决定能否联网。"""
    cands = _proxy_candidates()
    if not cands:
        return {"configured": False, "alive": None, "bypassed": False,
                "detail": "未配置代理"}
    dead = [(h, p) for h, p in cands if not _port_alive(h, p)]
    return {
        "configured": True,
        "alive": not dead,
        "bypassed": bool(dead),
        "detail": ("代理端口无监听，已绕行：" + ", ".join(f"{h}:{p}" for h, p in dead))
                  if dead else "代理可用",
    }


def _http_client_bypassing_dead_proxy() -> Any:
    """若配置的代理**端口无监听**，返回一个不信任环境代理的 httpx 客户端。

    为什么必须这样做：Windows 上系统代理（注册表 WinINET）常开着却指向
    一个没在运行的程序（例如代理软件退出后设置残留）。此时 httpx/openai
    会连到那个端口并被**主动拒绝**（``WinError 10061``），
    表现为「网络不通」，而实际上 API 完全可达。

    这个坑在本项目里出现过三次（Docker 拉镜像、RetailHero 下载、DeepSeek 调用），
    见 docs/日志.md L006 / L019 / L024。**一个连不上的代理不可能帮到任何人**，
    所以这里直接绕行，而不是让调用方去猜。

    代理活着时返回 None，表示沿用默认行为（走代理）。
    """
    cands = _proxy_candidates()
    if not cands:
        return None
    if any(_port_alive(h, p) for h, p in cands):
        return None  # 至少有一个可用，按正常路径走
    import httpx

    return httpx.Client(trust_env=False, timeout=60.0)


def get_llm(*, temperature: float = 0.0, **kw: Any) -> Any:
    """返回模型对象。

    优先级：**注入 > mock > 真实模型**。
    注入排在最前，是因为它必须能在"已配置真实密钥"的机器上
    也确定性地生效 —— 否则可靠性测试会随环境不同而改变行为。
    """
    if _INJECTED is not None:
        return _InjectedLlm(temperature=temperature, **kw)
    if is_mock():
        return MockLlm(temperature=temperature, **kw)
    return RealLlm(temperature=temperature, **kw)


def should_call_model() -> bool:
    """**调用方是否应该真的去取模型输出。**

    ⚠️ 判据不能只有 `not is_mock()`。

    `DSA_LLM_MODE=mock` 是**进程级全局状态**，而 `build_graph(use_llm=False)`
    会设置它。于是一个进程里只要先跑过任意一个"不用 LLM"的测试，
    后面的注入测试就会看到 `is_mock() == True` 而**提前跳过整个模型分支** ——
    注入设施在那种情况下**根本不可达**，测试表现为"守卫没有生效"，
    而真正的原因是被测代码压根没走到守卫。

    这类"环境开关把测试路径关掉"的失败很隐蔽：它看起来像功能坏了，
    实际是测试根本没跑到。判据写成 `注入激活 或 非 mock`，
    注入就永远可达，与进程历史无关。
    """
    return _INJECTED is not None or not is_mock()
