"""项目配置：集中读取环境变量，提供数据库连接。

用法：
    from dsa.config import get_settings, get_engine

    s = get_settings()
    eng = get_engine(readonly=True)      # Agent 查询一律用只读账号

设计说明：
- 环境变量来自项目根的 `.env`（已 gitignore）。
- **只读连接是默认推荐**：按 AGENTS.md，Agent 与查询工具必须使用 `dsa_ro`。
  只有建表与导入数据才允许用可写连接。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL, make_url

# 源码运行保留仓库默认值；安装包把工作文件放在显式目录或当前目录。
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_CHECKOUT = (_SOURCE_ROOT / "pyproject.toml").is_file() and (_SOURCE_ROOT / "src/dsa").is_dir()
PROJECT_ROOT = Path(os.environ.get("DSA_WORKSPACE_ROOT") or
                    (_SOURCE_ROOT if _SOURCE_CHECKOUT else Path.cwd())).resolve()
ENV_FILE = Path(os.environ.get("DSA_ENV_FILE") or PROJECT_ROOT / ".env").resolve()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
from importlib.resources import files
SQL_DIR = files("dsa.resources").joinpath("sql")
ARTIFACTS_DIR = Path(os.environ.get("DSA_ARTIFACTS_DIR") or PROJECT_ROOT / "artifacts").resolve()


class Settings(BaseSettings):
    """项目设置。字段名与 `.env` 中的变量名大小写不敏感地对应。"""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM ----
    deepseek_api_key: SecretStr = SecretStr("")
    default_model_key: str | None = Field(default=None,validation_alias="DSA_DEFAULT_MODEL_KEY")

    # ---- 数据库 ----
    database_url: str = Field(description="可写连接（dsa_app），仅用于建表与导入")
    database_url_ro: str = Field(description="只读连接（dsa_ro），分析查询一律用它")

    pghost: str = "127.0.0.1"
    pgport: int = 5432
    pgdatabase: str = "marketing_analytics"

    @field_validator("database_url", "database_url_ro")
    @classmethod
    def _must_be_postgres(cls, v: str) -> str:
        url = make_url(v)
        if not url.drivername.startswith("postgresql"):
            raise ValueError(f"只支持 PostgreSQL 连接串，收到: {url.drivername}")
        return v

    # ---- 派生属性 ----
    @property
    def writable_url(self) -> URL:
        return make_url(self.database_url)

    @property
    def readonly_url(self) -> URL:
        return make_url(self.database_url_ro)

    def describe(self) -> str:
        """用于日志的安全描述（不含密码）。"""
        w, r = self.writable_url, self.readonly_url
        return (
            f"host={w.host}:{w.port} db={w.database} "
            f"rw_user={w.username} ro_user={r.username} "
            f"llm_key={'已设置' if self.deepseek_api_key.get_secret_value() else '未设置'}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读取并缓存设置。"""
    return Settings()  # type: ignore[call-arg]


#: 数据库连接超时（秒）。
#:
#: psycopg 默认约 130 秒才放弃连接。数据库宕机时，每个工具调用与 ``/health``
#: 都会挂两分钟才返回 ``E_DB_001``，对交互式 Agent 等同于"卡死"。
#: 5 秒足以覆盖正常连接，又能把故障响应压进可用范围。
#: 可用环境变量 ``DSA_DB_CONNECT_TIMEOUT_S`` 覆盖。
DB_CONNECT_TIMEOUT_S: int = int(os.environ.get("DSA_DB_CONNECT_TIMEOUT_S", "5"))

#: **语句**执行超时（毫秒）—— 与上面的**连接**超时是两件事。
#:
#: 缺陷回顾（外部审查 5.1）
#: ------------------------
#: > 新库只读连接实测 `statement_timeout=0`，而配置中的 5 秒是**连接超时**。
#: > NL2SQL 执行路径有独立语句超时设置，普通分析工具路径没有统一设置。
#: > ……**不能把连接超时当作查询运行时限。**
#:
#: 这是两个不同的故障：连不上是"门打不开"，语句跑不完是"进去以后出不来"。
#: 只设前者，一条写错的聚合查询（比如漏了分组条件导致全表笛卡尔积）
#: 可以让交互卡到用户以为系统死了。
#:
#: 15 秒的取值理由：本项目**交互路径**上最大的合法查询
#: （`query_arm_aggregates` 对 64,000 行做分组聚合）实测在 1 秒内返回；
#: `analyze_experiment_effect` 的耗时主要在 Python 侧的 bootstrap，不在 SQL。
#: 因此 15 秒对交互查询有 15 倍以上余量。
#:
#: ⚠️ **但它不是"所有查询的上限"**（一次真实回归换来的教训）
#: --------------------------------------------------------
#: 第一版把它设成**全局默认**，结果 3 条既有用例必然失败：
#: 它们要扫 45,786,568 行的 `rh_purchases` 做去重核对，
#: 实测需要 **42～83 秒**。对照实验（`artifacts/_prove_timeout_regression.py`）：
#:
#: ```
#: statement_timeout=15s   → 15.1~15.3s 全部 QueryCanceled
#: statement_timeout=180s  → 57.4s / 83.4s / 42.0s 全部成功
#: ```
#:
#: 根因不是"超时值太小"，而是**把两种不同意图的查询塞进了同一个预算**：
#: 交互式问答应当时限，离线核对/批处理可以跑分钟级。
#: 用一个全局值，必然有一方是错的。
#:
#: 因此现在分两档，并且**默认仍是交互档**（安全的那一侧）：
#: 忘记指定的新代码得到的是"不会挂死"，而不是"能跑一小时"。
DB_STATEMENT_TIMEOUT_MS: int = int(os.environ.get("DSA_DB_STATEMENT_TIMEOUT_MS", "15000"))

#: **批处理/离线核对**档的语句超时（毫秒）。
#:
#: 用途：全表扫描类核对（RetailHero 4,578 万行去重验证、交叉核对脚本、慢速测试）。
#: **必须在调用点显式传入**，不能靠改默认值 —— 显式传入是**可 grep 的**：
#: `grep "BATCH_STATEMENT_TIMEOUT" ` 就能列出"谁申请了更多时间、为什么"。
#: 如果它是默认值，那些申请就隐没在配置里，没人知道哪些查询真的跑了很久。
DB_STATEMENT_TIMEOUT_MS_BATCH: int = int(
    os.environ.get("DSA_DB_STATEMENT_TIMEOUT_MS_BATCH", "900000"))


@lru_cache(maxsize=16)
def get_engine(readonly: bool = True, *, echo: bool = False,
               statement_timeout_ms: int | None = None) -> Engine:
    """创建 SQLAlchemy 引擎。

    Args:
        readonly: True 使用只读账号（默认）。**分析与 Agent 查询必须为 True。**
        echo: 是否打印 SQL，调试用。
        statement_timeout_ms: 覆盖语句超时。``None`` 用交互档
            :data:`DB_STATEMENT_TIMEOUT_MS`；全表扫描类核对传
            :data:`DB_STATEMENT_TIMEOUT_MS_BATCH`。

    Note:
        引擎按 (readonly, echo, statement_timeout_ms) 缓存。测试中如需隔离，
        调用 `dispose_engines()`。

    设置项（两者**都要**，见 :data:`DB_STATEMENT_TIMEOUT_MS` 的说明）：

    - `connect_timeout` —— 连不上时多久放弃；
    - `options=-c statement_timeout=<ms>` —— 语句跑多久被服务端取消。
      用连接参数而不是在执行前发一条 `SET`，是因为连接来自连接池：
      池化的连接会复用，靠"执行前 SET"就必须保证每一次借出都设过，
      漏一次就等于没设。放在连接参数里，**每条连接建立时都带上它**。
    """
    settings = get_settings()
    url = settings.readonly_url if readonly else settings.writable_url
    timeout_ms = (DB_STATEMENT_TIMEOUT_MS if statement_timeout_ms is None
                  else int(statement_timeout_ms))
    return create_engine(
        url,
        echo=echo,
        pool_pre_ping=True,
        future=True,
        # ⚠️ 显式连接超时**不可省略**：psycopg 默认约 130 秒才放弃。
        # 数据库宕机时，每个工具调用（以及 /health）都会挂两分钟才返回 E_DB_001，
        # 对交互式 Agent 而言等同于"卡死"。审计实测：不设超时的那次对照跑了 130 秒。
        # 见 docs/日志.md L023。
        connect_args={
            "connect_timeout": DB_CONNECT_TIMEOUT_S,
            "options": f"-c statement_timeout={timeout_ms}",
        },
    )


def get_batch_engine(*, readonly: bool = True) -> Engine:
    """全表扫描/离线核对用的引擎（语句超时为 :data:`DB_STATEMENT_TIMEOUT_MS_BATCH`）。

    存在的意义是**把例外变成一个具名入口**：看到
    `get_batch_engine()` 就知道"这里会跑很久，而且是有意的"。
    """
    return get_engine(readonly=readonly,
                      statement_timeout_ms=DB_STATEMENT_TIMEOUT_MS_BATCH)


def statement_timeout_ms(engine: Engine) -> int | None:
    """读取该引擎上**服务端实际生效**的 `statement_timeout`（毫秒）。

    刻意从服务端读回，而不是回显配置常量：
    配置写了不等于生效（连接参数拼错、被服务端覆盖、或走了别的连接池），
    而"我配了"和"它生效了"是两件事 —— 本项目已经在这类差别上栽过多次。
    0 表示**不限制**。
    """
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            raw = conn.execute(text("show statement_timeout")).scalar()
    except Exception:  # noqa: BLE001 —— 读不到就是读不到，不编一个值
        return None
    if raw is None:
        return None
    txt = str(raw).strip().lower()
    try:
        if txt.endswith("ms"):
            return int(float(txt[:-2]))
        if txt.endswith("s"):
            return int(float(txt[:-1]) * 1000)
        if txt.endswith("min"):
            return int(float(txt[:-3]) * 60_000)
        return int(float(txt))
    except ValueError:
        return None


def dispose_engines() -> None:
    """释放所有缓存的引擎（测试或长驻进程重启时使用）。"""
    get_engine.cache_clear()
    get_settings.cache_clear()


Role = Literal["readonly", "writable"]
