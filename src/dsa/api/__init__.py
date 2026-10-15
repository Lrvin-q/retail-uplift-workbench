"""HTTP 服务层：把工具层暴露为受控的 FastAPI 接口。

启动方式（本地调试）：

    python -m uvicorn dsa.api.main:app --port 8000

设计约束见 [docs/08-工具协议.md](../../../docs/08-工具协议.md) §7、§8：
错误码到 HTTP 状态码的映射必须可区分，日志与响应不得泄露密码/API key。
"""

from dsa.api.main import app, create_app

__all__ = ["app", "create_app"]
