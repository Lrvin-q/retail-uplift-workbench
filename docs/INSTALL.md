# 安装与运行

想先了解项目，不必先下载大数据或申请模型密钥。轻量演示会在本地生成虚构的营销实验，完成 SQL 对账、实验分析、建模、策略比较和请求路由。

## 轻量安装

要求 Python 3.12 或更高版本与 pip；首次安装需要访问 Python 包索引。推荐创建独立虚拟环境。以下命令均在解压后的项目根目录运行。

```bash
python -m venv .venv
```

Windows：

```powershell
.\.venv\Scripts\python.exe -m pip install ".[dev]"
.\.venv\Scripts\python.exe -m dsa.portfolio_demo --out artifacts/demo
.\.venv\Scripts\python.exe -m pytest
```

macOS / Linux：

```bash
.venv/bin/python -m pip install ".[dev]"
.venv/bin/python -m dsa.portfolio_demo --out artifacts/demo
.venv/bin/python -m pytest
```

安装后也可使用 `dsa-demo --out artifacts/demo`（需要虚拟环境的 Scripts/bin 目录在 PATH 中）。默认 2,000 名合成客户，200 次配对 bootstrap，固定 20% 比例。可用 `--n 4000 --seed 20261015` 改演示规模，但不能通过试种子把正结果写成业务成果。

演示不调用 `get_settings()`、不加载数据库、不调用模型。不要为了运行它而复制真实 `.env`。输出目录已存在时会报错，请改用 `--out artifacts/demo-2`；不会覆盖历史记录。

## 冻结研究与成本情景

基础安装即可在任意目录运行 `python -m dsa.research_portal --study starbucks --json`，或用 `--ledger` 查看探索账本。无需下载原始数据或配置 API；输出为保存好的证据及假设经济条件。

只看研究页面可安装 `python -m pip install ".[ui]"`，运行 `python -m streamlit run src/dsa/agent/workbench.py`，侧栏选择“研究证据与业务决策”。这个工作区不初始化数据库或模型；切回实验分析则需要完整服务。`[ui]` 只保证研究页面，不承诺完整产品依赖齐全。

可执行 `python -m pip install ".[dev,ui]"` 后运行包含 AppTest 的全部公开测试。仅基础 `[dev]` 且未安装 Streamlit 时，2 项 UI 测试会明确跳过；CI 安装 `[dev,ui]`，不以跳过代替 UI 验证。

## 完整产品（可选）

完整代码位于 `src/dsa`，轻量依赖只保证离线演示及公开测试所需路径。完整产品使用 PostgreSQL、FastAPI、Streamlit 与可选 DeepSeek；安装附加依赖：

```bash
python -m pip install ".[full]"
```

在自己的空 PostgreSQL 实例中初始化，**不要对既有业务数据库执行建库和迁移**。以有建角色权限的管理员创建 `dsa_app` 和 `dsa_ro`，使用交互式密码提示设置两个不同的密码，再创建由 `dsa_app` 拥有的 `marketing_analytics` 数据库。示意命令如下（主机、端口按自己的实例指定）：

```bash
createuser -h 127.0.0.1 -p 55432 -U postgres --pwprompt dsa_app
createuser -h 127.0.0.1 -p 55432 -U postgres --pwprompt dsa_ro
createdb -h 127.0.0.1 -p 55432 -U postgres -O dsa_app marketing_analytics
```

由管理员为只读账号设置：

```sql
ALTER ROLE dsa_ro SET default_transaction_read_only = on;
GRANT CONNECT ON DATABASE marketing_analytics TO dsa_ro;
```

按顺序以 `dsa_app` 身份在该空数据库运行下列 SQL：`001_schema.sql` → `003_models.sql` → `005_retailhero_schema.sql` → `006_policy_contrasts_and_identity.sql` → `007_retailhero_features_align.sql` → `008_run_parent.sql`。每次使用 `psql --set ON_ERROR_STOP=1 --file ...`，失败即停止。`002/004` 是查询模板，不是迁移。`dsa_ro` 必须事先存在，迁移末尾向它授予查询权限。

```bash
psql -h 127.0.0.1 -p 55432 -U dsa_app -d marketing_analytics --set ON_ERROR_STOP=1 --file sql/001_schema.sql
```

本次公开版未重新搭建这一整套 PostgreSQL 服务；上述步骤来自现有 SQL 结构，不构成新的冷启动验收声明。原先带固定开发密码的 Docker 初始化脚本和本机 Compose 未纳入公开版。

从 `.env.example` 复制为 `.env`，填写自己的连接信息，保留 `DSA_LLM_MODE=mock` 即可禁用真实模型。连接串中密码的特殊字符需 URL 编码。按[数据来源](DATA_SOURCES.md)取得 Hillstrom CSV，保存到 `data/raw/hillstrom_email_analytics_2008-03-20.csv`。不要改哈希以绕过版本校验。

```bash
python scripts/load_data.py --verify-only
python scripts/load_data.py
python -m streamlit run src/dsa/agent/workbench.py
```

API 入口：`python -m uvicorn dsa.api.main:app --port 8000`。总体实验分析不需先训练模型；客户名单等能力需要另行训练、登记模型及分数，见[模型协议](protocols/06-建模与策略评估协议.md)。数据导入与运行归档会写数据库，分析工具走只读账号。

只在需要真实 DeepSeek 联调时填写 API Key。不得把密钥、真实连接串或客户级工具载荷提交到仓库。公开版不内置服务账号，也不替用户自动调用收费 API。

## 常见问题

- `ModuleNotFoundError`：确认安装与运行使用同一个虚拟环境的 Python，完整网页服务须安装 `[full]`。
- `Output exists`：使用新的 `--out` 路径，不要删除需要保留的证据。
- PostgreSQL 连接失败：仅影响完整产品，先确认角色、数据库、端口、密码及只读权限，轻量演示无需数据库。
- 数值与样例略有差别：固定数据种子与流程，依赖版本仍可能影响拟合与浮点结果。本轮实测版本见[环境记录](ENVIRONMENT.md)。
- 安装慢或包下载失败：这是依赖下载问题，不需要配置模型 API；请检查包索引网络与 Python 版本。
