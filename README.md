# 客户分析与智能查询系统

> 基于 RFM 模型与 K-Means 聚类的航空公司客户价值分析平台
> 采用 **MCP（Model Context Protocol）工具调用架构**，支持自然语言查询与智能可视化

---

## 一、项目概述

本系统针对航空公司客户数据（2014 年观察期，约 6.3 万条记录），构建了完整的客户价值分层与智能分析体系：

- **RFM 模型** — 量化客户价值
- **K-Means 聚类** — 客户分群
- **MCP 工具调用** — 让 LLM 真正"看懂"数据库，避免幻觉
- **智能可视化** — 16 种 ECharts 图表类型 + 自动选型

通过大语言模型（DeepSeek、GPT 等）将用户的自然语言问题转换为数据查询与图表，输出可直接理解的分析报告。

---

## 二、核心特性

### 1. 真正的 MCP 架构

摒弃传统 Prompt 拼接，**MCP Server（5001 端口）** 与 **Flask Web（5000 端口）** 完全解耦：

```
┌──────────────────────┐         ┌──────────────────────┐
│  Flask Web (5000)    │  HTTP   │  MCP Server (5001)   │
│  - 仪表盘 UI         │ ──────► │  - 12 个工具         │
│  - /api/chat         │         │  - SSE 传输          │
│  - 报告导出          │         │  - FastMCP 框架      │
└──────────────────────┘         └──────────────────────┘
         ▲                                ▲
         │                                │
         └──────── LLM Agent ────────────┘
              (Function Calling / ReAct)
```

**MCP Server 暴露的 12 个工具**：

| 分类 | 工具 | 功能 |
| --- | --- | --- |
| **Schema 锚定** | `get_database_schema` | 获取完整表结构（防字段幻觉核心） |
| | `get_table_schema` | 获取单张表结构 |
| | `probe_distinct_values` | 嗅探字段真实枚举值（防 WHERE 条件错误） |
| | `get_business_rules` | 获取业务规则（价值标签分段等） |
| **SQL 沙箱** | `execute_secure_sql` | 沙箱安全执行 SELECT，黑名单拦截写操作 |
| | `validate_sql_syntax` | SQL 语法预校验 |
| **聚类分析** | `perform_dynamic_cluster` | 即时 K-Means 聚类 |
| | `analyze_cluster_profiles` | 聚类业务画像分析 |
| | `verify_data_logic` | SQL vs Pandas 双重逻辑校验 |
| **可视化** | `generate_visualization` | 16 种 ECharts 图表类型 |
| **预测** | `forecast_trend` | 趋势预测（线性回归 / 指数平滑） |
| | `detect_anomalies` | 异常检测（IQR / IsolationForest） |
| | `churn_risk_score` | 流失风险评分 |

### 2. 三表分层架构

数据仓库分层设计，业务主键 `member_no` 关联：

| 表 | 层级 | 内容 |
| --- | --- | --- |
| `customer_base` | Dim（维度） | 年龄、性别、会员等级、平均折扣、入会日期等 |
| `customer_flight_summary` | DWS（汇总） | 飞行次数、里程、积分、Recency 等 |
| `customer_analytics` | ADS（应用） | RFM 评分、聚类标签、价值分层 |
| `customer_rfm` | View（宽表） | 三表 JOIN，免写复杂 JOIN |

### 3. 智能可视化（16 种图表）

`generate_visualization` 支持 16 种 ECharts 图表，按数据特征自动选型：

| 类别 | 图表类型 |
| --- | --- |
| 基础 | 柱状图、折线图、饼图、散点图 |
| 高级 | 雷达图、热力图、漏斗图、箱线图、仪表盘 |
| 组合 | 堆叠柱状图、面积折线图、帕累托图、矩形树图 |
| 3D/拓扑 | 地图、3D 散点图、桑基图 |

**自动选型规则**：
- `composition`（占比）→ 饼图 / 树图
- `relationship`（关系）→ 散点图
- `distribution`（分布）→ 箱线图 / 柱状图
- `trend`（趋势）→ 面积折线图 / 折线图
- `ranking`（排名）→ 帕累托图

### 4. 地理分布可视化

在左侧菜单「地理分布」中，使用 ECharts 地图展示客户在全国 34 个省级行政区的分布：

- **地图数据**：前端自动从 `geojson.cn` 加载中国省级 GeoJSON，并缓存到 `localStorage`。
- **省份规范化**：后端将数据中的 `WORK_PROVINCE` 字段（可能混用简称、全称或脏数据）统一映射为标准全称，与 GeoJSON 中的 `fullname` 匹配。
- **多指标切换**：支持在客户数量、总飞行里程、总航班次数、总积分、平均折扣率之间切换展示。
- **自适应布局**：地图容器占满整个内容区，支持缩放与拖拽。
- **下钻客户明细**：点击任意省份，从右侧滑出抽屉，查看该地区客户明细。

### 5. 省份客户明细抽屉

点击地图省份后，右侧滑出客户明细面板：

- **分页加载**：默认每页 20 条，最大每页 100 条，避免一次性返回全量数据。
- **按当前指标排序**：支持升序/降序切换，排序字段与当前地图指标保持一致。
- **未知统计**：抽屉顶部显示该省份总记录数，并单独展示无法识别/无效的省份人数。
- **轻量字段**：返回会员号、城市、当前指标值、价值标签等核心字段。

### 6. 抗幻觉机制

| 机制 | 说明 |
| --- | --- |
| **Schema 强制锚定** | AI 编写 SQL 前必须调用 `get_database_schema` |
| **数据枚举值嗅探** | WHERE 条件前必须调用 `probe_distinct_values` |
| **完整字段白名单** | SQL 提示中明确禁用字段（city、update_time、姓名等） |
| **动态查询提示** | 根据关键词注入针对性指导（"年龄区间"→ CASE WHEN 分段） |
| **SQL 自纠错** | 执行失败时自动将错误堆栈反馈给 LLM 重新生成 |
| **三表别名规范** | 严格规范 b→customer_base、f→customer_flight_summary、a→customer_analytics |

### 7. 对话上下文记忆

Agent 自动注入最近 6 轮对话历史到 LLM，支持指代理解：
- 用户："高价值客户的年龄分布"
- 用户：**"根据上面**画个柱状图" ← Agent 能正确理解为上一条查询的可视化

### 8. 工具调用限流

每个工具最多调用 3 次（`MAX_TOOL_CALLS = 3`），避免 token 浪费与死循环。

### 9. 报告导出

| 格式 | 实现 | 特性 |
| --- | --- | --- |
| **PDF** | html2canvas + jsPDF | 按消息节点逐个截图 + 智能分页（每页 18mm 边距 + 页码） |
| **Excel** | openpyxl | 多表导出，每个查询一个 Sheet，含列宽与样式 |
| **图表 PNG** | ECharts `getDataURL` | 单图表导出 |

### 10. 移动端响应式

- 抽屉式侧边栏 + 遮罩
- 自适应断点
- 触摸友好交互

---

## 三、技术栈

| 类别 | 技术 |
| --- | --- |
| **后端** | Python 3.10 + Flask 3 + mcp SDK + FastMCP |
| **数据库** | MySQL 8.0+ / SQLite 沙箱（自动降级） |
| **算法** | scikit-learn（K-Means、StandardScaler、IsolationForest） |
| **数据处理** | pandas、numpy、SQLAlchemy |
| **可视化** | ECharts 5.5（前端）、matplotlib（后端） |
| **LLM 集成** | OpenAI 兼容 API（DeepSeek、GPT）、Ollama（本地） |
| **前端** | HTML + Tailwind CSS + Alpine.js 3.14 + Marked.js |
| **PDF/Excel** | html2canvas + jsPDF + openpyxl |

---

## 四、项目结构

```
代码/
├── app.py                      # Flask Web 服务主入口（5000 端口）
├── api.py                      # MCP Server 独立进程（5001 端口）
├── main.py                     # 数据分析流水线
├── config.py                   # 全局配置（API key 默认空，从 config.json 加载）
├── database.py                 # 数据库连接与会话管理
├── agent.py                    # LLM Agent 核心（Function Calling / ReAct）
├── mcp_client.py               # MCP HTTP 客户端
├── init.sql                    # MySQL 建表脚本
├── requirements.txt            # Python 依赖
│
├── core_algorithm/             # 核心算法模块
│   ├── data_cleaning.py        # 数据清洗流水线
│   └── rfm_analysis.py         # RFM 计算与 K-Means 聚类
│
├── mcp_tools/                  # MCP 工具实现
│   ├── probing_tools.py        # Schema 锚定 + 数据嗅探（4 函数）
│   ├── sql_tools.py            # SQL 沙箱 + 校验 + 自纠错装饰器
│   ├── analysis_tools.py       # 动态聚类 + 交叉校验 + 画像分析
│   ├── chart_tools.py          # 16 种 ECharts 图表构造器
│   └── forecast_tools.py       # 趋势预测 + 异常检测 + 流失风险
│
├── templates/
│   └── dashboard.html          # 单页应用（含 Alpine.js 状态、PDF/Excel 导出）
│
├── output/                     # 运行时输出（沙箱数据库、图表）
│
├── docs/                       # 项目文档
│   └── 数据库设计.md
│
├── readme合集/                 # 旧版本 README 与文档
│
└── 文档/                        # 课程设计文档
```

---

## 五、快速开始

### 1. 环境要求

- **Python 3.10**（必须，3.14+ 不兼容）
- MySQL 8.0+（可选，未配置时自动使用 SQLite 沙箱）

### 2. 安装依赖

```bash
py -3.10 -m pip install -r requirements.txt
```

### 3. 配置 LLM API

**方式 A：前端界面配置（推荐）**
1. 启动 Web 后访问 http://127.0.0.1:5000
2. 点击右上角"配置"按钮
3. 填入 API URL、API Key、模型名称
4. 自动保存到 `config.json`（已加入 `.gitignore`）

**方式 B：手动编辑 `config.json`**

```json
{
  "llm_provider": "openai_compatible",
  "openai_url": "https://api.deepseek.com/v1/chat/completions",
  "openai_key": "sk-your-key-here",
  "openai_model": "deepseek-chat"
}
```

### 4. 初始化数据库

```bash
# 方式 A：MySQL
mysql -u root -p < init.sql
py -3.10 main.py            # 跑数据分析流水线

# 方式 B：SQLite 沙箱（无需 MySQL）
py -3.10 main.py            # 自动降级到 SQLite
```

### 5. 启动服务

需要启动**两个进程**（MCP Server + Flask Web）：

**终端 1 — MCP Server**
```bash
py -3.10 api.py
# 输出: 航空公司客户分析 — MCP Server (FastMCP)
# 地址: http://127.0.0.1:5001
```

**终端 2 — Flask Web**
```bash
py -3.10 app.py
# 输出: Running on http://127.0.0.1:5000
```

### 6. 访问

打开浏览器访问 http://127.0.0.1:5000 即可使用智能仪表盘。

---

## 六、使用示例

### 自然语言查询

```
"高价值客户的平均年龄是多少？"
"各聚类的客户数量分布"
"飞行里程 Top 10 的客户信息"
"不同会员等级的 RFM 得分对比（柱状图）"
"分析客户流失风险"                          → 调用 churn_risk_score
"预测未来 3 个月的飞行趋势"                  → 调用 forecast_trend
"检测里程数据的异常值"                       → 调用 detect_anomalies
```

### 对话上下文记忆

```
轮 1: "各价值标签的客户数量"
轮 2: "根据上面画个饼图"                     ← 自动理解上文
轮 3: "流失客户的平均年龄是多少"
```

### 地理分布

```
1. 点击左侧菜单「地理分布」
2. 在顶部下拉框切换展示指标：客户数量 / 总飞行里程 / 总航班次数 / 总积分 / 平均折扣率
3. 鼠标悬停省份查看数值
4. 点击任意省份 → 右侧滑出该省客户明细抽屉
5. 在抽屉内切换升序/降序、翻页查看
```

### 图表类型动态切换

每条消息中的图表支持前端下拉菜单切换类型，实时调用 `/api/chart/generate` 重新渲染。

### 报告导出

- **PDF**：右上角菜单 → 导出报告 → PDF（按消息节点分块截图 + 智能分页）
- **Excel**：右上角菜单 → 导出报告 → Excel（多 Sheet 表格数据）
- **图表 PNG**：图表右上角下载按钮

---

## 七、API 端点

### Web（Flask，5000 端口）

| 端点 | 方法 | 功能 |
| --- | --- | --- |
| `/` | GET | 仪表盘首页 |
| `/api/health` | GET | 健康检查 |
| `/api/schema` | GET | 完整数据库 Schema |
| `/api/schema/<table>` | GET | 单表 Schema |
| `/api/probe/<column>` | GET | 嗅探字段枚举值 |
| `/api/business-rules` | GET | 业务规则 |
| `/api/stats` | GET | 概要统计 |
| `/api/data` | GET | 原始数据预览 |
| `/api/clusters` | GET | 聚类画像 |
| `/api/sql/execute` | POST | SQL 直接执行 |
| `/api/cluster/dynamic` | POST | 动态聚类 |
| `/api/tools/forecast` | POST | 趋势预测 |
| `/api/tools/anomaly` | POST | 异常检测 |
| `/api/tools/churn` | POST | 流失风险 |
| `/api/tools` | GET | 工具列表 |
| `/api/tools/<name>` | POST | 工具调用 |
| `/api/chat` | POST | **核心端点**：自然语言查询 |
| `/api/chat/page` | POST | 翻页查询（缓存） |
| `/api/chart/generate` | POST | 动态生成图表 |
| `/api/export/excel` | POST | 导出 Excel |
| `/api/mcp/health` | GET | MCP 健康代理（解决 CORS） |
| `/api/config` | GET/POST | LLM 配置 |
| `/api/pipeline/run` | POST | 重跑数据流水线 |
| `/api/map/province` | GET | 按省份聚合客户指标 |
| `/api/map/province/customers` | GET | 省份客户明细（分页、排序、未知统计） |

### MCP Server（5001 端口）

| 端点 | 方法 | 功能 |
| --- | --- | --- |
| `/sse` | GET | MCP SSE 连接 |
| `/messages/` | POST | MCP 消息端点 |
| `/tools/list` | GET | 工具列表 |
| `/tools/call` | POST | 工具调用 |
| `/health` | GET | 健康检查 |

---

## 八、关键工程决策

### 1. 为什么使用 MCP 工具调用而非 Prompt 拼接？

- **确定性**：工具返回结构化数据，减少 LLM 幻觉
- **可追溯**：每次工具调用记录完整参数与结果
- **可扩展**：新增工具不影响现有调用链
- **自纠错**：错误自动反馈并触发重试
- **可观测**：每个步骤可独立调试

### 2. 为什么禁用物理外键？

- 解耦表结构，支持独立更新
- 避免外键约束导致的写入失败
- 应用层通过 JOIN 实现逻辑关联

### 3. 为什么 MCP Server 与 Flask Web 分离部署？

- **职责清晰**：MCP 专注工具能力，Web 专注 UI 编排
- **可独立扩展**：未来可多 Web 客户端共享同一 MCP
- **协议标准化**：未来可被 Claude Desktop、Cherry Studio 等 MCP 客户端直接消费

### 4. 分页与图表优化

| 优化点 | 阈值 | 说明 |
| --- | --- | --- |
| 分页 | > 20 行 | 防止前端渲染卡顿 |
| 树图扁平化 | > 15 类 | Top 15 + "其他" |
| 工具调用 | ≤ 3 次/工具 | 避免 token 浪费 |
| Agent 迭代 | FC: 20 / ReAct: 15 | 防止死循环 |
| 历史注入 | 最近 6 轮 | 控制 prompt 长度 |

### 5. PDF 导出实现

按消息节点逐个截图，智能分页：
1. 每条消息（用户问题 / Agent 回答）作为一个节点
2. html2canvas 单独截屏
3. 计算高度，超长消息自动切分到多页
4. 每页 18mm 边距 + 页码 + 报告标题

---

## 九、安全与配置

### API Key 安全

- `config.py` 中所有 API key 默认值为空字符串
- 运行时配置持久化到 `config.json`（已加入 `.gitignore`）
- 前端配置保存逻辑：空输入会清空已有 key，掩码值（`sk-****`）不更新

### SQL 沙箱

- 仅允许 `SELECT` 语句
- 黑名单关键词：`DELETE/DROP/TRUNCATE/INSERT/UPDATE/ALTER/CREATE/GRANT/REPLACE/EXEC`
- 禁止多语句执行（sqlparse 校验）
- 沙箱文件自动重试释放，避免 PermissionError

### 数据库连接

- MySQL 优先，自动降级到 SQLite
- 沙箱文件删除失败时重命名备份
- 引擎句柄主动 `dispose()` 释放连接

---

## 十、常见问题

### Q: 为什么必须使用 Python 3.10？
部分依赖（pandas、scikit-learn 等）在 3.14+ 存在兼容性问题。Windows 下可用 `py -3.10` 指定版本。

### Q: LLM 生成的 SQL 包含不存在的字段（如 city、姓名）？
- 系统已在 SQL 提示中明确禁用字段列表
- Agent 必须先调用 `get_database_schema` 了解真实字段
- 仍出现时检查 LLM 是否真正调用了工具链

### Q: 启动 Flask 时报 `AssertionError: View function mapping is overwriting`？
重复定义了同名端点函数。检查 `@app.route("/api/...")` 装饰的函数名是否唯一。

### Q: MCP Server 启动后 Flask 报 `MCP Server 不可用`？
- 确认 5001 端口未被占用
- 访问 http://127.0.0.1:5001/health 检查健康状态
- 通过 `/api/mcp/health` 代理端点测试

### Q: 对话上下文记忆不生效？
- 确认前端 `submitQuery` 正确发送 `history` 数组
- Agent 自动注入最近 6 轮到 prompt
- 对话刷新后会重置（localStorage 持久化的是 messages，不是 history）

### Q: 树图显示拥挤？
已自动启用 Top 15 + Other 扁平化，可调整 `chart_tools.py` 中 `MAX_ITEMS`。

### Q: 地图页面空白或只显示轮廓？
- 检查浏览器能否访问 `https://geojson.cn/api/china/100000.json`
- 查看控制台是否有 CORS 或网络错误
- 如处于内网无法访问公网，可下载该 GeoJSON 保存到 `static/china.json`，并将 `dashboard.html` 中 `loadChinaMap()` 的 URL 改为 `/static/china.json`
- 确认 `final_result.csv` 中存在 `WORK_PROVINCE` 字段

### Q: 点击省份后抽屉没有数据？
- 确认 `/api/map/province/customers?province=...` 返回成功
- 检查省份名称是否已规范化（后台日志会打印未知/无效省份数量）

### Q: 如何修改聚类数 K？
`config.py` 中调整 `N_CLUSTERS = 4`，重跑 `python main.py`。

---

## 十一、扩展路线

- [x] MCP 工具调用架构
- [x] 16 种 ECharts 图表
- [x] PDF/Excel 报告导出
- [x] 移动端响应式
- [x] 对话上下文记忆
- [x] 工具调用限流
- [x] 地理分布可视化（中国地图 + 省份客户明细）
- [ ] 接入 Claude Desktop 等标准 MCP 客户端
- [ ] 多用户权限管理
- [ ] 流式响应（SSE）
- [ ] 定时任务（每日自动跑流水线）
- [ ] 看板自定义配置

---

## 十二、测试验证

```bash
# 1. 验证所有依赖与导入
py -3.10 -c "import app, api, agent; from mcp_tools import sql_tools, chart_tools, analysis_tools, forecast_tools, probing_tools"

# 2. 启动 MCP Server
py -3.10 api.py
# 终端 2: 测试健康
curl http://127.0.0.1:5001/health
curl http://127.0.0.1:5001/tools/list

# 3. 启动 Flask Web
py -3.10 app.py
# 终端 2: 测试首页
curl http://127.0.0.1:5000/api/tools
curl http://127.0.0.1:5000/api/schema
```

---

## 十三、许可证

本项目为暑期实践项目，仅供学习与研究使用。

---

## 十四、参考

- **MCP 协议**：https://modelcontextprotocol.io/
- **FastMCP**：[mcp SDK](https://github.com/modelcontextprotocol/python-sdk)
- **ECharts**：https://echarts.apache.org/
- **DeepSeek API**：https://platform.deepseek.com/
- **项目文档**：[docs/](docs/) 与 [readme合集/](readme合集/)
