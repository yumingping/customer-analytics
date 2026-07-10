# 客户分析与智能查询系统

基于 RFM 模型与 K-Means 聚类的航空公司客户价值分析平台，采用 MCP（Model Context Protocol）工具调用架构，支持自然语言查询与智能可视化。

## 项目概述

本系统针对航空公司客户数据（2014 年观察期），构建了完整的客户价值分层与智能分析体系。通过 RFM 模型量化客户价值，利用 K-Means 算法进行聚类分层，并结合大语言模型实现自然语言驱动的数据查询与可视化。

## 核心特性

### 1. MCP 工具调用架构

摒弃传统的 Prompt 拼接方式，采用结构化的工具调用机制：

- **Schema 锚定工具**：强制 AI 获取真实表结构，杜绝字段幻觉
- **数据嗅探工具**：枚举字段实际取值，避免 WHERE 条件错误
- **SQL 沙箱执行**：隔离环境验证 SQL，自动捕获并反馈错误
- **动态聚类工具**：支持即时 K-Means 聚类与交叉验证
- **智能图表工具**：根据数据特征自动选择图表类型，生成 ECharts 配置

### 2. 三表分层架构

采用数据仓库分层设计，业务主键 `member_no` 关联：

- **customer_base**（维度表）：客户基础信息（年龄、性别、会员等级等）
- **customer_flight_summary**（事实表）：飞行行为统计（飞行次数、里程、Recency）
- **customer_analytics**（应用表）：RFM 评分、聚类标签、价值分层

### 3. 智能查询与可视化

- 支持自然语言提问，自动转换为 SQL 查询
- 结果分页展示（每页 20 条）
- 自动生成 ECharts 图表（柱状图、饼图、树图等）
- 树图自动扁平化（Top 15 + Other）

### 4. 抗幻觉机制

- **完整 Schema 提示**：SQL 系统提示包含所有字段定义、禁用字段列表、别名规则
- **动态查询提示**：根据问题关键词（如"年龄区间"、"城市"）注入针对性指导
- **SQL 自纠功能**：执行失败时自动将错误反馈给 LLM 重新生成
- **交叉验证**：核心指标双重计算校验

## 技术栈

- **后端**：Python 3.10 + Flask
- **数据库**：MySQL 8.0+ / SQLite（沙箱）
- **算法**：scikit-learn（K-Means、StandardScaler）
- **数据处理**：pandas、numpy
- **可视化**：ECharts（前端）、matplotlib（后端）
- **LLM 集成**：OpenAI 兼容 API（DeepSeek 等）
- **前端**：HTML + Tailwind CSS + Alpine.js

## 项目结构

```
代码/
├── app.py                      # Flask Web 服务主入口
├── main.py                     # 数据分析流水线（清洗→RFM→聚类→入库）
├── config.py                   # 全局配置（数据库、LLM、聚类参数）
├── database.py                 # 数据库连接与会话管理
├── agent.py                    # LLM 调用封装
├── init.sql                    # MySQL 建表脚本
├── requirements.txt            # Python 依赖
│
├── core_algorithm/             # 核心算法模块
│   ├── data_cleaning.py        # 数据清洗流水线
│   └── rfm_analysis.py         # RFM 计算与 K-Means 聚类
│
├── mcp_tools/                  # MCP 工具集
│   ├── probing_tools.py        # Schema 查询与数据嗅探
│   ├── sql_tools.py            # SQL 沙箱执行与纠错
│   ├── analysis_tools.py       # 动态聚类与交叉验证
│   ├── chart_tools.py          # 智能图表生成
│   └── forecast_tools.py       # 趋势预测与异常检测
│
├── templates/
│   └── dashboard.html          # 前端仪表盘界面
│
└── output/                     # 运行时输出（图表、沙箱数据库）
```

## 快速开始

### 1. 环境要求

- Python 3.10（**必须**，不兼容 3.14+）
- MySQL 8.0+（可选，默认使用 SQLite）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置数据库

编辑 `config.py` 或通过前端界面配置：

```python
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "your_password",
    "database": "airline_analytics",
}
```

初始化数据库：

```bash
mysql -u root -p < init.sql
```

### 4. 运行数据分析流水线

执行数据清洗、RFM 计算、聚类分析并写入数据库：

```bash
python main.py
```

输出示例：
```
[数据清洗] 加载原始数据: 62987 条记录
[RFM 计算] 完成 Recency/Frequency/Monetary 特征提取
[K-Means] 最优 K=4, 轮廓系数=0.68
[数据入库] 三表写入完成: customer_base, customer_flight_summary, customer_analytics
```

### 5. 启动 Web 服务

```bash
python app.py
```

访问 http://127.0.0.1:5000 打开智能仪表盘。

## 使用指南

### 自然语言查询

在对话框中输入问题，系统自动解析并返回结果：

- "高价值客户的平均年龄是多少？"
- "各聚类的客户数量分布"
- "飞行里程 Top 10 的客户信息"
- "不同会员等级的 RFM 得分对比（柱状图）"

### 可视化功能

- **自动图表生成**：根据数据特征选择柱状图、饼图、折线图等
- **树图优化**：超过 15 个分类自动合并为 Top 15 + Other
- **结果分页**：大数据集自动分页展示（每页 20 条）

### API 配置

首次使用需配置 LLM API：

1. 访问 http://127.0.0.1:5000
2. 点击右上角"配置"按钮
3. 填入 API URL、API Key、模型名称
4. 配置保存在 `config.json`（已加入 .gitignore）

## 数据库设计

详见 [docs/数据库设计.md](docs/数据库设计.md)

### 核心表结构

**customer_base**（客户基础信息）
- member_no（会员编号，主键）
- age, gender, ffp_tier（会员等级）
- avg_discount, ffp_date, first_flight

**customer_flight_summary**（飞行行为统计）
- member_no（关联键）
- flight_count（飞行次数）
- seg_km_sum（飞行里程）
- recency（距观察期末天数）

**customer_analytics**（分析结果）
- member_no（关联键）
- r_score, f_score, m_score（RFM 评分 1-5）
- rfm_total（RFM 总分 3-15）
- cluster（聚类标签 0-3）
- value_label（价值标签：高价值/中价值/低价值/流失客户）

## 关键工程决策

### 1. 为什么使用业务主键而非自增 ID？

- 避免数据合并时的主键冲突
- 提升 ETL 写入效率
- 便于跨表关联与数据追踪

### 2. 为什么禁用物理外键？

- 解耦表结构，支持独立更新
- 避免外键约束导致的写入失败
- 应用层通过 JOIN 实现逻辑关联

### 3. 为什么采用 MCP 工具调用而非 Prompt 拼接？

- **确定性**：工具返回结构化数据，减少幻觉
- **可追溯**：每次调用记录完整上下文
- **可扩展**：新增工具不影响现有逻辑
- **自纠错**：错误自动反馈并触发重试

### 4. 分页与图表优化

- **分页**：防止前端渲染卡顿（>20 条数据自动分页）
- **树图扁平化**：超过 15 个分类时自动合并，避免视觉拥挤
- **缓存机制**：查询结果缓存 10 分钟，支持翻页浏览

## 常见问题

### Q: 为什么必须使用 Python 3.10？

部分依赖库（如特定版本的 scikit-learn、pandas）在 Python 3.14+ 存在兼容性问题。

### Q: LLM 生成的 SQL 包含不存在的字段（如 city、update_time）？

系统已在 SQL 提示中明确禁用字段列表。若仍出现，检查 LLM 是否正确使用 `get_database_schema` 工具。

### Q: 树图显示拥挤或重叠？

自动扁平化逻辑已启用（Top 15 + Other）。若仍有问题，调整 `chart_tools.py` 中的 `leafDepth` 参数。

### Q: 如何修改聚类数量？

编辑 `config.py` 中的 `N_CLUSTERS = 4`，或在前端配置界面调整。

## 开发计划

- [ ] 支持更多图表类型（散点图、雷达图）
- [ ] 增加时间序列分析工具
- [ ] 实现用户权限管理
- [ ] 导出分析报告（PDF/Word）
- [ ] 移动端适配

## 许可证

本项目为暑期实践项目，仅供学习与研究使用。

## 联系方式

如有问题或建议，请通过 GitHub Issues 反馈。
