"""
MCP 工具服务器 (Model Context Protocol) — api.py
===============================================
独立的 MCP Server，基于 mcp SDK 的 FastMCP。
运行在 5001 端口，与 Flask Web 仪表盘 (5000) 完全解耦。

启动方式:
  python api.py
  python api.py --port 5001

MCP 端点 (SSE 传输):
  /sse            — SSE 连接端点
  /messages/      — 消息端点
  /tools/list     — 工具列表 (HTTP GET)
  /tools/call     — 工具调用 (HTTP POST)
  /health         — 健康检查
"""
import sys
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import config
from mcp.server.fastmcp import FastMCP

# ==================== 解析 CLI 参数 ====================
parser = argparse.ArgumentParser(description="航空公司客户分析 MCP 服务器")
parser.add_argument("--port", type=int, default=5001, help="SSE 服务端口 (默认 5001)")
parser.add_argument("--host", type=str, default="127.0.0.1", help="绑定地址")
_cli_args, _ = parser.parse_known_args()

# ==================== 初始化 MCP Server ====================
mcp = FastMCP(
    name="airline-analytics-mcp",
    instructions="航空公司客户分析 MCP 服务器 — 提供 Schema 锚定、数据嗅探、SQL 沙箱、动态聚类、智能图表生成工具",
    host=_cli_args.host,
    port=_cli_args.port,
)


# ==================== 工具注册 ====================

@mcp.tool()
def get_database_schema() -> dict:
    """获取完整的数据库表结构、字段定义和业务注释。编写任何 SQL 前必须调用此工具防止字段名幻觉。"""
    from mcp_tools.probing_tools import get_database_schema as _fn
    return _fn()


@mcp.tool()
def get_table_schema(table_name: str) -> dict:
    """获取指定单张表的结构信息（字段名、类型、注释）。

    Args:
        table_name: 表名，可选值: customer_base, customer_flight_summary, customer_analytics
    """
    from mcp_tools.probing_tools import get_table_schema as _fn
    return _fn(table_name)


@mcp.tool()
def probe_distinct_values(column_name: str, table_name: str = None) -> dict:
    """嗅探指定字段的真实枚举值。构建 WHERE 条件前必须调用，确认数据中的实际取值。

    Args:
        column_name: 字段名，如 value_label, gender, cluster
        table_name: 表名（可选，系统自动推断）
    """
    from mcp_tools.probing_tools import probe_distinct_values as _fn
    return _fn(column_name, table_name)


@mcp.tool()
def execute_secure_sql(sql_query: str) -> dict:
    """在 SQLite 沙箱中安全执行 SELECT 查询。禁止 INSERT/UPDATE/DELETE 等写操作。
    如果 SQL 报错，返回完整错误堆栈供 AI 自动修正。

    Args:
        sql_query: 纯 SELECT SQL 语句
    """
    from mcp_tools.sql_tools import execute_secure_sql as _fn
    return _fn(sql_query, sandbox_db_path=config.SQLITE_PATH)


@mcp.tool()
def perform_dynamic_cluster(data_json: str, k: int = 4,
                            features: list = None) -> dict:
    """对传入的 JSON 数据集执行 Log1p + StandardScaler + KMeans 聚类分析。
    返回各簇标签、轮廓系数和统计信息。

    Args:
        data_json: JSON 数组字符串，如 '[{"Recency":10,"Frequency":5,"Monetary":1000},...]'
        k: 聚类簇数，默认 4
        features: 用于聚类的特征列名列表，默认 ["Recency","Frequency","Monetary"]
    """
    from mcp_tools.analysis_tools import perform_dynamic_cluster as _fn
    return _fn(data_json, k=k, features=features)


@mcp.tool()
def generate_visualization(data_json: str, chart_type: str = "auto",
                           title: str = "", purpose: str = "") -> dict:
    """根据数据自动选择最合适的图表类型，生成 ECharts 配置 JSON 供前端渲染。
    支持 bar / pie / line / scatter / radar / heatmap。

    Args:
        data_json: JSON 数组数据
        chart_type: 图表类型 (auto/bar/pie/line/scatter/radar/heatmap)，默认自动选型
        title: 图表标题
        purpose: 分析目的 (comparison/distribution/composition/relationship)
    """
    from mcp_tools.chart_tools import generate_visualization as _fn
    return _fn(data_json, chart_type=chart_type, title=title, purpose=purpose)


@mcp.tool()
def analyze_cluster_profiles(df_json: str) -> dict:
    """对聚类结果进行业务画像分析。输出每群客户的特征描述和推荐营销策略。

    Args:
        df_json: 包含 cluster 列的 JSON 数组字符串
    """
    from mcp_tools.analysis_tools import analyze_cluster_profiles as _fn
    return _fn(df_json)


@mcp.tool()
def verify_data_logic(sql_result_json: str, pandas_result_json: str,
                      tolerance: float = None) -> dict:
    """交叉校验：对比 SQL 聚合结果与 Pandas 计算结果是否一致。
    对 RFM 均值、聚类占比等核心指标进行双重逻辑验证。

    Args:
        sql_result_json: SQL 聚合结果的 JSON 字符串
        pandas_result_json: Pandas 计算结果的 JSON 字符串
        tolerance: 允许的差异百分比，默认 0.01 (1%)
    """
    from mcp_tools.analysis_tools import verify_data_logic as _fn
    return _fn(sql_result_json, pandas_result_json, tolerance)


@mcp.tool()
def get_business_rules() -> dict:
    """获取业务规则定义：价值标签分段逻辑、RFM 评分含义、聚类标签解释等。"""
    from mcp_tools.probing_tools import get_business_rules as _fn
    return _fn()


# ==================== 健康检查端点 ====================

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """健康检查端点，供 app.py 检测 MCP Server 状态"""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "status": "ok",
        "server": "airline-analytics-mcp",
        "tools_count": 9,
        "transport": "sse"
    })


# ==================== REST API 端点（供 app.py 调用） ====================

@mcp.custom_route("/tools/list", methods=["GET"])
async def list_tools_rest(request):
    """列出所有可用工具（REST API）"""
    from starlette.responses import JSONResponse
    try:
        # 获取所有注册的工具
        tools_info = []
        for name, tool in mcp._tool_manager._tools.items():
            tools_info.append({
                "name": name,
                "description": tool.description or "",
                "inputSchema": tool.parameters if hasattr(tool, 'parameters') else {}
            })
        return JSONResponse({"tools": tools_info, "count": len(tools_info)})
    except Exception as e:
        return JSONResponse({"error": str(e), "tools": []}, status_code=500)


@mcp.custom_route("/tools/call", methods=["POST"])
async def call_tool_rest(request):
    """调用工具（REST API）- 解析 MCP 标准格式，返回纯 JSON"""
    from starlette.responses import JSONResponse
    try:
        data = await request.json()
        tool_name = data.get("name")
        arguments = data.get("arguments", {})
        
        if not tool_name:
            return JSONResponse({"error": "缺少工具名称"}, status_code=400)
        
        # 检查工具是否存在
        if tool_name not in mcp._tool_manager._tools:
            return JSONResponse({"error": f"工具 {tool_name} 不存在"}, status_code=404)
        
        # 调用工具
        result = await mcp._tool_manager.call_tool(tool_name, arguments)
        
        # 解析 MCP 标准返回格式 [{"type":"text","text":"..."}]
        parsed_result = {}
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and item.get("type") == "text":
                    try:
                        parsed_result = json.loads(item["text"])
                    except (json.JSONDecodeError, TypeError):
                        parsed_result = {"result": item["text"]}
                    break
        elif isinstance(result, dict):
            parsed_result = result
        else:
            parsed_result = {"result": str(result)}
        
        return JSONResponse({
            "success": True,
            "tool": tool_name,
            "result": parsed_result  # 直接返回解析后的字典
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "tool": data.get("name") if 'data' in locals() else None
        }, status_code=500)


# ==================== 启动 ====================

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  航空公司客户分析 — MCP Server (FastMCP)")
    print(f"  地址: http://{_cli_args.host}:{_cli_args.port}")
    print(f"  SSE 端点: http://{_cli_args.host}:{_cli_args.port}/sse")
    print(f"  消息端点: http://{_cli_args.host}:{_cli_args.port}/messages/")
    print(f"{'='*55}\n")

    mcp.run(transport="sse")
