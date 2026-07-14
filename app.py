"""
Flask Web 后端 — MCP 工具调用架构版
===================================
- MCP 工具注册表 (Tool Registry)
- /api/chat 核心端点: 自然语言查询走完整 MCP 调用链
- /api/schema /api/probe /api/sql/execute 等独立工具端点
- 多 LLM 提供商支持 (Ollama / OpenAI 兼容 API)
"""
import os
import re
import json
import time
import traceback
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from pathlib import Path
import config

app = Flask(__name__)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

FINAL_PATH = config.OUTPUT_DIR / "final_result.csv"

# ==================== MCP 工具注册表 ====================

TOOL_REGISTRY = {}  # 延迟初始化

_QUERY_CACHE = {}
_QUERY_CACHE_MAX = 50
_QUERY_CACHE_TTL = 600

PIPELINE_STATUS = {"running": False, "progress": "", "error": None}


def _init_tool_registry():
    """延迟初始化工具注册表，避免循环引用"""
    global TOOL_REGISTRY
    if TOOL_REGISTRY:
        return

    from mcp_tools.probing_tools import get_database_schema, probe_distinct_values, get_table_schema, get_business_rules
    from mcp_tools.sql_tools import execute_secure_sql, validate_sql_syntax, llm_fix_sql, setup_sandbox_from_csv
    from mcp_tools.analysis_tools import perform_dynamic_cluster, verify_data_logic, analyze_cluster_profiles
    from mcp_tools.chart_tools import generate_visualization, recommend_chart_type
    from mcp_tools.forecast_tools import forecast_trend, detect_anomalies, churn_risk_score

    TOOL_REGISTRY.update({
        "get_database_schema": {
            "function": get_database_schema,
            "description": "获取完整的数据库表结构、字段定义和业务注释（防幻觉核心工具）",
            "parameters": {},
        },
        "get_table_schema": {
            "function": get_table_schema,
            "description": "获取单张表的结构信息",
            "parameters": {"table_name": "str — 表名"},
        },
        "probe_distinct_values": {
            "function": probe_distinct_values,
            "description": "嗅探指定列的真实枚举值，杜绝AI凭空构造 WHERE 条件值",
            "parameters": {"column_name": "str — 字段名", "table_name": "str (可选) — 表名"},
        },
        "execute_secure_sql": {
            "function": lambda sql_query=None, **kwargs: execute_secure_sql(
                sql_query or kwargs.get("sql_query") or kwargs.get("sql"),
                sandbox_db_path=config.SQLITE_PATH
            ),
            "description": "在 SQLite 沙箱安全执行 SELECT，捕获并返回错误堆栈供 AI 自纠",
            "parameters": {"sql_query": "str — 纯 SELECT SQL 语句"},
        },
        "perform_dynamic_cluster": {
            "function": perform_dynamic_cluster,
            "description": "对传入的 JSON 数据集即时执行 Log1p+StandardScaler+KMeans 聚类",
            "parameters": {"data_json": "str — JSON 数组", "k": "int (默认4) — 聚类数", "features": "list (可选) — 特征列"},
        },
        "generate_visualization": {
            "function": generate_visualization,
            "description": "根据数据自动选择图表类型并生成 ECharts 配置 JSON。支持 bar/pie/line/scatter/radar/heatmap/funnel/boxplot/gauge/stacked_bar/area_line/treemap/pareto。当用户要求可视化、画图、趋势、分布、排名、漏斗、仪表盘等场景时，必须调用此工具。",
            "parameters": {"data_json": "str — JSON 数组", "chart_type": "str (默认auto) — 图表类型", "title": "str — 标题", "purpose": "str (可选) — 分析目的: comparison/distribution/composition/relationship/trend/ranking/funnel"},
        },
        "verify_data_logic": {
            "function": verify_data_logic,
            "description": "交叉校验：对比 SQL 聚合与 Pandas 计算结果，确保数据准确",
            "parameters": {"sql_result_json": "str", "pandas_result_json": "str", "tolerance": "float (可选)"},
        },
        "analyze_cluster_profiles": {
            "function": analyze_cluster_profiles,
            "description": "对聚类结果进行业务画像分析，输出每群客户的描述和营销策略",
            "parameters": {"df_json": "str — 带 cluster 列的 JSON 数组"},
        },
        "get_business_rules": {
            "function": get_business_rules,
            "description": "获取业务规则（价值标签分段、评分含义等）",
            "parameters": {},
        },
        "forecast_trend": {
            "function": forecast_trend,
            "description": "对时序数据进行趋势预测（线性回归 / 指数平滑），返回历史值与预测值",
            "parameters": {"data_json": "str — JSON 数组", "date_col": "str (可选) — 时间列", "value_col": "str (可选) — 数值列", "periods": "int (默认3) — 预测期数", "method": "str (默认linear) — linear 或 exponential_smoothing"},
        },
        "detect_anomalies": {
            "function": detect_anomalies,
            "description": "检测数据中的异常值，支持 IQR 和 IsolationForest 方法",
            "parameters": {"data_json": "str — JSON 数组", "columns": "list (可选) — 检测列", "method": "str (默认iqr) — iqr 或 isolation_forest", "contamination": "float (默认0.05) — 异常比例"},
        },
        "churn_risk_score": {
            "function": churn_risk_score,
            "description": "基于 RFM 特征计算客户流失风险评分与风险分层",
            "parameters": {"data_json": "str — JSON 数组", "recency_col": "str (默认Recency)", "frequency_col": "str (默认Frequency)", "monetary_col": "str (默认Monetary)"},
        },
    })


# ==================== 工具描述 JSON（供 LLM Function Calling） ====================

def get_tools_for_llm() -> list:
    """返回 OpenAI Function Calling 格式的工具定义"""
    _init_tool_registry()
    return [
        {
            "type": "function",
            "function": {
                "name": "get_database_schema",
                "description": "获取完整的数据库表结构、字段定义和业务注释。编写任何 SQL 前必须调用。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        },
        {
            "type": "function",
            "function": {
                "name": "probe_distinct_values",
                "description": "嗅探指定字段的真实枚举值。构建 WHERE 条件前必须调用，防止值不存在。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column_name": {"type": "string", "description": "要嗅探的字段名，如 value_label, gender, cluster"},
                        "table_name": {"type": "string", "description": "表名（可选，自动推断）"},
                    },
                    "required": ["column_name"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "execute_secure_sql",
                "description": "在沙箱中安全执行 SQL SELECT 语句，返回结果数据。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_query": {"type": "string", "description": "要执行的纯 SELECT SQL"},
                    },
                    "required": ["sql_query"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_visualization",
                "description": "根据数据自动选择图表类型，生成 ECharts 配置 JSON 供前端渲染。当用户提到画图、可视化、趋势、漏斗、箱线图、仪表盘、帕累托、占比、分布等场景时必须调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_json": {"type": "string", "description": "JSON 数组数据"},
                        "chart_type": {"type": "string", "description": "图表类型: auto, bar, pie, line, scatter, radar, heatmap, funnel, boxplot, gauge, stacked_bar, area_line, treemap, pareto"},
                        "title": {"type": "string", "description": "图表标题"},
                        "purpose": {"type": "string", "description": "分析目的，可选 comparison/distribution/composition/relationship/trend/ranking/funnel"},
                    },
                    "required": ["data_json"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "perform_dynamic_cluster",
                "description": "对传入数据集执行 K-Means 聚类分析。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_json": {"type": "string", "description": "JSON 数组数据"},
                        "k": {"type": "integer", "description": "聚类数，默认4"},
                        "features": {"type": "array", "items": {"type": "string"}, "description": "用于聚类的特征列名"},
                    },
                    "required": ["data_json"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "forecast_trend",
                "description": "对时序数据进行趋势预测。当用户问未来趋势、预测、走势时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_json": {"type": "string", "description": "JSON 数组数据，必须包含时间列和数值列"},
                        "date_col": {"type": "string", "description": "时间列名"},
                        "value_col": {"type": "string", "description": "数值列名"},
                        "periods": {"type": "integer", "description": "预测期数，默认3"},
                        "method": {"type": "string", "description": "linear 或 exponential_smoothing"},
                    },
                    "required": ["data_json"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "detect_anomalies",
                "description": "检测数据中的异常值。当用户问异常、离群点、极端值时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_json": {"type": "string", "description": "JSON 数组数据"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "要检测的数值列"},
                        "method": {"type": "string", "description": "iqr 或 isolation_forest"},
                        "contamination": {"type": "number", "description": "异常比例，默认0.05"},
                    },
                    "required": ["data_json"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "churn_risk_score",
                "description": "基于 RFM 特征计算客户流失风险评分。当用户问谁会流失、流失风险、挽留客户时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_json": {"type": "string", "description": "JSON 数组数据，必须包含 Recency/Frequency/Monetary 列"},
                        "recency_col": {"type": "string", "description": "Recency 列名"},
                        "frequency_col": {"type": "string", "description": "Frequency 列名"},
                        "monetary_col": {"type": "string", "description": "Monetary 列名"},
                    },
                    "required": ["data_json"],
                },
            }
        },
    ]


# ==================== 帮助函数 ====================

def has_data() -> bool:
    return FINAL_PATH.exists()


def load_result() -> pd.DataFrame:
    return pd.read_csv(FINAL_PATH, encoding="utf-8-sig")


# 省份名称规范化（兼容简称、全称及少数民族自治区）
_PROVINCE_NAME_MAP = {
    '北京': '北京市', '北京市': '北京市',
    '天津': '天津市', '天津市': '天津市',
    '上海': '上海市', '上海市': '上海市',
    '重庆': '重庆市', '重庆市': '重庆市',
    '河北': '河北省', '河北省': '河北省',
    '山西': '山西省', '山西省': '山西省',
    '辽宁': '辽宁省', '辽宁省': '辽宁省',
    '吉林': '吉林省', '吉林省': '吉林省',
    '黑龙江': '黑龙江省', '黑龙江省': '黑龙江省',
    '江苏': '江苏省', '江苏省': '江苏省',
    '浙江': '浙江省', '浙江省': '浙江省',
    '安徽': '安徽省', '安徽省': '安徽省',
    '福建': '福建省', '福建省': '福建省',
    '江西': '江西省', '江西省': '江西省',
    '山东': '山东省', '山东省': '山东省',
    '河南': '河南省', '河南省': '河南省',
    '湖北': '湖北省', '湖北省': '湖北省',
    '湖南': '湖南省', '湖南省': '湖南省',
    '广东': '广东省', '广东省': '广东省',
    '海南': '海南省', '海南省': '海南省',
    '四川': '四川省', '四川省': '四川省',
    '贵州': '贵州省', '贵州省': '贵州省',
    '云南': '云南省', '云南省': '云南省',
    '陕西': '陕西省', '陕西省': '陕西省',
    '甘肃': '甘肃省', '甘肃省': '甘肃省',
    '青海': '青海省', '青海省': '青海省',
    '台湾': '台湾省', '台湾省': '台湾省',
    '内蒙古': '内蒙古自治区', '内蒙古自治区': '内蒙古自治区',
    '广西': '广西壮族自治区', '广西壮族自治区': '广西壮族自治区',
    '西藏': '西藏自治区', '西藏自治区': '西藏自治区',
    '宁夏': '宁夏回族自治区', '宁夏回族自治区': '宁夏回族自治区',
    '新疆': '新疆维吾尔自治区', '新疆维吾尔自治区': '新疆维吾尔自治区',
    '香港': '香港特别行政区', '香港特别行政区': '香港特别行政区',
    '澳门': '澳门特别行政区', '澳门特别行政区': '澳门特别行政区',
}


def _normalize_province(name: str) -> str:
    """将省份简称/不规范名称统一为 GeoJSON 标准全称；非标准名称返回 None。"""
    if not isinstance(name, str):
        return None
    name = name.strip()
    if name in ('', '未知', 'NA', 'N/A', 'NULL', 'None'):
        return None
    return _PROVINCE_NAME_MAP.get(name)


def _clean_query_cache():
    now = time.time()
    stale = [k for k, v in _QUERY_CACHE.items() if now - v["_time"] > _QUERY_CACHE_TTL]
    for k in stale:
        del _QUERY_CACHE[k]
    if len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
        sorted_keys = sorted(_QUERY_CACHE, key=lambda k: _QUERY_CACHE[k]["_time"])
        for k in sorted_keys[:len(_QUERY_CACHE) - _QUERY_CACHE_MAX]:
            del _QUERY_CACHE[k]


def _ensure_sandbox():
    """确保沙箱数据库有数据"""
    sandbox_path = Path(config.SQLITE_PATH)
    if not sandbox_path.exists() and has_data():
        from mcp_tools.sql_tools import setup_sandbox_from_csv
        setup_sandbox_from_csv()
    elif has_data() and sandbox_path.stat().st_size == 0:
        from mcp_tools.sql_tools import setup_sandbox_from_csv
        setup_sandbox_from_csv()


# ==================== 首页 ====================

@app.route("/")
def index():
    return render_template("dashboard.html")


# ==================== MCP 健康检查代理 ====================

@app.route("/api/mcp/health")
def api_mcp_health():
    """MCP 服务器健康检查代理（避免 CORS 问题）"""
    mcp_url = getattr(config, 'MCP_SERVER_URL', 'http://127.0.0.1:5001')
    try:
        r = requests.get(f"{mcp_url}/health", timeout=3)
        if r.status_code == 200:
            return jsonify(r.json())
        else:
            return jsonify({"status": "error", "message": f"HTTP {r.status_code}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ==================== 地理分布 API ====================

@app.route("/api/map/province", methods=["GET"])
def api_map_province():
    """按省份聚合客户指标，返回给 ECharts 地图使用。"""
    if not has_data():
        return jsonify({"success": False, "error": "暂无数据，请先上传或生成分析结果"}), 400

    try:
        df = load_result()
        province_col = None
        for col in ("WORK_PROVINCE", "PROVINCE", "work_province", "province"):
            if col in df.columns:
                province_col = col
                break

        if province_col is None:
            return jsonify({"success": False, "error": "缺少省份字段（WORK_PROVINCE/PROVINCE）"}), 400

        df = df.copy()
        df["_province_norm"] = df[province_col].apply(_normalize_province)
        df = df[df["_province_norm"].notna()]

        # 可选聚合指标
        agg = {"MEMBER_NO": "count"}
        rename = {"MEMBER_NO": "count", "_province_norm": "name"}

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if "SEG_KM_SUM" in numeric_cols:
            agg["SEG_KM_SUM"] = "sum"
            rename["SEG_KM_SUM"] = "total_km"
        if "avg_discount" in numeric_cols:
            agg["avg_discount"] = "mean"
            rename["avg_discount"] = "avg_discount"
        if "FLIGHT_COUNT" in numeric_cols:
            agg["FLIGHT_COUNT"] = "sum"
            rename["FLIGHT_COUNT"] = "total_flights"
        if "BP_SUM" in numeric_cols:
            agg["BP_SUM"] = "sum"
            rename["BP_SUM"] = "total_bp"

        stats = df.groupby("_province_norm").agg(agg).reset_index()
        stats = stats.rename(columns=rename)

        # 浮点精度处理
        for col in stats.columns:
            if col != "name":
                stats[col] = stats[col].round(2)

        return jsonify({"success": True, "data": stats.to_dict(orient="records")})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/map/province/customers", methods=["GET"])
def api_map_province_customers():
    """返回指定省份的客户明细，支持分页与按指标排序，同时返回未知/无效省份统计。"""
    if not has_data():
        return jsonify({"success": False, "error": "暂无数据"}), 400

    province = request.args.get("province", "").strip()
    metric = request.args.get("metric", "count")
    sort = request.args.get("sort", "desc")
    page = request.args.get("page", "1", type=int)
    page_size = request.args.get("page_size", "20", type=int)

    if not province:
        return jsonify({"success": False, "error": "缺少 province 参数"}), 400

    # 限制分页大小，防止内存/带宽问题
    page = max(1, page)
    page_size = min(max(1, page_size), 100)

    metric_to_column = {
        "count": "MEMBER_NO",
        "total_km": "SEG_KM_SUM",
        "total_flights": "FLIGHT_COUNT",
        "total_bp": "BP_SUM",
        "avg_discount": "avg_discount",
    }
    sort_col = metric_to_column.get(metric, "MEMBER_NO")
    ascending = sort.lower() == "asc"

    try:
        df = load_result()
        province_col = None
        for col in ("WORK_PROVINCE", "PROVINCE", "work_province", "province"):
            if col in df.columns:
                province_col = col
                break
        if province_col is None:
            return jsonify({"success": False, "error": "缺少省份字段"}), 400

        df = df.copy()
        df["_province_norm"] = df[province_col].apply(_normalize_province)

        # 未知/无效省份统计
        unknown_count = int(df["_province_norm"].isna().sum())

        # 筛选目标省份（已规范化的全称）
        mask = df["_province_norm"] == province
        filtered = df[mask].copy()

        # 排序
        if sort_col in filtered.columns:
            filtered = filtered.sort_values(by=sort_col, ascending=ascending)

        total = len(filtered)
        total_pages = max(1, (total + page_size - 1) // page_size)
        start = (page - 1) * page_size
        end = start + page_size
        page_df = filtered.iloc[start:end]

        # 选择返回字段
        output_cols = [
            "MEMBER_NO", "AGE", "GENDER", "FFP_TIER",
            "WORK_CITY", "WORK_PROVINCE",
            "FLIGHT_COUNT", "SEG_KM_SUM", "BP_SUM", "avg_discount",
            "Recency", "Frequency", "Monetary", "value_label"
        ]
        available_cols = [c for c in output_cols if c in page_df.columns]
        records = page_df[available_cols].fillna("-").to_dict(orient="records")

        return jsonify({
            "success": True,
            "province": province,
            "metric": metric,
            "sort": sort,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "unknown_count": unknown_count,
            "data": records,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== 配置读写 API ====================

@app.route("/api/config", methods=["GET"])
def api_get_config():
    """返回当前 LLM 提供商配置（隐藏 API Key 的中间字符）"""
    key = config.OPENAI_API_KEY or ""
    masked = key[:8] + "****" + key[-4:] if len(key) > 12 else ("***" if key else "")
    return jsonify({
        "llm_provider": config.LLM_PROVIDER,
        "ollama_url": config.OLLAMA_URL,
        "ollama_model": config.OLLAMA_MODEL,
        "openai_url": config.OPENAI_API_URL,
        "openai_model": config.OPENAI_MODEL,
        "openai_key_masked": masked,
        "mcp_server_url": getattr(config, 'MCP_SERVER_URL', 'http://127.0.0.1:5001'),
        "n_clusters": config.N_CLUSTERS,
        "cluster_random_state": config.CLUSTER_RANDOM_STATE,
        "cluster_range": [config.CLUSTER_RANGE.start, config.CLUSTER_RANGE.stop],
    })


@app.route("/api/config", methods=["POST"])
def api_save_config():
    """保存 LLM 提供商配置（运行时生效并持久化到 config.json）"""
    data = request.json or {}
    updates = {}

    if "llm_provider" in data:
        config.LLM_PROVIDER = data["llm_provider"]
        updates["llm_provider"] = data["llm_provider"]
    if "ollama_url" in data:
        config.OLLAMA_URL = data["ollama_url"]
        updates["ollama_url"] = data["ollama_url"]
    if "ollama_model" in data:
        config.OLLAMA_MODEL = data["ollama_model"]
        updates["ollama_model"] = data["ollama_model"]
    if "openai_url" in data:
        config.OPENAI_API_URL = data["openai_url"]
        updates["openai_url"] = data["openai_url"]
    if "openai_model" in data:
        config.OPENAI_MODEL = data["openai_model"]
        updates["openai_model"] = data["openai_model"]
    # 修改：允许清空 API key
    if "openai_key" in data:
        new_key = data["openai_key"]
        # 如果是掩码值，不更新
        if new_key.startswith("sk-****"):
            pass
        # 如果是空值，清除旧的 key
        elif not new_key or new_key.strip() == "":
            config.OPENAI_API_KEY = ""
            updates["openai_key"] = ""
        # 如果是新的有效 key，更新
        else:
            config.OPENAI_API_KEY = new_key
            updates["openai_key"] = new_key
    if "mcp_server_url" in data:
        config.MCP_SERVER_URL = data["mcp_server_url"]
        updates["mcp_server_url"] = data["mcp_server_url"]
    if "n_clusters" in data:
        config.N_CLUSTERS = int(data["n_clusters"])
        updates["n_clusters"] = int(data["n_clusters"])
    if "cluster_random_state" in data:
        config.CLUSTER_RANDOM_STATE = int(data["cluster_random_state"])
        updates["cluster_random_state"] = int(data["cluster_random_state"])

    config.save_runtime_config(updates)
    return jsonify({"success": True, "message": "配置已更新并已持久化"})


# ==================== 健康检查 ====================

@app.route("/api/health")
def api_health():
    _init_tool_registry()
    db_type = "none"
    try:
        from database import get_engine_type
        db_type = get_engine_type()
    except Exception:
        pass

    return jsonify({
        "status": "ok",
        "data_available": has_data(),
        "database": db_type,
        "tools_available": list(TOOL_REGISTRY.keys()),
        "llm_provider": config.LLM_PROVIDER,
    })


# ==================== Schema 与探测 ====================

@app.route("/api/schema")
def api_schema():
    _init_tool_registry()
    result = TOOL_REGISTRY["get_database_schema"]["function"]()
    return jsonify(result)


@app.route("/api/schema/<table_name>")
def api_table_schema(table_name):
    _init_tool_registry()
    result = TOOL_REGISTRY["get_table_schema"]["function"](table_name)
    return jsonify(result)


@app.route("/api/probe/<column_name>")
def api_probe(column_name):
    _init_tool_registry()
    table_name = request.args.get("table")
    _ensure_sandbox()
    result = TOOL_REGISTRY["probe_distinct_values"]["function"](column_name, table_name)
    return jsonify(result)


@app.route("/api/business-rules")
def api_business_rules():
    _init_tool_registry()
    result = TOOL_REGISTRY["get_business_rules"]["function"]()
    return jsonify(result)


# ==================== 数据 API ====================

@app.route("/api/stats")
def api_stats():
    if not has_data():
        return jsonify({"error": "无数据，请先运行分析流水线"}), 400
    df = load_result()
    return jsonify({
        "rows": len(df),
        "columns": len(df.columns),
        "cluster_counts": df["cluster"].value_counts().sort_index().to_dict()
        if "cluster" in df.columns else {},
        "value_labels": df["value_label"].value_counts().to_dict()
        if "value_label" in df.columns else {},
        "avg_rfm": {
            "R": round(float(df["R_score"].mean()), 2) if "R_score" in df.columns else None,
            "F": round(float(df["F_score"].mean()), 2) if "F_score" in df.columns else None,
            "M": round(float(df["M_score"].mean()), 2) if "M_score" in df.columns else None,
        },
    })


@app.route("/api/data")
def api_data():
    if not has_data():
        return jsonify({"error": "无数据"}), 400
    df = load_result()
    page = request.args.get("page", 1, type=int)
    page_size = request.args.get("page_size", 20, type=int)
    page_size = min(page_size, 500)
    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    return jsonify({
        "columns": df.columns.tolist(),
        "rows": df.iloc[start:end].fillna("").to_dict(orient="records"),
        "total": total, "page": page, "page_size": page_size,
    })


@app.route("/api/clusters")
def api_clusters():
    if not has_data():
        return jsonify({"error": "无数据"}), 400
    df = load_result()
    rfm_cols = ["R_score", "F_score", "M_score", "Recency", "Frequency", "Monetary"]
    existing = [c for c in rfm_cols if c in df.columns]
    summary = df.groupby("cluster")[existing].agg(["mean", "std"]).round(2)
    summary.columns = [f"{col}_{agg}" for col, agg in summary.columns]
    summary["count"] = df.groupby("cluster").size().values
    summary = summary.reset_index()
    summary["cluster"] = summary["cluster"].astype(int)
    return jsonify(summary.to_dict(orient="records"))


# ==================== SQL 执行端点 ====================

@app.route("/api/sql/execute", methods=["POST"])
def api_sql_execute():
    _init_tool_registry()
    sql = request.json.get("sql", "").strip()
    if not sql:
        return jsonify({"success": False, "error": "SQL 不能为空"}), 400

    _ensure_sandbox()
    result = TOOL_REGISTRY["execute_secure_sql"]["function"](sql)

    # 取样数据
    if result.get("success") and result.get("data"):
        try:
            sample_size = int(request.json.get("sample", 50))
        except (ValueError, TypeError):
            sample_size = 50
        if len(result["data"]) > sample_size:
            result["data"] = result["data"][:sample_size]
            result["truncated"] = True
            result["total_rows"] = result["row_count"]
            result["row_count"] = sample_size

    return jsonify(result)


# ==================== 动态聚类端点 ====================

@app.route("/api/cluster/dynamic", methods=["POST"])
def api_cluster_dynamic():
    _init_tool_registry()
    data_json = request.json.get("data", "[]")
    try:
        k = int(request.json.get("k", 4))
    except (ValueError, TypeError):
        k = 4
    features = request.json.get("features", None)

    if isinstance(data_json, list):
        data_json = json.dumps(data_json)

    result = TOOL_REGISTRY["perform_dynamic_cluster"]["function"](
        data_json, k=k, features=features
    )
    return jsonify(result)


# ==================== 预测与异常检测端点 ====================

@app.route("/api/tools/forecast", methods=["POST"])
def api_forecast():
    """趋势预测端点"""
    _init_tool_registry()
    data_json = request.json.get("data", "[]")
    date_col = request.json.get("date_col")
    value_col = request.json.get("value_col")
    periods = request.json.get("periods", 3)
    method = request.json.get("method", "linear")

    if isinstance(data_json, list):
        data_json = json.dumps(data_json)

    result = TOOL_REGISTRY["forecast_trend"]["function"](
        data_json, date_col=date_col, value_col=value_col,
        periods=int(periods), method=method
    )
    return jsonify(result)


@app.route("/api/tools/anomaly", methods=["POST"])
def api_anomaly():
    """异常检测端点"""
    _init_tool_registry()
    data_json = request.json.get("data", "[]")
    columns = request.json.get("columns")
    method = request.json.get("method", "iqr")
    contamination = request.json.get("contamination", 0.05)

    if isinstance(data_json, list):
        data_json = json.dumps(data_json)

    result = TOOL_REGISTRY["detect_anomalies"]["function"](
        data_json, columns=columns, method=method, contamination=float(contamination)
    )
    return jsonify(result)


@app.route("/api/tools/churn", methods=["POST"])
def api_churn():
    """流失风险评分端点"""
    _init_tool_registry()
    data_json = request.json.get("data", "[]")
    recency_col = request.json.get("recency_col", "Recency")
    frequency_col = request.json.get("frequency_col", "Frequency")
    monetary_col = request.json.get("monetary_col", "Monetary")

    if isinstance(data_json, list):
        data_json = json.dumps(data_json)

    result = TOOL_REGISTRY["churn_risk_score"]["function"](
        data_json, recency_col=recency_col,
        frequency_col=frequency_col, monetary_col=monetary_col
    )
    return jsonify(result)


# ==================== 工具调用端点（通用） ====================

@app.route("/api/tools", methods=["GET"])
def api_tools_list():
    """列出所有可用 MCP 工具"""
    _init_tool_registry()
    tools = []
    for name, info in TOOL_REGISTRY.items():
        tools.append({
            "name": name,
            "description": info["description"],
            "parameters": info["parameters"],
        })
    return jsonify({"tools": tools, "count": len(tools)})


@app.route("/api/tools/<tool_name>", methods=["POST"])
def api_tools_call(tool_name):
    """通用工具调用端点"""
    _init_tool_registry()
    if tool_name not in TOOL_REGISTRY:
        return jsonify({"success": False, "error": f"未知工具: {tool_name}",
                        "available": list(TOOL_REGISTRY.keys())}), 404

    params = request.json or {}
    tool_fn = TOOL_REGISTRY[tool_name]["function"]

    try:
        result = tool_fn(**params)
        return jsonify({"success": True, "tool": tool_name, "result": result})
    except Exception as e:
        return jsonify({"success": False, "tool": tool_name,
                        "error": str(e), "traceback": traceback.format_exc()})


# ==================== 核心: /api/chat — MCP 调用链编排 ====================

# ==================== 后端确定性编排（不依赖 LLM function calling） ====================

def _detect_intent(question: str) -> str:
    """
    后端确定性意图识别，不依赖 LLM。
    返回: sql | forecast | anomaly | churn | cluster | chat | help
    """
    q = question.lower()

    # 预测类
    if any(k in q for k in ["预测", "趋势", "未来", "走势", "下个月", "下个季度", "forecast"]):
        return "forecast"

    # 异常检测
    if any(k in q for k in ["异常", "离群", "极端值", "outlier", "anomaly"]):
        return "anomaly"

    # 流失风险
    if any(k in q for k in ["流失风险", "流失预警", "谁会流失", "churn", "挽留", "流失倾向"]):
        return "churn"

    # 动态聚类
    if any(k in q for k in ["重新聚类", "动态聚类", "聚成", "分群", "kmeans", "k-means", "分成几类"]):
        return "cluster"

    # 帮助
    if any(k in q for k in ["帮助", "怎么用", "功能", "能做什么", "help"]):
        return "help"

    # 对话/解释类 — 在 SQL 之前检测，避免解释性问题被强制生成 SQL
    # 触发词：是什么/什么意思/解释/含义/代表/聊聊/交流/帮我理解/区别/指的/说明
    if _is_chat_intent(q):
        return "chat"

    # 默认: SQL 查询
    return "sql"


def _is_chat_intent(q: str) -> bool:
    """检测是否为对话/解释类问题（而非数据查询）"""
    # 排除信号：包含可视化/图表关键词时，不是纯对话意图
    visualization_keywords = [
        "柱状图", "饼图", "折线图", "散点图", "雷达图", "热力图", "漏斗图",
        "箱线图", "仪表盘", "树图", "帕累托", "面积图", "堆叠图",
        "画图", "绘图", "可视化", "图表", "展示", "分布", "占比", "统计",
        "查询", "列出", "对比", "排名",
    ]
    if any(k in q for k in visualization_keywords):
        return False

    # 强信号：明确的解释/聊天意图
    chat_keywords = [
        "是什么", "什么意思", "什么含义", "含义是", "代表什么", "指的是",
        "指什么", "解释一下", "解释下", "帮我解释", "说明一下", "说明下",
        "聊聊", "交流", "聊一聊", "讨论一下", "讨论下",
        "帮我理解", "理解一下", "区别是", "区别在哪", "有什么区别",
        "怎么理解", "如何理解", "干嘛的", "干什么的", "做什么用",
        "你好", "在吗", "谢谢", "你是谁", "你能做什么",
    ]
    if any(k in q for k in chat_keywords):
        return True

    # 弱信号：包含解释性词但没有查询动作词
    explain_words = ["含义", "意思", "解释", "代表", "说明", "含义", "定义", "概念"]
    query_action_words = [
        "统计", "查询", "查出", "列出", "显示", "查找", "筛选", "排序",
        "分组", "计数", "求和", "平均", "最大", "最小", "画", "绘图",
        "可视化", "展示", "对比", "排名", "分布", "占比", "多少", "几个",
        "哪些", "排行", "top", "前几",
    ]
    has_explain = any(k in q for k in explain_words)
    has_action = any(k in q for k in query_action_words)
    # 有解释词且无查询动作词 → 对话
    if has_explain and not has_action:
        return True

    return False


def _detect_chart_type(question: str) -> str:
    """从用户问题中提取图表类型偏好"""
    q = question.lower()
    chart_map = {
        "柱状图": "bar",
        "柱图": "bar",
        "条形图": "bar",
        "区间": "bar",
        "分布图": "bar",
        "饼图": "pie",
        "环形图": "pie",
        "占比图": "pie",
        "折线图": "line",
        "趋势图": "line",
        "散点图": "scatter",
        "雷达图": "radar",
        "热力图": "heatmap",
        "漏斗": "funnel",
        "箱线图": "boxplot",
        "仪表盘": "gauge",
        "堆叠": "stacked_bar",
        "面积图": "area_line",
        "树图": "treemap",
        "帕累托": "pareto",
    }
    for kw, ctype in chart_map.items():
        if kw in q:
            return ctype
    return "auto"


def _detect_chart_purpose(question: str) -> str:
    """从用户问题中提取分析目的"""
    q = question.lower()
    purpose_map = {
        "占比": "composition",
        "构成": "composition",
        "比例": "composition",
        "分布": "distribution",
        "趋势": "trend",
        "变化": "trend",
        "排名": "ranking",
        "排序": "ranking",
        "对比": "comparison",
        "比较": "comparison",
        "关系": "relationship",
        "相关": "relationship",
        "漏斗": "funnel",
        "转化": "funnel",
    }
    for kw, purpose in purpose_map.items():
        if kw in q:
            return purpose
    return "auto"


def _has_visualization_intent(question: str) -> bool:
    """检测用户是否有可视化/画图意图"""
    keywords = [
        "图", "图表", "画图", "可视化", "绘制", "展示", "看一下", "看下",
        "趋势", "占比", "分布", "排名", "漏斗", "仪表盘", "箱线图", "帕累托",
        "树图", "热力图", "散点图", "柱状图", "饼图", "折线图", "面积图",
        "区间", "柱", "画", "直方图", "频次",
    ]
    q = question.lower()
    return any(k in q for k in keywords)


def _build_sql_system_prompt(question: str = "") -> str:
    """构造 SQL 生成的 system prompt（不依赖 function calling，低智力模型也能理解）"""

    # 检测当前数据库引擎
    engine_type = "sqlite"
    try:
        from database import get_engine_type
        engine_type = get_engine_type()
    except Exception:
        pass

    sql_dialect = "MySQL" if engine_type == "mysql" else "SQLite"
    if engine_type == "mysql":
        dialect_tips = "可用反引号 `列名`，支持 DATEDIFF、DATE_FORMAT、STR_TO_DATE 等函数"
    else:
        dialect_tips = "用双引号或直接写列名，日期函数用 STRFTIME('%Y', col)，不要用 DATEDIFF/DATE_FORMAT"

    # 按问题附加针对性提示
    hints = _build_query_hints(question)
    hints_block = f"\n【本次查询提示】\n{hints}" if hints else ""

    return f"""你是 SQL 生成助手。数据库为 {sql_dialect}，主题是「航空公司客户分析」。
你的任务：根据用户问题，生成一条 {sql_dialect} SELECT 语句。只返回 SQL 本身，不要 markdown，不要解释。

═══════════════════════════════════════════
【数据库结构 — 严格按此结构写 SQL，不要编造任何列名】
═══════════════════════════════════════════

共有 3 张表 + 1 个视图，通过 member_no（会员编号）关联：

■ 表1: customer_base（客户基础信息）
  - member_no     会员编号（主键）
  - age           年龄（整数，如 25, 40, 55）
  - gender        性别（取值: 男 / 女 / 未知）
  - ffp_tier      会员等级（整数，如 4, 5, 6）
  - avg_discount  平均折扣率（小数，如 0.85）
  - ffp_date      入会日期
  - first_flight  首次飞行日期

■ 表2: customer_flight_summary（飞行行为汇总）
  - member_no     会员编号（主键）
  - flight_count  总飞行次数（整数）
  - seg_km_sum    总飞行里程（整数）
  - bp_sum        总积分（整数）
  - last_flight   最后飞行日期
  - recency       距观察期末天数（越小=越近期=越活跃）

■ 表3: customer_analytics（AI分析结果 — RFM + 聚类）
  - member_no     会员编号（主键）
  - r_score       R评分（1-5）
  - f_score       F评分（1-5）
  - m_score       M评分（1-5）
  - rfm_total     RFM总分（3-15）
  - cluster       聚类标签（取值: 0, 1, 2, 3）
  - value_label   价值标签（取值: 高价值客户 / 中价值客户 / 低价值客户 / 流失客户）

■ 视图: customer_rfm（三表 JOIN 的大宽表，含上述所有字段）
  可直接 SELECT * FROM customer_rfm，免写 JOIN。

═══════════════════════════════════════════
【绝对禁止 — 以下列名不存在，用了必报错】
═══════════════════════════════════════════
  ✗ city / 工作地点 / 城市 / 地区 / province / 省份 — 数据库中没有地理位置字段！
  ✗ name / 姓名 / phone / 电话 / email — 没有个人信息字段！
  ✗ work_location / company / 职位 — 没有职业信息！
  ✗ order_count / revenue / amount — 没有订单/金额字段！
  ✗ 任何上面 schema 中没有列名的字段 — 一律不许使用！

  如果用户问的内容不在上述字段范围内，请返回:
  SELECT '该数据不在当前数据库字段范围内' AS hint

═══════════════════════════════════════════
【别名规则 — 最易犯的错误】
═══════════════════════════════════════════
  用别名时，列必须属于对应别名所在的表：
  ✗ SELECT c.cluster FROM customer_base c  — cluster 不在 customer_base！
  ✓ SELECT a.cluster FROM customer_base b JOIN customer_analytics a ON b.member_no = a.member_no

  记住别名口诀：
    b = customer_base      → age, gender, ffp_tier, avg_discount
    f = customer_flight_summary → flight_count, seg_km_sum, recency, bp_sum
    a = customer_analytics → cluster, value_label, rfm_total, r_score, f_score, m_score

═══════════════════════════════════════════
【示例 SQL — 参考这些模式写】
═══════════════════════════════════════════

例1: 统计各价值层级的客户数量
SELECT value_label, COUNT(*) AS cnt FROM customer_analytics GROUP BY value_label

例2: 各聚类簇的平均飞行次数（需跨表 JOIN）
SELECT a.cluster, AVG(f.flight_count) AS avg_flights
FROM customer_analytics a
JOIN customer_flight_summary f ON a.member_no = f.member_no
GROUP BY a.cluster

例3: 高价值客户的年龄分布（用 CASE WHEN 分段）
SELECT
  CASE
    WHEN age < 20 THEN '<20'
    WHEN age BETWEEN 20 AND 29 THEN '20-29'
    WHEN age BETWEEN 30 AND 39 THEN '30-39'
    WHEN age BETWEEN 40 AND 49 THEN '40-49'
    WHEN age >= 50 THEN '50+'
  END AS age_range,
  COUNT(*) AS cnt
FROM customer_base b
JOIN customer_analytics a ON b.member_no = a.member_no
WHERE a.value_label = '高价值客户'
GROUP BY age_range
ORDER BY age_range

例4: 各会员等级的平均消费金额
SELECT b.ffp_tier, AVG(f.seg_km_sum) AS avg_km
FROM customer_base b
JOIN customer_flight_summary f ON b.member_no = f.member_no
GROUP BY b.ffp_tier

例5: 用 customer_rfm 宽表免 JOIN 查询
SELECT cluster, AVG(flight_count) AS avg_fc FROM customer_rfm GROUP BY cluster

═══════════════════════════════════════════
【其他规则】
═══════════════════════════════════════════
1. {dialect_tips}
2. 只用 SELECT，禁止 INSERT/UPDATE/DELETE/DROP。
3. 聚合查询记得 GROUP BY。
4. "高水平人群"="高价值客户"，"各阶层"=按 value_label 或 ffp_tier 分组。
5. 只返回一条 SQL 语句，不要分号结尾，不要 markdown 代码块。{hints_block}"""


def _build_query_hints(question: str) -> str:
    """根据用户问题关键词，附加针对性 SQL 生成提示"""
    q = question.lower()
    hints = []

    # 年龄分段
    if any(k in q for k in ["年龄区间", "年龄段", "年龄分布", "年龄结构"]):
        hints.append("用户想看年龄分段统计。请用 CASE WHEN 将 age 分为 <20, 20-29, 30-39, 40-49, 50+ 几段，再 GROUP BY 该分段。")

    # 地理位置相关 — 数据库无此字段
    if any(k in q for k in ["城市", "工作地点", "地区", "省份", "地点", "哪里人", "哪个城市", "居住"]):
        hints.append("⚠ 数据库中没有城市/工作地点/地理位置字段。如果用户问地理位置相关内容，请返回: SELECT '当前数据库无地理位置字段' AS hint")

    # 价值层级
    if any(k in q for k in ["高水平", "高层", "高价值", "高消费", "优质客户"]):
        hints.append("用户说的'高水平/高层'对应 value_label='高价值客户'。")

    # 聚类分析
    if any(k in q for k in ["聚簇", "聚类", "cluster", "分群", "哪一类"]):
        hints.append("聚类标签在 customer_analytics.cluster（取值0-3）。跨表查 base 或 flight 数据时需 JOIN customer_analytics。")

    # 占比/比例
    if any(k in q for k in ["占比", "比例", "百分比", "构成"]):
        hints.append("用户想看占比。SQL 中先 COUNT(*) 求各组数量，前端会自动算百分比。")

    # 排名/Top N
    import re as _re
    if _re.search(r"top\s*\d+|前\d+|最高|最多|排名", q):
        hints.append("用户想看排名。请加 ORDER BY ... DESC LIMIT N。")

    return "\n".join(hints)


def _generate_sql_via_llm(question: str) -> tuple[str | None, str]:
    """
    用 LLM 生成单条 SQL（单次调用，不依赖 function calling）。
    优先 OpenAI 兼容 API，降级 Ollama。
    返回 (sql, source)
    """
    system_prompt = _build_sql_system_prompt(question)

    # 尝试 OpenAI 兼容 API
    if config.LLM_PROVIDER == "openai_compatible" and config.OPENAI_API_KEY:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.OPENAI_API_KEY}"}
        try:
            r = requests.post(
                config.OPENAI_API_URL,
                json={
                    "model": config.OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"问题: {question}\nSQL:"},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500,
                },
                headers=headers,
                timeout=30,
            )
            if r.status_code == 200:
                content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                sql = _extract_sql(content)
                if sql:
                    return sql, f"{config.OPENAI_MODEL} (one-shot)"
        except Exception as e:
            print(f"[LLM] OpenAI API 失败: {e}")

    # 降级 Ollama
    try:
        r = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": f"{system_prompt}\n\n用户问题: {question}\nSQL:",
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=60,
        )
        if r.status_code == 200:
            raw = r.json().get("response", "")
            sql = _extract_sql(raw)
            if sql:
                return sql, f"{config.OLLAMA_MODEL} (one-shot)"
    except Exception as e:
        print(f"[LLM] Ollama 失败: {e}")

    return None, "none"


def _generate_fixed_sql(question: str, bad_sql: str, error: str) -> str | None:
    """让 LLM 修正 SQL（单次调用，复用完整 schema prompt）"""
    system_prompt = _build_sql_system_prompt(question)
    fix_instruction = (
        f"以下是执行失败的 SQL，请根据错误信息修正。\n"
        f"失败 SQL: {bad_sql}\n"
        f"错误: {error}\n\n"
        f"请仔细检查：1)列名是否属于正确的表 2)别名是否对应 3)是否编造了不存在的列。\n"
        f"只返回修正后的 SQL，不要解释。"
    )

    # 尝试 OpenAI 兼容 API
    if config.LLM_PROVIDER == "openai_compatible" and config.OPENAI_API_KEY:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.OPENAI_API_KEY}"}
        try:
            r = requests.post(
                config.OPENAI_API_URL,
                json={
                    "model": config.OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": fix_instruction},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500,
                },
                headers=headers,
                timeout=30,
            )
            if r.status_code == 200:
                content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                sql = _extract_sql(content)
                if sql:
                    return sql
        except Exception:
            pass

    # 降级 Ollama
    try:
        r = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": f"{system_prompt}\n\n{fix_instruction}",
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=60,
        )
        if r.status_code == 200:
            raw = r.json().get("response", "")
            sql = _extract_sql(raw)
            if sql:
                return sql
    except Exception:
        pass

    return None


def _orchestrate_sql_query(question: str, page: int = 1, page_size: int = 20,
                           paginate: bool = True) -> dict:
    """
    后端确定性编排 SQL 查询流程：
    1. LLM 生成 SQL（单次调用）
    2. 沙箱执行
    3. 失败自动修正（最多 3 次）
    4. 自动判断是否需要可视化
    5. 分页：缓存完整结果，仅返回当前页（paginate=True 时）
    """
    steps = []

    # Step 1: 获取 schema（记录步骤但不依赖 LLM 调用）
    steps.append({"tool": "get_database_schema", "args": {}, "success": True})

    # Step 2: LLM 生成 SQL
    sql, source = _generate_sql_via_llm(question)
    if not sql:
        # LLM 全部不可用，降级关键词
        return _fallback_regex(question)

    # Step 3: 执行 SQL 带自纠错
    from mcp_tools.sql_tools import execute_secure_sql
    current_sql = sql
    attempts_log = []
    last_result = None

    for attempt in range(3):
        last_result = execute_secure_sql(current_sql, sandbox_db_path=config.SQLITE_PATH)
        attempts_log.append({
            "sql": current_sql,
            "success": last_result.get("success", False),
            "error": last_result.get("error"),
        })

        if last_result.get("success"):
            full_data = last_result.get("data", [])
            total_rows = last_result.get("row_count", len(full_data))
            columns = last_result.get("columns", [])

            # 分页：缓存完整数据，仅返回当前页给前端
            if paginate and total_rows > page_size:
                query_id = f"q{int(time.time()*1000)}_{abs(hash(question))%100000}"
                _QUERY_CACHE[query_id] = {
                    "data": full_data,
                    "columns": columns,
                    "sql": current_sql,
                    "total": total_rows,
                    "_time": time.time(),
                }
                _clean_query_cache()

                start = (page - 1) * page_size
                page_data = full_data[start:start + page_size]
                total_pages = max(1, (total_rows + page_size - 1) // page_size)

                paged_result = {
                    **last_result,
                    "data": page_data,
                    "row_count": len(page_data),
                    "total_rows": total_rows,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": total_pages,
                    "query_id": query_id,
                }
            else:
                paged_result = last_result

            steps.append({
                "tool": "execute_secure_sql",
                "args": {"sql_query": current_sql},
                "success": True,
                "result": paged_result,
            })

            # Step 4: 自动判断是否需要可视化（用完整数据聚合，避免分页影响）
            if _has_visualization_intent(question):
                data = full_data
                if data:
                    chart_type = _detect_chart_type(question)
                    purpose = _detect_chart_purpose(question)
                    # 截断大数据
                    chart_data = data[:50] if len(data) > 50 else data
                    viz_result = TOOL_REGISTRY["generate_visualization"]["function"](
                        json.dumps(chart_data, ensure_ascii=False),
                        chart_type=chart_type,
                        title=question[:30],
                        purpose=purpose,
                    )
                    steps.append({
                        "tool": "generate_visualization",
                        "args": {"chart_type": chart_type, "purpose": purpose},
                        "success": viz_result.get("success", False),
                        "result": viz_result,
                    })

            # 生成文字摘要
            answer = _build_answer(question, current_sql, last_result)
            return {
                "question": question,
                "mode": "orchestrated",
                "steps": steps,
                "answer": answer,
                "source": f"{source} (attempts={attempt + 1})",
                "self_heal_attempts": attempts_log,
            }

        # 尝试修正
        if attempt < 2:
            fixed_sql = _generate_fixed_sql(question, current_sql, last_result.get("error", "未知错误"))
            if fixed_sql and fixed_sql.lower().strip() != current_sql.lower().strip():
                current_sql = fixed_sql
            else:
                break

    # 全部失败
    steps.append({
        "tool": "execute_secure_sql",
        "args": {"sql_query": current_sql},
        "success": False,
        "result": last_result,
        "error": last_result.get("error") if last_result else "执行失败",
    })

    return {
        "question": question,
        "mode": "orchestrated",
        "steps": steps,
        "answer": f"SQL 执行失败（已尝试 {len(attempts_log)} 次）：\n```sql\n{current_sql}\n```\n错误：{last_result.get('error', '未知错误') if last_result else '未知错误'}",
        "source": f"{source} (self-heal failed)",
        "self_heal_attempts": attempts_log,
    }


def _orchestrate_forecast(question: str) -> dict:
    """编排趋势预测流程"""
    # 先通过 SQL 获取时序数据（不分页，需完整数据做预测）
    sql_result = _orchestrate_sql_query(question, paginate=False)
    sql_data = None
    for s in sql_result.get("steps", []):
        if s.get("tool") == "execute_secure_sql" and s.get("success"):
            sql_data = s.get("result", {}).get("data")
            break

    if not sql_data:
        return sql_result

    # 调用预测工具
    forecast_result = TOOL_REGISTRY["forecast_trend"]["function"](
        json.dumps(sql_data, ensure_ascii=False), periods=3
    )

    sql_result["steps"].append({
        "tool": "forecast_trend",
        "args": {"periods": 3},
        "success": forecast_result.get("success", False),
        "result": forecast_result,
    })

    if forecast_result.get("success"):
        historical = forecast_result.get("historical", [])
        forecast = forecast_result.get("forecast", [])
        sql_result["answer"] = (
            f"趋势预测结果（{forecast_result.get('method')}）：\n"
            f"历史数据 {len(historical)} 个点，预测未来 {len(forecast)} 期：\n"
            + "\n".join([f"  {f['x']}: {f['y']}" for f in forecast])
        )

    return sql_result


def _orchestrate_anomaly(question: str) -> dict:
    """编排异常检测流程"""
    sql_result = _orchestrate_sql_query(question, paginate=False)
    sql_data = None
    for s in sql_result.get("steps", []):
        if s.get("tool") == "execute_secure_sql" and s.get("success"):
            sql_data = s.get("result", {}).get("data")
            break

    if not sql_data:
        return sql_result

    anomaly_result = TOOL_REGISTRY["detect_anomalies"]["function"](
        json.dumps(sql_data, ensure_ascii=False)
    )

    sql_result["steps"].append({
        "tool": "detect_anomalies",
        "args": {"method": "iqr"},
        "success": anomaly_result.get("success", False),
        "result": anomaly_result,
    })

    if anomaly_result.get("success"):
        sql_result["answer"] = (
            f"异常检测完成（{anomaly_result.get('method')}）：\n"
            f"总样本 {anomaly_result.get('total')} 条，发现异常 {anomaly_result.get('anomaly_count')} 条。"
        )

    return sql_result


def _orchestrate_churn(question: str) -> dict:
    """编排流失风险评分流程"""
    # 直接从 final_result.csv 读取 RFM 数据
    df = load_result()
    rfm_cols = []
    for col in ["Recency", "Frequency", "Monetary", "MEMBER_NO", "value_label"]:
        if col in df.columns:
            rfm_cols.append(col)

    if not rfm_cols or "Recency" not in df.columns:
        return {
            "question": question,
            "mode": "orchestrated",
            "steps": [],
            "answer": "数据中未找到 RFM 字段，无法计算流失风险。",
            "source": "backend",
        }

    # 取前 500 条样本
    sample_df = df[rfm_cols].head(500)
    data_json = sample_df.to_json(orient="records", force_ascii=False)

    churn_result = TOOL_REGISTRY["churn_risk_score"]["function"](data_json)

    steps = [{
        "tool": "churn_risk_score",
        "args": {"sample_size": 500},
        "success": churn_result.get("success", False),
        "result": churn_result,
    }]

    # 自动生成分布图
    if churn_result.get("success"):
        dist = churn_result.get("distribution", {})
        chart_data = [{"label": k, "count": v} for k, v in dist.items()]
        viz_result = TOOL_REGISTRY["generate_visualization"]["function"](
            json.dumps(chart_data, ensure_ascii=False),
            chart_type="pie",
            title="流失风险分布",
        )
        steps.append({
            "tool": "generate_visualization",
            "args": {"chart_type": "pie"},
            "success": viz_result.get("success", False),
            "result": viz_result,
        })

    answer = "流失风险评分完成。" if churn_result.get("success") else "流失风险评分失败。"
    if churn_result.get("distribution"):
        dist = churn_result["distribution"]
        answer += f"\n风险分布：高风险 {dist.get('高风险', 0)} 人，中风险 {dist.get('中风险', 0)} 人，低风险 {dist.get('低风险', 0)} 人。"

    return {
        "question": question,
        "mode": "orchestrated",
        "steps": steps,
        "answer": answer,
        "source": "backend",
    }


def _build_chat_system_prompt() -> str:
    """构造对话模式 system prompt，让 LLM 了解项目数据字典，能解释数据含义"""
    return """你是航空公司客户分析平台的智能助手。你可以与用户交流、解释数据含义、回答关于项目的问题。

【平台数据字典 — 回答数据含义类问题时参考】

这是一个航空公司会员客户分析系统，数据存储在 3 张表中：

1. customer_base（客户基础信息）
   - member_no: 会员编号（唯一标识）
   - age: 年龄（已清洗异常值）
   - gender: 性别（男/女/未知）
   - ffp_tier: 会员等级（如4、5、6，数字越大等级越高）
   - avg_discount: 平均折扣率（反映购票价格敏感度，0-1之间）
   - ffp_date: 入会日期
   - first_flight: 首次飞行日期

2. customer_flight_summary（飞行行为汇总）
   - flight_count: 总飞行次数
   - seg_km_sum: 总飞行里程（公里）
   - bp_sum: 总基本积分
   - last_flight: 最后一次飞行日期
   - recency: 距观察期末(2014-03-31)的天数，越小表示越近期、越活跃

3. customer_analytics（AI分析结果）
   - r_score: 最近消费评分(1-5)，基于Recency百分位排名，越高越活跃
   - f_score: 消费频率评分(1-5)，基于Flight_count百分位排名，越高越频繁
   - m_score: 消费金额评分(1-5)，基于Seg_km_sum百分位排名，越高消费越多
   - rfm_total: RFM总分(3-15)，三单项分之和
   - cluster: K-Means聚类标签(0-3)，基于Log1p+StandardScaler预处理后的RFM特征
   - value_label: 价值标签，由rfm_total分段得到：
     · 高价值客户: rfm_total >= 13
     · 中价值客户: 9 <= rfm_total < 13
     · 低价值客户: 6 <= rfm_total < 9
     · 流失客户: rfm_total < 6

【平台功能】
- 数据查询：自然语言查客户数据
- 可视化：柱状图、饼图、折线图等13种图表
- 趋势预测：线性回归/指数平滑
- 异常检测：IQR/IsolationForest
- 流失风险：基于RFM的流失评分

【回答要求】
1. 用中文回答，简洁明了。
2. 解释数据字段时，说明它的业务含义和取值范围。
3. 如果用户问的数据不在上述字段范围内，明确告知。
4. 不要编造数据，不确定时说明。
5. 如果用户想查具体数据，引导他们用查询语句，如"统计各价值层级客户数量"。"""


def _orchestrate_chat(question: str) -> dict:
    """编排对话/聊天流程：调用 LLM 做自然对话，不生成 SQL"""
    system_prompt = _build_chat_system_prompt()
    answer = None
    source = "none"

    # 尝试 OpenAI 兼容 API
    if config.LLM_PROVIDER == "openai_compatible" and config.OPENAI_API_KEY:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.OPENAI_API_KEY}"}
        try:
            r = requests.post(
                config.OPENAI_API_URL,
                json={
                    "model": config.OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    "temperature": 0.5,
                    "max_tokens": 800,
                },
                headers=headers,
                timeout=30,
            )
            if r.status_code == 200:
                content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                if content.strip():
                    answer = content.strip()
                    source = f"{config.OPENAI_MODEL} (chat)"
        except Exception as e:
            print(f"[Chat] OpenAI API 失败: {e}")

    # 降级 Ollama
    if not answer:
        try:
            r = requests.post(
                f"{config.OLLAMA_URL}/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": f"{system_prompt}\n\n用户问题: {question}",
                    "stream": False,
                    "options": {"temperature": 0.5},
                },
                timeout=60,
            )
            if r.status_code == 200:
                raw = r.json().get("response", "")
                if raw.strip():
                    answer = raw.strip()
                    source = f"{config.OLLAMA_MODEL} (chat)"
        except Exception as e:
            print(f"[Chat] Ollama 失败: {e}")

    # LLM 全部不可用 — 降级静态回答
    if not answer:
        answer = _build_fallback_chat(question)

    return {
        "question": question,
        "mode": "chat",
        "steps": [],
        "answer": answer,
        "source": source,
    }


def _build_fallback_chat(question: str) -> str:
    """LLM 不可用时的静态对话兜底"""
    q = question.lower()
    if any(k in q for k in ["rfm", "rfm_total"]):
        return ("RFM 是一种客户价值分析模型：\n"
                "- R (Recency): 最近消费时间，距观察期末天数，越小越活跃\n"
                "- F (Frequency): 消费频率，即总飞行次数\n"
                "- M (Monetary): 消费金额，用总飞行里程代替\n\n"
                "每项评分1-5分，总分rfm_total为3-15。根据总分分为：\n"
                "高价值(>=13) / 中价值(9-12) / 低价值(6-8) / 流失(<6)")
    if "cluster" in q or "聚类" in q or "聚簇" in q:
        return ("cluster 是 K-Means 聚类标签(0-3)，基于 Log1p+StandardScaler "
                "预处理后的 RFM 特征聚类得到。同簇客户具有相似的消费行为模式。")
    if "value_label" in q or "价值标签" in q or "价值层级" in q:
        return ("value_label 是客户价值标签，由 rfm_total 分段得到：\n"
                "高价值客户(>=13) / 中价值客户(9-12) / 低价值客户(6-8) / 流失客户(<6)")
    if "recency" in q:
        return "recency 是距观察期末(2014-03-31)的天数，越小表示客户越近期活跃。"
    if "ffp_tier" in q or "会员等级" in q:
        return "ffp_tier 是会员等级(如4/5/6)，数字越大等级越高。"
    if any(k in q for k in ["你好", "在吗", "hi", "hello"]):
        return "你好！我是客户分析助手。你可以问我数据字段的含义，或者查询客户数据、画图等。"
    if "谢谢" in q:
        return "不客气！还有其他问题随时问我。"
    return ("我是客户分析助手，可以解释数据字段含义、聊天交流。"
            "如果你想查具体数据，可以直接提问，如'统计各价值层级客户数量'。")


def _build_answer(question: str, sql: str, result: dict) -> str:
    """根据 SQL 查询结果生成文字摘要（不依赖 LLM）"""
    if not result.get("success"):
        return "查询失败。"

    data = result.get("data", [])
    row_count = result.get("row_count", len(data))
    columns = result.get("columns", [])

    if not data:
        return "查询结果为空。"

    # 单值结果
    if len(data) == 1 and len(columns) == 1:
        return f"查询结果：{data[0][columns[0]]}"

    # 聚合结果（GROUP BY）
    if len(data) <= 20:
        lines = [f"查询完成，共 {row_count} 条记录：\n"]
        for row in data[:10]:
            parts = [f"{col}={row.get(col)}" for col in columns]
            lines.append("  " + ", ".join(parts))
        if len(data) > 10:
            lines.append(f"  ...（共 {len(data)} 条）")
        return "\n".join(lines)

    return f"查询完成，共 {row_count} 条记录。"


def _build_help_answer() -> str:
    """返回帮助信息"""
    return """可用功能：

1. **数据查询** — 用自然语言查询客户数据
   例: "统计各价值层级的客户数量"
   例: "找出年龄大于50岁的高价值客户"

2. **可视化** — 自动生成图表
   例: "用柱状图展示各聚类簇的平均消费金额"
   例: "画一个客户价值占比的饼图"
   例: "各会员等级的飞行次数分布箱线图"

3. **趋势预测** — 预测未来走势
   例: "预测各月新增客户趋势"

4. **异常检测** — 识别离群值
   例: "检测飞行距离的异常值"

5. **流失风险** — 客户流失预警
   例: "评估客户流失风险"

支持的图表类型: 柱状图、饼图、折线图、散点图、雷达图、热力图、漏斗图、箱线图、仪表盘、堆叠柱状图、面积图、树图、帕累托图"""


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    核心端点：优先使用 MCP Agent，不可用时降级到后端编排。
    """
    if not has_data():
        return jsonify({"error": "无数据，请先运行分析流水线"}), 400

    _init_tool_registry()
    _ensure_sandbox()

    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"error": "请输入问题"}), 400

    # 获取历史对话上下文（用于理解"上文"、"上面"等指代）
    history = request.json.get("history", [])

    mode = request.json.get("mode", "auto")

    # 手动模式：前端指定工具调用链
    if mode == "manual":
        return _handle_manual_chat(request.json)

    # 尝试使用 MCP Agent
    try:
        from agent import run_agent, is_mcp_available
        if is_mcp_available():
            print("[/api/chat] 使用 MCP Agent 模式")
            agent_result = run_agent(question, history=history)
            if agent_result.get("mode") != "error":
                # Agent 成功，处理分页逻辑
                return _process_agent_result(agent_result, request.json)
            else:
                print(f"[/api/chat] Agent 失败: {agent_result.get('answer')}")
    except Exception as e:
        print(f"[/api/chat] Agent 调用异常: {e}")

    # 降级到后端编排
    print("[/api/chat] 降级到后端编排模式")
    return _fallback_orchestration(question, request.json, history)


def _process_agent_result(agent_result: dict, request_data: dict) -> dict:
    """处理 Agent 结果，提取分页数据"""
    steps = agent_result.get("steps", [])
    answer = agent_result.get("answer", "")
    mode = agent_result.get("mode", "agent")

    # 查找 SQL 查询结果用于分页
    sql_step = None
    for step in reversed(steps):
        if step.get("tool") == "execute_secure_sql" and step.get("success"):
            sql_step = step
            break

    result = {
        "question": request_data.get("question", ""),
        "mode": mode,
        "answer": answer,
        "steps": steps,
    }

    if sql_step:
        sql_result = sql_step.get("result", {})
        data = sql_result.get("data", [])
        columns = sql_result.get("columns", [])
        total_rows = sql_result.get("row_count", len(data))

        # 分页处理
        page = max(1, int(request_data.get("page", 1)))
        page_size = max(5, min(200, int(request_data.get("page_size", 20))))

        if total_rows > page_size:
            query_id = f"q{int(time.time()*1000)}_{abs(hash(request_data.get('question', '')))%100000}"
            _QUERY_CACHE[query_id] = {
                "data": data,
                "columns": columns,
                "total": total_rows,
                "_time": time.time(),
            }
            _clean_query_cache()

            start = (page - 1) * page_size
            page_data = data[start:start + page_size]
            total_pages = max(1, (total_rows + page_size - 1) // page_size)

            # 更新步骤中的结果
            sql_step["result"] = {
                **sql_result,
                "data": page_data,
                "row_count": len(page_data),
                "total_rows": total_rows,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "query_id": query_id,
            }

    return jsonify(result)


def _fallback_orchestration(question: str, request_data: dict, history: list = None) -> dict:
    """降级到后端编排模式（接收 history 以保持对话连贯性）"""
    intent = _detect_intent(question)

    if intent == "help":
        return jsonify({
            "question": question,
            "mode": "help",
            "answer": _build_help_answer(),
            "steps": [],
        })

    # 分页参数
    page = max(1, int(request_data.get("page", 1)))
    page_size = max(5, min(200, int(request_data.get("page_size", 20))))

    if intent == "forecast":
        result = _orchestrate_forecast(question)
    elif intent == "anomaly":
        result = _orchestrate_anomaly(question)
    elif intent == "churn":
        result = _orchestrate_churn(question)
    elif intent == "chat":
        result = _orchestrate_chat(question)
    else:
        result = _orchestrate_sql_query(question, page=page, page_size=page_size)

    return jsonify(result)


@app.route("/api/chat/page", methods=["POST"])
def api_chat_page():
    """分页获取已缓存查询的结果（避免重复执行 SQL）"""
    data = request.json or {}
    query_id = data.get("query_id")
    page = max(1, int(data.get("page", 1)))
    page_size = max(5, min(200, int(data.get("page_size", 20))))

    if not query_id:
        return jsonify({"error": "缺少 query_id"}), 400

    _clean_query_cache()
    cached = _QUERY_CACHE.get(query_id)
    if not cached:
        return jsonify({"error": "查询已过期，请重新提问"}), 404

    full_data = cached["data"]
    total = cached["total"]
    start = (page - 1) * page_size
    page_data = full_data[start:start + page_size]
    total_pages = max(1, (total + page_size - 1) // page_size)

    return jsonify({
        "success": True,
        "data": page_data,
        "columns": cached["columns"],
        "page": page,
        "page_size": page_size,
        "total_rows": total,
        "total_pages": total_pages,
    })


@app.route("/api/chart/generate", methods=["POST"])
def api_chart_generate():
    """动态生成图表（用于图表类型切换）"""
    data = request.json or {}
    data_json = data.get("data_json")
    chart_type = data.get("chart_type", "auto")
    title = data.get("title", "")
    purpose = data.get("purpose", "")

    if not data_json:
        return jsonify({"error": "缺少 data_json"}), 400

    try:
        # 直接调用 chart_tools
        from mcp_tools.chart_tools import generate_visualization
        result = generate_visualization(data_json, chart_type, title, purpose)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


def _handle_manual_chat(payload: dict):
    """手动模式：前端指定工具调用序列"""
    tool_calls = payload.get("tool_calls", [])
    question = payload.get("question", "")

    results = []
    data_context = None  # 前一步的结果可被后一步引用

    for tc in tool_calls:
        tool_name = tc.get("tool")
        params = tc.get("params", {})

        if tool_name not in TOOL_REGISTRY:
            results.append({"tool": tool_name, "success": False,
                            "error": f"未知工具: {tool_name}"})
            continue

        # 替换引用（$prev.data → 前一步的 data）
        resolved_params = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("$prev."):
                ref_path = v[6:]
                resolved_params[k] = _resolve_ref(data_context, ref_path)
            else:
                resolved_params[k] = v

        try:
            fn = TOOL_REGISTRY[tool_name]["function"]
            result = fn(**resolved_params)
            results.append({"tool": tool_name, "success": True, "result": result})
            data_context = result
        except Exception as e:
            results.append({"tool": tool_name, "success": False,
                            "error": str(e)})

    return jsonify({
        "question": question,
        "mode": "manual",
        "steps": results,
        "final_result": results[-1] if results else None,
    })


def _resolve_ref(ctx, path: str):
    """解析 $prev.data 引用"""
    parts = path.split(".")
    val = ctx
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return None
    return val


def _fallback_regex(question: str):
    """纯关键词兜底（无 LLM 时使用）"""
    df = load_result()
    q = question.lower().strip()
    conditions = []

    nums = set(re.findall(r"(?<=簇)\s*(\d)", q)) | set(re.findall(r"(\d)\s*号簇", q))
    if nums and "cluster" in df.columns:
        conditions.append(df["cluster"].isin([int(n) for n in nums]))

    label_map = {"高价值": "高价值客户", "中价值": "中价值客户", "低价值": "低价值客户", "流失": "流失客户"}
    for kw, label in label_map.items():
        if kw in q and "value_label" in df.columns:
            conditions.append(df["value_label"] == label)
            break

    if ("男" in q or "男性" in q) and "GENDER" in df.columns:
        conditions.append(df["GENDER"] == "男")
    elif ("女" in q or "女性" in q) and "GENDER" in df.columns:
        conditions.append(df["GENDER"] == "女")

    ages = re.findall(r"(\d+)\s*岁", q) or re.findall(r"年龄[以大].*?(\d+)", q)
    if ages and "AGE" in df.columns:
        v = int(ages[0])
        conditions.append(df["AGE"] >= v if ("大于" in q or "以上" in q) else df["AGE"] <= v if ("小于" in q or "以下" in q) else df["AGE"] >= v)

    days = re.findall(r"近[的]?(\d+)天", q) or re.findall(r"最近(\d+)天", q)
    if days and "Recency" in df.columns:
        conditions.append(df["Recency"] <= int(days[0]))

    if conditions:
        mask = conditions[0]
        for c in conditions[1:]:
            mask &= c
        df = df[mask]

    page = int(request.json.get("page", 1))
    page_size = int(request.json.get("page_size", 20))
    page_size = min(page_size, 500)
    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size

    return jsonify({
        "question": question,
        "mode": "keyword",
        "columns": df.columns.tolist(),
        "rows": df.iloc[start:end].fillna("").to_dict(orient="records"),
        "total": total, "page": page, "page_size": page_size,
    })


# ==================== 流水线控制 ====================

@app.route("/api/pipeline/status")
def api_pipeline_status():
    return jsonify(PIPELINE_STATUS)


@app.route("/api/pipeline/run", methods=["POST"])
def api_pipeline_run():
    """触发分析流水线"""
    global PIPELINE_STATUS
    if PIPELINE_STATUS["running"]:
        return jsonify({"success": False, "error": "流水线正在运行中"}), 409

    PIPELINE_STATUS = {"running": True, "progress": "启动中...", "error": None}

    try:
        from core_algorithm.data_cleaning import load_data, clean_data, engineer_features
        from core_algorithm.rfm_analysis import (
            compute_rfm, label_customer_value, perform_clustering,
            preprocess_for_clustering, find_optimal_k,
            plot_cluster_radar, plot_pca_scatter, plot_cluster_composition,
        )

        PIPELINE_STATUS["progress"] = "加载数据..."
        df = load_data()

        PIPELINE_STATUS["progress"] = "清洗数据..."
        df = clean_data(df)

        PIPELINE_STATUS["progress"] = "特征工程..."
        df = engineer_features(df)

        PIPELINE_STATUS["progress"] = "计算 RFM..."
        df = compute_rfm(df)
        df = label_customer_value(df)

        PIPELINE_STATUS["progress"] = "K-Means 聚类..."
        X_scaled, _, _ = preprocess_for_clustering(df)
        opt_k = find_optimal_k(X_scaled)
        df = perform_clustering(df, n_clusters=opt_k["k"])

        PIPELINE_STATUS["progress"] = "生成可视化..."
        plot_cluster_radar(df)
        plot_pca_scatter(df)
        plot_cluster_composition(df)

        # 保存结果
        PIPELINE_STATUS["progress"] = "保存结果..."
        df.to_csv(FINAL_PATH, index=False, encoding="utf-8-sig")

        # 入库
        skip_db = request.json.get("skip_db", False)
        if not skip_db:
            PIPELINE_STATUS["progress"] = "数据入库..."
            try:
                from database import store_to_tables, verify_data
                store_to_tables(df)
                verify_data()
            except Exception as e:
                PIPELINE_STATUS["error"] = f"入库失败: {e}"
                print(f"[流水线] {e}")

        # 初始化沙箱
        PIPELINE_STATUS["progress"] = "初始化沙箱..."
        from mcp_tools.sql_tools import setup_sandbox_from_csv
        setup_sandbox_from_csv()

        PIPELINE_STATUS["running"] = False
        PIPELINE_STATUS["progress"] = "完成"
        return jsonify({"success": True, "rows": len(df),
                        "clusters": int(df["cluster"].nunique()),
                        "k": opt_k["k"]})
    except Exception as e:
        PIPELINE_STATUS["running"] = False
        PIPELINE_STATUS["error"] = str(e)
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== SQL 提取 ====================

def _extract_sql(text: str) -> str | None:
    text = re.sub(r"(?i)```sql", "", text)
    text = text.replace("```", "").strip()
    match = re.search(r"(?i)(SELECT\s+.+)", text, re.DOTALL)
    if not match:
        return None
    sql = match.group(1).strip().rstrip(";").strip()
    if not sql.lower().startswith("select"):
        return None
    for kw in config.SQL_BLACKLIST:
        if re.search(rf"\b{kw}\b", sql.lower()):
            return None
    return sql


# ==================== 启动 ====================

if __name__ == "__main__":
    _init_tool_registry()

    if not has_data():
        print("[启动] 未检测到分析结果，正在运行流水线...")
        try:
            from core_algorithm.data_cleaning import load_data, clean_data, engineer_features
            from core_algorithm.rfm_analysis import (
                compute_rfm, label_customer_value, perform_clustering,
                preprocess_for_clustering, find_optimal_k,
                plot_cluster_radar, plot_pca_scatter, plot_cluster_composition,
            )
            df = load_data()
            df = clean_data(df)
            df = engineer_features(df)
            df = compute_rfm(df)
            df = label_customer_value(df)
            X_scaled, _, _ = preprocess_for_clustering(df)
            opt_k = find_optimal_k(X_scaled)
            df = perform_clustering(df, n_clusters=opt_k["k"])
            plot_cluster_radar(df)
            plot_pca_scatter(df)
            plot_cluster_composition(df)
            df.to_csv(FINAL_PATH, index=False, encoding="utf-8-sig")

            try:
                from database import store_to_tables, verify_data
                store_to_tables(df)
                verify_data()
            except Exception as e:
                print(f"[启动] 数据库入库失败: {e}")

            from mcp_tools.sql_tools import setup_sandbox_from_csv
            setup_sandbox_from_csv()

            print(f"[启动] 流水线完成，共 {len(df)} 条记录")
        except Exception as e:
            traceback.print_exc()
            print(f"[启动] 流水线失败: {e}")

    print(f"\n{'='*50}")
    print(f"  客户分析智能查询系统 — MCP 架构版")
    print(f"  http://127.0.0.1:5000")
    print(f"  工具数: {len(TOOL_REGISTRY)}")
    print(f"  LLM: {config.LLM_PROVIDER}")
    print(f"{'='*50}\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
