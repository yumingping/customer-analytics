"""
全局配置文件 — MCP 架构版
=====================
支持 MySQL（生产）/ SQLite（沙箱）双模式，
多 LLM 提供商（Ollama / OpenAI 兼容 API）配置。
"""
import os
from pathlib import Path

# ==================== 项目路径 ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "项目5_客户分析与智能查询"
OUTPUT_DIR = BASE_DIR / "output"
SQLITE_PATH = str(OUTPUT_DIR / "sandbox.db")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 数据文件 ====================
CSV_FILE = str(DATA_DIR / "air.csv")

# ==================== 数据库配置（MySQL） ====================
# 若 MySQL 不可用，系统自动降级到 SQLite
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "airline_analytics",
}

# SQLAlchemy 连接 URL
DATABASE_URL = "mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4".format(
    **DB_CONFIG
)

# ==================== 三表映射 ====================
TABLES = {
    "base": "customer_base",
    "flight": "customer_flight_summary",
    "analytics": "customer_analytics",
}

# 兼容旧代码的别名
TABLE_NAME = "customer_rfm"

# ==================== 只读沙箱连接（MCP 查询工具专用） ====================
SQLITE_READONLY_URL = f"sqlite:///{SQLITE_PATH}"

# ==================== MCP Server 地址 ====================
MCP_SERVER_URL = "http://127.0.0.1:5001"

# ==================== LLM 提供商配置 ====================
# provider: "ollama" | "openai_compatible" | "none"
LLM_PROVIDER = "openai_compatible"  # 👈 第一处修改：启用 OpenAI 兼容模式

# Ollama (保持不变即可，因为上面已经改成了 openai_compatible)
OLLAMA_URL = "http://localhost:55555"
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_OPTIONS = {"temperature": 0.1}

# OpenAI 兼容 API（以 DeepSeek 为例）
OPENAI_API_URL = ""   # 填入云端 API 地址，如 https://api.deepseek.com/v1/chat/completions
OPENAI_API_KEY = ""   # 填入 API KEY（建议通过 config.json 或环境变量配置）
OPENAI_MODEL = ""     # 填入模型名称，如 deepseek-v4-flash

# ==================== 聚类参数 ====================
CLUSTER_RANGE = range(2, 9)   # K 值搜索范围
CLUSTER_RANDOM_STATE = 42
N_CLUSTERS = 4                # 默认聚类数

# ==================== 交叉校验容差 ====================
CROSS_VALIDATION_TOLERANCE = 0.01  # 1% 差异容忍度

# ==================== 运行时配置持久化 ====================
CONFIG_FILE = BASE_DIR / "config.json"


def _load_runtime_config():
    """从 config.json 加载运行时配置，覆盖 config.py 中的默认值"""
    if not CONFIG_FILE.exists():
        return {}
    try:
        import json
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_runtime_config(updates: dict):
    """将配置变更写入 config.json（不覆盖 config.py 中的代码默认值）"""
    import json
    current = _load_runtime_config()
    # 过滤掉空值和掩码占位符
    clean = {}
    for k, v in updates.items():
        if v is None:
            continue
        if isinstance(v, str) and "****" in v:
            continue
        clean[k] = v
    current.update(clean)
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[配置] 保存 config.json 失败: {e}")


_RUNTIME_CONFIG = _load_runtime_config()

# 用持久化配置覆盖默认值（如果存在）
LLM_PROVIDER = _RUNTIME_CONFIG.get("llm_provider", LLM_PROVIDER)
OLLAMA_URL = _RUNTIME_CONFIG.get("ollama_url", OLLAMA_URL)
OLLAMA_MODEL = _RUNTIME_CONFIG.get("ollama_model", OLLAMA_MODEL)
OPENAI_API_URL = _RUNTIME_CONFIG.get("openai_url", OPENAI_API_URL)
OPENAI_MODEL = _RUNTIME_CONFIG.get("openai_model", OPENAI_MODEL)
OPENAI_API_KEY = _RUNTIME_CONFIG.get("openai_key", OPENAI_API_KEY)
MCP_SERVER_URL = _RUNTIME_CONFIG.get("mcp_server_url", MCP_SERVER_URL)
N_CLUSTERS = _RUNTIME_CONFIG.get("n_clusters", N_CLUSTERS)
CLUSTER_RANDOM_STATE = _RUNTIME_CONFIG.get("cluster_random_state", CLUSTER_RANDOM_STATE)


# ==================== 沙箱安全 ====================
# SQL 黑名单关键词（禁止执行）
SQL_BLACKLIST = [
    "delete", "drop", "truncate", "insert", "update",
    "alter", "create", "grant", "replace", "exec",
]
