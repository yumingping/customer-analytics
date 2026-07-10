"""
MCP 工具: Schema 锚定与数据嗅探 (probing_tools)
=============================================
实现 readme.md 规定的两个抗幻觉核心工具：
- get_database_schema(): 返回三表字段定义，强制 AI 锚定真实结构
- probe_distinct_values(): 嗅探分类字段枚举值，杜绝 WHERE value_label='VIP' 类幻觉
"""
import json
from sqlalchemy import create_engine, text
import config


# ==================== 数据字典（硬编码，保证确定性） ====================

SCHEMA_DEFINITION = {
    "database": "airline_analytics",
    "description": "航空公司客户分析系统 — 三表分层架构",
    "tables": {
        "customer_base": {
            "description": "客户基础信息维度表（Dim），存储静态档案",
            "primary_key": "member_no",
            "columns": [
                {"name": "member_no",     "type": "VARCHAR(20)",  "nullable": False, "comment": "会员编号（唯一业务标识）"},
                {"name": "age",           "type": "TINYINT",      "nullable": True,  "comment": "年龄（已处理异常值）"},
                {"name": "gender",        "type": "VARCHAR(10)",  "nullable": True,  "comment": "性别（男/女/未知）"},
                {"name": "ffp_tier",      "type": "TINYINT",      "nullable": True,  "comment": "会员等级（如4,5,6等）"},
                {"name": "avg_discount",  "type": "DECIMAL(5,2)", "nullable": True,  "comment": "平均折扣率（反映购票价格敏感度）"},
                {"name": "ffp_date",      "type": "DATE",         "nullable": True,  "comment": "入会日期"},
                {"name": "first_flight",  "type": "DATE",         "nullable": True,  "comment": "首次飞行日期"},
                {"name": "create_time",   "type": "DATETIME",     "nullable": False, "comment": "记录首次创建时间"},
                {"name": "update_time",   "type": "DATETIME",     "nullable": False, "comment": "记录最近更新时间"},
            ],
            "indexes": [
                {"name": "idx_gender_tier", "columns": ["gender", "ffp_tier"], "type": "复合索引"}
            ]
        },
        "customer_flight_summary": {
            "description": "客户飞行事实汇总表（DWS），存储观察期内客观行为统计",
            "primary_key": "member_no",
            "columns": [
                {"name": "member_no",     "type": "VARCHAR(20)", "nullable": False, "comment": "会员编号（逻辑关联键）"},
                {"name": "flight_count",  "type": "INT",         "nullable": True,  "comment": "总飞行次数（Frequency原始来源）"},
                {"name": "seg_km_sum",    "type": "INT",         "nullable": True,  "comment": "总飞行里程（Monetary原始来源）"},
                {"name": "bp_sum",        "type": "INT",         "nullable": True,  "comment": "总基本积分"},
                {"name": "last_flight",   "type": "DATE",        "nullable": True,  "comment": "最后一次飞行日期"},
                {"name": "recency",       "type": "INT",         "nullable": True,  "comment": "距观察期末天数（越小越活跃）"},
                {"name": "create_time",   "type": "DATETIME",    "nullable": False, "comment": "记录首次创建时间"},
                {"name": "update_time",   "type": "DATETIME",    "nullable": False, "comment": "记录最近更新时间"},
            ],
            "indexes": [
                {"name": "idx_recency", "columns": ["recency"], "type": "普通索引"}
            ]
        },
        "customer_analytics": {
            "description": "客户AI分析结果表（ADS），存储RFM/K-Means算法生成的特征标签",
            "primary_key": "member_no",
            "columns": [
                {"name": "member_no",     "type": "VARCHAR(20)", "nullable": False, "comment": "会员编号（逻辑关联键）"},
                {"name": "r_score",       "type": "TINYINT",     "nullable": True,  "comment": "最近消费评分（1-5，基于百分位排名）"},
                {"name": "f_score",       "type": "TINYINT",     "nullable": True,  "comment": "消费频率评分（1-5，基于百分位排名）"},
                {"name": "m_score",       "type": "TINYINT",     "nullable": True,  "comment": "消费金额评分（1-5，基于百分位排名）"},
                {"name": "rfm_total",     "type": "TINYINT",     "nullable": True,  "comment": "RFM总分（3-15）"},
                {"name": "cluster",       "type": "TINYINT",     "nullable": True,  "comment": "聚类标签（0,1,2,3）"},
                {"name": "value_label",   "type": "VARCHAR(20)", "nullable": True,  "comment": "客户价值标签（高价值/中价值/低价值/流失客户）"},
                {"name": "update_time",   "type": "DATETIME",    "nullable": False, "comment": "算法模型最近一次更新时间"},
            ],
            "indexes": [
                {"name": "idx_cluster",   "columns": ["cluster"],                      "type": "普通索引"},
                {"name": "idx_rfm_label", "columns": ["rfm_total", "value_label"],     "type": "复合索引"}
            ]
        }
    },
    "join_relationships": [
        {
            "description": "三表通过 member_no 进行 1:1 逻辑关联",
            "sql_template": "SELECT ... FROM customer_base b JOIN customer_flight_summary f ON b.member_no = f.member_no JOIN customer_analytics a ON b.member_no = a.member_no"
        }
    ],
    "business_rules": {
        "value_label": {
            "高价值客户": "rfm_total >= 13",
            "中价值客户": "rfm_total >= 9 AND rfm_total < 13",
            "低价值客户": "rfm_total >= 6 AND rfm_total < 9",
            "流失客户": "rfm_total < 6",
        },
        "r_score": "1-5，基于 Recency 百分位排名，分数越高表示最近越活跃",
        "f_score": "1-5，基于 Frequency 百分位排名，分数越高表示飞行越频繁",
        "m_score": "1-5，基于 Monetary 百分位排名，分数越高表示消费越高",
        "cluster": "0-3，基于连续特征 Log1p+StandardScaler+KMeans 聚类",
        "recency": "距观察期末（2014-03-31）天数，越小越活跃",
    }
}


def get_database_schema() -> dict:
    """
    获取完整的数据库 Schema 定义。
    AI 在编写任何查询前必须调用此工具，获取精准的表结构和业务字段释义。
    （readme.md 第3.1节 — Schema 强制锚定 / Metadata Grounding）
    """
    return SCHEMA_DEFINITION


def get_table_schema(table_name: str) -> dict:
    """获取单张表的结构信息"""
    for key, table_info in SCHEMA_DEFINITION["tables"].items():
        if key == table_name:
            return table_info
    return {"error": f"表 '{table_name}' 不存在", "available_tables": list(SCHEMA_DEFINITION["tables"].keys())}


def probe_distinct_values(column_name: str, table_name: str = None) -> dict:
    """
    嗅探指定列的枚举值（实际数据中的取值）。
    AI 在构建 WHERE 条件前必须调用此工具确认真实数据取值。
    （readme.md 第3.2节 — 数据枚举值嗅探 / Data Probing）

    参数:
        column_name: 列名，如 "value_label", "gender", "cluster"
        table_name: 目标表名，不指定则自动寻找

    返回:
        {"column": "value_label", "table": "customer_analytics",
         "values": ["高价值客户", "中价值客户", "低价值客户", "流失客户"],
         "counts": {...}, "distinct_count": 4}
    """
    # ── 兜底值（Schema 理论取值）──
    _FALLBACK_VALUES = {
        "value_label": ["高价值客户", "中价值客户", "低价值客户", "流失客户"],
        "gender": ["男", "女", "未知"],
        "cluster": [0, 1, 2, 3],
        "r_score": [1, 2, 3, 4, 5],
        "f_score": [1, 2, 3, 4, 5],
        "m_score": [1, 2, 3, 4, 5],
        "ffp_tier": [4, 5, 6],
    }
    _COLUMN_TO_RFM = {
        # 三表列名 → customer_rfm 宽表中的对应列名
        "member_no": "member_no",
        "age": "age", "gender": "gender", "ffp_tier": "ffp_tier",
        "avg_discount": "avg_discount",
        "recency": "recency",
        "r_score": "r_score", "f_score": "f_score", "m_score": "m_score",
        "rfm_total": "rfm_total", "cluster": "cluster",
        "value_label": "value_label",
    }

    # 定位表
    if table_name is None:
        for tkey, tname in config.TABLES.items():
            for col_def in SCHEMA_DEFINITION["tables"].get(tname, {}).get("columns", []):
                if col_def["name"] == column_name:
                    table_name = tname
                    break
            if table_name:
                break

    if table_name is None:
        return {"error": f"无法定位列 '{column_name}'，请先用 get_database_schema 确认字段名"}

    # ── 查询（三表优先 → customer_rfm 宽表回退 → Schema 兜底）──
    values = []
    counts = {}
    error_msg = None

    try:
        from database import get_engine
        engine = get_engine(readonly=True)

        # 尝试1：查指定表
        values, counts = _query_column(engine, column_name, table_name)

        # 尝试2：三表为空 → 回退查 customer_rfm 宽表
        if not values and table_name != "customer_rfm":
            rfm_col = _COLUMN_TO_RFM.get(column_name, column_name)
            print(f"[探测] {table_name}.{column_name} 为空，回退查 customer_rfm.{rfm_col}")
            values, counts = _query_column(engine, rfm_col, "customer_rfm")

    except Exception as e:
        error_msg = str(e)

    # ── 尝试3：数据库也失败/为空 → Schema 兜底 ──
    if not values:
        fallback = _FALLBACK_VALUES.get(column_name)
        if fallback:
            counts = {str(v): 0 for v in fallback}
        return {
            "column": column_name,
            "table": table_name,
            "values": [str(v) for v in fallback] if fallback else [],
            "counts": counts,
            "distinct_count": len(fallback) if fallback else 0,
            "source": "schema_fallback",
            "note": "数据库无数据或查询失败，返回 Schema 理论取值",
            "error": error_msg,
        }

    return {
        "column": column_name,
        "table": table_name,
        "values": values,
        "counts": counts,
        "distinct_count": len(values),
    }


def _query_column(engine, column_name: str, table_name: str) -> tuple:
    """在指定表中查询列的 DISTINCT 值，返回 (values, counts)"""
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT DISTINCT `{column_name}` FROM `{table_name}` "
                     f"WHERE `{column_name}` IS NOT NULL LIMIT 50")
            ).fetchall()
        values = [row[0] for row in rows]

        counts = {}
        if values:
            try:
                with engine.connect() as conn:
                    count_rows = conn.execute(
                        text(f"SELECT `{column_name}`, COUNT(*) as cnt FROM `{table_name}` "
                             f"WHERE `{column_name}` IS NOT NULL GROUP BY `{column_name}`")
                    ).fetchall()
                counts = {str(row[0]): int(row[1]) for row in count_rows}
            except Exception:
                pass

        return values, counts
    except Exception:
        return [], {}


def get_business_rules() -> dict:
    """返回业务规则（价值标签分段逻辑等）"""
    return SCHEMA_DEFINITION.get("business_rules", {})
