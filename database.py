"""
数据库模块 — MCP 架构版
=====================
- MySQL（生产）/ SQLite（沙箱）双模式，连接失败自动降级
- 三表分层：customer_base / customer_flight_summary / customer_analytics
- MCP 查询工具使用只读连接，防止误写
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, inspect, event
from sqlalchemy.engine import Engine
from sqlalchemy import String
from contextlib import contextmanager
from pathlib import Path
import config


# ==================== 引擎管理 ====================

_engine = None
_engine_type = None  # "mysql" | "sqlite"


def get_engine(readonly: bool = False) -> "Engine":
    """
    获取数据库引擎。
    - 优先连接 MySQL
    - 连接失败自动降级到 SQLite
    - readonly=True 用于 MCP 查询工具（事务级只读保护）
    """
    global _engine, _engine_type

    if _engine is not None:
        return _engine

    # 尝试 MySQL
    try:
        engine = create_engine(config.DATABASE_URL, echo=False, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _engine = engine
        _engine_type = "mysql"
        print(f"[数据库] MySQL 连接成功 ({config.DB_CONFIG['host']}:{config.DB_CONFIG['port']})")

        if readonly:
            _set_readonly_mysql(engine)

        return _engine
    except Exception as e:
        print(f"[数据库] MySQL 不可用 ({e})，降级到 SQLite")

    # 降级 SQLite
    sqlite_url = config.SQLITE_READONLY_URL if readonly else f"sqlite:///{config.SQLITE_PATH}"
    engine = create_engine(sqlite_url, echo=False)

    if readonly:
        _set_readonly_sqlite(engine)

    _engine = engine
    _engine_type = "sqlite"
    print(f"[数据库] 使用 SQLite ({config.SQLITE_PATH})")

    return _engine


def _set_readonly_mysql(engine: Engine):
    """MySQL 只读事务保护"""
    @event.listens_for(engine, "connect")
    def set_readonly(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("SET SESSION TRANSACTION READ ONLY")
        cursor.close()


def _set_readonly_sqlite(engine: Engine):
    """SQLite 只读连接"""
    @event.listens_for(engine, "connect")
    def set_readonly(dbapi_conn, connection_record):
        dbapi_conn.execute("PRAGMA query_only = ON")


def get_engine_type() -> str:
    """返回当前引擎类型: mysql | sqlite"""
    global _engine_type
    if _engine_type is None:
        get_engine()
    return _engine_type


def reset_engine():
    """重置引擎（配置变更后调用）"""
    global _engine, _engine_type
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _engine_type = None


@contextmanager
def get_connection(readonly: bool = False):
    """上下文管理器：获取数据库连接，自动关闭"""
    engine = get_engine(readonly=readonly)
    conn = engine.connect()
    try:
        yield conn
        if not readonly:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ==================== 数据库初始化 ====================

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS customer_base (
        member_no      VARCHAR(20)   NOT NULL,
        age            INTEGER       DEFAULT NULL,
        gender         VARCHAR(10)   DEFAULT NULL,
        ffp_tier       INTEGER       DEFAULT NULL,
        avg_discount   DECIMAL(5,2)  DEFAULT NULL,
        ffp_date       DATE          DEFAULT NULL,
        first_flight   DATE          DEFAULT NULL,
        create_time    DATETIME      DEFAULT CURRENT_TIMESTAMP,
        update_time    DATETIME      DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (member_no)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS customer_flight_summary (
        member_no      VARCHAR(20)   NOT NULL,
        flight_count   INTEGER       DEFAULT 0,
        seg_km_sum     INTEGER       DEFAULT 0,
        bp_sum         INTEGER       DEFAULT 0,
        last_flight    DATE          DEFAULT NULL,
        recency        INTEGER       DEFAULT NULL,
        create_time    DATETIME      DEFAULT CURRENT_TIMESTAMP,
        update_time    DATETIME      DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (member_no)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS customer_analytics (
        member_no      VARCHAR(20)   NOT NULL,
        r_score        INTEGER       DEFAULT NULL,
        f_score        INTEGER       DEFAULT NULL,
        m_score        INTEGER       DEFAULT NULL,
        rfm_total      INTEGER       DEFAULT NULL,
        cluster        INTEGER       DEFAULT NULL,
        value_label    VARCHAR(20)   DEFAULT NULL,
        update_time    DATETIME      DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (member_no)
    )
    """,
]

# 索引（MySQL 用 CREATE INDEX，SQLite 用 IF NOT EXISTS）
INDEX_STATEMENTS_MYSQL = [
    "CREATE INDEX idx_base_gender_tier ON customer_base (gender, ffp_tier)",
    "CREATE INDEX idx_flight_recency ON customer_flight_summary (recency)",
    "CREATE INDEX idx_analytics_cluster ON customer_analytics (cluster)",
    "CREATE INDEX idx_analytics_rfm_label ON customer_analytics (rfm_total, value_label)",
]

INDEX_STATEMENTS_SQLITE = [
    "CREATE INDEX IF NOT EXISTS idx_base_gender_tier ON customer_base (gender, ffp_tier)",
    "CREATE INDEX IF NOT EXISTS idx_flight_recency ON customer_flight_summary (recency)",
    "CREATE INDEX IF NOT EXISTS idx_analytics_cluster ON customer_analytics (cluster)",
    "CREATE INDEX IF NOT EXISTS idx_analytics_rfm_label ON customer_analytics (rfm_total, value_label)",
]


def init_database(drop_if_exists: bool = False):
    """
    初始化数据库：建表 + 索引。
    drop_if_exists=True 时先删表重建（确保数据干净）。
    """
    engine_type = get_engine_type()
    engine = get_engine()

    with engine.connect() as conn:
        if drop_if_exists:
            # 先删视图再删表
            for obj in [config.TABLE_NAME] + list(config.TABLES.values()):
                try:
                    conn.execute(text(f"DROP VIEW IF EXISTS `{obj}`"))
                except Exception:
                    pass
            for table in reversed(list(config.TABLES.values())):
                try:
                    conn.execute(text(f"DROP TABLE IF EXISTS `{table}`"))
                    print(f"[数据库] 已删除旧表: {table}")
                except Exception as e:
                    print(f"[数据库] 删表警告: {e}")

        for stmt in DDL_STATEMENTS:
            try:
                conn.execute(text(stmt))
            except Exception as e:
                print(f"[数据库] 建表警告: {e}")

        # 索引
        if engine_type == "mysql":
            for stmt in INDEX_STATEMENTS_MYSQL:
                try:
                    conn.execute(text(stmt))
                except Exception as e:
                    print(f"[数据库] 索引警告: {e}")
        else:
            for stmt in INDEX_STATEMENTS_SQLITE:
                try:
                    conn.execute(text(stmt))
                except Exception:
                    pass

        conn.commit()

    print(f"[数据库] 初始化完成 (引擎: {engine_type})")


# ==================== 数据入库（三表拆分） ====================

def store_to_tables(df: pd.DataFrame) -> dict:
    """
    将分析结果拆分写入三张表。
    返回每条表的写入行数。
    """
    init_database(drop_if_exists=True)
    engine = get_engine()
    engine_type = get_engine_type()
    # MySQL 用 SQLAlchemy String 类型对象，SQLite 不指定 dtype
    dtype_member_no = {"member_no": String(20)} if engine_type == "mysql" else None
    result = {}

    # --- 表 1: customer_base ---
    base_cols_map = {
        "MEMBER_NO": "member_no",
        "AGE": "age",
        "GENDER": "gender",
        "FFP_TIER": "ffp_tier",
        "avg_discount": "avg_discount",
        "FFP_DATE": "ffp_date",
        "FIRST_FLIGHT_DATE": "first_flight",
    }
    base_df = _extract_columns(df, base_cols_map)
    if not base_df.empty:
        base_df.to_sql(
            config.TABLES["base"], engine,
            if_exists="replace", index=False,
            dtype=dtype_member_no,
            chunksize=1000,
        )
        result["base"] = len(base_df)
        print(f"[入库] customer_base: {len(base_df)} 条")

    # --- 表 2: customer_flight_summary ---
    flight_cols_map = {
        "MEMBER_NO": "member_no",
        "FLIGHT_COUNT": "flight_count",
        "SEG_KM_SUM": "seg_km_sum",
        "BP_SUM": "bp_sum",
        "LAST_FLIGHT_DATE": "last_flight",
        "Recency": "recency",
    }
    flight_df = _extract_columns(df, flight_cols_map)
    if not flight_df.empty:
        flight_df.to_sql(
            config.TABLES["flight"], engine,
            if_exists="replace", index=False,
            dtype=dtype_member_no,
            chunksize=1000,
        )
        result["flight"] = len(flight_df)
        print(f"[入库] customer_flight_summary: {len(flight_df)} 条")

    # --- 表 3: customer_analytics ---
    analytics_cols_map = {
        "MEMBER_NO": "member_no",
        "R_score": "r_score",
        "F_score": "f_score",
        "M_score": "m_score",
        "RFM_total": "rfm_total",
        "cluster": "cluster",
        "value_label": "value_label",
    }
    analytics_df = _extract_columns(df, analytics_cols_map)
    if not analytics_df.empty:
        analytics_df.to_sql(
            config.TABLES["analytics"], engine,
            if_exists="replace", index=False,
            dtype=dtype_member_no,
            chunksize=1000,
        )
        result["analytics"] = len(analytics_df)
        print(f"[入库] customer_analytics: {len(analytics_df)} 条")

    # 创建宽表视图（兼容旧版 SQL 查询）
    _ensure_wide_view()

    return result


def _extract_columns(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """从 DataFrame 中提取并重命名列，处理缺失列"""
    result = pd.DataFrame()
    for src, dst in col_map.items():
        if src in df.columns:
            result[dst] = df[src]
    return result


# ==================== 数据读取 ====================

def load_from_tables() -> pd.DataFrame:
    """三表 JOIN 读取全部数据"""
    engine = get_engine(readonly=True)
    sql = """
        SELECT
            b.member_no, b.age, b.gender, b.ffp_tier,
            b.avg_discount, b.ffp_date, b.first_flight,
            f.flight_count, f.seg_km_sum, f.bp_sum,
            f.last_flight, f.recency,
            a.r_score, a.f_score, a.m_score,
            a.rfm_total, a.cluster, a.value_label
        FROM customer_base b
        JOIN customer_flight_summary f ON b.member_no = f.member_no
        JOIN customer_analytics a ON b.member_no = a.member_no
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    print(f"[读取] 三表 JOIN: {len(df)} 条记录")
    return df


def load_table(table_key: str) -> pd.DataFrame:
    """读取单张表"""
    table_name = config.TABLES.get(table_key, table_key)
    engine = get_engine(readonly=True)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {table_name}"), conn)
    return df


# ==================== 数据验证 ====================

def verify_data() -> dict:
    """验证三表数据完整性"""
    engine = get_engine(readonly=True)
    result = {}

    with engine.connect() as conn:
        for key, table in config.TABLES.items():
            try:
                count = conn.execute(
                    text(f"SELECT COUNT(*) FROM `{table}`")
                ).scalar()
                result[key] = {"table": table, "row_count": count}
            except Exception as e:
                result[key] = {"table": table, "error": str(e)}

    print("\n[数据验证]")
    for key, info in result.items():
        if "error" in info:
            print(f"  {info['table']}: 错误 - {info['error']}")
        else:
            print(f"  {info['table']}: {info['row_count']} 条")

    # 交叉验证：三表 member_no 应一致
    try:
        with engine.connect() as conn:
            base_set = set(row[0] for row in conn.execute(
                text(f"SELECT member_no FROM `{config.TABLES['base']}`")
            ).fetchall())
            analytics_set = set(row[0] for row in conn.execute(
                text(f"SELECT member_no FROM `{config.TABLES['analytics']}`")
            ).fetchall())
            result["cross_check"] = {
                "base_only": len(base_set - analytics_set),
                "analytics_only": len(analytics_set - base_set),
                "common": len(base_set & analytics_set),
            }
    except Exception:
        result["cross_check"] = {"error": "无法执行交叉验证"}

    return result


def verify_table_exists(table_key: str) -> bool:
    """检查指定表是否存在"""
    table_name = config.TABLES.get(table_key, table_key)
    engine = get_engine()
    inspector = inspect(engine)
    return inspector.has_table(table_name)


# ==================== MySQL 兼容的大宽表视图（兼容旧版查询） ====================

def _ensure_wide_view():
    """
    在 MySQL/SQLite 中创建 customer_rfm 视图，
    将三表 JOIN 为单张大宽表，兼容旧版 SQL 查询模式。
    """
    engine = get_engine()
    view_sql = f"""
    CREATE VIEW {config.TABLE_NAME} AS
    SELECT
        b.member_no, b.age, b.gender, b.ffp_tier,
        b.avg_discount,
        f.flight_count, f.seg_km_sum, f.bp_sum,
        f.last_flight, f.recency AS recency,
        f.flight_count AS frequency,
        f.seg_km_sum AS monetary,
        a.r_score, a.f_score, a.m_score,
        a.rfm_total, a.cluster, a.value_label
    FROM {config.TABLES['base']} b
    JOIN {config.TABLES['flight']} f ON b.member_no = f.member_no
    JOIN {config.TABLES['analytics']} a ON b.member_no = a.member_no
    """
    try:
        with engine.connect() as conn:
            conn.execute(text(f"DROP VIEW IF EXISTS {config.TABLE_NAME}"))
            conn.execute(text(view_sql))
            conn.commit()
            print(f"[视图] customer_rfm 视图已创建 (三表 JOIN)")
    except Exception as e:
        print(f"[视图] 创建视图失败 (可能引擎不支持): {e}")


# ==================== 独立测试入口 ====================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(config.BASE_DIR))

    from data_cleaning import load_data, clean_data, engineer_features
    from rfm_analysis import compute_rfm, label_customer_value
    from clustering import perform_clustering

    print("=" * 50)
    print("数据库模块测试")
    print("=" * 50)

    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = compute_rfm(df)
    df = label_customer_value(df)
    df = perform_clustering(df)

    store_to_tables(df)
    verify_data()
    _ensure_wide_view()

    print("\n测试完成!")
