"""
MCP 工具: 安全 SQL 执行与沙箱 (sql_tools)
=========================================
实现 readme.md 规定的沙箱隔离与自适应纠错：
- execute_secure_sql(): 沙箱执行 SQL，防注入，捕获并返回错误堆栈
- validate_sql_syntax(): 语法校验
- auto_retry 装饰器: 执行失败时自动反馈错误给 AI 重试
"""
import re
import json
import traceback
import pandas as pd
from functools import wraps
from sqlalchemy import create_engine, text
from pathlib import Path
import config


# ==================== SQL 安全检查 ====================

def validate_sql_syntax(sql_query: str) -> dict:
    """
    校验 SQL 语法（基础检查 + sqlparse）。
    返回 {"valid": bool, "errors": [...]}
    """
    errors = []

    # 1. 类型检查
    if not sql_query or not isinstance(sql_query, str):
        return {"valid": False, "errors": ["SQL 查询不能为空"]}

    sql_stripped = sql_query.strip().lower()

    # 2. 只允许 SELECT
    if not sql_stripped.startswith("select"):
        errors.append("仅允许执行 SELECT 语句")
        return {"valid": False, "errors": errors}

    # 3. 黑名单关键词检测
    for kw in config.SQL_BLACKLIST:
        if re.search(rf"\b{kw}\b", sql_stripped):
            errors.append(f"禁止使用关键词: {kw}")

    # 4. sqlparse 语法解析（可选）
    try:
        import sqlparse
        parsed = sqlparse.parse(sql_query)
        if not parsed:
            errors.append("SQL 解析失败: 无法识别")
        else:
            # 检查是否有多个语句
            statements = [s for s in parsed if s.tokens and str(s).strip()]
            if len(statements) > 1:
                errors.append("不允许同时执行多条 SQL 语句")
    except ImportError:
        pass  # sqlparse 未安装则跳过
    except Exception as e:
        pass  # 解析失败不影响执行

    return {"valid": len(errors) == 0, "errors": errors}


# ==================== 沙箱执行 ====================

def execute_secure_sql(sql_query: str, sandbox_db_path: str = None,
                       data_source: pd.DataFrame = None) -> dict:
    """
    安全执行 SQL SELECT。
    优先在 MySQL 上执行（更快），MySQL 不可用时降级到 SQLite 沙箱。

    参数:
        sql_query: 要执行的 SQL SELECT 语句
        sandbox_db_path: 沙箱 SQLite 路径（降级时使用）
        data_source: 可选，通过 DataFrame 传入数据（写入 SQLite 沙箱）

    返回:
        {
            "success": bool,
            "data": [...],       # 结果行列表（JSON）
            "columns": [...],    # 列名
            "row_count": int,    # 返回行数
            "error": str | None, # 错误信息（含堆栈）
        }
    """
    # 1. 语法校验
    validation = validate_sql_syntax(sql_query)
    if not validation["valid"]:
        return {
            "success": False,
            "data": None,
            "columns": None,
            "row_count": 0,
            "error": "; ".join(validation["errors"]),
            "error_type": "validation_error",
        }

    # 2. 优先尝试 MySQL
    engine = None
    try:
        from database import get_engine_type, get_engine as _get_db_engine
        engine_type = get_engine_type()
        if engine_type == "mysql":
            engine = _get_db_engine(readonly=True)
            with engine.connect() as conn:
                result_df = pd.read_sql(text(sql_query), conn)

            return {
                "success": True,
                "data": result_df.fillna("").to_dict(orient="records"),
                "columns": result_df.columns.tolist(),
                "row_count": len(result_df),
                "error": None,
                "engine": "mysql",
            }
    except Exception as e:
        # MySQL 失败，降级到 SQLite 沙箱
        print(f"[SQL] MySQL 执行失败 ({e})，降级到 SQLite 沙箱")
        engine = None

    # 3. 降级：SQLite 沙箱
    db_path = sandbox_db_path or config.SQLITE_PATH
    sandbox_url = f"sqlite:///{db_path}"

    try:
        engine = create_engine(sandbox_url, echo=False)

        # 如果传入了 data_source，写入沙箱
        if data_source is not None and isinstance(data_source, pd.DataFrame):
            _write_dataframe_to_sandbox(engine, data_source)

        with engine.connect() as conn:
            result_df = pd.read_sql(text(sql_query), conn)

        return {
            "success": True,
            "data": result_df.fillna("").to_dict(orient="records"),
            "columns": result_df.columns.tolist(),
            "row_count": len(result_df),
            "error": None,
            "engine": "sqlite",
        }

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "success": False,
            "data": None,
            "columns": None,
            "row_count": 0,
            "error": str(e),
            "error_traceback": tb,
            "error_type": type(e).__name__,
        }
    finally:
        if engine is not None:
            engine.dispose()


# ── 三表列映射（CSV 原始列名 → DB 列名）──

_BASE_COL_MAP = {
    "MEMBER_NO": "member_no", "AGE": "age", "GENDER": "gender",
    "FFP_TIER": "ffp_tier", "avg_discount": "avg_discount",
    "FFP_DATE": "ffp_date", "FIRST_FLIGHT_DATE": "first_flight",
}

_FLIGHT_COL_MAP = {
    "MEMBER_NO": "member_no", "FLIGHT_COUNT": "flight_count",
    "SEG_KM_SUM": "seg_km_sum", "BP_SUM": "bp_sum",
    "LAST_FLIGHT_DATE": "last_flight", "Recency": "recency",
}

_ANALYTICS_COL_MAP = {
    "MEMBER_NO": "member_no",
    "R_score": "r_score", "F_score": "f_score", "M_score": "m_score",
    "RFM_total": "rfm_total", "cluster": "cluster",
    "value_label": "value_label",
}

_WIDE_COL_MAP = {
    "MEMBER_NO": "member_no", "AGE": "age", "GENDER": "gender",
    "FFP_TIER": "ffp_tier", "Recency": "recency",
    "Frequency": "frequency", "Monetary": "monetary",
    "R_score": "r_score", "F_score": "f_score", "M_score": "m_score",
    "RFM_total": "rfm_total", "cluster": "cluster",
    "avg_discount": "avg_discount", "value_label": "value_label",
}


def _map_columns(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """从 DataFrame 提取并重命名列，不依赖 SQLAlchemy 类型"""
    data = {}
    for src, dst in col_map.items():
        if src in df.columns:
            data[dst] = df[src]
    return pd.DataFrame(data) if data else pd.DataFrame()


def _write_dataframe_to_sandbox(engine, df: pd.DataFrame):
    """将 DataFrame 写入 SQLite 沙箱：三表 + customer_rfm 宽表"""
    # 1. customer_base
    base_df = _map_columns(df, _BASE_COL_MAP)
    if not base_df.empty:
        base_df.to_sql("customer_base", engine, if_exists="replace", index=False, chunksize=5000)
        print(f"  [沙箱] customer_base: {len(base_df)} 条")

    # 2. customer_flight_summary
    flight_df = _map_columns(df, _FLIGHT_COL_MAP)
    if not flight_df.empty:
        flight_df.to_sql("customer_flight_summary", engine, if_exists="replace", index=False, chunksize=5000)
        print(f"  [沙箱] customer_flight_summary: {len(flight_df)} 条")

    # 3. customer_analytics
    analytics_df = _map_columns(df, _ANALYTICS_COL_MAP)
    if not analytics_df.empty:
        analytics_df.to_sql("customer_analytics", engine, if_exists="replace", index=False, chunksize=5000)
        print(f"  [沙箱] customer_analytics: {len(analytics_df)} 条")

    # 4. customer_rfm 宽表（兼容旧查询）
    wide_df = _map_columns(df, _WIDE_COL_MAP)
    if not wide_df.empty:
        wide_df.to_sql("customer_rfm", engine, if_exists="replace", index=False, chunksize=5000)
        print(f"  [沙箱] customer_rfm (宽表): {len(wide_df)} 条")


def _remove_or_rename_sandbox(sandbox_file: Path):
    """
    删除旧的沙箱文件。若 Windows 下文件被占用无法删除，
    则重命名为带时间戳的备份，保证新文件能创建成功。
    """
    if not sandbox_file.exists():
        return

    # 1. 释放当前进程中可能持有的数据库连接
    try:
        import database
        database.reset_engine()
    except Exception:
        pass

    # 2. 尝试删除，带重试
    import time
    last_err = None
    for attempt in range(5):
        try:
            sandbox_file.unlink()
            print(f"[沙箱] 已删除旧沙箱: {sandbox_file}")
            return
        except PermissionError as e:
            last_err = e
            print(f"[沙箱] 删除旧沙箱失败 (尝试 {attempt + 1}/5): {e}")
            time.sleep(0.2 * (attempt + 1))

    # 3. 删除失败则重命名备份
    backup_name = sandbox_file.with_suffix(f".backup_{int(time.time())}.db")
    try:
        sandbox_file.rename(backup_name)
        print(f"[沙箱] 旧沙箱被占用，已重命名为备份: {backup_name}")
    except Exception as e:
        print(f"[沙箱] 重命名旧沙箱也失败: {e}")
        if last_err:
            raise last_err


def setup_sandbox_from_csv(csv_path: str = None) -> str:
    """
    从 CSV 文件初始化沙箱数据库（流水线运行后调用）。
    直接写 SQLite，不走 database.store_to_tables（那是 MySQL 路径）。
    返回沙箱路径。
    """
    csv_path = csv_path or str(config.OUTPUT_DIR / "final_result.csv")
    sandbox_path = config.SQLITE_PATH

    # 删除旧沙箱，全量重建；若被占用则重命名备份
    _remove_or_rename_sandbox(Path(sandbox_path))

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        print(f"[沙箱] 读取 CSV: {csv_path} ({len(df)} 条)")
    except Exception as e:
        print(f"[沙箱] 无法读取 CSV: {e}")
        return sandbox_path

    engine = create_engine(f"sqlite:///{sandbox_path}", echo=False)
    try:
        _write_dataframe_to_sandbox(engine, df)
    except Exception as e:
        print(f"[沙箱] 写入失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.dispose()

    print(f"[沙箱] SQLite 沙箱已就绪: {sandbox_path}")
    return sandbox_path


# ==================== 自动重试装饰器 ====================

def auto_retry(max_attempts: int = 3):
    """
    自动重试装饰器。
    当 SQL 执行失败时，将错误堆栈回传给调用方（AI），
    让 AI 根据错误信息修正后重新执行。
    （readme.md 第8节 — 沙箱自愈组件）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(sql_query: str, *args, **kwargs):
            last_error = None
            current_sql = sql_query

            for attempt in range(max_attempts):
                result = func(current_sql, *args, **kwargs)

                if result.get("success"):
                    result["attempts"] = attempt + 1
                    return result

                last_error = result
                current_sql = kwargs.pop("_retry_sql", None)
                if current_sql is None:
                    break

            # 所有尝试失败
            if last_error:
                last_error["attempts"] = max_attempts
                last_error["error"] += f" (已重试 {max_attempts} 次)"
            return last_error

        return wrapper
    return decorator


def llm_fix_sql(original_sql: str, error_message: str,
                schema: dict = None) -> dict:
    """
    将错误堆栈组织为 AI 可理解的修正提示。
    供 app.py 中的 LLM 调用工具链使用。
    返回修正提示词，让 AI 据此重新生成 SQL。
    """
    if schema is None:
        from mcp_tools.probing_tools import get_database_schema
        schema = get_database_schema()

    # 提取关键表字段（简化版）
    tables_info = schema.get("tables", {})
    field_list = []
    for tname, tinfo in tables_info.items():
        cols = [c["name"] for c in tinfo.get("columns", [])]
        field_list.append(f"  {tname}: {', '.join(cols)}")

    fix_prompt = f"""之前的 SQL 执行失败，请修正。

原始 SQL:
```sql
{original_sql}
```

错误信息:
{error_message}

可用的数据表与字段:
{chr(10).join(field_list)}

业务规则:
- value_label 取值: '高价值客户', '中价值客户', '低价值客户', '流失客户'
- gender 取值: '男', '女', '未知'
- cluster 取值: 0, 1, 2, 3

请只返回修正后的纯 SELECT SQL 语句。"""
    return {"fix_prompt": fix_prompt, "original_sql": original_sql}
