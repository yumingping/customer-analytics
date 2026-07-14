"""
MCP 工具: 预测与异常检测 (forecast_tools)
============================================
扩展 MCP 能力，支持：
- forecast_trend(): 简单趋势预测（线性回归 / 指数平滑）
- detect_anomalies(): 基于 IQR / Z-Score 的异常值检测
- churn_risk_score(): 基于 RFM 特征的流失风险评分

【Compute-over-Data 架构】
所有工具优先支持 sql_query / table_name 参数，在工具内部直连数据库加载全量数据，
绕过 LLM 上下文传输，彻底消除数据截断问题。
data_json 仅作为降级通道（小数据量测试用）。
"""
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest


# ==================== 共享数据加载器 ====================

def _load_dataframe(sql_query: str = None, table_name: str = None,
                    data_json: str = None, max_rows: int = None) -> pd.DataFrame:
    """
    统一数据加载器：优先 sql_query → table_name → data_json。
    从数据库加载时，数据不经过 LLM 上下文，不受截断限制。
    返回 (DataFrame, source_description)。
    """
    # 通道 1：SQL 直连数据库（全量，无截断）
    if sql_query:
        try:
            from database import get_engine
            from sqlalchemy import text as sa_text
            engine = get_engine(readonly=True)
            with engine.connect() as conn:
                df = pd.read_sql(sa_text(sql_query), conn)
            print(f"[数据加载] SQL 查询返回 {len(df)} 行")
            if max_rows and len(df) > max_rows:
                df = df.head(max_rows)
            return df
        except Exception as e:
            raise ValueError(f"SQL 查询失败: {e}")

    # 通道 2：表名全量读取（全量，无截断）
    if table_name:
        try:
            from database import get_engine
            from sqlalchemy import text as sa_text
            engine = get_engine(readonly=True)
            with engine.connect() as conn:
                df = pd.read_sql(sa_text(f"SELECT * FROM `{table_name}`"), conn)
            print(f"[数据加载] 表 {table_name} 返回 {len(df)} 行")
            if max_rows and len(df) > max_rows:
                df = df.head(max_rows)
            return df
        except Exception as e:
            raise ValueError(f"读取表 {table_name} 失败: {e}")

    # 通道 3：JSON 降级（小数据，兼容旧逻辑）
    if data_json:
        try:
            data = json.loads(data_json) if isinstance(data_json, str) else data_json
            df = pd.DataFrame(data)
            print(f"[数据加载] JSON 解析返回 {len(df)} 行")
            return df
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"JSON 解析失败: {e}")

    raise ValueError("必须提供 sql_query、table_name 或 data_json 之一")


def _build_data_source_info(sql_query: str, table_name: str, data_json: str) -> str:
    """构建数据来源描述"""
    if sql_query:
        return f"SQL: {sql_query[:80]}..."
    if table_name:
        return f"表: {table_name}"
    return "JSON 数据"


# ==================== 趋势预测 ====================


def forecast_trend(date_col: str = None, value_col: str = None,
                   periods: int = 3, method: str = "linear",
                   sql_query: str = None, table_name: str = None,
                   data_json: str = None) -> dict:
    """
    对时序数据进行趋势预测。优先使用 sql_query/table_name 从数据库直连加载全量数据。

    参数:
        sql_query: 【推荐】SQL 查询语句，工具内部直接查库加载全量数据（不经过 LLM，无截断）
        table_name: 表名，直接读取全表
        data_json: JSON 字符串（降级通道，仅用于小数据量测试）
        date_col: 时间列名，为空则自动识别含 date/time/year/month 的列
        value_col: 数值列名，为空则自动识别第一个数值列
        periods: 向后预测期数，默认 3
        method: "linear" 或 "exponential_smoothing"

    返回:
        { "success": bool, "method": str, "historical": [...], "forecast": [...],
          "total_rows": int, "data_source": str }
    """
    try:
        df = _load_dataframe(sql_query=sql_query, table_name=table_name, data_json=data_json)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    if df.empty:
        return {"success": False, "error": "数据为空"}

    total_rows = len(df)

    # 自动识别列
    if value_col is None or value_col not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return {"success": False, "error": "未找到数值列", "columns": df.columns.tolist()}
        value_col = numeric_cols[0]

    if date_col is None or date_col not in df.columns:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "year", "month", "period", "ds"])]
        date_col = candidates[0] if candidates else df.columns[0]

    df = df[[date_col, value_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna()

    if len(df) < 2:
        return {"success": False, "error": "有效数据点不足 2 个，无法预测"}

    # 用索引作为 X
    df["_x"] = np.arange(len(df))
    X = df[["_x"]].values
    y = df[value_col].values

    if method == "exponential_smoothing":
        alpha = 0.3
        smoothed = [y[0]]
        for val in y[1:]:
            smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
        last_level = smoothed[-1]
        trend = last_level - smoothed[-2] if len(smoothed) >= 2 else 0
        forecast_values = [last_level + (i + 1) * trend for i in range(periods)]
        model_desc = f"指数平滑(α={alpha})"
    else:
        model = LinearRegression()
        model.fit(X, y)
        forecast_values = [model.predict([[len(df) + i]])[0] for i in range(periods)]
        trend = {"slope": round(float(model.coef_[0]), 4), "intercept": round(float(model.intercept_), 4)}
        model_desc = "线性回归"

    historical = [
        {"x": str(df.iloc[i][date_col]), "y": round(float(y[i]), 2)}
        for i in range(len(df))
    ]

    forecast = [
        {"x": f"+{i+1}", "y": round(float(v), 2)}
        for i, v in enumerate(forecast_values)
    ]

    return {
        "success": True,
        "method": model_desc,
        "date_col": date_col,
        "value_col": value_col,
        "historical": historical,
        "forecast": forecast,
        "trend": trend if method == "linear" else {"last_level": round(float(last_level), 2), "trend": round(float(trend), 4)},
        "total_rows": total_rows,
        "data_source": _build_data_source_info(sql_query, table_name, data_json),
    }


def detect_anomalies(columns: list = None,
                     method: str = "iqr", contamination: float = 0.05,
                     sql_query: str = None, table_name: str = None,
                     data_json: str = None) -> dict:
    """
    检测数据中的异常值。优先使用 sql_query/table_name 从数据库直连加载全量数据。

    参数:
        sql_query: 【推荐】SQL 查询语句，工具内部直接查库加载全量数据（不经过 LLM，无截断）
        table_name: 表名，直接读取全表
        data_json: JSON 字符串（降级通道，仅用于小数据量测试）
        columns: 要检测的数值列，为空则检测所有数值列
        method: "iqr" 或 "isolation_forest"
        contamination: IsolationForest 的异常比例，默认 0.05

    返回:
        { "success": bool, "method": str, "total": int, "anomaly_count": int,
          "anomaly_rate": float, "summary": {...}, "data_source": str }
    """
    try:
        df = _load_dataframe(sql_query=sql_query, table_name=table_name, data_json=data_json)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    if df.empty:
        return {"success": False, "error": "数据为空"}

    total_rows = len(df)

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        columns = [c for c in columns if c in df.columns]

    if not columns:
        return {"success": False, "error": "未找到可检测的数值列"}

    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")

    if method == "isolation_forest":
        filled = numeric_df.fillna(numeric_df.mean())
        if len(filled) < 2:
            return {"success": False, "error": "有效样本不足"}
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(filled)
        mask = preds == -1
    else:
        # IQR 方法
        mask = pd.Series(False, index=df.index)
        bounds = {}
        for col in columns:
            q1 = numeric_df[col].quantile(0.25)
            q3 = numeric_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            bounds[col] = {"lower": round(float(lower), 2), "upper": round(float(upper), 2)}
            mask = mask | numeric_df[col].lt(lower) | numeric_df[col].gt(upper)

    anomaly_count = int(mask.sum())

    summary = {}
    for col in columns:
        summary[col] = {
            "mean": round(float(numeric_df[col].mean()), 2),
            "std": round(float(numeric_df[col].std()), 2),
            "min": round(float(numeric_df[col].min()), 2),
            "max": round(float(numeric_df[col].max()), 2),
        }
        if method == "iqr":
            summary[col].update(bounds[col])

    return {
        "success": True,
        "method": method,
        "total": total_rows,
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(anomaly_count / total_rows * 100, 2) if total_rows > 0 else 0,
        "summary": summary,
        "columns_checked": columns,
        "data_source": _build_data_source_info(sql_query, table_name, data_json),
    }


def churn_risk_score(recency_col: str = "recency",
                     frequency_col: str = "flight_count",
                     monetary_col: str = "seg_km_sum",
                     sql_query: str = None, table_name: str = None,
                     data_json: str = None) -> dict:
    """
    基于 RFM 特征计算客户流失风险评分。优先使用 sql_query/table_name 从数据库直连加载全量数据。

    【Compute-over-Data】: 工具内部直连数据库加载全量数据（62,987 行），
    不经过 LLM 上下文，彻底消除 30 行截断问题。仅返回聚合摘要。

    参数:
        sql_query: 【推荐】SQL 查询语句，工具内部直接查库加载全量数据（不经过 LLM，无截断）
                   示例: "SELECT member_no, recency, flight_count, seg_km_sum FROM customer_flight_summary"
        table_name: 表名，直接读取全表（如 "customer_flight_summary"）
        data_json: JSON 字符串（降级通道，仅用于小数据量测试）
        recency_col: Recency 列名（默认 recency）
        frequency_col: Frequency 列名（默认 flight_count）
        monetary_col: Monetary 列名（默认 seg_km_sum）

    返回:
        { "success": bool, "total_modeled": int, "distribution": {...},
          "avg_score": float, "data_source": str }
    """
    try:
        df = _load_dataframe(sql_query=sql_query, table_name=table_name, data_json=data_json)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    if df.empty:
        return {"success": False, "error": "数据为空"}

    total_rows = len(df)

    # 模糊匹配列名（大小写不敏感）
    col_lower = {c.lower(): c for c in df.columns}

    def find_col(preferred, aliases):
        """多级匹配：精确 → 大小写不敏感 → 别名列表"""
        if preferred in df.columns:
            return preferred
        if preferred.lower() in col_lower:
            return col_lower[preferred.lower()]
        for a in aliases:
            for c in df.columns:
                if c.lower() == a.lower():
                    return c
        return None

    alias_map = {
        recency_col: ["Recency", "recency", "R", "last_flight_days"],
        frequency_col: ["Frequency", "frequency", "F", "flight_count"],
        monetary_col: ["Monetary", "monetary", "M", "seg_km_sum", "bp_sum"],
    }

    r_col = find_col(recency_col, alias_map[recency_col])
    f_col = find_col(frequency_col, alias_map[frequency_col])
    m_col = find_col(monetary_col, alias_map[monetary_col])

    missing = [n for n, c in [("Recency", r_col), ("Frequency", f_col), ("Monetary", m_col)] if c is None]
    if missing:
        return {"success": False, "error": f"缺少列: {missing}", "available_columns": df.columns.tolist()}

    r = pd.to_numeric(df[r_col], errors="coerce").fillna(0)
    f = pd.to_numeric(df[f_col], errors="coerce").fillna(0)
    m = pd.to_numeric(df[m_col], errors="coerce").fillna(0)

    # 归一化到 [0, 1]
    def norm(s):
        min_v, max_v = s.min(), s.max()
        if max_v == min_v:
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - min_v) / (max_v - min_v)

    r_norm = norm(r)      # Recency 越高越差
    f_norm = 1 - norm(f)  # Frequency 越低越差
    m_norm = 1 - norm(m)  # Monetary 越低越差

    score = (r_norm * 0.4 + f_norm * 0.3 + m_norm * 0.3) * 100
    avg_score = round(float(score.mean()), 1)

    def label(s):
        if s >= 70:
            return "高风险"
        elif s >= 40:
            return "中风险"
        else:
            return "低风险"

    labels = score.apply(label)
    distribution = labels.value_counts().to_dict()
    # 确保三个等级都有
    for level in ["高风险", "中风险", "低风险"]:
        if level not in distribution:
            distribution[level] = 0

    return {
        "success": True,
        "total_modeled": total_rows,
        "avg_score": avg_score,
        "distribution": distribution,
        "score_range": {"min": round(float(score.min()), 1), "max": round(float(score.max()), 1)},
        "data_source": _build_data_source_info(sql_query, table_name, data_json),
        "note": f"成功对全量 {total_rows} 行数据完成流失风险建模，返回聚合摘要（非明细数据）",
    }
