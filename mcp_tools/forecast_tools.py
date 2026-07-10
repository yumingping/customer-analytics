"""
MCP 工具: 预测与异常检测 (forecast_tools)
============================================
扩展 MCP 能力，支持：
- forecast_trend(): 简单趋势预测（线性回归 / 指数平滑）
- detect_anomalies(): 基于 IQR / Z-Score 的异常值检测
- churn_risk_score(): 基于 RFM 特征的流失风险评分
"""
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest


def forecast_trend(data_json: str, date_col: str = None, value_col: str = None,
                   periods: int = 3, method: str = "linear") -> dict:
    """
    对时序数据进行趋势预测。

    参数:
        data_json: JSON 字符串，格式 [{"date": "2024-01", "value": 100}, ...]
        date_col: 时间列名，为空则自动识别含 date/time/year/month 的列
        value_col: 数值列名，为空则自动识别第一个数值列
        periods: 向后预测期数，默认 3
        method: "linear" 或 "exponential_smoothing"

    返回:
        {
            "success": bool,
            "method": str,
            "historical": [{"x": ..., "y": ...}],
            "forecast": [{"x": ..., "y": ...}],
            "trend": {"slope": float, "intercept": float},
        }
    """
    try:
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        df = pd.DataFrame(data)
    except (json.JSONDecodeError, ValueError) as e:
        return {"success": False, "error": f"数据解析失败: {e}"}

    if df.empty:
        return {"success": False, "error": "数据为空"}

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
        # 简单指数平滑：alpha=0.3
        alpha = 0.3
        smoothed = [y[0]]
        for val in y[1:]:
            smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
        last_level = smoothed[-1]
        trend = last_level - smoothed[-2] if len(smoothed) >= 2 else 0
        forecast_values = [last_level + (i + 1) * trend for i in range(periods)]
        model_desc = f"指数平滑(α={alpha})"
    else:
        # 线性回归
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
    }


def detect_anomalies(data_json: str, columns: list = None,
                     method: str = "iqr", contamination: float = 0.05) -> dict:
    """
    检测数据中的异常值。

    参数:
        data_json: JSON 字符串
        columns: 要检测的数值列，为空则检测所有数值列
        method: "iqr" 或 "isolation_forest"
        contamination: IsolationForest 的异常比例，默认 0.05

    返回:
        {
            "success": bool,
            "method": str,
            "total": int,
            "anomaly_count": int,
            "anomaly_indices": [...],
            "summary": { col: {lower, upper, mean, std} },
            "anomalies": [...]  # 异常样本
        }
    """
    try:
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        df = pd.DataFrame(data)
    except (json.JSONDecodeError, ValueError) as e:
        return {"success": False, "error": f"数据解析失败: {e}"}

    if df.empty:
        return {"success": False, "error": "数据为空"}

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        columns = [c for c in columns if c in df.columns]

    if not columns:
        return {"success": False, "error": "未找到可检测的数值列"}

    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")

    if method == "isolation_forest":
        # 需要完整数值，缺失值填充均值
        filled = numeric_df.fillna(numeric_df.mean())
        if len(filled) < 2:
            return {"success": False, "error": "有效样本不足"}
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(filled)
        mask = preds == -1
    else:
        # IQR 方法：任一列为异常即标记
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

    anomaly_indices = df[mask].index.tolist()
    anomalies = df[mask].head(50).fillna("").to_dict(orient="records")

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
        "total": len(df),
        "anomaly_count": int(mask.sum()),
        "anomaly_indices": anomaly_indices,
        "summary": summary,
        "anomalies": anomalies,
        "columns_checked": columns,
    }


def churn_risk_score(data_json: str, recency_col: str = "Recency",
                     frequency_col: str = "Frequency",
                     monetary_col: str = "Monetary") -> dict:
    """
    基于 RFM 特征计算每个客户的流失风险评分。
    规则：Recency 越高、Frequency/Monetary 越低，流失风险越高。

    参数:
        data_json: JSON 字符串，必须包含 R/F/M 列
        recency_col, frequency_col, monetary_col: 列名

    返回:
        {
            "success": bool,
            "risk_col": "churn_risk_score",
            "distribution": {"低风险": n, "中风险": n, "高风险": n},
            "data": [...]  # 前 20 条带评分样本
        }
    """
    try:
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        df = pd.DataFrame(data)
    except (json.JSONDecodeError, ValueError) as e:
        return {"success": False, "error": f"数据解析失败: {e}"}

    if df.empty:
        return {"success": False, "error": "数据为空"}

    # 允许别名
    alias_map = {
        recency_col: ["Recency", "recency", "R", "last_flight_days"],
        frequency_col: ["Frequency", "frequency", "F", "flight_count"],
        monetary_col: ["Monetary", "monetary", "M", "seg_km_sum", "bp_sum"],
    }

    def find_col(preferred, aliases):
        if preferred in df.columns:
            return preferred
        for a in aliases:
            candidates = [c for c in df.columns if c.lower() == a.lower()]
            if candidates:
                return candidates[0]
        return None

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
    df["churn_risk_score"] = score.round(1)

    def label(s):
        if s >= 70:
            return "高风险"
        elif s >= 40:
            return "中风险"
        else:
            return "低风险"

    df["churn_risk_label"] = df["churn_risk_score"].apply(label)
    distribution = df["churn_risk_label"].value_counts().to_dict()

    return {
        "success": True,
        "risk_col": "churn_risk_score",
        "risk_label_col": "churn_risk_label",
        "distribution": distribution,
        "data": df.head(20).fillna("").to_dict(orient="records"),
        "total": len(df),
    }
