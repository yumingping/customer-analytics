"""
MCP 工具: 动态分析与交叉验证 (analysis_tools)
============================================
实现 readme.md 规定的分析类 MCP 工具：
- perform_dynamic_cluster(): 即时 K-Means 聚类
- verify_data_logic(): 双重逻辑交叉校验
"""
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import config


def perform_dynamic_cluster(data_json: str, k: int = 4,
                            features: list = None) -> dict:
    """
    对传入的 JSON 数据集执行即时 K-Means 聚类。
    （readme.md 第6节 / algorithm.md 第4节）

    参数:
        data_json: JSON 字符串，格式 [{"col1": val, "col2": val}, ...]
        k: 聚类数，默认 4
        features: 用于聚类的特征列名，默认 ["Recency", "Frequency", "Monetary"]

    返回:
        {
            "success": bool,
            "k": int,
            "labels": [...],
            "silhouette_score": float,
            "cluster_stats": {...},
            "data_with_labels": [...],
        }
    """
    if features is None:
        features = ["Recency", "Frequency", "Monetary"]

    try:
        # 解析输入
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        df = pd.DataFrame(data)

        # 检查特征列
        missing = [f for f in features if f not in df.columns]
        if missing:
            return {
                "success": False,
                "error": f"缺少特征列: {missing}",
                "available_columns": df.columns.tolist(),
            }

        # Log1p + StandardScaler 预处理（按 algorithm.md 要求）
        X = df[features].copy().fillna(0)
        X_log = np.log1p(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_log)

        # K-Means
        km = KMeans(n_clusters=k, random_state=config.CLUSTER_RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        df["cluster"] = labels

        # 轮廓系数
        sil = None
        if k >= 2 and len(set(labels)) > 1:
            sil = round(float(silhouette_score(X_scaled, labels)), 4)

        # 聚类统计
        cluster_stats = {}
        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            cluster_stats[int(cluster_id)] = {
                "count": int(mask.sum()),
                "pct": round(float(mask.sum() / len(df) * 100), 1),
                "means": {f: round(float(df.loc[mask, f].mean()), 2) for f in features},
            }

        # 价值标签（如果有离散评分）
        if "R_score" in df.columns and "F_score" in df.columns and "M_score" in df.columns:
            for cluster_id in cluster_stats:
                mask = labels == cluster_id
                cluster_stats[cluster_id]["rfm_means"] = {
                    "R": round(float(df.loc[mask, "R_score"].mean()), 2),
                    "F": round(float(df.loc[mask, "F_score"].mean()), 2),
                    "M": round(float(df.loc[mask, "M_score"].mean()), 2),
                }

        return {
            "success": True,
            "k": k,
            "labels": labels.tolist(),
            "silhouette_score": sil,
            "cluster_stats": cluster_stats,
            "data_with_labels": df.fillna("").to_dict(orient="records"),
            "features_used": features,
        }

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON 解析失败: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def verify_data_logic(sql_result_json: str, pandas_result_json: str,
                      tolerance: float = None) -> dict:
    """
    交叉校验：对比 SQL 聚合结果与 Pandas 计算结果。
    对 RFM 均值、聚类占比等核心指标进行双重逻辑校验。
    （readme.md 第3.4节）

    参数:
        sql_result_json: SQL 聚合结果的 JSON 字符串
        pandas_result_json: Pandas 计算结果的 JSON 字符串
        tolerance: 允许的差异百分比（默认从 config 读取）

    返回:
        {
            "match": bool,
            "details": { metric: { sql_value, pandas_value, diff_pct, match } },
            "summary": str
        }
    """
    tolerance = tolerance if tolerance is not None else config.CROSS_VALIDATION_TOLERANCE

    try:
        sql_data = json.loads(sql_result_json) if isinstance(sql_result_json, str) else sql_result_json
        pd_data = json.loads(pandas_result_json) if isinstance(pandas_result_json, str) else pandas_result_json
    except json.JSONDecodeError as e:
        return {"match": False, "error": f"JSON 解析失败: {e}"}

    details = {}
    all_match = True

    # 对比所有共同的键
    common_keys = set(sql_data.keys()) & set(pd_data.keys())
    if not common_keys:
        # 尝试作为 list of dicts 处理
        if isinstance(sql_data, list) and isinstance(pd_data, list):
            common_keys = {"_aggregated"}

    for key in common_keys:
        sql_val = sql_data.get(key) if isinstance(sql_data, dict) else sql_data
        pd_val = pd_data.get(key) if isinstance(pd_data, dict) else pd_data

        if sql_val is None or pd_val is None:
            continue

        try:
            sql_num = float(sql_val) if not isinstance(sql_val, (list, dict)) else float(
                np.mean([float(v) for v in (sql_val if isinstance(sql_val, list) else sql_val.values())]))
            pd_num = float(pd_val) if not isinstance(pd_val, (list, dict)) else float(
                np.mean([float(v) for v in (pd_val if isinstance(pd_val, list) else pd_val.values())]))
        except (ValueError, TypeError):
            continue

        if pd_num == 0:
            diff_pct = abs(sql_num)
        else:
            diff_pct = abs(sql_num - pd_num) / abs(pd_num)

        is_match = diff_pct <= tolerance
        if not is_match:
            all_match = False

        details[str(key)] = {
            "sql_value": round(sql_num, 4),
            "pandas_value": round(pd_num, 4),
            "diff_pct": round(float(diff_pct * 100), 2),
            "match": is_match,
        }

    summary = "✓ 校验通过: SQL 与 Pandas 计算结果一致" if all_match else "✗ 校验失败: 存在超出容差的差异"

    return {
        "match": all_match,
        "details": details,
        "summary": summary,
        "tolerance_pct": round(float(tolerance * 100), 1),
    }


def analyze_cluster_profiles(df_json: str) -> dict:
    """
    对聚类结果进行业务画像分析。
    返回每个簇的 RFM 特征、推荐营销策略。
    """
    try:
        data = json.loads(df_json) if isinstance(df_json, str) else df_json
        df = pd.DataFrame(data)
    except (json.JSONDecodeError, ValueError) as e:
        return {"success": False, "error": f"数据解析失败: {e}"}

    if "cluster" not in df.columns:
        return {"success": False, "error": "数据中缺少 'cluster' 列"}

    rfm_cols = ["R_score", "F_score", "M_score"]
    existing_rfm = [c for c in rfm_cols if c in df.columns]

    profiles = {}
    for cluster_id in sorted(df["cluster"].unique()):
        mask = df["cluster"] == cluster_id
        cluster_df = df[mask]

        profile = {
            "count": int(mask.sum()),
            "pct": round(float(mask.sum() / len(df) * 100), 1),
        }

        if existing_rfm:
            r_avg = float(cluster_df["R_score"].mean())
            f_avg = float(cluster_df["F_score"].mean())
            m_avg = float(cluster_df["M_score"].mean())
            rfm_sum = r_avg + f_avg + m_avg
            profile["rfm_avg"] = {"R": round(r_avg, 2), "F": round(f_avg, 2), "M": round(m_avg, 2)}

            # 业务解读
            if rfm_sum >= 12 and f_avg >= 4 and m_avg >= 4:
                profile["description"] = "高价值常旅客 — 频繁出行、高里程贡献"
                profile["strategy"] = "提供VIP服务、积分翻倍、专属客服"
            elif rfm_sum >= 9 and f_avg >= 3:
                profile["description"] = "潜力客户 — 有一定活跃度和消费能力"
                profile["strategy"] = "定向推送升舱优惠、里程加速计划"
            elif r_avg <= 2 and f_avg <= 2:
                profile["description"] = "沉睡客户 — 近期未出行、活跃度低"
                profile["strategy"] = "发送回归优惠、限时里程兑换活动"
            else:
                profile["description"] = "一般客户 — 处于中等水平"
                profile["strategy"] = "常规关怀、节日问候、积分到期提醒"

        profiles[int(cluster_id)] = profile

    return {"success": True, "profiles": profiles}
