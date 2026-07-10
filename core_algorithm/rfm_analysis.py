"""
RFM 分析与客户聚类 — core_algorithm.rfm_analysis
==============================================
算法改进（按 algorithm.md 规范）：
- 百分位评分法（已有，保留）
- Log1p + StandardScaler 预处理（替代离散评分聚类）
- Silhouette Score + SSE 双指标寻优最佳 K
- 局部最大值 + SSE 拐点组合决策（不是简单取全局最大）
- 结果缓存：首次计算后缓存 K 值和聚类结果，后续运行直接复用
- PCA 降维可视化
- 雷达图使用离散评分（业务可读）
- Jitter 防散点重叠
"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import config

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 缓存路径 ====================

CACHE_DIR = Path(config.OUTPUT_DIR)
K_CACHE_FILE = CACHE_DIR / "_optimal_k_cache.json"
CLUSTER_CACHE_FILE = CACHE_DIR / "_cluster_labels.npy"
PREPROCESS_CACHE_FILE = CACHE_DIR / "_X_scaled.npy"


# ==================== RFM 评分 ====================

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """计算 R、F、M 得分（1-5分），采用百分位排名法"""
    print("\n[RFM] 计算RFM得分...")

    def _score(col: pd.Series, ascending: bool = True) -> pd.Series:
        """百分位排名 → 5级评分"""
        valid = col.dropna()
        pct = valid.rank(method="average", ascending=ascending) / len(valid)
        result = pd.cut(
            pct, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=[1, 2, 3, 4, 5], include_lowest=True
        )
        result = result.astype(int)
        return result.reindex(col.index, fill_value=3)

    df["R_score"] = _score(df["Recency"], ascending=False)
    df["F_score"] = _score(df["Frequency"], ascending=True)
    df["M_score"] = _score(df["Monetary"], ascending=True)
    df["RFM_total"] = df["R_score"] + df["F_score"] + df["M_score"]

    print(f"  R_score 分布: {df['R_score'].value_counts().sort_index().to_dict()}")
    print(f"  F_score 分布: {df['F_score'].value_counts().sort_index().to_dict()}")
    print(f"  M_score 分布: {df['M_score'].value_counts().sort_index().to_dict()}")
    return df


def label_customer_value(df: pd.DataFrame) -> pd.DataFrame:
    """基于 RFM 总分进行客户价值分层"""
    print("\n[价值分层] ...")

    def _value_label(total):
        if total >= 13:
            return "高价值客户"
        elif total >= 9:
            return "中价值客户"
        elif total >= 6:
            return "低价值客户"
        else:
            return "流失客户"

    df["value_label"] = df["RFM_total"].apply(_value_label)

    label_counts = df["value_label"].value_counts()
    print("  各层级客户数量:")
    for label, count in label_counts.items():
        print(f"    {label}: {count} ({count / len(df) * 100:.1f}%)")
    return df


# ==================== 聚类的工业级预处理 ====================

def preprocess_for_clustering(df: pd.DataFrame, features: list = None,
                              force_recompute: bool = False) -> tuple:
    """
    Log1p 变换 + Z-Score 标准化，支持缓存。
    按 algorithm.md 第2.1节要求。
    """
    if features is None:
        features = ["Recency", "Frequency", "Monetary"]

    # ── 缓存检查 ──
    if not force_recompute and PREPROCESS_CACHE_FILE.exists():
        try:
            X_scaled = np.load(PREPROCESS_CACHE_FILE)
            if X_scaled.shape[1] == len(features):
                print(f"[缓存] 命中预处理缓存 ({PREPROCESS_CACHE_FILE})")
                return X_scaled, None, features
        except Exception:
            PASS

    X = df[features].copy().fillna(0)
    X_log = np.log1p(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    print(f"[预处理] Log1p + StandardScaler 完成, 特征: {features}")

    # 写入缓存
    np.save(PREPROCESS_CACHE_FILE, X_scaled)
    print(f"[缓存] 预处理结果已缓存: {PREPROCESS_CACHE_FILE}")

    return X_scaled, scaler, features


# ==================== 最优 K 值搜索 ====================

def find_optimal_k(X_scaled: np.ndarray, k_range: range = None,
                   force_recompute: bool = False) -> dict:
    """
    SSE (Inertia) + Silhouette Score 双指标寻优，支持缓存。
    按 algorithm.md 第2.2节要求：
    - 寻找 Silhouette Score 的局部最大值
    - 结合 SSE 曲线的拐点（曲率最大处）
    - 组合评分 = 归一化轮廓系数 × 0.5 + 肘部曲率 × 0.5
    返回 {"k": 推荐K, "inertias": [...], "silhouettes": [...], "k_values": [...]}
    """
    # ── 缓存检查 ──
    if not force_recompute and K_CACHE_FILE.exists():
        try:
            cached = json.loads(K_CACHE_FILE.read_text(encoding="utf-8"))
            print(f"[缓存] 命中 K 值缓存: K={cached['k']} (清除缓存请删除 {K_CACHE_FILE})")
            return cached
        except Exception:
            pass

    k_range = k_range or config.CLUSTER_RANGE
    k_values = list(k_range)
    inertias = []
    silhouettes = []

    print(f"\n[寻优K值] 搜索范围 K={min(k_values)}~{max(k_values)}...")
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=config.CLUSTER_RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        if k >= 2 and len(set(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)
            silhouettes.append(sil)
        else:
            silhouettes.append(0)
        print(f"  K={k}: Inertia={km.inertia_:.1f}, Silhouette={silhouettes[-1]:.4f}")

    # ── 组合决策 ──
    recommended_k = _combined_k_decision(k_values, inertias, silhouettes)

    # 保存肘部图
    _plot_elbow_with_silhouette(k_values, inertias, silhouettes, recommended_k)

    result = {
        "k": recommended_k,
        "inertias": [round(v, 1) for v in inertias],
        "silhouettes": [round(v, 4) for v in silhouettes],
        "k_values": k_values,
    }

    # 写入缓存
    K_CACHE_FILE.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"[缓存] K 值结果已缓存: {K_CACHE_FILE}")

    return result


def _combined_k_decision(k_values: list, inertias: list,
                         silhouettes: list) -> int:
    """
    组合决策引擎：
    1. 找 Silhouette 的局部最大值
    2. 计算 SSE 肘部曲率
    3. 每个局部最大 K 的评分 = 0.5 × sil_norm + 0.5 × elbow_norm
    4. 返回得分最高的 K
    若局部最大值只有一个且是 K=2，跳过 K=2 取次优（业务要求 ≥3 簇）
    """
    n = len(k_values)

    # 1. 找 Silhouette 局部最大值（含端点）
    local_max_indices = _find_local_maxima(silhouettes)

    # 2. 计算 SSE 变化率（一阶差分率）
    sse_deltas = []
    for i in range(1, n):
        rate = (inertias[i - 1] - inertias[i]) / max(inertias[i - 1], 1)
        sse_deltas.append(rate)
    # 补第一个点为 NaN → 用第二个点代替
    sse_deltas = [sse_deltas[0]] + sse_deltas

    # 3. 归一化两个指标到 [0, 1]
    sil_arr = np.array(silhouettes)
    sse_arr = np.array(sse_deltas)

    sil_min, sil_max = sil_arr.min(), sil_arr.max()
    sse_min, sse_max = sse_arr.min(), sse_arr.max()

    sil_norm = (sil_arr - sil_min) / max(sil_max - sil_min, 1e-10)
    sse_norm = (sse_arr - sse_min) / max(sse_max - sse_min, 1e-10)

    # 4. 每个局部最大 K 的得分
    scores = {}
    for idx in local_max_indices:
        k = k_values[idx]
        score = 0.5 * sil_norm[idx] + 0.5 * sse_norm[idx]
        scores[k] = score

    # 5. 选择最佳 K
    if len(scores) >= 2:
        # 如果有多个局部最大值，跳过 K=2（业务上太粗糙）
        candidates = {k: v for k, v in scores.items() if k > 2}
        if not candidates:
            candidates = scores  # 如果没有 >2 的，回退全部
        best_k = max(candidates, key=candidates.get)
    elif k_values[local_max_indices[0]] == 2 and n >= 3:
        # 只有 K=2 一个局部最大，取 K=4（业务默认）
        best_k = 4
    else:
        best_k = k_values[local_max_indices[0]]

    # 打印决策过程
    print(f"\n  局部最大 Silhouette 点: K={[k_values[i] for i in local_max_indices]}")
    print(f"  各候选得分: { {k: round(v, 4) for k, v in scores.items()} }")
    print(f"  最终选择: K = {best_k}")

    return best_k


def _find_local_maxima(values: list) -> list:
    """找局部最大值的索引（含首尾端点）"""
    n = len(values)
    if n <= 2:
        return [int(np.argmax(values))]
    indices = []
    # 首端点
    if values[0] > values[1]:
        indices.append(0)
    # 中间点
    for i in range(1, n - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            indices.append(i)
    # 尾端点
    if values[-1] > values[-2]:
        indices.append(n - 1)
    # 兜底：没找到局部最大就返回全局最大
    if not indices:
        indices.append(int(np.argmax(values)))
    return indices


def _plot_elbow_with_silhouette(k_values: list, inertias: list,
                                silhouettes: list, recommended_k: int):
    """绘制双 Y 轴肘部 + 轮廓系数图，标注推荐 K 值"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "steelblue"
    ax1.set_xlabel("K (聚类数)")
    ax1.set_ylabel("SSE (簇内误差平方和)", color=color1)
    ax1.plot(k_values, inertias, "o-", color=color1, linewidth=2, label="SSE")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "coral"
    ax2.set_ylabel("Silhouette Score (轮廓系数)", color=color2)
    ax2.plot(k_values, silhouettes, "s--", color=color2, linewidth=2, label="Silhouette")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1)

    # 标注推荐 K
    if recommended_k in k_values:
        idx = k_values.index(recommended_k)
        ax1.annotate(f"推荐 K={recommended_k}",
                     xy=(recommended_k, inertias[idx]),
                     xytext=(recommended_k + 0.8, inertias[idx] * 1.15),
                     fontsize=11, fontweight="bold", color="darkred",
                     arrowprops=dict(arrowstyle="->", color="darkred"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.title("最优 K 值搜索 — 肘部法则 + 轮廓系数 (组合决策)")
    fig.tight_layout()
    plt.savefig(str(config.OUTPUT_DIR / "elbow_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  双指标图已保存: {config.OUTPUT_DIR / 'elbow_plot.png'}")


# ==================== 聚类执行 ====================

def perform_clustering(df: pd.DataFrame, n_clusters: int = None,
                       features: list = None,
                       force_recompute: bool = False) -> pd.DataFrame:
    """
    执行 K-Means 聚类（使用 Log1p 预处理后的连续特征），支持缓存。
    按 algorithm.md 第2节要求：输入特征是 [Recency, Frequency, Monetary]，
    不是离散的 [R_score, F_score, M_score]。
    """
    n_clusters = n_clusters or config.N_CLUSTERS
    if features is None:
        features = ["Recency", "Frequency", "Monetary"]

    # ── 缓存检查 ──
    if not force_recompute and CLUSTER_CACHE_FILE.exists():
        try:
            cached_labels = np.load(CLUSTER_CACHE_FILE)
            if len(cached_labels) == len(df):
                df["cluster"] = cached_labels
                print(f"[缓存] 命中聚类标签缓存 ({CLUSTER_CACHE_FILE})")
                cluster_counts = df["cluster"].value_counts().sort_index()
                print("  各簇客户数量 (来自缓存):")
                for c, count in cluster_counts.items():
                    print(f"    簇 {c}: {count} ({count / len(df) * 100:.1f}%)")
                return df
        except Exception:
            pass

    print(f"\n[聚类] K-Means (K={n_clusters}, 特征={features})...")

    X_scaled, scaler, _ = preprocess_for_clustering(df, features, force_recompute)

    km = KMeans(n_clusters=n_clusters, random_state=config.CLUSTER_RANDOM_STATE, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    # 计算轮廓系数
    if n_clusters >= 2 and len(set(df["cluster"])) > 1:
        sil = silhouette_score(X_scaled, df["cluster"])
        print(f"  轮廓系数 (Silhouette): {sil:.4f}")

    # 保存模型参数
    df.attrs["cluster_centers"] = km.cluster_centers_
    df.attrs["scaler"] = scaler

    # 写入缓存
    np.save(CLUSTER_CACHE_FILE, df["cluster"].values)
    print(f"[缓存] 聚类标签已缓存: {CLUSTER_CACHE_FILE}")

    cluster_counts = df["cluster"].value_counts().sort_index()
    print("  各簇客户数量:")
    for c, count in cluster_counts.items():
        print(f"    簇 {c}: {count} ({count / len(df) * 100:.1f}%)")

    return df


def analyze_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """分析聚类结果：各簇在连续特征和离散评分上的统计"""
    print("\n[聚类分析] ...")

    rfm_cols = ["R_score", "F_score", "M_score", "Recency", "Frequency", "Monetary"]
    existing = [c for c in rfm_cols if c in df.columns]
    cluster_stats = df.groupby("cluster")[existing].mean().round(2)
    cluster_stats["count"] = df.groupby("cluster").size()

    print("  各簇 RFM 均值:")
    print(cluster_stats.to_string())
    return cluster_stats


# ==================== PCA 降维 ====================

def pca_reduce(df: pd.DataFrame, features: list = None, n_components: int = 2) -> dict:
    """
    PCA 降维，验证 explained_variance > 0.8。
    返回 {"coords": (N,2), "var_ratio": [float,float], "total_var": float}
    """
    if features is None:
        features = ["Recency", "Frequency", "Monetary"]

    X = df[features].copy().fillna(0)
    X_log = np.log1p(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    pca = PCA(n_components=n_components, random_state=config.CLUSTER_RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    total_var = sum(pca.explained_variance_ratio_)

    print(f"[PCA] {n_components}D 累计方差解释率: {total_var:.1%}")
    if total_var < 0.8:
        print(f"  ⚠️ 警告: 方差解释率低于 80%，建议保留更多主成分")

    return {
        "coords": X_pca,
        "var_ratio": [round(float(r), 4) for r in pca.explained_variance_ratio_],
        "total_var": round(float(total_var), 4),
    }


# ==================== 交叉验证 ====================

def cross_validate_rfm(df_sql: pd.DataFrame, df_pandas: pd.DataFrame,
                       metrics: list = None) -> dict:
    """
    双重逻辑交叉校验：对比 SQL 聚合结果与 Pandas 计算结果。
    按 readme.md 第3.4节要求。
    """
    if metrics is None:
        metrics = ["R_score", "F_score", "M_score", "RFM_total"]

    results = {}
    all_match = True

    for col in metrics:
        if col not in df_sql.columns or col not in df_pandas.columns:
            continue
        sql_mean = df_sql[col].mean()
        pd_mean = df_pandas[col].mean()
        if pd_mean == 0:
            diff_pct = abs(sql_mean)
        else:
            diff_pct = abs(sql_mean - pd_mean) / abs(pd_mean)
        is_match = diff_pct <= config.CROSS_VALIDATION_TOLERANCE
        if not is_match:
            all_match = False
        results[col] = {
            "sql_mean": round(float(sql_mean), 4),
            "pandas_mean": round(float(pd_mean), 4),
            "diff_pct": round(float(diff_pct * 100), 2),
            "match": is_match,
        }

    print(f"\n[交叉验证] {'✓ 通过' if all_match else '✗ 存在差异'}")
    for col, r in results.items():
        status = "✓" if r["match"] else "✗"
        print(f"  {status} {col}: SQL={r['sql_mean']:.2f} vs Pandas={r['pandas_mean']:.2f} "
              f"(差异 {r['diff_pct']:.2f}%)")

    return {"all_match": all_match, "details": results}


# ==================== 可视化 ====================

def plot_cluster_radar(df: pd.DataFrame):
    """
    各簇雷达图 — 使用离散评分 R_score/F_score/M_score（业务可读）。
    按 algorithm.md 第2.3节要求: 雷达图不应对数变换后的值。
    """
    rfm_cols = ["R_score", "F_score", "M_score"]
    cluster_means = df.groupby("cluster")[rfm_cols].mean()

    # 归一化到 [0, 1]
    cluster_norm = (cluster_means - cluster_means.min()) / (
        cluster_means.max() - cluster_means.min() + 1e-10
    )

    angles = np.linspace(0, 2 * np.pi, len(rfm_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(cluster_norm.index)))

    for i, cluster_id in enumerate(cluster_norm.index):
        values = cluster_norm.loc[cluster_id].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", label=f"簇 {int(cluster_id)}",
                linewidth=2, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(rfm_cols, fontsize=12)
    ax.set_title("各簇 RFM 特征雷达图（离散评分）", pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(str(config.OUTPUT_DIR / "cluster_radar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  雷达图已保存: {config.OUTPUT_DIR / 'cluster_radar.png'}")


def plot_pca_scatter(df: pd.DataFrame):
    """
    PCA 降维散点图 — 使用连续特征 + Jitter 防重叠。
    按 algorithm.md 第3.2节要求。
    """
    features = ["Recency", "Frequency", "Monetary"]
    pca_result = pca_reduce(df, features)
    X_pca = pca_result["coords"]

    # Jitter 防重叠（采样不超过5000点）
    n_sample = min(5000, len(X_pca))
    indices = np.random.choice(len(X_pca), n_sample, replace=False)
    X_sample = X_pca[indices]
    clusters = df["cluster"].values[indices]

    jitter = np.random.uniform(-0.08, 0.08, size=X_sample.shape)
    X_jittered = X_sample + jitter * np.std(X_sample, axis=0)

    plt.figure(figsize=(10, 7))
    unique_clusters = sorted(set(clusters))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(unique_clusters)))

    for i, c in enumerate(unique_clusters):
        mask = clusters == c
        plt.scatter(
            X_jittered[mask, 0], X_jittered[mask, 1],
            c=[colors[i]], alpha=0.5, s=8,
            label=f"簇 {int(c)}",
        )

    plt.colorbar = None
    plt.xlabel(f"PC1 ({pca_result['var_ratio'][0]:.1%})", fontsize=12)
    plt.ylabel(f"PC2 ({pca_result['var_ratio'][1]:.1%})", fontsize=12)
    plt.title(f"PCA 降维客户聚类可视化 (累计解释率: {pca_result['total_var']:.1%})", fontsize=14)
    plt.legend(markerscale=3, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(config.OUTPUT_DIR / "pca_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PCA散点图已保存: {config.OUTPUT_DIR / 'pca_clusters.png'}")


def plot_cluster_composition(df: pd.DataFrame):
    """各簇占比饼图"""
    counts = df["cluster"].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(counts)))

    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(
        counts.values,
        labels=[f"簇 {int(c)}" for c in counts.index],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    for t in autotexts:
        t.set_fontsize(10)
    plt.title("各簇客户占比", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(config.OUTPUT_DIR / "cluster_composition.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  簇占比图已保存: {config.OUTPUT_DIR / 'cluster_composition.png'}")


# ==================== 独立测试入口 ====================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(config.BASE_DIR))
    from data_cleaning import load_data, clean_data, engineer_features

    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = compute_rfm(df)
    df = label_customer_value(df)

    # 预处理
    X_scaled, scaler, feats = preprocess_for_clustering(df)

    # K值搜索
    opt_result = find_optimal_k(X_scaled)

    # 聚类
    df = perform_clustering(df, n_clusters=opt_result["k"])

    # 分析
    stats = analyze_clusters(df)

    # 交叉验证（演示：同一份数据作为 SQL 和 Pandas 的代表）
    result = cross_validate_rfm(df, df)

    # 可视化
    plot_cluster_radar(df)
    plot_pca_scatter(df)
    plot_cluster_composition(df)

    print(f"\nRFM + 聚类完成, 数据形状: {df.shape}")
