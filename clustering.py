"""K-Means 客户聚类"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import config

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def find_optimal_k(X_scaled: np.ndarray, k_range: range = None) -> int:
    """肘部法则确定最优K值，返回最佳K并绘图"""
    k_range = k_range or config.CLUSTER_RANGE
    inertias = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=config.CLUSTER_RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # 绘制肘部图
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, "bo-")
    plt.xlabel("K (聚类数)")
    plt.ylabel("SSE (簇内误差平方和)")
    plt.title("肘部法则确定最优 K 值")
    plt.grid(True, alpha=0.3)
    plt.savefig(str(config.OUTPUT_DIR / "elbow_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"肘部图已保存: {config.OUTPUT_DIR / 'elbow_plot.png'}")

    # 自动推荐 K（二阶差分最大点 + 2）
    deltas = np.diff(inertias)
    delta2 = np.diff(deltas)
    recommended_k = list(k_range)[np.argmin(delta2) + 1] if len(delta2) > 0 else 4

    print(f"  各K值SSE: {dict(zip(k_range, [round(v, 1) for v in inertias]))}")
    print(f"  推荐 K = {recommended_k}")

    return recommended_k


def perform_clustering(df: pd.DataFrame, n_clusters: int = None) -> pd.DataFrame:
    """执行 K-Means 聚类"""
    n_clusters = n_clusters or config.N_CLUSTERS
    print(f"\nK-Means 聚类 (K={n_clusters})...")

    feature_cols = ["R_score", "F_score", "M_score"]
    X = df[feature_cols].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 聚类
    km = KMeans(n_clusters=n_clusters, random_state=config.CLUSTER_RANDOM_STATE, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    # 保存模型参数供后续使用
    df.attrs["cluster_centers"] = km.cluster_centers_
    df.attrs["scaler_mean"] = scaler.mean_
    df.attrs["scaler_scale"] = scaler.scale_

    cluster_counts = df["cluster"].value_counts().sort_index()
    print("  各簇客户数量:")
    for c, count in cluster_counts.items():
        print(f"    簇 {c}: {count} ({count / len(df) * 100:.1f}%)")

    return df


def analyze_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """分析聚类结果：各簇RFM均值"""
    print("\n聚类结果分析...")

    rfm_cols = ["R_score", "F_score", "M_score", "Recency", "Frequency", "Monetary"]
    cluster_stats = df.groupby("cluster")[rfm_cols].mean().round(2)
    cluster_stats["count"] = df.groupby("cluster").size()

    print("  各簇 RFM 均值:")
    print(cluster_stats.to_string())

    return cluster_stats


def plot_cluster_radar(df: pd.DataFrame):
    """绘制各簇雷达图"""
    rfm_cols = ["R_score", "F_score", "M_score"]
    cluster_means = df.groupby("cluster")[rfm_cols].mean()

    # 归一化到 [0, 1] 便于对比
    cluster_norm = (cluster_means - cluster_means.min()) / (
        cluster_means.max() - cluster_means.min() + 1e-10
    )

    angles = np.linspace(0, 2 * np.pi, len(rfm_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for cluster_id in cluster_norm.index:
        values = cluster_norm.loc[cluster_id].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", label=f"簇 {cluster_id}", linewidth=2)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(rfm_cols)
    ax.set_title("各簇 RFM 特征雷达图", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(str(config.OUTPUT_DIR / "cluster_radar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"雷达图已保存: {config.OUTPUT_DIR / 'cluster_radar.png'}")


def plot_pca_scatter(df: pd.DataFrame):
    """PCA 降维后可视化聚类"""
    feature_cols = ["R_score", "F_score", "M_score"]
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=config.CLUSTER_RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=df["cluster"], cmap="viridis", alpha=0.6, s=10,
    )
    plt.colorbar(scatter, label="簇")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("PCA 降维可视化客户聚类")
    plt.grid(True, alpha=0.3)
    plt.savefig(str(config.OUTPUT_DIR / "pca_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PCA聚类图已保存: {config.OUTPUT_DIR / 'pca_clusters.png'}")


def plot_cluster_composition(df: pd.DataFrame):
    """绘制簇占比饼图"""
    counts = df["cluster"].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(counts)))

    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(
        counts.values,
        labels=[f"簇 {c}" for c in counts.index],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    plt.title("各簇客户占比")
    plt.savefig(str(config.OUTPUT_DIR / "cluster_composition.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"簇占比图已保存: {config.OUTPUT_DIR / 'cluster_composition.png'}")


if __name__ == "__main__":
    from data_cleaning import load_data, clean_data, engineer_features
    from rfm_analysis import compute_rfm

    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = compute_rfm(df)
    df = perform_clustering(df)
    cluster_stats = analyze_clusters(df)
    plot_cluster_radar(df)
    plot_pca_scatter(df)
    plot_cluster_composition(df)
