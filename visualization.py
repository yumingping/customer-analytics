"""数据可视化模块"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_age_distribution(df: pd.DataFrame):
    """年龄分布直方图"""
    plt.figure(figsize=(10, 6))
    plt.hist(df["AGE"].dropna(), bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    plt.xlabel("年龄")
    plt.ylabel("人数")
    plt.title("客户年龄分布")
    plt.grid(True, alpha=0.3)
    plt.savefig(str(config.OUTPUT_DIR / "age_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_gender_distribution(df: pd.DataFrame):
    """性别分布饼图"""
    gender_counts = df["GENDER"].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(
        gender_counts.values,
        labels=gender_counts.index.tolist(),
        autopct="%1.1f%%",
        startangle=90,
        colors=["skyblue", "lightcoral", "lightgray"],
    )
    plt.title("客户性别分布")
    plt.savefig(str(config.OUTPUT_DIR / "gender_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_tier_distribution(df: pd.DataFrame):
    """会员等级分布"""
    plt.figure(figsize=(8, 5))
    tier_counts = df["FFP_TIER"].value_counts().sort_index()
    plt.bar(tier_counts.index.astype(str), tier_counts.values, color="mediumseagreen")
    plt.xlabel("会员等级")
    plt.ylabel("人数")
    plt.title("会员等级分布")
    for i, v in enumerate(tier_counts.values):
        plt.text(i, v + max(tier_counts.values) * 0.01, str(v), ha="center")
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig(str(config.OUTPUT_DIR / "tier_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_flight_count_distribution(df: pd.DataFrame):
    """飞行次数分布"""
    plt.figure(figsize=(10, 6))
    plt.hist(df["FLIGHT_COUNT"], bins=50, color="coral", edgecolor="white", alpha=0.8)
    plt.xlabel("飞行次数")
    plt.ylabel("人数")
    plt.title("客户飞行次数分布")
    plt.grid(True, alpha=0.3)
    plt.savefig(str(config.OUTPUT_DIR / "flight_count_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    """数值特征相关性热力图"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 选取关键特征
    key_cols = [
        c for c in num_cols
        if c in ["AGE", "FLIGHT_COUNT", "SEG_KM_SUM", "BP_SUM", "avg_discount",
                 "Recency", "Frequency", "Monetary", "R_score", "F_score", "M_score",
                 "RFM_total"]
    ]
    corr = df[key_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

    # 设置刻度
    ax.set_xticks(np.arange(len(key_cols)))
    ax.set_yticks(np.arange(len(key_cols)))
    ax.set_xticklabels(key_cols, rotation=45, ha="right")
    ax.set_yticklabels(key_cols)

    # 在每个单元格标注相关系数
    for i in range(len(key_cols)):
        for j in range(len(key_cols)):
            if i != j:
                text = ax.text(
                    j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=8,
                )

    ax.set_title("数值特征相关性热力图")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(str(config.OUTPUT_DIR / "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_plots(df: pd.DataFrame):
    """生成全部可视化图表"""
    print("\n生成可视化图表...")
    plot_age_distribution(df)
    plot_gender_distribution(df)
    plot_tier_distribution(df)
    plot_flight_count_distribution(df)
    plot_correlation_heatmap(df)

    print(f"  所有图表已保存至: {config.OUTPUT_DIR}/")
    print(f"    1. age_distribution.png - 年龄分布")
    print(f"    2. gender_distribution.png - 性别分布")
    print(f"    3. tier_distribution.png - 会员等级分布")
    print(f"    4. flight_count_distribution.png - 飞行次数分布")
    print(f"    5. correlation_heatmap.png - 相关性热力图")


if __name__ == "__main__":
    from data_cleaning import load_data, clean_data, engineer_features

    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    generate_all_plots(df)
