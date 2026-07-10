"""
主流程入口 — MCP 架构版
=====================
分析流水线: 清洗 → RFM计算 → Log1p+KMeans聚类 → 三表入库 → 沙箱初始化

用法:
  python main.py               # 完整流水线
  python main.py --skip-db     # 跳过数据库入库
  python main.py --skip-viz    # 跳过可视化
  python main.py --query       # 完成后进入查询模式
"""
import sys
import traceback
import pandas as pd
from pathlib import Path
import config


def run_pipeline(skip_db: bool = False, skip_viz: bool = False) -> pd.DataFrame:
    """执行完整分析流水线"""
    print("=" * 50)
    print("  客户分析与智能查询 — MCP 架构版")
    print("=" * 50)

    # ── Step 1: 数据清洗 ──
    print("\n" + "=" * 30)
    print("Step 1: 数据加载与清洗")
    print("=" * 30)
    from core_algorithm.data_cleaning import load_data, clean_data, engineer_features

    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)

    # ── Step 2: RFM 计算 ──
    print("\n" + "=" * 30)
    print("Step 2: RFM 模型计算（百分位排名法）")
    print("=" * 30)
    from core_algorithm.rfm_analysis import compute_rfm, label_customer_value
    df = compute_rfm(df)
    df = label_customer_value(df)

    # ── Step 3: K-Means 聚类（Log1p + StandardScaler） ──
    print("\n" + "=" * 30)
    print("Step 3: K-Means 聚类（Log1p + StandardScaler）")
    print("=" * 30)
    from core_algorithm.rfm_analysis import (
        preprocess_for_clustering, find_optimal_k, perform_clustering,
        analyze_clusters,
    )

    # 预处理：Log1p + StandardScaler
    X_scaled, scaler, features = preprocess_for_clustering(df)

    # 最优 K 值搜索（SSE + Silhouette Score 双指标）
    opt_result = find_optimal_k(X_scaled)
    best_k = opt_result["k"]

    # 执行聚类
    df = perform_clustering(df, n_clusters=best_k, features=features)
    cluster_stats = analyze_clusters(df)

    # 聚类业务解读
    print("\n  [业务解读] 聚类结果:")
    for cluster_id in sorted(cluster_stats.index):
        try:
            r_mean = cluster_stats.loc[cluster_id, "R_score"]
            f_mean = cluster_stats.loc[cluster_id, "F_score"]
            m_mean = cluster_stats.loc[cluster_id, "M_score"]
        except KeyError:
            continue

        rfm_sum = r_mean + f_mean + m_mean
        if rfm_sum >= 12 and f_mean >= 4 and m_mean >= 4:
            desc = "高价值常旅客 — 频繁出行、高里程贡献"
        elif rfm_sum >= 9 and f_mean >= 3:
            desc = "潜力客户 — 有一定活跃度和消费能力"
        elif r_mean <= 2 and f_mean <= 2:
            desc = "沉睡客户 — 近期未出行、活跃度低"
        else:
            desc = "一般客户 — 处于中等水平"
        print(f"    簇 {int(cluster_id)}: {desc} (n={int(cluster_stats.loc[cluster_id, 'count'])})")

    # ── Step 4: 可视化 ──
    if not skip_viz:
        print("\n" + "=" * 30)
        print("Step 4: 可视化分析")
        print("=" * 30)
        from core_algorithm.rfm_analysis import (
            plot_cluster_radar, plot_pca_scatter, plot_cluster_composition,
        )
        from visualization import generate_all_plots

        generate_all_plots(df)
        plot_cluster_radar(df)
        plot_pca_scatter(df)
        plot_cluster_composition(df)

    # ── Step 5: 数据入库（三表存储） ──
    if not skip_db:
        print("\n" + "=" * 30)
        print("Step 5: 数据入库（三表分层存储）")
        print("=" * 30)
        try:
            from database import store_to_tables, verify_data
            store_to_tables(df)
            verify_data()
        except Exception as e:
            print(f"  [警告] 数据库操作失败: {e}")
            print(f"  系统将自动降级使用 SQLite 沙箱")
    else:
        print("\n  [跳过] 数据库入库")

    # ── Step 6: 初始化沙箱 ──
    print("\n" + "=" * 30)
    print("Step 6: 初始化 SQLite 沙箱")
    print("=" * 30)
    from mcp_tools.sql_tools import setup_sandbox_from_csv

    # 保存最终结果
    output_path = config.OUTPUT_DIR / "final_result.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  最终结果已保存: {output_path}")
    print(f"  共 {len(df)} 条记录, {len(df.columns)} 个字段")

    # 初始化沙箱
    setup_sandbox_from_csv(str(output_path))

    print("\n" + "=" * 50)
    print("  ✓ 流水线完成!")
    print("=" * 50)

    return df


def interactive_query():
    """进入大模型查询交互模式"""
    print("\n进入大模型查询模式...")
    try:
        from llm_query import interactive_mode
        interactive_mode()
    except ImportError:
        print("[提示] llm_query 模块不可用，请通过 Web API 查询")
    except Exception as e:
        print(f"[错误] {e}")


if __name__ == "__main__":
    args = set(sys.argv[1:])
    skip_db = "--skip-db" in args
    skip_viz = "--skip-viz" in args

    try:
        df = run_pipeline(skip_db=skip_db, skip_viz=skip_viz)

        if "--query" in args:
            interactive_query()
    except FileNotFoundError as e:
        print(f"\n[错误] 找不到数据文件: {e}")
        print(f"请确保 {config.CSV_FILE} 存在")
        sys.exit(1)
    except Exception as e:
        print(f"\n[错误] 流水线执行失败:")
        traceback.print_exc()
        sys.exit(1)
