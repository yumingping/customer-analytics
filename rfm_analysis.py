"""RFM 模型计算与客户价值分层"""
import pandas as pd
import numpy as np
from pathlib import Path
import config


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """计算 R、F、M 得分（1~5分）"""
    print("\n计算RFM得分...")

    def _score(col: pd.Series, ascending: bool = True) -> pd.Series:
        """将连续变量分5等份评分（1~5分），兼容重复值情况"""
        valid = col.dropna()
        # 用百分比排名替代 qcut，避免重复分位数问题
        pct = valid.rank(method="average", ascending=ascending) / len(valid)
        # 将 [0,1] 映射到 {1,2,3,4,5}
        result = pd.cut(pct, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        labels=[1, 2, 3, 4, 5], include_lowest=True)
        result = result.astype(int)
        # 填充原始列中的 NaN 为中位分 3
        return result.reindex(col.index, fill_value=3)

    # R 越小越好 → 降序（ascending=False 让值小的得高分）
    df["R_score"] = _score(df["Recency"], ascending=False)

    # F 越大越好 → 升序
    df["F_score"] = _score(df["Frequency"], ascending=True)

    # M 越大越好 → 升序
    df["M_score"] = _score(df["Monetary"], ascending=True)

    # RFM 总分
    df["RFM_total"] = df["R_score"] + df["F_score"] + df["M_score"]

    print(f"  R_score 分布: {df['R_score'].value_counts().sort_index().to_dict()}")
    print(f"  F_score 分布: {df['F_score'].value_counts().sort_index().to_dict()}")
    print(f"  M_score 分布: {df['M_score'].value_counts().sort_index().to_dict()}")
    return df


def label_customer_value(df: pd.DataFrame) -> pd.DataFrame:
    """基于RFM总分做简单客户价值分层"""
    print("\n客户价值分层...")

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


if __name__ == "__main__":
    from data_cleaning import load_data, clean_data, engineer_features

    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = compute_rfm(df)
    df = label_customer_value(df)
    print(f"\nRFM 计算完成，数据形状: {df.shape}")
