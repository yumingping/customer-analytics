"""
数据清洗与预处理 — core_algorithm.data_cleaning
=============================================
从原始 CSV 加载、清洗、特征工程，输出可用于 RFM 分析的干净 DataFrame。
(从根目录 data_cleaning.py 移入，功能不变)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import config


def load_data(file_path=None) -> pd.DataFrame:
    """加载原始CSV数据"""
    path = file_path or config.CSV_FILE
    print(f"[数据清洗] 加载数据: {path}")
    try:
        df = pd.read_csv(path, encoding="gb18030", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gb18030", low_memory=False, errors="ignore")
    print(f"  原始数据形状: {df.shape}")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """数据概览，返回基本信息"""
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "numeric_stats": df.describe().to_dict(),
    }
    print(f"\n[数据概览]")
    print(f"  行数: {df.shape[0]}, 列数: {df.shape[1]}")
    print(f"  缺失值统计:")
    for col, miss in info["missing"].items():
        if miss > 0:
            print(f"    {col}: {miss} ({info['missing_pct'][col]}%)")
    return info


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """执行数据清洗"""
    print("\n[数据清洗] 开始...")

    # 1. 处理日期列
    date_cols = ["FFP_DATE", "FIRST_FLIGHT_DATE", "LAST_FLIGHT_DATE", "LOAD_TIME"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 2. 处理缺失值
    for col in ["WORK_CITY", "WORK_PROVINCE", "WORK_COUNTRY"]:
        if col in df.columns:
            df[col] = df[col].fillna("未知")
            df[col] = df[col].replace(".", "未知").replace("", "未知")

    if "AGE" in df.columns:
        age_median = df["AGE"].median()
        df["AGE"] = df["AGE"].fillna(age_median)
        print(f"  AGE 缺失值填充为中位数: {age_median:.1f}")

    if "avg_discount" in df.columns:
        discount_mean = df["avg_discount"].mean()
        df["avg_discount"] = df["avg_discount"].fillna(discount_mean)
        print(f"  avg_discount 缺失值填充为均值: {discount_mean:.4f}")

    if "GENDER" in df.columns:
        df["GENDER"] = df["GENDER"].fillna("未知")

    # 3. 异常值处理
    before = len(df)
    if "FLIGHT_COUNT" in df.columns and "SEG_KM_SUM" in df.columns:
        df = df[(df["FLIGHT_COUNT"] > 0) & (df["SEG_KM_SUM"] > 0)]
        print(f"  剔除无效记录(飞行次数/里程=0): {before - len(df)} 条")

    before = len(df)
    if "AGE" in df.columns:
        df = df[(df["AGE"] > 0) & (df["AGE"] <= 100)]
        print(f"  剔除年龄异常记录: {before - len(df)} 条")

    df = df.reset_index(drop=True)
    print(f"  清洗后数据形状: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """衍生特征工程"""
    print("\n[特征工程] ...")
    obs_date = pd.Timestamp("2014-03-31")

    if "LAST_FLIGHT_DATE" in df.columns:
        df["Recency"] = (obs_date - df["LAST_FLIGHT_DATE"]).dt.days

    if "FLIGHT_COUNT" in df.columns:
        df["Frequency"] = df["FLIGHT_COUNT"]

    if "SEG_KM_SUM" in df.columns:
        df["Monetary"] = df["SEG_KM_SUM"]

    if "FFP_DATE" in df.columns:
        df["membership_years"] = (obs_date - df["FFP_DATE"]).dt.days / 365.25

    print(f"  新增特征: Recency, Frequency, Monetary, membership_years")
    return df


def save_clean_data(df: pd.DataFrame, path: str = None) -> str:
    """保存清洗后的数据"""
    path = path or str(config.OUTPUT_DIR / "cleaned_data.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[数据清洗] 已保存: {path}")
    return path


if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
    df = clean_data(df)
    df = engineer_features(df)
    save_clean_data(df)
