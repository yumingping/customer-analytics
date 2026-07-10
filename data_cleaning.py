"""数据清洗与预处理"""
import pandas as pd
import numpy as np
from pathlib import Path
import config


def load_data(file_path=None) -> pd.DataFrame:
    """加载原始CSV数据"""
    path = file_path or config.CSV_FILE
    print(f"加载数据: {path}")
    try:
        df = pd.read_csv(path, encoding="gb18030", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gb18030", low_memory=False, errors="ignore")
    print(f"原始数据形状: {df.shape}")
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
    print(f"\n数据概览:")
    print(f"  行数: {df.shape[0]}, 列数: {df.shape[1]}")
    print(f"  缺失值统计:")
    for col, miss in info["missing"].items():
        if miss > 0:
            print(f"    {col}: {miss} ({info['missing_pct'][col]}%)")
    return info


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """执行数据清洗"""
    print("\n开始数据清洗...")

    # 1. 处理日期列
    date_cols = ["FFP_DATE", "FIRST_FLIGHT_DATE", "LAST_FLIGHT_DATE", "LOAD_TIME"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 2. 处理缺失值
    # 地址信息缺失较多，填充为"未知"
    for col in ["WORK_CITY", "WORK_PROVINCE", "WORK_COUNTRY"]:
        if col in df.columns:
            df[col] = df[col].fillna("未知")
            df[col] = df[col].replace(".", "未知").replace("", "未知")

    # AGE 缺失用中位数填充
    if "AGE" in df.columns:
        age_median = df["AGE"].median()
        df["AGE"] = df["AGE"].fillna(age_median)
        print(f"  AGE 缺失值填充为中位数: {age_median}")

    # avg_discount 缺失用均值填充
    if "avg_discount" in df.columns:
        discount_mean = df["avg_discount"].mean()
        df["avg_discount"] = df["avg_discount"].fillna(discount_mean)
        print(f"  avg_discount 缺失值填充为均值: {discount_mean:.4f}")

    # GENDER 缺失填"未知"
    if "GENDER" in df.columns:
        df["GENDER"] = df["GENDER"].fillna("未知")

    # 3. 异常值处理
    # 剔除飞行次数或里程为0的无效记录
    before = len(df)
    if "FLIGHT_COUNT" in df.columns and "SEG_KM_SUM" in df.columns:
        df = df[(df["FLIGHT_COUNT"] > 0) & (df["SEG_KM_SUM"] > 0)]
        removed = before - len(df)
        print(f"  剔除无效记录(FLIGHT_COUNT=0或SEG_KM_SUM=0): {removed} 条")

    # 剔除年龄异常
    before = len(df)
    if "AGE" in df.columns:
        df = df[(df["AGE"] > 0) & (df["AGE"] <= 100)]
        removed = before - len(df)
        print(f"  剔除年龄异常记录: {removed} 条")

    df = df.reset_index(drop=True)
    print(f"清洗后数据形状: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """衍生特征工程"""
    print("\n特征工程...")
    obs_date = pd.Timestamp("2014-03-31")

    # Recency: 距观察期最后一天的天数
    if "LAST_FLIGHT_DATE" in df.columns and "LOAD_TIME" in df.columns:
        df["Recency"] = (obs_date - df["LAST_FLIGHT_DATE"]).dt.days
    elif "LAST_FLIGHT_DATE" in df.columns:
        df["Recency"] = (obs_date - df["LAST_FLIGHT_DATE"]).dt.days

    # Frequency: 总飞行次数
    if "FLIGHT_COUNT" in df.columns:
        df["Frequency"] = df["FLIGHT_COUNT"]

    # Monetary: 总里程
    if "SEG_KM_SUM" in df.columns:
        df["Monetary"] = df["SEG_KM_SUM"]

    # 会员时长（年）
    if "FFP_DATE" in df.columns:
        df["membership_years"] = (obs_date - df["FFP_DATE"]).dt.days / 365.25

    print(f"  新增特征: Recency, Frequency, Monetary, membership_years")
    return df


def save_clean_data(df: pd.DataFrame, path: str = None) -> str:
    """保存清洗后的数据"""
    path = path or str(config.OUTPUT_DIR / "cleaned_data.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"清洗数据已保存: {path}")
    return path


if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
    df = clean_data(df)
    df = engineer_features(df)
    save_clean_data(df)
