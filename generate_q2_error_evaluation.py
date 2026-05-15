#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成问题二最终模型的验证集误差评价表。

输出：
1. q2_error_evaluation.csv
2. q2_error_evaluation_latex.txt

运行：
    /Users/linjiamin/venv/bin/python generate_q2_error_evaluation.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

data_path = "daily_summary.csv"
output_csv = "q2_error_evaluation.csv"
output_latex = "q2_error_evaluation_latex.txt"

TARGETS = [
    "就餐人数",
    "销售总额",
    "热量需求",
    "碳水需求",
    "蛋白质需求",
    "脂肪需求",
    "膳食纤维需求",
]

FINAL_MODELS = {
    "就餐人数": "KNN",
    "销售总额": "ExtraTrees",
    "热量需求": "KNN",
    "碳水需求": "KNN",
    "蛋白质需求": "ElasticNet",
    "脂肪需求": "优化KNN",
    "膳食纤维需求": "KNN",
}

# 正文已有参考值，用于运行后提示差异；最终输出以本脚本复现值为准。
REFERENCE_RMSE = {
    "就餐人数": 50.399,
    "销售总额": 870.654,
    "热量需求": 42493.580,
    "碳水需求": 4075.666,
    "蛋白质需求": 2558.724,
    "脂肪需求": 1980.421,
    "膳食纤维需求": 369.901,
}


def read_data(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("仅支持 csv、xlsx、xls 数据文件")

    df.columns = [str(c).strip() for c in df.columns]
    print("数据列名：", list(df.columns))

    date_col = None
    for col in df.columns:
        if col.lower() in {"date", "day"} or "日期" in col:
            date_col = col
            break
    if date_col is None:
        raise ValueError("未识别到日期列，请检查数据字段。")

    # 字段名映射：优先精确匹配，若不一致则根据关键词识别。
    col_map = {"date": date_col}
    for target in TARGETS:
        if target in df.columns:
            col_map[target] = target
            continue
        key = target.replace("需求", "")
        candidates = [c for c in df.columns if key in c]
        if candidates:
            col_map[target] = candidates[0]
        else:
            raise ValueError(f"未找到目标字段：{target}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    for target, col in col_map.items():
        if target != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, col_map


def add_date_features(date, start_date):
    day = int(date.day)
    return {
        "date_ordinal": float((date - start_date).days),
        "dayofweek": float(date.dayofweek),
        "is_workday": float(date.dayofweek < 5),
        "month": float(date.month),
        "is_month_start": float(day <= 10),
        "is_month_mid": float(11 <= day <= 20),
        "is_month_end": float(day >= 21),
    }


def make_feature_frame(df, date_col, target_col):
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "y": pd.to_numeric(df[target_col], errors="coerce"),
    })
    start_date = out["date"].min()
    date_features = out["date"].apply(lambda d: pd.Series(add_date_features(d, start_date)))
    out = pd.concat([out, date_features], axis=1)

    # 先 shift 再 rolling，确保只使用历史数据，不泄露当前日与未来数据。
    for lag in [1, 3, 7]:
        out[f"lag_{lag}"] = out["y"].shift(lag)
    shifted = out["y"].shift(1)
    for window in [3, 7]:
        out[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        out[f"rolling_std_{window}"] = shifted.rolling(window).std()

    return out.dropna().reset_index(drop=True)


def make_model(model_name):
    if model_name == "KNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=7, weights="distance", p=2)),
        ])
    if model_name == "优化KNN":
        # 根据前序调参结果，脂肪需求最优参数为 k=11、uniform、Manhattan 距离。
        return Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=11, weights="uniform", p=1)),
        ])
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42)
    if model_name == "ElasticNet":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("elasticnet", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)),
        ])
    raise ValueError(f"未知模型：{model_name}")


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = np.abs(y_true) > 1e-12
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return mae, rmse, mape


def make_latex_table(result_df):
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{问题二模型预测误差评价结果}",
        r"    \label{tab:q2_error_evaluation}",
        r"    \begin{tabular}{ccccc}",
        r"        \toprule",
        r"        预测指标 & 最终模型 & MAE & RMSE & MAPE/\% \\",
        r"        \midrule",
    ]
    for _, row in result_df.iterrows():
        lines.append(
            f"        {row['预测指标']} & {row['最终模型']} & "
            f"{row['MAE']:.3f} & {row['RMSE']:.3f} & {row['MAPE/%']:.2f} \\\\"
        )
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    df, col_map = read_data(data_path)
    date_col = col_map["date"]

    rows = []
    raw_split_idx = int(len(df) * 0.8)
    split_date = pd.to_datetime(df[date_col]).iloc[raw_split_idx]
    for target in TARGETS:
        target_col = col_map[target]
        model_name = FINAL_MODELS[target]
        feat_df = make_feature_frame(df, date_col, target_col)
        feature_cols = [c for c in feat_df.columns if c not in {"date", "y"}]

        train_df = feat_df[feat_df["date"] < split_date].copy()
        valid_df = feat_df[feat_df["date"] >= split_date].copy()

        model = make_model(model_name)
        model.fit(train_df[feature_cols], train_df["y"])
        pred = np.maximum(model.predict(valid_df[feature_cols]), 0)
        mae, rmse, mape = calc_metrics(valid_df["y"], pred)

        rows.append({
            "预测指标": target,
            "最终模型": model_name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "MAPE/%": round(mape, 2),
        })

    result_df = pd.DataFrame(rows, columns=["预测指标", "最终模型", "MAE", "RMSE", "MAPE/%"])
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    latex = make_latex_table(result_df)
    Path(output_latex).write_text(latex, encoding="utf-8")

    print("\n模型预测误差评价表：")
    print(result_df.to_string(index=False))
    print(f"\nCSV 已保存：{output_csv}")
    print(f"LaTeX 表格已保存：{output_latex}")
    print("\nLaTeX 表格代码：")
    print(latex)

    print("\n与正文已有 RMSE 的差异提醒：")
    for _, row in result_df.iterrows():
        target = row["预测指标"]
        ref = REFERENCE_RMSE.get(target)
        if ref is None:
            continue
        diff = row["RMSE"] - ref
        print(f" - {target}: 本次 RMSE={row['RMSE']:.3f}, 正文参考 RMSE={ref:.3f}, 差异={diff:.3f}")


if __name__ == "__main__":
    main()
