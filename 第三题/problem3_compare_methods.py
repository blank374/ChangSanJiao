#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题三三类方法对比：
1. 基准模型：贪心算法
2. 主模型：多目标整数规划（MILP）
3. 改进模型：NSGA-II

输出：
    problem3_method_comparison.csv
    problem3_method_comparison.xlsx
    problem3_method_comparison_latex.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd


METHOD_FILES = {
    "贪心算法": "problem3_greedy_lunch_menu_plan.xlsx",
    "多目标整数规划": "problem3_lunch_menu_plan.xlsx",
    "NSGA-II": "problem3_nsga2_lunch_menu_plan.xlsx",
}

NUTRIENT_PAIRS = [
    ("热量供给", "预测热量需求"),
    ("碳水供给", "预测碳水需求"),
    ("蛋白质供给", "预测蛋白质需求"),
    ("脂肪供给", "预测脂肪需求"),
    ("膳食纤维供给", "预测膳食纤维需求"),
]


def read_summary(method: str, path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="每日汇总")
    df["方法"] = method
    return df


def compute_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    nutrient_devs = []
    for supply, target in NUTRIENT_PAIRS:
        dev = (out[supply] - out[target]).abs() / (out[target].abs() + 1e-12)
        nutrient_devs.append(dev)
    out["平均营养相对偏差"] = np.vstack([d.to_numpy() for d in nutrient_devs]).mean(axis=0)
    out["销售额相对偏差"] = (out["预计销售额"] - out["预测销售总额"]).abs() / (out["预测销售总额"].abs() + 1e-12)
    # 100g 单位下，总备菜重量 kg = unit * 0.1，因此推回总单位。
    out["总备菜单位偏差"] = (out["总备菜重量kg"] * 10 - out["预测就餐人数"] * 4.784).abs() / (out["预测就餐人数"] * 4.784 + 1e-12)
    out["利润"] = out["预计利润"]
    out["菜品多样性"] = out["推荐菜品数"]
    return out


def normalize_positive(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


def main() -> None:
    frames = []
    for method, path in METHOD_FILES.items():
        frames.append(compute_daily_metrics(read_summary(method, path)))
    daily = pd.concat(frames, ignore_index=True)

    method_summary = daily.groupby("方法", as_index=False).agg(
        平均预计利润=("利润", "mean"),
        平均销售额相对偏差=("销售额相对偏差", "mean"),
        平均营养相对偏差=("平均营养相对偏差", "mean"),
        平均备菜量相对偏差=("总备菜单位偏差", "mean"),
        平均菜品数=("菜品多样性", "mean"),
    )

    # 综合评分：利润和多样性越高越好，偏差越低越好。
    score_df = method_summary.copy()
    profit_score = normalize_positive(score_df["平均预计利润"])
    diversity_score = normalize_positive(score_df["平均菜品数"])
    sales_score = 1 - normalize_positive(score_df["平均销售额相对偏差"])
    nutrition_score = 1 - normalize_positive(score_df["平均营养相对偏差"])
    quantity_score = 1 - normalize_positive(score_df["平均备菜量相对偏差"])
    score_df["综合评分"] = (
        0.25 * profit_score
        + 0.15 * diversity_score
        + 0.20 * sales_score
        + 0.30 * nutrition_score
        + 0.10 * quantity_score
    )
    method_summary["综合评分"] = score_df["综合评分"]
    method_summary = method_summary.sort_values("综合评分", ascending=False).reset_index(drop=True)

    daily_out = daily[[
        "方法", "日期", "星期", "推荐菜品数", "总备菜重量kg", "预计销售额", "预计利润",
        "平均营养相对偏差", "销售额相对偏差", "总备菜单位偏差",
    ]].copy()
    for col in daily_out.columns:
        if pd.api.types.is_float_dtype(daily_out[col]):
            daily_out[col] = daily_out[col].round(4)
    for col in method_summary.columns:
        if pd.api.types.is_float_dtype(method_summary[col]):
            method_summary[col] = method_summary[col].round(4)

    with pd.ExcelWriter("problem3_method_comparison.xlsx") as writer:
        method_summary.to_excel(writer, sheet_name="方法汇总比较", index=False)
        daily_out.to_excel(writer, sheet_name="每日指标比较", index=False)
    method_summary.to_csv("problem3_method_comparison.csv", index=False, encoding="utf-8-sig")

    latex_lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{问题三不同优化方法效果比较}",
        r"    \label{tab:q3_method_comparison}",
        r"    \begin{tabular}{cccccc}",
        r"        \toprule",
        r"        方法 & 平均预计利润 & 销售额偏差 & 营养偏差 & 平均菜品数 & 综合评分 \\",
        r"        \midrule",
    ]
    for _, row in method_summary.iterrows():
        latex_lines.append(
            f"        {row['方法']} & {row['平均预计利润']:.2f} & "
            f"{row['平均销售额相对偏差']:.4f} & {row['平均营养相对偏差']:.4f} & "
            f"{row['平均菜品数']:.2f} & {row['综合评分']:.4f} \\\\"
        )
    latex_lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ])
    Path("problem3_method_comparison_latex.txt").write_text("\n".join(latex_lines), encoding="utf-8")

    print("问题三三类方法汇总比较：")
    print(method_summary.to_string(index=False))
    print("\n推荐主结果：", method_summary.iloc[0]["方法"])
    print("\n输出文件：")
    print(" - problem3_method_comparison.xlsx")
    print(" - problem3_method_comparison.csv")
    print(" - problem3_method_comparison_latex.txt")


if __name__ == "__main__":
    main()
