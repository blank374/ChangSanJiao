#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题三基准模型：贪心算法午餐备菜方案。

思路：
1. 根据利润、历史偏好和营养缺口覆盖能力构造菜品综合得分；
2. 先选择偏好/利润较高的最低菜品种类数；
3. 再按边际得分逐个增加 100g 备菜单位，直到总备菜量接近预测就餐人数需求；
4. 输出与 MILP、NSGA-II 相同结构的午餐备菜方案。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from problem3_lunch_optimization import (
    DATA_PATH,
    PRED_PATH,
    TARGET_DATES,
    COST_RATE,
    TOP_DISH_LIMIT,
    MIN_DISH_TYPES,
    MIN_UNITS_PER_DISH,
    MAX_SHARE_PER_DISH,
    read_prediction,
    load_lunch_detail,
    build_dish_stats,
    get_day_targets,
    summarize_plan,
)


OUTPUT_XLSX = Path("problem3_greedy_lunch_menu_plan.xlsx")
OUTPUT_CSV = Path("problem3_greedy_lunch_menu_plan.csv")
OUTPUT_LATEX = Path("problem3_greedy_lunch_latex_tables.txt")


def minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return (v - v.min()) / (v.max() - v.min() + 1e-12)


def greedy_solve_day(dishes: pd.DataFrame, targets: Dict[str, float]) -> np.ndarray:
    n = len(dishes)
    target_units = int(round(targets["total_units"]))
    max_units = max(MIN_UNITS_PER_DISH + 1, int(np.ceil(targets["total_units"] * MAX_SHARE_PER_DISH)))

    profit_norm = minmax(dishes["profit_per_100g"].to_numpy())
    pref_norm = dishes["preference_norm"].to_numpy()
    nutrients = np.vstack([
        dishes["calories_100g"].to_numpy(),
        dishes["carbohydrates_100g"].to_numpy(),
        dishes["protein_100g"].to_numpy(),
        dishes["fat_100g"].to_numpy(),
        dishes["fiber_100g"].to_numpy(),
    ]).T
    nutrient_targets = np.array([
        targets["calories"],
        targets["carbohydrates"],
        targets["protein"],
        targets["fat"],
        targets["fiber"],
    ])
    target_profile = nutrient_targets / (nutrient_targets.sum() + 1e-12)
    dish_profile = nutrients / (nutrients.sum(axis=1, keepdims=True) + 1e-12)
    nutrition_similarity = 1 - np.mean(np.abs(dish_profile - target_profile), axis=1)
    nutrition_similarity = minmax(nutrition_similarity)

    base_score = 0.40 * pref_norm + 0.35 * profit_norm + 0.25 * nutrition_similarity
    selected = np.argsort(base_score)[::-1][:MIN_DISH_TYPES]
    x = np.zeros(n, dtype=int)
    x[selected] = MIN_UNITS_PER_DISH

    while x.sum() < target_units:
        supply = x @ nutrients
        deficit = np.maximum(nutrient_targets - supply, 0)
        deficit_ratio = deficit / (nutrient_targets + 1e-12)
        marginal_nutrition = nutrients @ deficit_ratio
        marginal_nutrition = minmax(marginal_nutrition)
        marginal_score = 0.35 * pref_norm + 0.30 * profit_norm + 0.35 * marginal_nutrition
        marginal_score[x >= max_units] = -np.inf
        idx = int(np.argmax(marginal_score))
        if not np.isfinite(marginal_score[idx]):
            break
        x[idx] += 1
    return x


def make_latex(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{基于贪心算法的午餐备菜方案汇总}",
        r"    \label{tab:q3_greedy_lunch_summary}",
        r"    \resizebox{\textwidth}{!}{",
        r"    \begin{tabular}{ccccccccccc}",
        r"        \toprule",
        r"        日期 & 星期 & 菜品数 & 总备菜重量/kg & 预计销售额 & 预计利润 & 热量 & 碳水 & 蛋白质 & 脂肪 & 膳食纤维 \\",
        r"        \midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"        {row['日期']} & {row['星期']} & {int(row['推荐菜品数'])} & "
            f"{row['总备菜重量kg']:.2f} & {row['预计销售额']:.2f} & {row['预计利润']:.2f} & "
            f"{row['热量供给']:.2f} & {row['碳水供给']:.2f} & {row['蛋白质供给']:.2f} & "
            f"{row['脂肪供给']:.2f} & {row['膳食纤维供给']:.2f} \\\\"
        )
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main() -> None:
    pred = read_prediction()
    detail = load_lunch_detail()
    dishes, avg_units_per_customer = build_dish_stats(detail)

    all_details: List[pd.DataFrame] = []
    summaries: List[Dict[str, float]] = []
    for _, pred_row in pred.iterrows():
        targets = get_day_targets(pred_row, avg_units_per_customer)
        x = greedy_solve_day(dishes, targets)
        y = (x > 0).astype(int)
        day_detail, day_summary = summarize_plan(pred_row["日期"], pred_row["星期"], dishes, x, y)
        day_summary.update({
            "预测就餐人数": targets["people"],
            "预测销售总额": targets["sales"],
            "预测热量需求": targets["calories"],
            "预测碳水需求": targets["carbohydrates"],
            "预测蛋白质需求": targets["protein"],
            "预测脂肪需求": targets["fat"],
            "预测膳食纤维需求": targets["fiber"],
            "算法": "Greedy",
        })
        all_details.append(day_detail)
        summaries.append(day_summary)

    detail_df = pd.concat(all_details, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    for df in [detail_df, summary_df]:
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].round(3)

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:
        summary_df.to_excel(writer, sheet_name="每日汇总", index=False)
        detail_df.to_excel(writer, sheet_name="备菜明细", index=False)
        dishes.to_excel(writer, sheet_name="菜品统计", index=False)
    detail_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    OUTPUT_LATEX.write_text(make_latex(summary_df, detail_df), encoding="utf-8")

    print("贪心算法午餐备菜方案汇总：")
    print(summary_df[["日期", "星期", "推荐菜品数", "总备菜重量kg", "预计销售额", "预计利润", "热量供给", "碳水供给", "蛋白质供给", "脂肪供给", "膳食纤维供给"]].to_string(index=False))
    print("\n输出文件：")
    print(" -", OUTPUT_XLSX)
    print(" -", OUTPUT_CSV)
    print(" -", OUTPUT_LATEX)


if __name__ == "__main__":
    main()
