#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题三：2025 年 5 月 6 日至 5 月 12 日工作日午餐备菜优化模型。

建模说明：
1. 优先读取 merged_order_dish.csv 统计午餐菜品历史销售、价格、营养与偏好；
2. 读取 problem2_prediction_results.xlsx 中 2025-05-06 至 2025-05-12 工作日预测需求；
3. 决策变量 x_{i,d} 为菜品 i 在日期 d 的 100g 备菜单位数，整数；
4. 决策变量 y_{i,d} 表示菜品 i 当天是否备菜，0-1 变量；
5. 使用 scipy.optimize.milp 求解混合整数线性规划；若不可用，退化为 linprog 后整数化。

输出：
    problem3_lunch_menu_plan.xlsx
    problem3_lunch_menu_plan.csv
    problem3_lunch_latex_tables.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog

try:
    from scipy.optimize import LinearConstraint as ScipyLinearConstraint
    from scipy.optimize import milp

    SCIPY_MILP_AVAILABLE = True
except Exception:
    ScipyLinearConstraint = None
    milp = None
    SCIPY_MILP_AVAILABLE = False


warnings.filterwarnings("ignore")

DATA_PATH = Path("merged_order_dish.csv")
PRED_PATH = Path("problem2_prediction_results.xlsx")
OUTPUT_XLSX = Path("problem3_lunch_menu_plan.xlsx")
OUTPUT_CSV = Path("problem3_lunch_menu_plan.csv")
OUTPUT_LATEX = Path("problem3_lunch_latex_tables.txt")

TARGET_DATES = pd.to_datetime([
    "2025-05-06",
    "2025-05-07",
    "2025-05-08",
    "2025-05-09",
    "2025-05-12",
])

# 可按论文口径微调的参数。
COST_RATE = 0.65                 # 未知成本时，假设菜品原料/加工成本约占售价 65%。
TOP_DISH_LIMIT = 60              # 候选菜品数量，避免模型过大且提高可执行性。
MIN_DISH_TYPES = 18              # 每日最少菜品种类。
MAX_DISH_TYPES = 32              # 每日最多菜品种类。
MIN_UNITS_PER_DISH = 2           # 入选菜品至少备 2 个 100g 单位。
MAX_SHARE_PER_DISH = 0.16        # 单菜品备菜量不超过总 100g 单位需求的比例。
TOTAL_UNITS_TOL = 0.08           # 总备菜量相对预测需求允许上下浮动比例。
NUTRIENT_LOWER = 0.86            # 营养供给下限。
NUTRIENT_UPPER = 1.18            # 营养供给上限。
SALES_LOWER = 0.82               # 预计销售额下限。
SALES_UPPER = 1.25               # 预计销售额上限。
PREFERENCE_WEIGHT = 1.20         # 消费偏好奖励权重。
DIVERSITY_REWARD = 0.08          # 菜品多样性奖励权重。


def read_prediction() -> pd.DataFrame:
    if PRED_PATH.exists():
        pred = pd.read_excel(PRED_PATH)
        pred.columns = [str(c).strip() for c in pred.columns]
        pred["日期"] = pd.to_datetime(pred["日期"])
        pred = pred[pred["日期"].isin(TARGET_DATES)].copy()
        if pred.empty:
            raise ValueError("问题二预测结果中未找到 2025-05-06 至 2025-05-12 的工作日预测。")
        return pred

    # 手工接口：若没有问题二预测结果文件，可在此替换为人工预测值。
    manual = pd.DataFrame({
        "日期": TARGET_DATES,
        "星期": ["星期二", "星期三", "星期四", "星期五", "星期一"],
        "预测就餐人数": [300, 299, 293, 302, 304],
        "预测销售总额": [3445, 3440, 3439, 3080, 3477],
        "预测热量需求": [199440, 193230, 194456, 199309, 209630],
        "预测碳水需求": [21119, 20535, 20673, 19677, 23501],
        "预测蛋白质需求": [11951, 12074, 11891, 11637, 12050],
        "预测脂肪需求": [7770, 7673, 7844, 7672, 8204],
        "预测膳食纤维需求": [1809, 1701, 1738, 1805, 1959],
    })
    return manual


def load_lunch_detail() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到 merged_order_dish.csv，请检查当前目录。")
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    required = [
        "indent_id", "dish_serial", "dish_name", "total_price", "weight", "unit_price",
        "calories_dish", "carbohydrates_dish", "protein_dish", "fat_dish", "fiber_dish",
        "meal_period",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"merged_order_dish.csv 缺少必要字段：{missing}")
    df = df[df["meal_period"].astype(str).str.lower().eq("lunch")].copy()
    for col in [
        "total_price", "weight", "unit_price", "calories_dish", "carbohydrates_dish",
        "protein_dish", "fat_dish", "fiber_dish",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["dish_name", "total_price", "weight", "unit_price"])
    df = df[(df["weight"] > 0) & (df["unit_price"] > 0) & (df["total_price"] >= 0)].copy()
    return df


def build_dish_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """统计菜品历史销量、平均价格、每 100g 营养、偏好权重。"""
    total_orders = df["indent_id"].nunique()
    avg_units_per_customer = df.groupby("indent_id")["weight"].sum().mean() / 100.0
    total_weight = df["weight"].sum()

    grouped = df.groupby(["dish_serial", "dish_name"], as_index=False).agg(
        sales_count=("indent_details_id", "count") if "indent_details_id" in df.columns else ("indent_id", "count"),
        order_count=("indent_id", "nunique"),
        total_weight_g=("weight", "sum"),
        total_sales=("total_price", "sum"),
        avg_unit_price=("unit_price", "mean"),
        calories_total=("calories_dish", "sum"),
        carbohydrates_total=("carbohydrates_dish", "sum"),
        protein_total=("protein_dish", "sum"),
        fat_total=("fat_dish", "sum"),
        fiber_total=("fiber_dish", "sum"),
    )
    grouped = grouped[grouped["total_weight_g"] > 0].copy()
    grouped["price_per_100g"] = grouped["total_sales"] / (grouped["total_weight_g"] / 100.0)
    for src, dst in [
        ("calories_total", "calories_100g"),
        ("carbohydrates_total", "carbohydrates_100g"),
        ("protein_total", "protein_100g"),
        ("fat_total", "fat_100g"),
        ("fiber_total", "fiber_100g"),
    ]:
        grouped[dst] = grouped[src] / grouped["total_weight_g"] * 100.0

    grouped["weight_share"] = grouped["total_weight_g"] / total_weight
    grouped["order_penetration"] = grouped["order_count"] / max(total_orders, 1)
    grouped["preference_weight"] = 0.65 * grouped["weight_share"] + 0.35 * grouped["order_penetration"]

    # 清理异常菜品，保留有销量、有价格、有基础营养记录的菜品。
    nutrient_cols = ["calories_100g", "carbohydrates_100g", "protein_100g", "fat_100g", "fiber_100g"]
    grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=["price_per_100g"] + nutrient_cols)
    grouped = grouped[grouped["price_per_100g"] > 0].copy()

    # 选择偏好高且营养信息完整的候选菜品。
    grouped = grouped.sort_values("preference_weight", ascending=False).head(TOP_DISH_LIMIT).reset_index(drop=True)
    pref_min, pref_max = grouped["preference_weight"].min(), grouped["preference_weight"].max()
    grouped["preference_norm"] = (grouped["preference_weight"] - pref_min) / (pref_max - pref_min + 1e-12)
    grouped["profit_per_100g"] = grouped["price_per_100g"] * (1.0 - COST_RATE)
    return grouped, float(avg_units_per_customer)


def get_day_targets(pred_row: pd.Series, avg_units_per_customer: float) -> Dict[str, float]:
    people = float(pred_row["预测就餐人数"])
    return {
        "people": people,
        "sales": float(pred_row["预测销售总额"]),
        "calories": float(pred_row["预测热量需求"]),
        "carbohydrates": float(pred_row["预测碳水需求"]),
        "protein": float(pred_row["预测蛋白质需求"]),
        "fat": float(pred_row["预测脂肪需求"]),
        "fiber": float(pred_row["预测膳食纤维需求"]),
        "total_units": max(people * avg_units_per_customer, MIN_DISH_TYPES * MIN_UNITS_PER_DISH),
    }


def solve_day_milp(dishes: pd.DataFrame, targets: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, str]:
    n = len(dishes)
    total_units = targets["total_units"]
    max_units = max(MIN_UNITS_PER_DISH + 1, int(np.ceil(total_units * MAX_SHARE_PER_DISH)))

    profit = dishes["profit_per_100g"].to_numpy()
    pref = dishes["preference_norm"].to_numpy()
    # scipy 求最小化，因此目标函数取负。
    c = np.r_[-(profit + PREFERENCE_WEIGHT * pref), -np.repeat(DIVERSITY_REWARD, n)]

    constraints = []
    lb = []
    ub = []

    # 总备菜量与预测人数匹配。
    row = np.r_[np.ones(n), np.zeros(n)]
    constraints.append(row)
    lb.append(total_units * (1 - TOTAL_UNITS_TOL))
    ub.append(total_units * (1 + TOTAL_UNITS_TOL))

    # 预计销售额区间。
    row = np.r_[dishes["price_per_100g"].to_numpy(), np.zeros(n)]
    constraints.append(row)
    lb.append(targets["sales"] * SALES_LOWER)
    ub.append(targets["sales"] * SALES_UPPER)

    # 营养供给区间。
    nutrient_map = [
        ("calories_100g", "calories"),
        ("carbohydrates_100g", "carbohydrates"),
        ("protein_100g", "protein"),
        ("fat_100g", "fat"),
        ("fiber_100g", "fiber"),
    ]
    for col, key in nutrient_map:
        row = np.r_[dishes[col].to_numpy(), np.zeros(n)]
        constraints.append(row)
        lb.append(targets[key] * NUTRIENT_LOWER)
        ub.append(targets[key] * NUTRIENT_UPPER)

    # 菜品种类数量约束。
    row = np.r_[np.zeros(n), np.ones(n)]
    constraints.append(row)
    lb.append(MIN_DISH_TYPES)
    ub.append(MAX_DISH_TYPES)

    # x_i 与 y_i 联动：x_i - max*y_i <= 0；x_i - min*y_i >= 0。
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = 1
        row[n + i] = -max_units
        constraints.append(row)
        lb.append(-np.inf)
        ub.append(0)

        row = np.zeros(2 * n)
        row[i] = 1
        row[n + i] = -MIN_UNITS_PER_DISH
        constraints.append(row)
        lb.append(0)
        ub.append(np.inf)

    A = np.vstack(constraints)
    lower = np.array(lb, dtype=float)
    upper = np.array(ub, dtype=float)

    if SCIPY_MILP_AVAILABLE and milp is not None:
        integrality = np.r_[np.ones(n), np.ones(n)]
        bounds = Bounds(np.zeros(2 * n), np.r_[np.repeat(max_units, n), np.ones(n)])
        res = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=ScipyLinearConstraint(A, lower, upper),
            options={"time_limit": 30.0, "mip_rel_gap": 0.02},
        )
        if res.success:
            x = np.rint(res.x[:n]).astype(int)
            y = np.rint(res.x[n:]).astype(int)
            return x, y, "scipy.optimize.milp"

    # 兜底：连续线性规划求解后四舍五入，并按联动关系修正。
    res = linprog(
        c=c,
        A_ub=np.vstack([A, -A]),
        b_ub=np.r_[upper, -lower],
        bounds=[(0, max_units)] * n + [(0, 1)] * n,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"优化模型求解失败：{res.message}")
    x = np.rint(res.x[:n]).astype(int)
    y = (x >= MIN_UNITS_PER_DISH).astype(int)
    x[y == 0] = 0
    return x, y, "scipy.optimize.linprog + round"


def summarize_plan(date, weekday, dishes: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    chosen = dishes.copy()
    chosen["备菜单位_100g"] = x
    chosen = chosen[chosen["备菜单位_100g"] > 0].copy()
    chosen["备菜重量kg"] = chosen["备菜单位_100g"] * 0.1
    chosen["预计销售额"] = chosen["备菜单位_100g"] * chosen["price_per_100g"]
    chosen["预计利润"] = chosen["备菜单位_100g"] * chosen["profit_per_100g"]
    for src, dst in [
        ("calories_100g", "热量供给"),
        ("carbohydrates_100g", "碳水供给"),
        ("protein_100g", "蛋白质供给"),
        ("fat_100g", "脂肪供给"),
        ("fiber_100g", "膳食纤维供给"),
    ]:
        chosen[dst] = chosen["备菜单位_100g"] * chosen[src]

    detail_cols = [
        "dish_name", "备菜单位_100g", "备菜重量kg", "price_per_100g",
        "预计销售额", "预计利润", "preference_weight",
        "热量供给", "碳水供给", "蛋白质供给", "脂肪供给", "膳食纤维供给",
    ]
    detail = chosen[detail_cols].rename(columns={
        "dish_name": "菜品名称",
        "price_per_100g": "历史均价_元每100g",
        "preference_weight": "消费偏好权重",
    })
    detail.insert(0, "日期", pd.to_datetime(date).strftime("%Y-%m-%d"))
    detail.insert(1, "星期", weekday)
    detail = detail.sort_values(["备菜单位_100g", "预计销售额"], ascending=False).reset_index(drop=True)

    summary = {
        "日期": pd.to_datetime(date).strftime("%Y-%m-%d"),
        "星期": weekday,
        "推荐菜品数": int(len(detail)),
        "总备菜重量kg": float(detail["备菜重量kg"].sum()),
        "预计销售额": float(detail["预计销售额"].sum()),
        "预计利润": float(detail["预计利润"].sum()),
        "热量供给": float(detail["热量供给"].sum()),
        "碳水供给": float(detail["碳水供给"].sum()),
        "蛋白质供给": float(detail["蛋白质供给"].sum()),
        "脂肪供给": float(detail["脂肪供给"].sum()),
        "膳食纤维供给": float(detail["膳食纤维供给"].sum()),
    }
    return detail, summary


def make_latex_tables(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> str:
    summary_show = summary_df[[
        "日期", "星期", "推荐菜品数", "总备菜重量kg", "预计销售额", "预计利润",
        "热量供给", "碳水供给", "蛋白质供给", "脂肪供给", "膳食纤维供给",
    ]].copy()
    for col in summary_show.columns:
        if col not in {"日期", "星期", "推荐菜品数"}:
            summary_show[col] = summary_show[col].map(lambda x: f"{float(x):.2f}")

    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{2025年5月6日至5月12日工作日午餐备菜方案汇总}",
        r"    \label{tab:q3_lunch_summary}",
        r"    \resizebox{\textwidth}{!}{",
        r"    \begin{tabular}{ccccccccccc}",
        r"        \toprule",
        r"        日期 & 星期 & 菜品数 & 总备菜重量/kg & 预计销售额 & 预计利润 & 热量 & 碳水 & 蛋白质 & 脂肪 & 膳食纤维 \\",
        r"        \midrule",
    ]
    for _, row in summary_show.iterrows():
        lines.append(
            f"        {row['日期']} & {row['星期']} & {row['推荐菜品数']} & "
            f"{row['总备菜重量kg']} & {row['预计销售额']} & {row['预计利润']} & "
            f"{row['热量供给']} & {row['碳水供给']} & {row['蛋白质供给']} & "
            f"{row['脂肪供给']} & {row['膳食纤维供给']} \\\\"
        )
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}}",
        r"\end{table}",
        "",
    ])

    # 每天给出前 10 个备菜量最大的菜品，适合正文展示；完整明细见 Excel/CSV。
    top_detail = (
        detail_df.sort_values(["日期", "备菜单位_100g"], ascending=[True, False])
        .groupby("日期")
        .head(10)
        .copy()
    )
    for col in ["备菜重量kg", "预计销售额", "预计利润"]:
        top_detail[col] = top_detail[col].map(lambda x: f"{float(x):.2f}")

    lines.extend([
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{各工作日午餐主要推荐菜品及备菜量}",
        r"    \label{tab:q3_lunch_detail_top}",
        r"    \resizebox{\textwidth}{!}{",
        r"    \begin{tabular}{ccccc}",
        r"        \toprule",
        r"        日期 & 菜品名称 & 备菜单位(100g) & 备菜重量/kg & 预计销售额 \\",
        r"        \midrule",
    ])
    for _, row in top_detail.iterrows():
        lines.append(
            f"        {row['日期']} & {row['菜品名称']} & {int(row['备菜单位_100g'])} & "
            f"{row['备菜重量kg']} & {row['预计销售额']} \\\\"
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
    print("午餐历史订单数：", detail["indent_id"].nunique())
    print("候选菜品数：", len(dishes))
    print("历史人均午餐取餐量约为 100g 单位：", round(avg_units_per_customer, 3))

    all_details: List[pd.DataFrame] = []
    summaries: List[Dict[str, float]] = []
    solver_names = []
    for _, pred_row in pred.iterrows():
        targets = get_day_targets(pred_row, avg_units_per_customer)
        x, y, solver_name = solve_day_milp(dishes, targets)
        solver_names.append(solver_name)
        day_detail, day_summary = summarize_plan(pred_row["日期"], pred_row["星期"], dishes, x, y)
        # 加入预测目标，便于检查供需偏差。
        day_summary.update({
            "预测就餐人数": targets["people"],
            "预测销售总额": targets["sales"],
            "预测热量需求": targets["calories"],
            "预测碳水需求": targets["carbohydrates"],
            "预测蛋白质需求": targets["protein"],
            "预测脂肪需求": targets["fat"],
            "预测膳食纤维需求": targets["fiber"],
            "求解器": solver_name,
        })
        all_details.append(day_detail)
        summaries.append(day_summary)

    detail_df = pd.concat(all_details, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    # 四舍五入输出，保持论文表格可读。
    for df in [detail_df, summary_df]:
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].round(3)

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:
        summary_df.to_excel(writer, sheet_name="每日汇总", index=False)
        detail_df.to_excel(writer, sheet_name="备菜明细", index=False)
        dishes.to_excel(writer, sheet_name="菜品统计", index=False)
    detail_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    latex = make_latex_tables(summary_df, detail_df)
    OUTPUT_LATEX.write_text(latex, encoding="utf-8")

    print("\n问题三午餐备菜方案汇总：")
    show_cols = ["日期", "星期", "推荐菜品数", "总备菜重量kg", "预计销售额", "预计利润", "热量供给", "碳水供给", "蛋白质供给", "脂肪供给", "膳食纤维供给"]
    print(summary_df[show_cols].to_string(index=False))
    print("\n每日备菜量最大的前 8 个菜品：")
    print(
        detail_df.sort_values(["日期", "备菜单位_100g"], ascending=[True, False])
        .groupby("日期")
        .head(8)[["日期", "菜品名称", "备菜单位_100g", "备菜重量kg", "预计销售额", "预计利润"]]
        .to_string(index=False)
    )
    print("\n输出文件：")
    print(" -", OUTPUT_XLSX)
    print(" -", OUTPUT_CSV)
    print(" -", OUTPUT_LATEX)
    print("使用求解器：", sorted(set(solver_names)))


if __name__ == "__main__":
    main()
