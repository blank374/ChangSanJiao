#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题三：基于 NSGA-II 的午餐备菜多目标优化。

与 MILP 版不同，本脚本将备菜方案视为约束多目标优化问题：
    f1: 最小化 -利润，即最大化利润
    f2: 最小化营养偏差
    f3: 最小化 -偏好得分，即最大化顾客偏好
    f4: 最小化过量/不足备菜偏差

约束：
    1. 总备菜量接近预测就餐人数对应需求；
    2. 菜品种类数不低于下限且不高于上限；
    3. 单菜品备菜量不超过上限；
    4. 营养供给偏差通过目标函数惩罚。

输出：
    problem3_nsga2_lunch_menu_plan.xlsx
    problem3_nsga2_lunch_menu_plan.csv
    problem3_nsga2_lunch_latex_tables.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize


warnings.filterwarnings("ignore")

DATA_PATH = Path("merged_order_dish.csv")
PRED_PATH = Path("problem2_prediction_results.xlsx")
OUTPUT_XLSX = Path("problem3_nsga2_lunch_menu_plan.xlsx")
OUTPUT_CSV = Path("problem3_nsga2_lunch_menu_plan.csv")
OUTPUT_LATEX = Path("problem3_nsga2_lunch_latex_tables.txt")

TARGET_DATES = pd.to_datetime([
    "2025-05-06",
    "2025-05-07",
    "2025-05-08",
    "2025-05-09",
    "2025-05-12",
])

COST_RATE = 0.65
TOP_DISH_LIMIT = 45
MIN_DISH_TYPES = 18
MAX_DISH_TYPES = 32
MIN_UNITS_PER_DISH = 2
MAX_SHARE_PER_DISH = 0.16
TOTAL_UNITS_TOL = 0.12
RANDOM_SEED = 42


def read_prediction() -> pd.DataFrame:
    pred = pd.read_excel(PRED_PATH)
    pred.columns = [str(c).strip() for c in pred.columns]
    pred["日期"] = pd.to_datetime(pred["日期"])
    pred = pred[pred["日期"].isin(TARGET_DATES)].copy()
    if pred.empty:
        raise ValueError("问题二预测结果中未找到目标日期。")
    return pred


def load_lunch_detail() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df["meal_period"].astype(str).str.lower().eq("lunch")].copy()
    for col in [
        "total_price", "weight", "unit_price", "calories_dish", "carbohydrates_dish",
        "protein_dish", "fat_dish", "fiber_dish",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["dish_name", "total_price", "weight", "unit_price"])
    df = df[(df["weight"] > 0) & (df["unit_price"] > 0)].copy()
    return df


def build_dish_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    total_orders = df["indent_id"].nunique()
    total_weight = df["weight"].sum()
    avg_units_per_customer = df.groupby("indent_id")["weight"].sum().mean() / 100.0

    grouped = df.groupby(["dish_serial", "dish_name"], as_index=False).agg(
        sales_count=("indent_details_id", "count") if "indent_details_id" in df.columns else ("indent_id", "count"),
        order_count=("indent_id", "nunique"),
        total_weight_g=("weight", "sum"),
        total_sales=("total_price", "sum"),
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
    grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna()
    grouped = grouped[grouped["price_per_100g"] > 0].copy()
    grouped = grouped.sort_values("preference_weight", ascending=False).head(TOP_DISH_LIMIT).reset_index(drop=True)

    pref_min = grouped["preference_weight"].min()
    pref_max = grouped["preference_weight"].max()
    grouped["preference_norm"] = (grouped["preference_weight"] - pref_min) / (pref_max - pref_min + 1e-12)
    grouped["profit_per_100g"] = grouped["price_per_100g"] * (1.0 - COST_RATE)
    return grouped, float(avg_units_per_customer)


def get_targets(row: pd.Series, avg_units_per_customer: float) -> Dict[str, float]:
    people = float(row["预测就餐人数"])
    return {
        "people": people,
        "sales": float(row["预测销售总额"]),
        "calories": float(row["预测热量需求"]),
        "carbohydrates": float(row["预测碳水需求"]),
        "protein": float(row["预测蛋白质需求"]),
        "fat": float(row["预测脂肪需求"]),
        "fiber": float(row["预测膳食纤维需求"]),
        "total_units": people * avg_units_per_customer,
    }


class LunchMenuProblem(ElementwiseProblem):
    def __init__(self, dishes: pd.DataFrame, targets: Dict[str, float]):
        self.dishes = dishes.reset_index(drop=True)
        self.targets = targets
        self.n = len(dishes)
        self.max_units = max(MIN_UNITS_PER_DISH + 1, int(np.ceil(targets["total_units"] * MAX_SHARE_PER_DISH)))
        self.price = dishes["price_per_100g"].to_numpy()
        self.profit = dishes["profit_per_100g"].to_numpy()
        self.pref = dishes["preference_norm"].to_numpy()
        self.nutrients = np.vstack([
            dishes["calories_100g"].to_numpy(),
            dishes["carbohydrates_100g"].to_numpy(),
            dishes["protein_100g"].to_numpy(),
            dishes["fat_100g"].to_numpy(),
            dishes["fiber_100g"].to_numpy(),
        ]).T
        self.nutrient_targets = np.array([
            targets["calories"],
            targets["carbohydrates"],
            targets["protein"],
            targets["fat"],
            targets["fiber"],
        ])
        super().__init__(
            n_var=self.n,
            n_obj=4,
            n_ieq_constr=5,
            xl=np.zeros(self.n, dtype=int),
            xu=np.repeat(self.max_units, self.n).astype(int),
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.rint(x).astype(int)
        x[x < MIN_UNITS_PER_DISH] = 0
        total_units = x.sum()
        selected = int(np.sum(x > 0))
        profit = float(np.dot(x, self.profit))
        pref_score = float(np.dot(x, self.pref)) / max(total_units, 1)
        nutrient_supply = x @ self.nutrients
        nutrient_deviation = float(np.mean(np.abs(nutrient_supply - self.nutrient_targets) / (self.nutrient_targets + 1e-9)))
        units_deviation = abs(total_units - self.targets["total_units"]) / (self.targets["total_units"] + 1e-9)

        # 四个目标均为最小化。
        out["F"] = [
            -profit / 1000.0,
            nutrient_deviation,
            -pref_score,
            units_deviation,
        ]

        # 约束 g <= 0。
        out["G"] = [
            abs(total_units - self.targets["total_units"]) - self.targets["total_units"] * TOTAL_UNITS_TOL,
            MIN_DISH_TYPES - selected,
            selected - MAX_DISH_TYPES,
            np.max(x) - self.max_units,
            -total_units,
        ]


def repair_solution(x: np.ndarray, targets: Dict[str, float], max_units: int) -> np.ndarray:
    x = np.rint(x).astype(int)
    x[x < MIN_UNITS_PER_DISH] = 0
    if np.sum(x > 0) < MIN_DISH_TYPES:
        zero_idx = np.where(x == 0)[0]
        for i in zero_idx[: MIN_DISH_TYPES - np.sum(x > 0)]:
            x[i] = MIN_UNITS_PER_DISH
    x = np.clip(x, 0, max_units)
    target_units = targets["total_units"]
    # 简单调节总量到允许区间。
    while x.sum() < target_units * (1 - TOTAL_UNITS_TOL):
        i = int(np.argmin(x + (x == 0) * 9999))
        x[i] = min(x[i] + 1, max_units)
    while x.sum() > target_units * (1 + TOTAL_UNITS_TOL):
        candidates = np.where(x > MIN_UNITS_PER_DISH)[0]
        if len(candidates) == 0:
            break
        i = int(candidates[np.argmax(x[candidates])])
        x[i] -= 1
    return x


def choose_compromise(problem: LunchMenuProblem, X: np.ndarray, F: np.ndarray, targets: Dict[str, float]) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(1, -1)
    rows = []
    for x in X:
        x = repair_solution(x, targets, problem.max_units)
        total_units = x.sum()
        selected = np.sum(x > 0)
        profit = np.dot(x, problem.profit)
        pref_score = np.dot(x, problem.pref) / max(total_units, 1)
        nutrient_supply = x @ problem.nutrients
        nutrient_deviation = np.mean(np.abs(nutrient_supply - problem.nutrient_targets) / (problem.nutrient_targets + 1e-9))
        units_deviation = abs(total_units - targets["total_units"]) / (targets["total_units"] + 1e-9)
        diversity_score = selected / MAX_DISH_TYPES
        rows.append([profit, pref_score, diversity_score, -nutrient_deviation, -units_deviation])
    score_matrix = np.asarray(rows, dtype=float)
    mins = score_matrix.min(axis=0)
    maxs = score_matrix.max(axis=0)
    norm = (score_matrix - mins) / (maxs - mins + 1e-12)
    weights = np.array([0.30, 0.25, 0.15, 0.20, 0.10])
    scores = norm @ weights
    return repair_solution(X[int(np.argmax(scores))], targets, problem.max_units)


def solve_day_nsga2(dishes: pd.DataFrame, targets: Dict[str, float], seed: int) -> Tuple[np.ndarray, int]:
    problem = LunchMenuProblem(dishes, targets)
    algorithm = NSGA2(
        pop_size=120,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=int, repair=None),
        mutation=PM(prob=0.12, eta=20, vtype=int, repair=None),
        eliminate_duplicates=True,
    )
    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", 180),
        seed=seed,
        verbose=False,
    )
    X = res.X
    F = res.F
    if X is None:
        raise RuntimeError("NSGA-II 未获得可行解")
    x = choose_compromise(problem, X, F, targets)
    pareto_size = len(X) if getattr(X, "ndim", 1) > 1 else 1
    return x, pareto_size


def summarize_plan(date, weekday, dishes: pd.DataFrame, x: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    chosen = dishes.copy()
    chosen["备菜单位_100g"] = np.rint(x).astype(int)
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


def make_latex(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{基于NSGA-II的午餐备菜方案汇总}",
        r"    \label{tab:q3_nsga2_lunch_summary}",
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
        "",
    ])
    top_detail = detail_df.sort_values(["日期", "备菜单位_100g"], ascending=[True, False]).groupby("日期").head(8)
    lines.extend([
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{NSGA-II折中解下主要推荐菜品及备菜量}",
        r"    \label{tab:q3_nsga2_lunch_detail}",
        r"    \resizebox{\textwidth}{!}{",
        r"    \begin{tabular}{ccccc}",
        r"        \toprule",
        r"        日期 & 菜品名称 & 备菜单位(100g) & 备菜重量/kg & 预计利润 \\",
        r"        \midrule",
    ])
    for _, row in top_detail.iterrows():
        lines.append(
            f"        {row['日期']} & {row['菜品名称']} & {int(row['备菜单位_100g'])} & "
            f"{row['备菜重量kg']:.2f} & {row['预计利润']:.2f} \\\\"
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
    pareto_sizes = []
    for idx, (_, pred_row) in enumerate(pred.iterrows()):
        targets = get_targets(pred_row, avg_units_per_customer)
        x, pareto_size = solve_day_nsga2(dishes, targets, RANDOM_SEED + idx)
        pareto_sizes.append(pareto_size)
        day_detail, day_summary = summarize_plan(pred_row["日期"], pred_row["星期"], dishes, x)
        day_summary.update({
            "预测就餐人数": targets["people"],
            "预测销售总额": targets["sales"],
            "预测热量需求": targets["calories"],
            "预测碳水需求": targets["carbohydrates"],
            "预测蛋白质需求": targets["protein"],
            "预测脂肪需求": targets["fat"],
            "预测膳食纤维需求": targets["fiber"],
            "Pareto解数量": pareto_size,
            "算法": "NSGA-II",
        })
        all_details.append(day_detail)
        summaries.append(day_summary)
        print(f"[OK] {day_summary['日期']} Pareto解数量：{pareto_size}")

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

    print("\nNSGA-II 午餐备菜方案汇总：")
    show_cols = ["日期", "星期", "推荐菜品数", "总备菜重量kg", "预计销售额", "预计利润", "热量供给", "碳水供给", "蛋白质供给", "脂肪供给", "膳食纤维供给", "Pareto解数量"]
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


if __name__ == "__main__":
    main()
