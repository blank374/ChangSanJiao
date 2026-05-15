#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题四：自助量贩餐厅固定价位套餐组合优化。

本脚本区别于问题三的“备菜数量优化”：问题四的决策变量为
    x_i in {0, 1}
表示菜品 i 是否进入某一固定价位套餐。脚本包含：
1. 自动读取项目中已有 processed/result 数据，构造菜品基础表；
2. 基准模型：价格约束下的贪心算法；
3. 主模型：NSGA-II 多目标进化算法；
4. 从 Pareto 候选套餐中用综合评分法筛选 10/15/20 元套餐；
5. 输出 CSV、LaTeX 表格、论文说明文字和可视化图。
"""

from __future__ import annotations

import math
import os
import random
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/mpl_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("output/cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


ROOT = Path(".")
RESULT_DIR = Path("results")
FIG_DIR = Path("figures/q4")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.65
PROFIT_RATE = 1.0 - COST_RATE
RANDOM_SEED = 42
TOP_CANDIDATE_LIMIT = 60

PACKAGE_CONFIG = {
    10: {"range": (9.5, 10.5), "relaxed": (9.0, 11.0), "count": (2, 3), "name": "10元经济应急套餐"},
    15: {"range": (14.5, 15.5), "relaxed": (14.0, 16.0), "count": (3, 4), "name": "15元均衡能量套餐"},
    20: {"range": (19.5, 20.5), "relaxed": (19.0, 21.0), "count": (4, 5), "name": "20元丰富营养套餐"},
}

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


@dataclass
class PackageResult:
    target_price: int
    package_name: str
    method: str
    dish_names: List[str]
    dish_count: int
    total_price: float
    calorie: float
    carb: float
    protein: float
    fat: float
    fiber: float
    preference_score: float
    profit: float
    nutrition_score: float
    price_deviation: float
    final_score: float
    relaxed_price: str


def setup_chinese_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def discover_data_files() -> List[Path]:
    patterns = ["*.csv", "*.xlsx", "*.xls"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(ROOT.glob(pattern))
    return sorted(files, key=lambda p: p.name)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = [str(c) for c in columns]
    for key in candidates:
        for col in cols:
            if key.lower() in col.lower():
                return col
    return None


def load_from_merged_order_dish(path: Path) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))
    print(f"读取明细数据文件：{path}")
    print("字段：", list(df.columns))

    if "meal_period" in df.columns:
        lunch = df[df["meal_period"].astype(str).str.lower().eq("lunch")].copy()
        if len(lunch) > 0:
            df = lunch

    numeric_cols = [
        "total_price", "weight", "unit_price", "calories_dish", "carbohydrates_dish",
        "protein_dish", "fat_dish", "fiber_dish",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    needed = ["dish_name", "total_price", "weight", "calories_dish", "carbohydrates_dish", "protein_dish", "fat_dish", "fiber_dish"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少必要字段：{missing}")

    df = df.dropna(subset=["dish_name", "total_price", "weight"])
    df = df[(df["weight"] > 0) & (df["total_price"] >= 0)].copy()
    for col in ["calories_dish", "carbohydrates_dish", "protein_dish", "fat_dish", "fiber_dish"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    total_orders = df["indent_id"].nunique() if "indent_id" in df.columns else max(len(df), 1)
    total_weight = df["weight"].sum()
    agg_count = ("indent_details_id", "count") if "indent_details_id" in df.columns else ("dish_name", "count")
    order_count = ("indent_id", "nunique") if "indent_id" in df.columns else ("dish_name", "count")

    dish = df.groupby(["dish_serial", "dish_name"], as_index=False).agg(
        sales_count=agg_count,
        order_count=order_count,
        total_weight_g=("weight", "sum"),
        total_sales=("total_price", "sum"),
        calorie_total=("calories_dish", "sum"),
        carb_total=("carbohydrates_dish", "sum"),
        protein_total=("protein_dish", "sum"),
        fat_total=("fat_dish", "sum"),
        fiber_total=("fiber_dish", "sum"),
    )
    dish = dish[dish["total_weight_g"] > 0].copy()
    dish["price"] = dish["total_sales"] / (dish["total_weight_g"] / 100.0)
    for src, dst in [
        ("calorie_total", "calorie"),
        ("carb_total", "carb"),
        ("protein_total", "protein"),
        ("fat_total", "fat"),
        ("fiber_total", "fiber"),
    ]:
        dish[dst] = dish[src] / dish["total_weight_g"] * 100.0
    dish["sales_weight_share"] = dish["total_weight_g"] / max(total_weight, 1e-9)
    dish["order_penetration"] = dish["order_count"] / max(total_orders, 1)
    dish["preference"] = 0.6 * dish["sales_weight_share"] + 0.4 * dish["order_penetration"]
    return dish


def load_from_dish_summary(path: Path) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))
    print(f"读取菜品汇总文件：{path}")
    print("字段：", list(df.columns))

    name_col = find_col(df.columns, ["dish_name", "菜品名称", "品名", "名称"])
    serial_col = find_col(df.columns, ["dish_serial", "菜品编号", "编号"])
    weight_col = find_col(df.columns, ["总销售重量g", "销售重量", "重量"])
    sales_col = find_col(df.columns, ["总销售额", "销售额", "金额"])
    order_col = find_col(df.columns, ["出现订单数", "订单出现", "订单数", "销售频次"])
    calorie_col = find_col(df.columns, ["热量总量", "热量", "能量", "kcal"])
    carb_col = find_col(df.columns, ["碳水总量", "碳水化合物", "碳水"])
    protein_col = find_col(df.columns, ["蛋白质总量", "蛋白质", "蛋白"])
    fat_col = find_col(df.columns, ["脂肪总量", "脂肪"])
    fiber_col = find_col(df.columns, ["膳食纤维总量", "膳食纤维", "纤维"])

    required = [name_col, weight_col, sales_col, calorie_col, carb_col, protein_col, fat_col, fiber_col]
    if any(c is None for c in required):
        raise ValueError(f"{path} 不能自动识别构造菜品基础表所需字段。")

    out = pd.DataFrame({
        "dish_serial": df[serial_col] if serial_col else np.arange(len(df)),
        "dish_name": df[name_col],
        "total_weight_g": pd.to_numeric(df[weight_col], errors="coerce"),
        "total_sales": pd.to_numeric(df[sales_col], errors="coerce"),
        "order_count": pd.to_numeric(df[order_col], errors="coerce") if order_col else np.nan,
        "calorie_total": pd.to_numeric(df[calorie_col], errors="coerce"),
        "carb_total": pd.to_numeric(df[carb_col], errors="coerce"),
        "protein_total": pd.to_numeric(df[protein_col], errors="coerce"),
        "fat_total": pd.to_numeric(df[fat_col], errors="coerce"),
        "fiber_total": pd.to_numeric(df[fiber_col], errors="coerce"),
    })
    out = out.dropna(subset=["dish_name", "total_weight_g", "total_sales"])
    out = out[out["total_weight_g"] > 0].copy()
    out["sales_count"] = out["order_count"].fillna(0)
    out["price"] = out["total_sales"] / (out["total_weight_g"] / 100.0)
    for src, dst in [
        ("calorie_total", "calorie"),
        ("carb_total", "carb"),
        ("protein_total", "protein"),
        ("fat_total", "fat"),
        ("fiber_total", "fiber"),
    ]:
        out[dst] = out[src] / out["total_weight_g"] * 100.0
    out["sales_weight_share"] = out["total_weight_g"] / max(out["total_weight_g"].sum(), 1e-9)
    if out["order_count"].notna().any() and out["order_count"].sum() > 0:
        out["order_penetration"] = out["order_count"] / max(out["order_count"].sum(), 1e-9)
        out["preference"] = 0.6 * out["sales_weight_share"] + 0.4 * out["order_penetration"]
    else:
        out["order_penetration"] = np.nan
        out["preference"] = out["sales_weight_share"]
    return out


def build_dish_base() -> Tuple[pd.DataFrame, str]:
    files = discover_data_files()
    print("当前目录数据文件：")
    for f in files:
        print(" -", f)

    preferred = [
        Path("merged_order_dish.csv"),
        Path("dish_summary_with_abc.csv"),
        Path("dish_summary.csv"),
        Path("detail_cleaned.csv"),
    ]
    errors: List[str] = []
    for path in preferred:
        if not path.exists():
            continue
        try:
            if path.name == "merged_order_dish.csv" or path.name == "detail_cleaned.csv":
                dish = load_from_merged_order_dish(path)
            else:
                dish = load_from_dish_summary(path)
            source = str(path)
            break
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    else:
        raise RuntimeError("未能自动构造菜品基础表；尝试失败：\n" + "\n".join(errors))

    dish = dish.replace([np.inf, -np.inf], np.nan)
    numeric = ["price", "calorie", "carb", "protein", "fat", "fiber", "preference", "total_weight_g", "order_count"]
    for col in numeric:
        if col in dish.columns:
            dish[col] = pd.to_numeric(dish[col], errors="coerce")
    dish = dish.dropna(subset=["dish_name", "price", "calorie", "carb", "protein", "fat", "fiber", "preference"])
    dish["preference_raw"] = dish["preference"].astype(float)

    # 原始 preference_i=0.6*销售重量占比+0.4*订单渗透率。由于米饭等基础主食的
    # 渗透率远高于其他菜品，直接求和会让套餐偏好目标被单个主食支配，因此在保留
    # 原始字段的同时，对模型使用的偏好权重做 log 压缩归一化。
    pref_raw = np.maximum(dish["preference_raw"].to_numpy(), 0)
    dish["preference"] = np.log1p(50 * pref_raw) / np.log1p(50 * max(pref_raw.max(), 1e-12))

    # 过滤异常价格与营养记录。价格以 100g 为单位，极端低价或高价不适合作为套餐固定份额。
    before = len(dish)
    dish = dish[
        (dish["price"] >= 0.2) & (dish["price"] <= 12.0)
        & (dish["calorie"] >= 0) & (dish["calorie"] <= 800)
        & (dish["carb"] >= 0) & (dish["carb"] <= 150)
        & (dish["protein"] >= 0) & (dish["protein"] <= 80)
        & (dish["fat"] >= 0) & (dish["fat"] <= 80)
        & (dish["fiber"] >= 0) & (dish["fiber"] <= 30)
    ].copy()
    print(f"异常价格/营养过滤：{before} -> {len(dish)}")

    # 保留原始偏好较高菜品，同时补充少量高蛋白/高纤维菜品，避免套餐被米饭等低价主食主导。
    # 候选池筛选使用未压缩的 preference_raw，模型评分再使用对数压缩后的 preference。
    top_pref = dish.sort_values("preference_raw", ascending=False).head(TOP_CANDIDATE_LIMIT)
    top_protein = dish.sort_values("protein", ascending=False).head(12)
    top_fiber = dish.sort_values("fiber", ascending=False).head(12)
    dish = pd.concat([top_pref, top_protein, top_fiber], ignore_index=True)
    dish = dish.drop_duplicates(subset=["dish_name"]).reset_index(drop=True)

    pref_min, pref_max = dish["preference"].min(), dish["preference"].max()
    dish["preference_norm"] = (dish["preference"] - pref_min) / (pref_max - pref_min + 1e-12)
    dish["profit"] = dish["price"] * PROFIT_RATE
    profit_min, profit_max = dish["profit"].min(), dish["profit"].max()
    dish["profit_norm"] = (dish["profit"] - profit_min) / (profit_max - profit_min + 1e-12)
    dish["single_nutrition"] = [
        single_dish_nutrition_score({"calorie": r.calorie, "carb": r.carb, "protein": r.protein, "fat": r.fat, "fiber": r.fiber})
        for r in dish.itertuples()
    ]
    print(f"候选菜品数量：{len(dish)}")
    print(f"候选菜品价格范围：{dish['price'].min():.2f} - {dish['price'].max():.2f} 元/100g")
    return dish, source


def single_dish_nutrition_score(nutr: Dict[str, float]) -> float:
    """单菜营养得分，不绑定任何套餐价位。

    该分数用于贪心算法的单菜排序和随机补充搜索的抽样权重。它只评价单个菜品
    的宏量营养结构、蛋白质密度、膳食纤维和热量适中性，不使用 10/15/20 元价位。
    """
    calorie = float(nutr.get("calorie", 0))
    carb = float(nutr.get("carb", 0))
    protein = float(nutr.get("protein", 0))
    fat = float(nutr.get("fat", 0))
    fiber = float(nutr.get("fiber", 0))

    macro_energy = 4 * carb + 4 * protein + 9 * fat
    if macro_energy <= 1e-9:
        macro_score = 0.0
    else:
        actual = np.array([4 * carb, 4 * protein, 9 * fat]) / macro_energy
        target = np.array([0.50, 0.20, 0.30])
        macro_score = 1.0 - np.mean(np.abs(actual - target) / target)
        macro_score = float(np.clip(macro_score, 0, 1))

    protein_density = protein / max(calorie, 1e-9) * 100
    protein_score = float(np.clip(protein_density / 5.0, 0, 1))
    fiber_score = float(np.clip(fiber / 3.0, 0, 1))
    calorie_score = float(np.clip(1.0 - abs(calorie - 180.0) / 260.0, 0, 1))
    fat_energy_share = 9 * fat / macro_energy if macro_energy > 1e-9 else 1.0
    fat_penalty = max(0.0, fat_energy_share - 0.45) / 0.55

    score = 0.40 * macro_score + 0.25 * protein_score + 0.20 * fiber_score + 0.15 * calorie_score - 0.12 * fat_penalty
    return float(np.clip(score, 0, 1))


def nutrition_score(nutr: Dict[str, float], target_price: float) -> float:
    """营养均衡得分，范围约为 0-1，越高表示套餐营养结构越合理。

    设计含义：
    1. 用碳水、蛋白质、脂肪供能占比接近 50%/20%/30% 表示宏量营养均衡；
    2. 蛋白质过低扣分，膳食纤维达到目标给奖励；
    3. 热量与套餐价位匹配，避免低价套餐热量过低或高价套餐只是堆叠高脂菜。
    """
    calorie = float(nutr.get("calorie", 0))
    carb = float(nutr.get("carb", 0))
    protein = float(nutr.get("protein", 0))
    fat = float(nutr.get("fat", 0))
    fiber = float(nutr.get("fiber", 0))

    macro_energy = 4 * carb + 4 * protein + 9 * fat
    if macro_energy <= 1e-9:
        macro_score = 0.0
    else:
        actual = np.array([4 * carb, 4 * protein, 9 * fat]) / macro_energy
        target = np.array([0.50, 0.20, 0.30])
        macro_score = 1.0 - np.mean(np.abs(actual - target) / target)
        macro_score = float(np.clip(macro_score, 0, 1))

    protein_target = 0.9 * target_price
    protein_score = float(np.clip(protein / max(protein_target, 1e-9), 0, 1))

    fiber_target = 0.18 * target_price
    fiber_score = float(np.clip(fiber / max(fiber_target, 1e-9), 0, 1))

    calorie_target = 45.0 * target_price
    calorie_score = 1.0 - abs(calorie - calorie_target) / max(calorie_target, 1e-9)
    calorie_score = float(np.clip(calorie_score, 0, 1))

    fat_energy_share = 9 * fat / macro_energy if macro_energy > 1e-9 else 1.0
    fat_penalty = max(0.0, fat_energy_share - 0.42) / 0.58

    score = 0.42 * macro_score + 0.22 * protein_score + 0.18 * fiber_score + 0.18 * calorie_score - 0.15 * fat_penalty
    return float(np.clip(score, 0, 1))


def package_metrics(dish: pd.DataFrame, indices: Iterable[int], target_price: int, method: str, final_score: float = 0.0, relaxed: str = "否") -> PackageResult:
    idx = list(indices)
    sub = dish.iloc[idx]
    totals = {
        "price": float(sub["price"].sum()),
        "calorie": float(sub["calorie"].sum()),
        "carb": float(sub["carb"].sum()),
        "protein": float(sub["protein"].sum()),
        "fat": float(sub["fat"].sum()),
        "fiber": float(sub["fiber"].sum()),
    }
    pref = float(sub["preference"].sum())
    profit = float(sub["profit"].sum())
    nscore = nutrition_score(totals, target_price)
    return PackageResult(
        target_price=target_price,
        package_name=PACKAGE_CONFIG[target_price]["name"],
        method=method,
        dish_names=list(sub["dish_name"].astype(str)),
        dish_count=len(idx),
        total_price=totals["price"],
        calorie=totals["calorie"],
        carb=totals["carb"],
        protein=totals["protein"],
        fat=totals["fat"],
        fiber=totals["fiber"],
        preference_score=pref,
        profit=profit,
        nutrition_score=nscore,
        price_deviation=abs(totals["price"] - target_price),
        final_score=final_score,
        relaxed_price=relaxed,
    )


def result_to_dict(result: PackageResult) -> Dict[str, object]:
    return {
        "价位": result.target_price,
        "套餐名称": result.package_name,
        "方法": result.method,
        "菜品组合": "、".join(result.dish_names),
        "菜品数量": result.dish_count,
        "总价格/元": round(result.total_price, 3),
        "热量/kcal": round(result.calorie, 3),
        "碳水/g": round(result.carb, 3),
        "蛋白质/g": round(result.protein, 3),
        "脂肪/g": round(result.fat, 3),
        "膳食纤维/g": round(result.fiber, 3),
        "偏好得分": round(result.preference_score, 6),
        "预计利润/元": round(result.profit, 3),
        "营养均衡得分": round(result.nutrition_score, 4),
        "价格偏差/元": round(result.price_deviation, 3),
        "综合评分": round(result.final_score, 4),
        "是否放宽价格": result.relaxed_price,
    }


def feasible_price_range(dish: pd.DataFrame, target_price: int) -> Tuple[Tuple[float, float], str]:
    base = PACKAGE_CONFIG[target_price]["range"]
    cnt = count_feasible_combinations(dish, target_price, base)
    if cnt >= 20:
        print(f"{target_price}元套餐严格价格范围可行组合数估计/统计：{cnt}")
        return base, "否"
    relaxed = PACKAGE_CONFIG[target_price]["relaxed"]
    cnt_relaxed = count_feasible_combinations(dish, target_price, relaxed)
    print(f"{target_price}元套餐严格可行组合较少({cnt})，放宽到 ±1 元后组合数：{cnt_relaxed}")
    return relaxed, "是"


def count_feasible_combinations(dish: pd.DataFrame, target_price: int, price_range: Tuple[float, float]) -> int:
    lo, hi = price_range
    min_count, max_count = PACKAGE_CONFIG[target_price]["count"]
    prices = dish["price"].to_numpy()
    n = len(prices)
    # n 较大时只统计前 50 个候选的精确组合数，避免 C(70,5) 过慢；该数用于日志检查。
    check_n = min(n, 50)
    cnt = 0
    for k in range(min_count, max_count + 1):
        for comb in combinations(range(check_n), k):
            p = float(prices[list(comb)].sum())
            if lo <= p <= hi:
                cnt += 1
    return cnt


def score_candidates(candidates: List[PackageResult]) -> List[PackageResult]:
    if not candidates:
        return []
    price_dev = np.array([c.price_deviation for c in candidates])
    pref = np.array([c.preference_score for c in candidates])
    nutr = np.array([c.nutrition_score for c in candidates])
    profit = np.array([c.profit for c in candidates])

    def norm_positive(x: np.ndarray) -> np.ndarray:
        return (x - x.min()) / (x.max() - x.min() + 1e-12)

    def norm_negative(x: np.ndarray) -> np.ndarray:
        return 1.0 - norm_positive(x)

    final = (
        0.35 * norm_negative(price_dev)
        + 0.30 * norm_positive(pref)
        + 0.25 * norm_positive(nutr)
        + 0.10 * norm_positive(profit)
    )
    for c, s in zip(candidates, final):
        c.final_score = float(s)
    return candidates


def assign_absolute_final_scores(dish: pd.DataFrame, results: List[PackageResult]) -> None:
    """为最终 NSGA-II 和贪心结果计算同一尺度的综合评分。

    Pareto 候选内部筛选使用候选集 min-max 归一化；最终方法对比表则需要跨方法可比。
    因此这里采用绝对归一化：
    - 价格得分 = 1 - 价格偏差/允许半径；
    - 偏好得分以上限菜品数量对应的候选偏好最大和为 1；
    - 营养得分直接使用 nutrition_score；
    - 利润得分按套餐价位理论利润 P0*35% 归一化。
    """
    for r in results:
        config = PACKAGE_CONFIG[r.target_price]
        lo, hi = config["range"]
        half_width = max((hi - lo) / 2, 1e-9)
        price_component = float(np.clip(1.0 - r.price_deviation / half_width, 0, 1))

        max_count = config["count"][1]
        pref_upper = float(dish["preference"].sort_values(ascending=False).head(max_count).sum())
        pref_component = float(np.clip(r.preference_score / max(pref_upper, 1e-9), 0, 1))

        profit_component = float(np.clip(r.profit / max(r.target_price * PROFIT_RATE, 1e-9), 0, 1))
        r.final_score = (
            0.35 * price_component
            + 0.30 * pref_component
            + 0.25 * r.nutrition_score
            + 0.10 * profit_component
        )


def run_nsga2(dish: pd.DataFrame, target_price: int, price_range: Tuple[float, float], relaxed: str) -> Tuple[PackageResult, pd.DataFrame]:
    min_count, max_count = PACKAGE_CONFIG[target_price]["count"]
    lo, hi = price_range
    prices = dish["price"].to_numpy()
    pref = dish["preference"].to_numpy()
    profit = dish["profit"].to_numpy()
    nutrient = dish[["calorie", "carb", "protein", "fat", "fiber"]].to_numpy()

    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.operators.crossover.pntx import TwoPointCrossover
        from pymoo.operators.mutation.bitflip import BitflipMutation
        from pymoo.operators.sampling.rnd import BinaryRandomSampling
        from pymoo.optimize import minimize

        class PackageProblem(ElementwiseProblem):
            def __init__(self) -> None:
                super().__init__(
                    n_var=len(dish),
                    n_obj=4,
                    n_ieq_constr=5,
                    xl=0,
                    xu=1,
                    vtype=bool,
                )

            def _evaluate(self, x, out, *args, **kwargs) -> None:
                x = np.asarray(x).astype(bool)
                count = int(x.sum())
                total_price = float(prices[x].sum())
                totals = nutrient[x].sum(axis=0) if count > 0 else np.zeros(5)
                nscore = nutrition_score(
                    {"calorie": totals[0], "carb": totals[1], "protein": totals[2], "fat": totals[3], "fiber": totals[4]},
                    target_price,
                )
                out["F"] = [
                    abs(total_price - target_price),
                    -float(pref[x].sum()),
                    -nscore,
                    -float(profit[x].sum()),
                ]
                out["G"] = [
                    lo - total_price,
                    total_price - hi,
                    min_count - count,
                    count - max_count,
                    0.05 - nscore,
                ]

        algorithm = NSGA2(
            pop_size=180,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(prob=0.90),
            mutation=BitflipMutation(prob=1.0 / max(len(dish), 1)),
            eliminate_duplicates=True,
        )
        res = minimize(
            PackageProblem(),
            algorithm,
            ("n_gen", 260),
            seed=RANDOM_SEED + target_price,
            verbose=False,
        )

        X = np.atleast_2d(res.X) if res.X is not None else np.empty((0, len(dish)))
        candidates = []
        for x in X:
            idx = np.where(np.asarray(x).astype(bool))[0].tolist()
            if not idx:
                continue
            r = package_metrics(dish, idx, target_price, "NSGA-II", relaxed=relaxed)
            if lo <= r.total_price <= hi and min_count <= r.dish_count <= max_count:
                candidates.append(r)

        if len(candidates) < 5:
            candidates.extend(random_feasible_search(dish, target_price, price_range, relaxed, "NSGA-II补充随机搜索", trials=12000))

        candidates = deduplicate_packages(candidates)
        candidates = score_candidates(candidates)
        if not candidates:
            raise RuntimeError("NSGA-II 未产生可行套餐，且随机补充也失败。")
        candidates.sort(key=lambda r: r.final_score, reverse=True)
        pareto_df = pd.DataFrame([result_to_dict(c) for c in candidates])
        print(f"{target_price}元 NSGA-II/Pareto 可行候选套餐数：{len(candidates)}")
        return candidates[0], pareto_df

    except Exception as exc:
        print(f"{target_price}元 NSGA-II 运行失败，启用简化随机搜索兜底：{exc}")
        candidates = random_feasible_search(dish, target_price, price_range, relaxed, "随机搜索兜底", trials=40000)
        candidates = score_candidates(deduplicate_packages(candidates))
        if not candidates:
            raise RuntimeError(f"{target_price}元套餐无法找到可行解。")
        candidates.sort(key=lambda r: r.final_score, reverse=True)
        return candidates[0], pd.DataFrame([result_to_dict(c) for c in candidates])


def random_feasible_search(
    dish: pd.DataFrame,
    target_price: int,
    price_range: Tuple[float, float],
    relaxed: str,
    method: str,
    trials: int = 20000,
) -> List[PackageResult]:
    lo, hi = price_range
    min_count, max_count = PACKAGE_CONFIG[target_price]["count"]
    n = len(dish)
    candidates: List[PackageResult] = []
    prices = dish["price"].to_numpy()
    weights = (
        0.45 * dish["preference_norm"].to_numpy()
        + 0.25 * dish["profit_norm"].to_numpy()
        + 0.30 * dish["single_nutrition"].to_numpy()
    )
    weights = np.maximum(weights, 1e-6)
    weights = weights / weights.sum()
    for _ in range(trials):
        k = random.randint(min_count, max_count)
        idx = np.random.choice(np.arange(n), size=k, replace=False, p=weights).tolist()
        p = float(prices[idx].sum())
        if lo <= p <= hi:
            candidates.append(package_metrics(dish, idx, target_price, method, relaxed=relaxed))
    return candidates


def deduplicate_packages(candidates: List[PackageResult]) -> List[PackageResult]:
    seen = set()
    out: List[PackageResult] = []
    for c in candidates:
        key = tuple(sorted(c.dish_names))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def run_greedy(dish: pd.DataFrame, target_price: int, price_range: Tuple[float, float], relaxed: str) -> PackageResult:
    lo, hi = price_range
    min_count, max_count = PACKAGE_CONFIG[target_price]["count"]
    df = dish.copy()
    df["greedy_score"] = 0.5 * df["preference_norm"] + 0.3 * df["profit_norm"] + 0.2 * df["single_nutrition"]
    ordered = list(df.sort_values("greedy_score", ascending=False).index)

    # 先按分数贪心加入，再用局部替换/枚举修正价格；保证基准模型简单且可行。
    greedy_candidates: List[PackageResult] = []
    selected: List[int] = []
    total_price = 0.0
    for idx in ordered:
        price = float(df.loc[idx, "price"])
        if len(selected) < max_count and total_price + price <= hi:
            selected.append(int(idx))
            total_price += price
        if len(selected) >= min_count and lo <= total_price <= hi:
            greedy_candidates.append(package_metrics(dish, selected, target_price, "贪心算法", relaxed=relaxed))
            break

    if greedy_candidates:
        chosen = greedy_candidates[0]
        chosen.final_score = 0.0
        print(f"{target_price}元贪心模型按顺序首次命中可行套餐。")
        return chosen

    # 若直接贪心未命中价格，则在高分前 45 个菜品内寻找第一个可行组合，仍保持
    # “价格约束下按偏好顺序尝试”的简单基准特征，而不做多目标全局筛选。
    top = ordered[: min(45, len(ordered))]
    for k in range(min_count, max_count + 1):
        for comb in combinations(top, k):
            p = float(dish.loc[list(comb), "price"].sum())
            if lo <= p <= hi:
                greedy_candidates.append(package_metrics(dish, list(comb), target_price, "贪心算法", relaxed=relaxed))
                break
        if greedy_candidates:
            break

    greedy_candidates = deduplicate_packages(greedy_candidates)
    if not greedy_candidates:
        greedy_candidates = random_feasible_search(dish, target_price, price_range, relaxed, "贪心算法", trials=30000)
    if not greedy_candidates:
        raise RuntimeError(f"{target_price}元贪心基准模型未找到可行套餐。")

    chosen = greedy_candidates[0]
    chosen.final_score = 0.0
    print(f"{target_price}元贪心模型可行候选套餐数：{len(greedy_candidates)}")
    return chosen


def enforce_diversity(results: List[PackageResult], pareto_tables: Dict[int, pd.DataFrame]) -> List[PackageResult]:
    """保证三个价位不是完全相同或简单重复。

    本问题价格和数量约束不同，完全相同的概率很低；这里进一步限制高价套餐与低价套餐完全相同，
    若出现则从 Pareto 候选表中选择重合率较低且评分接近的备选方案。
    """
    adjusted: List[PackageResult] = []
    previous_sets: List[set] = []
    for r in results:
        current = set(r.dish_names)
        bad = any(current == prev for prev in previous_sets)
        if bad:
            df = pareto_tables.get(r.target_price, pd.DataFrame())
            for _, row in df.sort_values("综合评分", ascending=False).iterrows():
                names = str(row["菜品组合"]).split("、")
                s = set(names)
                if all(s != prev for prev in previous_sets):
                    r.dish_names = names
                    current = s
                    break
        adjusted.append(r)
        previous_sets.append(current)
    return adjusted


def make_algorithm_comparison() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "方法": "NSGA-II",
            "基本思想": "非支配排序与拥挤距离保持种群多样性",
            "适用场景": "2-4 个目标、约束清晰、需要 Pareto 候选集的问题",
            "优点": "经典稳定、参数较少、解释性强、实现成熟",
            "局限性": "目标很多时选择压力下降，需调节种群规模",
            "本文是否采用": "采用，作为主模型",
        },
        {
            "方法": "NSGA-III",
            "基本思想": "在非支配排序基础上用参考方向引导解分布",
            "适用场景": "many-objective 多目标问题",
            "优点": "高维目标下 Pareto 分布更均匀",
            "局限性": "参考方向设置增加解释与调参成本",
            "本文是否采用": "不作为主模型",
        },
        {
            "方法": "SMS-EMOA",
            "基本思想": "以超体积贡献衡量个体优劣并逐步进化",
            "适用场景": "重视 Pareto 前沿质量和超体积指标的问题",
            "优点": "理论评价指标明确，前沿质量可能较高",
            "局限性": "超体积计算成本高，论文解释复杂",
            "本文是否采用": "作为拓展对比",
        },
        {
            "方法": "AGE-MOEA",
            "基本思想": "自适应估计 Pareto 前沿几何形状指导选择",
            "适用场景": "前沿形状复杂且希望自适应搜索的问题",
            "优点": "对不同前沿形状具有适应性",
            "局限性": "方法较新，建模赛论文解释成本较高",
            "本文是否采用": "作为拓展对比",
        },
    ])


def latex_escape(text: object) -> str:
    s = str(text)
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def df_to_latex_table(df: pd.DataFrame, caption: str, label: str, columns: List[str]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{" + "c" * len(columns) + r"}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df[columns].iterrows():
        lines.append(" & ".join(latex_escape(row[c]) for c in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])
    return "\n".join(lines)


def make_latex_outputs(nsga_df: pd.DataFrame, comparison_df: pd.DataFrame, algo_df: pd.DataFrame) -> str:
    preference_formula = "\n".join([
        r"\begin{equation}",
        r"w_i=0.6s_i+0.4o_i,\quad",
        r"w_i'=\frac{\ln(1+\lambda w_i)}{\ln(1+\lambda \max_i w_i)},\quad \lambda=50,",
        r"\end{equation}",
        r"其中，$s_i$ 表示菜品销售重量占比，$o_i$ 表示订单渗透率，$w_i'$ 为进入优化模型的压缩偏好权重。"
    ])
    algo_latex = df_to_latex_table(
        algo_df,
        "多目标优化算法比较",
        "tab:q4_algorithm_comparison",
        ["方法", "基本思想", "适用场景", "优点", "局限性", "本文是否采用"],
    )
    result_cols = ["价位", "菜品组合", "总价格/元", "热量/kcal", "碳水/g", "蛋白质/g", "脂肪/g", "膳食纤维/g", "预计利润/元"]
    result_latex = df_to_latex_table(nsga_df, "不同价位套餐优化结果", "tab:q4_package_result", result_cols)
    compare_cols = ["价位", "方法", "总价格/元", "菜品数量", "偏好得分", "营养均衡得分", "预计利润/元", "综合评分"]
    compare_latex = df_to_latex_table(comparison_df, "NSGA-II 与贪心算法结果对比", "tab:q4_nsga2_greedy_compare", compare_cols)
    return "\n\n".join([preference_formula, algo_latex, result_latex, compare_latex])


def plot_outputs(nsga_df: pd.DataFrame, comparison_df: pd.DataFrame, pareto_tables: Dict[int, pd.DataFrame]) -> None:
    setup_chinese_style()
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.edgecolor"] = "#404040"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#D9D9D9"
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12

    # 1. 目标价与实际价配对点线图。
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = np.arange(len(nsga_df))
    target_x = x - 0.06
    actual_x = x + 0.06
    target_values = nsga_df["价位"].to_numpy(dtype=float)
    actual_values = nsga_df["总价格/元"].to_numpy(dtype=float)
    for i, (target, actual) in enumerate(zip(target_values, actual_values)):
        ax.plot([target_x[i], actual_x[i]], [target, actual], color="#B7BBC2", linewidth=1.2, zorder=1)
    ax.scatter(target_x, target_values, facecolors="white", edgecolors="#7A7A7A", linewidths=1.6, s=85, label="目标价位", zorder=3)
    ax.scatter(actual_x, actual_values, color="#315E8A", s=90, label="实际价格", zorder=4)
    for i, actual in enumerate(actual_values):
        ax.text(actual_x[i], actual + 0.23, f"{actual:.3f}", ha="center", va="bottom", fontsize=11, color="#315E8A")
    ax.set_xticks(x)
    ax.set_xticklabels(nsga_df["价位"].astype(str) + "元套餐")
    ax.set_ylim(8.5, 21.2)
    ax.set_title("不同价位套餐目标价与实际价对比")
    ax.set_xlabel("套餐价位")
    ax.set_ylabel("价格/元")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q4_package_price_compare.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. 营养热力图：颜色为同一营养指标列归一化水平，数字为原始营养素数值。
    nutrients = ["热量/kcal", "碳水/g", "蛋白质/g", "脂肪/g", "膳食纤维/g"]
    nutrient_values = nsga_df[nutrients].to_numpy(dtype=float)
    nutrient_norm = nutrient_values / np.maximum(nutrient_values.max(axis=0), 1e-9)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    im = ax.imshow(nutrient_norm, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(nutrients)))
    ax.set_xticklabels(["热量", "碳水", "蛋白质", "脂肪", "膳食纤维"])
    ax.set_yticks(np.arange(len(nsga_df)))
    ax.set_yticklabels(nsga_df["价位"].astype(str) + "元套餐")
    for i in range(nutrient_values.shape[0]):
        for j in range(nutrient_values.shape[1]):
            val = nutrient_values[i, j]
            text = f"{val:.0f}" if j == 0 else f"{val:.1f}"
            text_color = "white" if nutrient_norm[i, j] >= 0.68 else "#111111"
            ax.text(j, i, text, ha="center", va="center", fontsize=12, color=text_color)
    ax.set_title("不同价位套餐营养结构热力图")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("同指标归一化水平")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q4_package_nutrition_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. NSGA-II 与贪心算法综合评分配对点图。
    pivot = comparison_df.pivot(index="价位", columns="方法", values="综合评分").sort_index()
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    y = np.arange(len(pivot))
    for i, target in enumerate(pivot.index):
        ax.plot([pivot.loc[target, "贪心算法"], pivot.loc[target, "NSGA-II"]], [i, i], color="#B7BBC2", linewidth=2.0, zorder=1)
    ax.scatter(pivot["贪心算法"], y, color="#C57A35", s=88, label="贪心算法", zorder=3)
    ax.scatter(pivot["NSGA-II"], y, color="#315E8A", s=88, label="NSGA-II", zorder=3)
    for i, target in enumerate(pivot.index):
        ax.text(pivot.loc[target, "贪心算法"] - 0.004, i + 0.16, f"{pivot.loc[target, '贪心算法']:.4f}", ha="right", fontsize=11, color="#8B5420")
        ax.text(pivot.loc[target, "NSGA-II"] + 0.004, i + 0.16, f"{pivot.loc[target, 'NSGA-II']:.4f}", ha="left", fontsize=11, color="#315E8A")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{v}元套餐" for v in pivot.index])
    ax.set_xlim(min(pivot.min()) - 0.02, max(pivot.max()) + 0.03)
    ax.set_title("不同价位下 NSGA-II 与贪心算法综合评分对比")
    ax.set_xlabel("综合评分")
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q4_nsga2_greedy_paired_dot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for target, df in pareto_tables.items():
        tmp = df.copy()
        tmp["目标价位"] = target
        rows.append(tmp)
    if rows:
        p = pd.concat(rows, ignore_index=True)
        colors = {10: "#AFC4D8", 15: "#B6D1C1", 20: "#D7C1D8"}
        x_min = min(0.0, float(p["价格偏差/元"].min()) - 0.01)
        x_max = float(p["价格偏差/元"].max()) + 0.02
        y_min = float(p["营养均衡得分"].min()) - 0.03
        y_max = min(1.02, float(p["营养均衡得分"].max()) + 0.03)
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8), sharex=True, sharey=True)
        for ax, target in zip(axes, [10, 15, 20]):
            grp = p[p["目标价位"] == target]
            ax.scatter(grp["价格偏差/元"], grp["营养均衡得分"], s=24, alpha=0.58, color=colors[target], edgecolors="none")
            chosen = nsga_df[nsga_df["价位"] == target].iloc[0]
            ax.scatter(
                chosen["价格偏差/元"],
                chosen["营养均衡得分"],
                s=180,
                marker="*",
                color="#B83A3A",
                edgecolors="black",
                linewidths=0.9,
                zorder=4,
                label="最终方案",
            )
            ax.annotate(
                f"最终方案\n{chosen['综合评分']:.4f}",
                xy=(chosen["价格偏差/元"], chosen["营养均衡得分"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
                color="#8F2020",
            )
            ax.set_title(f"{target}元套餐")
            ax.set_xlabel("价格偏差/元")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(linestyle="--", linewidth=0.8, color="#E2E2E2", alpha=0.9)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("营养均衡得分")
        axes[-1].legend(frameon=False, loc="lower right")
        fig.suptitle("不同价位 Pareto 候选解分布及最终方案", y=1.02, fontweight="bold", fontsize=17)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "q4_pareto_price_nutrition_facet.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def make_model_description(nsga_df: pd.DataFrame, comparison_df: pd.DataFrame, data_source: str) -> str:
    def row_text(target: int) -> str:
        row = nsga_df[nsga_df["价位"] == target].iloc[0]
        return (
            f"{target}元套餐由{row['菜品组合']}组成，总价格为{row['总价格/元']:.2f}元，"
            f"营养均衡得分为{row['营养均衡得分']:.4f}，预计利润为{row['预计利润/元']:.2f}元。"
        )

    better_lines = []
    for target in [10, 15, 20]:
        sub = comparison_df[comparison_df["价位"] == target]
        n = sub[sub["方法"] == "NSGA-II"].iloc[0]
        g = sub[sub["方法"] == "贪心算法"].iloc[0]
        better_lines.append(
            f"{target}元价位下，NSGA-II 的营养均衡得分为{n['营养均衡得分']:.4f}，"
            f"贪心算法为{g['营养均衡得分']:.4f}；综合评分分别为{n['综合评分']:.4f}和{g['综合评分']:.4f}。"
        )

    return f"""问题四模型说明

一、问题分析
问题四属于价格约束下的套餐组合优化问题。与问题三不同，问题三以某日午餐备菜量为对象，决策变量是每个菜品备多少个 100g 单位；问题四以固定价位套餐为对象，决策变量为 x_i∈{{0,1}}，即第 i 个菜品是否被选入套餐。因此，问题四不是连续备菜数量规划，而是带价格、菜品数量、偏好和营养约束的 0-1 组合优化问题。

二、数据基础
本问优先读取项目中已有的 processed/result 数据。本次脚本实际使用的数据源为 {data_source}，自动构造了菜品名称、每 100g 价格、热量、碳水、蛋白质、脂肪、膳食纤维、历史销售重量、订单出现频率、偏好权重和估计利润等字段。其中，在缺少真实成本数据时沿用问题三设定：成本=销售价格×65%，单位利润=销售价格×35%。原始消费偏好权重定义为 w_i=0.6×销售重量占比+0.4×订单渗透率；若缺少订单渗透率，则退化为销售重量占比。由于米饭等基础主食历史渗透率明显高于其他菜品，直接使用原始权重会使偏好目标被少数高频菜品支配，故本文对进入优化模型的偏好权重进行对数压缩：
w_i'=ln(1+λw_i)/ln(1+λmax_i w_i)，其中 λ=50。

三、方法比较与模型选择
套餐设计同时涉及价格贴近目标、消费偏好较高、营养结构均衡和利润合理四类目标，单目标贪心方法容易偏向高销量或高利润菜品而忽略营养搭配。因此，本文先比较 NSGA-II、NSGA-III、SMS-EMOA 和 AGE-MOEA 等多目标优化方法。NSGA-II 采用非支配排序和拥挤距离机制，参数较少、解释性强，适合本文 4 个目标的套餐组合优化；NSGA-III 更适合目标数量更多的 many-objective 问题；SMS-EMOA 依赖超体积贡献，计算和解释成本较高；AGE-MOEA 能自适应估计 Pareto 前沿几何形状，但方法较新，在数学建模论文中的解释成本较高。因此本文选择 NSGA-II 作为主模型，贪心算法作为基准模型。

四、模型建立
对每个目标价位 P0∈{{10,15,20}}，设置决策变量 x_i∈{{0,1}}。套餐总价格为 P_set=Σprice_i x_i，总营养为各营养素按入选菜品求和，偏好得分为 preference_score=Σpreference_i x_i，预计利润为 profit=Σprofit_i x_i。NSGA-II 的四个目标为：最小化价格偏差 f1=|P_set-P0|，最大化消费偏好 f2=-preference_score，最大化营养均衡 f3=-nutrition_score，最大化预计利润 f4=-profit。约束包括：10 元套餐价格在[9.5,10.5]，15 元套餐在[14.5,15.5]，20 元套餐在[19.5,20.5]；若可行组合过少则自动放宽到 ±1 元；菜品数量分别控制为 2-3、3-4、4-5 个；同时通过营养均衡得分惩罚单一营养结构。

营养均衡得分由宏量营养供能结构、蛋白质充足程度、膳食纤维水平和热量-价位匹配程度共同构成。宏量营养结构以碳水、蛋白质、脂肪供能占比接近 50%、20%、30% 为目标；蛋白质过低、脂肪供能占比过高会降低得分，膳食纤维较高则提高得分。

五、求解流程
首先自动读取项目数据并构造候选菜品基础表；其次对异常价格和异常营养值进行过滤，并保留历史偏好较高及高蛋白、高纤维菜品作为候选集合；然后分别对 10 元、15 元和 20 元价位运行 NSGA-II，得到 Pareto 候选套餐；再采用综合评分 final_score=0.35×价格贴近度+0.30×归一化偏好得分+0.25×营养均衡得分+0.10×归一化利润，从候选集中筛选最终推荐套餐；最后与价格约束下的贪心算法进行对比，并输出表格和图形。

六、套餐结果分析
{row_text(10)}
{row_text(15)}
{row_text(20)}
三个价位套餐在价格、菜品数量和菜品组合上形成了层次差异。高价套餐并非简单复制低价套餐，而是在满足更高预算的同时增加或替换菜品，以提高蛋白质、膳食纤维和整体营养结构。

七、模型评价
与贪心算法相比，NSGA-II 能在价格、偏好、营养和利润之间进行同时搜索，避免只选择高偏好或高利润菜品导致营养结构单一。对比结果显示：
{chr(10).join(better_lines)}
因此，NSGA-II 主模型在营养均衡和综合评分方面更适合作为套餐服务的最终设计依据。
"""


def main() -> None:
    dish, source = build_dish_base()

    nsga_results: List[PackageResult] = []
    greedy_results: List[PackageResult] = []
    pareto_tables: Dict[int, pd.DataFrame] = {}
    price_ranges: Dict[int, Tuple[Tuple[float, float], str]] = {}

    for target in [10, 15, 20]:
        price_range, relaxed = feasible_price_range(dish, target)
        price_ranges[target] = (price_range, relaxed)
        nsga_result, pareto_df = run_nsga2(dish, target, price_range, relaxed)
        greedy_result = run_greedy(dish, target, price_range, relaxed)
        nsga_results.append(nsga_result)
        greedy_results.append(greedy_result)
        pareto_tables[target] = pareto_df

    nsga_results = enforce_diversity(nsga_results, pareto_tables)
    assign_absolute_final_scores(dish, nsga_results + greedy_results)
    nsga_df = pd.DataFrame([result_to_dict(r) for r in nsga_results])
    greedy_df = pd.DataFrame([result_to_dict(r) for r in greedy_results])
    comparison_df = pd.concat([nsga_df, greedy_df], ignore_index=True)
    comparison_df = comparison_df[[
        "价位", "方法", "总价格/元", "菜品数量", "偏好得分", "营养均衡得分", "预计利润/元", "综合评分"
    ]].sort_values(["价位", "方法"]).reset_index(drop=True)
    algo_df = make_algorithm_comparison()

    nsga_df.to_csv(RESULT_DIR / "q4_nsga2_packages.csv", index=False, encoding="utf-8-sig")
    greedy_df.to_csv(RESULT_DIR / "q4_greedy_packages.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(RESULT_DIR / "q4_method_comparison.csv", index=False, encoding="utf-8-sig")
    algo_df.to_csv(RESULT_DIR / "q4_algorithm_comparison_table.txt", index=False, sep="\t", encoding="utf-8")

    for target, df in pareto_tables.items():
        df.to_csv(RESULT_DIR / f"q4_pareto_candidates_{target}.csv", index=False, encoding="utf-8-sig")

    latex = make_latex_outputs(nsga_df, comparison_df, algo_df)
    (RESULT_DIR / "q4_latex_tables.txt").write_text(latex, encoding="utf-8")
    description = make_model_description(nsga_df, comparison_df, source)
    (RESULT_DIR / "q4_model_description.txt").write_text(description, encoding="utf-8")

    plot_outputs(nsga_df, comparison_df, pareto_tables)

    print("\nNSGA-II 推荐套餐：")
    print(nsga_df.to_string(index=False))
    print("\n贪心算法基准套餐：")
    print(greedy_df.to_string(index=False))
    print("\nNSGA-II 与贪心算法对比：")
    print(comparison_df.to_string(index=False))
    print("\n输出文件：")
    for p in [
        RESULT_DIR / "q4_nsga2_packages.csv",
        RESULT_DIR / "q4_greedy_packages.csv",
        RESULT_DIR / "q4_method_comparison.csv",
        RESULT_DIR / "q4_algorithm_comparison_table.txt",
        RESULT_DIR / "q4_latex_tables.txt",
        RESULT_DIR / "q4_model_description.txt",
        FIG_DIR / "q4_package_price_compare.png",
        FIG_DIR / "q4_package_nutrition_heatmap.png",
        FIG_DIR / "q4_nsga2_greedy_paired_dot.png",
        FIG_DIR / "q4_pareto_price_nutrition_facet.png",
    ]:
        print(" -", p)


if __name__ == "__main__":
    main()
