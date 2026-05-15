#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题三补充实验：将问题一的销售结构、ABC 分类和与米饭相关的关联规则
引入午餐备菜优化模型。

本脚本不改动原有问题三脚本，只在其基础上补充：
1. 问题一结果整理；
2. 关联强度 g_i 与修正偏好权重 w_i'；
3. ABC 分类驱动的单菜品备菜上下限；
4. 新方案、对比表、图表和论文说明文本输出。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import ast
import warnings
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog

import problem3_lunch_optimization as base_q3


warnings.filterwarnings("ignore")

# -----------------------------
# 路径与参数
# -----------------------------
ROOT = Path(".")
DATA_PATH = ROOT / "merged_order_dish.csv"
BASKET_PATH = ROOT / "basket.csv"
DISH_SUMMARY_PATH = ROOT / "dish_summary.csv"
DISH_SUMMARY_ABC_PATH = ROOT / "dish_summary_with_abc.csv"
ASSOCIATION_RULES_PATH = ROOT / "association_rules.csv"

PREFERENCE_OUTPUT = ROOT / "q3_preference_weight_with_association.csv"
PLAN_OUTPUT = ROOT / "q3_lunch_plan_with_association.csv"
SUMMARY_OUTPUT = ROOT / "q3_lunch_summary_with_association.csv"
COMPARISON_OUTPUT = ROOT / "q3_before_after_comparison.csv"
NOTES_OUTPUT = ROOT / "q3_association_experiment_notes.md"

FIG_DIR = ROOT / "第三题" / "figures"
FIG_PREF = FIG_DIR / "q3_preference_weight_compare.png"
FIG_ABC = FIG_DIR / "q3_abc_selected_distribution.png"
FIG_ASSOC = FIG_DIR / "q3_association_strength_top10.png"
FIG_COMPARE = FIG_DIR / "q3_before_after_comparison.png"

OLD_PLAN_CANDIDATES = [
    ROOT / "第三题" / "problem3_lunch_menu_plan.csv",
    ROOT / "problem3_lunch_menu_plan.csv",
]

GAMMA = 0.15
RICE_KEYWORDS = ("米饭",)
TOP_DISH_LIMIT = base_q3.TOP_DISH_LIMIT
MIN_RULE_SUPPORT = 0.01
MIN_RULE_CONFIDENCE = 0.60
MIN_RULE_LIFT = 1.00

# 字段映射集中管理，便于兼容现有结果文件的中文列名。
SUMMARY_RENAME = {
    "总销售重量g": "total_weight",
    "总销售额": "total_amount",
    "出现订单数": "order_count",
    "重量占比": "weight_share",
}


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return pd.Series(np.zeros(len(values)), index=series.index, dtype=float)
    return (values - lo) / (hi - lo)


def read_lunch_detail() -> pd.DataFrame:
    detail = base_q3.load_lunch_detail().copy()
    detail["date"] = pd.to_datetime(detail["date"])
    return detail


def read_prediction_for_experiment() -> pd.DataFrame:
    """优先读取问题二预测结果；缺少 openpyxl 时直接解析 xlsx 的首张工作表。"""
    candidates = sorted(
        Path(".").rglob("problem2_prediction_results.xlsx"),
        key=lambda p: ("归档" in str(p), len(str(p))),
    )
    if not candidates:
        print("未找到问题二预测结果文件，改用原问题三脚本中的手工预测接口。")
        return base_q3.read_prediction()

    path = candidates[0]
    try:
        pred = pd.read_excel(path)
        print(f"已读取问题二预测结果：{path}")
    except ImportError:
        ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        with ZipFile(path) as zf:
            root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in root.findall(".//a:sheetData/a:row", ns):
            values = []
            for cell in row.findall("a:c", ns):
                inline = cell.find("a:is/a:t", ns)
                numeric = cell.find("a:v", ns)
                values.append(inline.text if inline is not None else (numeric.text if numeric is not None else None))
            rows.append(values)
        pred = pd.DataFrame(rows[1:], columns=rows[0])
        for col in pred.columns:
            if col not in {"日期", "星期"}:
                pred[col] = pd.to_numeric(pred[col], errors="coerce")
        print(f"已通过轻量级 xlsx 解析读取问题二预测结果：{path}")

    pred["日期"] = pd.to_datetime(pred["日期"])
    pred = pred[pred["日期"].isin(base_q3.TARGET_DATES)].copy()
    if pred.empty:
        raise ValueError("问题二预测结果中未找到 2025-05-06 至 2025-05-12 的普通工作日。")
    return pred


def rebuild_dish_summary(detail: pd.DataFrame) -> pd.DataFrame:
    total_orders = max(detail["indent_id"].nunique(), 1)
    summary = detail.groupby(["dish_serial", "dish_name"], as_index=False).agg(
        total_weight=("weight", "sum"),
        total_amount=("total_price", "sum"),
        order_count=("indent_id", "nunique"),
    )
    summary["weight_share"] = summary["total_weight"] / summary["total_weight"].sum()
    summary["order_penetration"] = summary["order_count"] / total_orders
    return summary


def load_or_build_dish_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if DISH_SUMMARY_PATH.exists():
        summary = pd.read_csv(DISH_SUMMARY_PATH)
        summary = summary.rename(columns=SUMMARY_RENAME)
        needed = {"dish_serial", "dish_name", "total_weight", "total_amount", "order_count", "weight_share"}
        if needed.issubset(summary.columns):
            total_orders = max(detail["indent_id"].nunique(), 1)
            summary["order_penetration"] = pd.to_numeric(summary["order_count"], errors="coerce") / total_orders
            return summary[list(needed) + ["order_penetration"]].copy()
        print("dish_summary.csv 字段不完整，改为根据清洗数据重新计算。")
    else:
        print("未找到 dish_summary.csv，改为根据清洗数据重新计算。")
    return rebuild_dish_summary(detail)


def attach_abc_class(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.sort_values("weight_share", ascending=False).reset_index(drop=True).copy()
    result["cumulative_weight_share"] = result["weight_share"].cumsum()
    result["abc_class"] = np.select(
        [
            result["cumulative_weight_share"] <= 0.70,
            result["cumulative_weight_share"] <= 0.90,
        ],
        ["A", "B"],
        default="C",
    )
    return result


def parse_itemset(value) -> Tuple[str, ...]:
    if isinstance(value, (set, frozenset, list, tuple)):
        return tuple(str(x) for x in value)
    text = str(value).strip()
    if not text:
        return tuple()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (set, frozenset, list, tuple)):
            return tuple(str(x) for x in parsed)
    except Exception:
        pass
    return tuple(part.strip() for part in text.replace("{", "").replace("}", "").split("|") if part.strip())


def compute_rice_rules_from_basket() -> pd.DataFrame:
    if not BASKET_PATH.exists():
        raise FileNotFoundError("未找到 basket.csv，无法重新计算关联规则。")
    basket = pd.read_csv(BASKET_PATH)
    if not {"indent_id", "dish_set"}.issubset(basket.columns):
        raise ValueError("basket.csv 缺少 indent_id 或 dish_set 字段。")

    transactions = basket["dish_set"].fillna("").astype(str).map(
        lambda x: {item.strip() for item in x.split("|") if item.strip()}
    )
    n_orders = len(transactions)
    rice_orders = sum(any(keyword in items for keyword in RICE_KEYWORDS) for items in transactions)
    rice_support = rice_orders / max(n_orders, 1)

    item_orders: Dict[str, int] = {}
    pair_orders: Dict[str, int] = {}
    for items in transactions:
        has_rice = any(keyword in items for keyword in RICE_KEYWORDS)
        for item in items:
            if item in RICE_KEYWORDS:
                continue
            item_orders[item] = item_orders.get(item, 0) + 1
            if has_rice:
                pair_orders[item] = pair_orders.get(item, 0) + 1

    rows = []
    for item, antecedent_count in item_orders.items():
        joint_count = pair_orders.get(item, 0)
        support = joint_count / max(n_orders, 1)
        confidence = joint_count / max(antecedent_count, 1)
        lift = confidence / rice_support if rice_support > 0 else 0.0
        rows.append(
            {
                "antecedents": item,
                "consequents": "米饭",
                "support": support,
                "confidence": confidence,
                "lift": lift,
            }
        )
    rules = pd.DataFrame(rows).sort_values(["confidence", "lift", "support"], ascending=False)
    rules.to_csv(ASSOCIATION_RULES_PATH, index=False, encoding="utf-8-sig")
    print(f"未找到 association_rules.csv，已基于 basket.csv 重新计算与米饭相关规则：{ASSOCIATION_RULES_PATH}")
    return rules


def load_or_build_association_rules() -> pd.DataFrame:
    if not ASSOCIATION_RULES_PATH.exists():
        return compute_rice_rules_from_basket()
    rules = pd.read_csv(ASSOCIATION_RULES_PATH)
    needed = {"antecedents", "consequents", "support", "confidence", "lift"}
    if not needed.issubset(rules.columns):
        print("association_rules.csv 字段不完整，改为根据 basket.csv 重新计算。")
        return compute_rice_rules_from_basket()
    return rules


def build_association_strength(summary: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    work = rules.copy()
    work["antecedent_items"] = work["antecedents"].map(parse_itemset)
    work["consequent_items"] = work["consequents"].map(parse_itemset)
    rice_rules = work[
        work["consequent_items"].map(lambda items: any(keyword in items for keyword in RICE_KEYWORDS))
        & work["antecedent_items"].map(lambda items: len(items) == 1)
    ].copy()
    rice_rules = rice_rules[
        (pd.to_numeric(rice_rules["support"], errors="coerce").fillna(0.0) >= MIN_RULE_SUPPORT)
        & (pd.to_numeric(rice_rules["confidence"], errors="coerce").fillna(0.0) >= MIN_RULE_CONFIDENCE)
        & (pd.to_numeric(rice_rules["lift"], errors="coerce").fillna(0.0) > MIN_RULE_LIFT)
    ].copy()
    rice_rules["dish_name"] = rice_rules["antecedent_items"].map(lambda items: items[0] if items else "")
    rice_rules["association_strength"] = (
        pd.to_numeric(rice_rules["confidence"], errors="coerce").fillna(0.0)
        * pd.to_numeric(rice_rules["lift"], errors="coerce").fillna(0.0)
    )
    strongest = rice_rules.groupby("dish_name", as_index=False)["association_strength"].max()

    merged = summary.merge(strongest, on="dish_name", how="left")
    merged["association_strength"] = merged["association_strength"].fillna(0.0)
    merged["association_strength_norm"] = minmax(merged["association_strength"])
    return merged


def build_candidate_dishes(detail: pd.DataFrame, preference: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    total_orders = max(detail["indent_id"].nunique(), 1)
    avg_units_per_customer = detail.groupby("indent_id")["weight"].sum().mean() / 100.0

    grouped = detail.groupby(["dish_serial", "dish_name"], as_index=False).agg(
        order_count=("indent_id", "nunique"),
        total_weight_g=("weight", "sum"),
        total_sales=("total_price", "sum"),
        avg_unit_price=("unit_price", "mean"),
        calories_total=("calories_dish", "sum"),
        carbohydrates_total=("carbohydrates_dish", "sum"),
        protein_total=("protein_dish", "sum"),
        fat_total=("fat_dish", "sum"),
        fiber_total=("fiber_dish", "sum"),
        active_days=("date", "nunique"),
    )
    grouped["price_per_100g"] = grouped["total_sales"] / (grouped["total_weight_g"] / 100.0)
    for src, dst in [
        ("calories_total", "calories_100g"),
        ("carbohydrates_total", "carbohydrates_100g"),
        ("protein_total", "protein_100g"),
        ("fat_total", "fat_100g"),
        ("fiber_total", "fiber_100g"),
    ]:
        grouped[dst] = grouped[src] / grouped["total_weight_g"] * 100.0
    grouped["historical_avg_units"] = (grouped["total_weight_g"] / grouped["active_days"].clip(lower=1)) / 100.0
    grouped["profit_per_100g"] = grouped["price_per_100g"] * (1.0 - base_q3.COST_RATE)

    dishes = grouped.merge(
        preference[
            [
                "dish_serial",
                "weight_share",
                "order_penetration",
                "abc_class",
                "association_strength",
                "association_strength_norm",
                "preference_weight_old",
                "preference_weight_new",
            ]
        ],
        on="dish_serial",
        how="left",
    )
    numeric_cols = ["price_per_100g", "calories_100g", "carbohydrates_100g", "protein_100g", "fat_100g", "fiber_100g"]
    dishes = dishes.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_cols)
    dishes = dishes[dishes["price_per_100g"] > 0].copy()
    dishes = dishes.sort_values("preference_weight_new", ascending=False).head(TOP_DISH_LIMIT).reset_index(drop=True)
    dishes["preference_norm_new"] = minmax(dishes["preference_weight_new"])
    dishes["total_orders"] = total_orders
    return dishes, float(avg_units_per_customer)


def solve_day_milp_with_abc(dishes: pd.DataFrame, targets: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, str]:
    n = len(dishes)
    total_units = targets["total_units"]
    default_max_units = max(base_q3.MIN_UNITS_PER_DISH + 1, int(np.ceil(total_units * base_q3.MAX_SHARE_PER_DISH)))

    lower_units = np.repeat(base_q3.MIN_UNITS_PER_DISH, n).astype(float)
    upper_units = np.repeat(default_max_units, n).astype(float)
    a_mask = dishes["abc_class"].eq("A").to_numpy()
    c_mask = dishes["abc_class"].eq("C").to_numpy()
    lower_units[a_mask] = np.maximum(lower_units[a_mask], np.ceil(dishes.loc[a_mask, "historical_avg_units"].to_numpy() * 0.5))
    upper_units[c_mask] = np.minimum(upper_units[c_mask], np.maximum(lower_units[c_mask], np.floor(dishes.loc[c_mask, "historical_avg_units"].to_numpy() * 1.2)))
    upper_units = np.maximum(upper_units, lower_units)

    profit = dishes["profit_per_100g"].to_numpy()
    pref = dishes["preference_norm_new"].to_numpy()
    c = np.r_[-(profit + base_q3.PREFERENCE_WEIGHT * pref), -np.repeat(base_q3.DIVERSITY_REWARD, n)]

    constraints: List[np.ndarray] = []
    lb: List[float] = []
    ub: List[float] = []

    constraints.append(np.r_[np.ones(n), np.zeros(n)])
    lb.append(total_units * (1 - base_q3.TOTAL_UNITS_TOL))
    ub.append(total_units * (1 + base_q3.TOTAL_UNITS_TOL))

    constraints.append(np.r_[dishes["price_per_100g"].to_numpy(), np.zeros(n)])
    lb.append(targets["sales"] * base_q3.SALES_LOWER)
    ub.append(targets["sales"] * base_q3.SALES_UPPER)

    for col, key in [
        ("calories_100g", "calories"),
        ("carbohydrates_100g", "carbohydrates"),
        ("protein_100g", "protein"),
        ("fat_100g", "fat"),
        ("fiber_100g", "fiber"),
    ]:
        constraints.append(np.r_[dishes[col].to_numpy(), np.zeros(n)])
        lb.append(targets[key] * base_q3.NUTRIENT_LOWER)
        ub.append(targets[key] * base_q3.NUTRIENT_UPPER)

    constraints.append(np.r_[np.zeros(n), np.ones(n)])
    lb.append(base_q3.MIN_DISH_TYPES)
    ub.append(base_q3.MAX_DISH_TYPES)

    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = 1
        row[n + i] = -upper_units[i]
        constraints.append(row)
        lb.append(-np.inf)
        ub.append(0)

        row = np.zeros(2 * n)
        row[i] = 1
        row[n + i] = -lower_units[i]
        constraints.append(row)
        lb.append(0)
        ub.append(np.inf)

    A = np.vstack(constraints)
    lower = np.array(lb, dtype=float)
    upper = np.array(ub, dtype=float)

    if base_q3.SCIPY_MILP_AVAILABLE and base_q3.milp is not None:
        integrality = np.r_[np.ones(n), np.ones(n)]
        bounds = Bounds(np.zeros(2 * n), np.r_[upper_units, np.ones(n)])
        res = base_q3.milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=LinearConstraint(A, lower, upper),
            options={"time_limit": 30.0, "mip_rel_gap": 0.02},
        )
        if res.success:
            return np.rint(res.x[:n]).astype(int), np.rint(res.x[n:]).astype(int), "scipy.optimize.milp"

    res = linprog(
        c=c,
        A_ub=np.vstack([A, -A]),
        b_ub=np.r_[upper, -lower],
        bounds=[(0, upper_units[i]) for i in range(n)] + [(0, 1)] * n,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"优化模型求解失败：{res.message}")
    x = np.rint(res.x[:n]).astype(int)
    y = (x >= lower_units).astype(int)
    x[y == 0] = 0
    return x, y, "scipy.optimize.linprog + round"


def summarize_day(date, dishes: pd.DataFrame, x: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    chosen = dishes.copy()
    chosen["prepare_units_100g"] = x
    chosen = chosen[chosen["prepare_units_100g"] > 0].copy()
    chosen["prepare_weight_kg"] = chosen["prepare_units_100g"] * 0.1
    chosen["estimated_profit"] = chosen["prepare_units_100g"] * chosen["profit_per_100g"]

    detail = chosen[
        [
            "dish_name",
            "abc_class",
            "prepare_weight_kg",
            "price_per_100g",
            "estimated_profit",
            "preference_weight_new",
            "association_strength_norm",
        ]
    ].copy()
    detail.insert(0, "date", pd.to_datetime(date).strftime("%Y-%m-%d"))
    detail = detail.sort_values(["prepare_weight_kg", "estimated_profit"], ascending=False).reset_index(drop=True)

    counts = detail["abc_class"].value_counts()
    summary = {
        "date": pd.to_datetime(date).strftime("%Y-%m-%d"),
        "total_prepare_weight_kg": detail["prepare_weight_kg"].sum(),
        "total_estimated_profit": detail["estimated_profit"].sum(),
        "selected_dish_count": len(detail),
        "A_class_count": int(counts.get("A", 0)),
        "B_class_count": int(counts.get("B", 0)),
        "C_class_count": int(counts.get("C", 0)),
        "average_preference_weight": detail["preference_weight_new"].mean(),
        "average_association_strength": detail["association_strength_norm"].mean(),
    }
    return detail, summary


def load_old_plan() -> pd.DataFrame | None:
    for path in OLD_PLAN_CANDIDATES:
        if path.exists():
            print(f"已找到原问题三方案：{path}")
            old = pd.read_csv(path)
            return old.rename(
                columns={
                    "日期": "date",
                    "菜品名称": "dish_name",
                    "预计利润": "estimated_profit",
                    "消费偏好权重": "preference_weight_old",
                }
            )
    print("未找到原问题三方案文件，将只输出新方案并跳过前后对比图。")
    return None


def build_comparison(old_plan: pd.DataFrame | None, new_plan: pd.DataFrame, preference: pd.DataFrame) -> pd.DataFrame:
    if old_plan is None:
        return pd.DataFrame(
            [
                {
                    "metric": "comparison_unavailable",
                    "before": np.nan,
                    "after": np.nan,
                    "change": np.nan,
                    "note": "未找到原问题三方案文件",
                }
            ]
        )

    old = old_plan.merge(preference[["dish_name", "abc_class", "association_strength_norm"]], on="dish_name", how="left")
    new = new_plan.copy()
    old["is_associated"] = old["association_strength_norm"].fillna(0) > 0
    new["is_associated"] = new["association_strength_norm"].fillna(0) > 0

    old_selected_daily = old.groupby("date")["dish_name"].count().mean()
    new_selected_daily = new.groupby("date")["dish_name"].count().mean()
    metrics = [
        ("total_profit", old["estimated_profit"].sum(), new["estimated_profit"].sum()),
        ("average_preference_weight", old["preference_weight_old"].mean(), new["preference_weight_new"].mean()),
        ("A_class_share", old["abc_class"].eq("A").mean(), new["abc_class"].eq("A").mean()),
        ("associated_dish_share", old["is_associated"].mean(), new["is_associated"].mean()),
        ("average_selected_dish_count_per_day", old_selected_daily, new_selected_daily),
    ]
    return pd.DataFrame(
        [{"metric": name, "before": before, "after": after, "change": after - before, "note": ""} for name, before, after in metrics]
    )


def save_figures(preference: pd.DataFrame, plan: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    set_chinese_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    top_pref = preference.sort_values("preference_weight_new", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(top_pref))
    ax.barh(y - 0.2, top_pref["preference_weight_old"], height=0.38, label="修正前")
    ax.barh(y + 0.2, top_pref["preference_weight_new"], height=0.38, label="修正后")
    ax.set_yticks(y)
    ax.set_yticklabels(top_pref["dish_name"])
    ax.set_xlabel("消费偏好权重")
    ax.set_title("修正前后消费偏好权重前15菜品对比")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_PREF, dpi=300)
    plt.close(fig)

    abc = summary.set_index("date")[["A_class_count", "B_class_count", "C_class_count"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    abc.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("日期")
    ax.set_ylabel("菜品数量")
    ax.set_title("加入关联规则修正后的每日ABC类菜品数量分布")
    ax.legend(["A类", "B类", "C类"], title="ABC类别")
    fig.tight_layout()
    fig.savefig(FIG_ABC, dpi=300)
    plt.close(fig)

    top_assoc = preference.sort_values("association_strength", ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_assoc["dish_name"], top_assoc["association_strength"])
    ax.set_xlabel("关联强度")
    ax.set_title("与米饭关联强度排名前10菜品")
    fig.tight_layout()
    fig.savefig(FIG_ASSOC, dpi=300)
    plt.close(fig)

    usable = comparison[comparison["metric"].isin(["total_profit", "average_preference_weight", "A_class_share", "associated_dish_share"])]
    if not usable.empty and usable["before"].notna().all():
        plot_df = usable.set_index("metric")[["before", "after"]]
        plot_df.index = ["总利润", "平均偏好权重", "A类菜品占比", "关联菜品占比"]
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_df.plot(kind="bar", ax=ax)
        ax.set_ylabel("指标值")
        ax.set_title("原方案与修正方案核心指标对比")
        ax.legend(["原方案", "修正方案"])
        fig.tight_layout()
        fig.savefig(FIG_COMPARE, dpi=300)
        plt.close(fig)
    else:
        print("未生成 q3_before_after_comparison.png：缺少可用的原方案对比数据。")


def write_notes(comparison: pd.DataFrame, comparison_available: bool) -> None:
    change_lines = []
    if comparison_available:
        readable = {
            "total_profit": "总利润",
            "average_preference_weight": "平均消费偏好权重",
            "A_class_share": "A类菜品占比",
            "associated_dish_share": "关联菜品占比",
            "average_selected_dish_count_per_day": "日均选择菜品数",
        }
        for _, row in comparison.iterrows():
            if row["metric"] in readable:
                change_lines.append(
                    f"- {readable[row['metric']]}：由 {row['before']:.4f} 变为 {row['after']:.4f}，变化 {row['change']:+.4f}"
                )
    else:
        change_lines.append("- 未找到原问题三方案文件，本次仅输出修正后方案，对比接口已保留。")

    text = f"""# 问题三补充实验说明：问题一结果驱动的消费偏好修正

## 1. 本次补充实验做了什么
在原问题三午餐备菜优化模型不变的基础上，补充引入问题一得到的菜品销售结构、ABC 分类以及与“米饭”相关的关联规则。实验新增了关联强度指标、修正后的消费偏好权重，并据此重新求解 2025 年 5 月 6 日至 5 月 12 日普通工作日的午餐备菜方案。

## 2. 为什么要引入问题一的关联规则结果
仅使用销量占比和订单渗透率，能够描述“单品受欢迎程度”，但无法体现顾客常见的搭配习惯。将与米饭相关的强关联规则引入后，模型可以进一步识别“既常卖、又常与主食共同出现”的菜品，使问题一发现的消费结构和搭配规律真正进入问题三的决策过程。

## 3. 修正偏好权重定义
设销售重量占比为 `s_i`，订单渗透率为 `o_i`，与米饭的归一化关联强度为 `g_i_norm`，则：

`w_i = 0.6 s_i + 0.4 o_i`

`w_i' = 0.6 s_i + 0.4 o_i + {GAMMA} g_i_norm`

其中，`g_i = confidence(i => 米饭) * lift(i => 米饭)`，若菜品没有与米饭相关的规则，则 `g_i = 0`。
本实验将“强关联”定义为：`support >= {MIN_RULE_SUPPORT}`、`confidence >= {MIN_RULE_CONFIDENCE}` 且 `lift > {MIN_RULE_LIFT}`。

## 4. ABC 分类如何影响备菜上下限
- A 类菜品：在原下限基础上，最低备菜量提高到历史平均备菜量的 50% 与原下限中的较大值。
- B 类菜品：保持原上下限不变。
- C 类菜品：在原上限基础上，最高备菜量压缩到历史平均备菜量的 120% 与原上限中的较小值。

这种处理保持了原模型主体结构，同时让核心菜品更稳定进入方案、长尾菜品不过度占用备菜量。

## 5. 新方案相比原方案的主要变化
{chr(10).join(change_lines)}

## 6. 可插入论文的图表
- `第三题/figures/q3_preference_weight_compare.png`
- `第三题/figures/q3_abc_selected_distribution.png`
- `第三题/figures/q3_association_strength_top10.png`
{"- `第三题/figures/q3_before_after_comparison.png`" if comparison_available else "- 原方案文件缺失，因此未生成前后对比图。"}
"""
    NOTES_OUTPUT.write_text(text, encoding="utf-8")


def main() -> None:
    detail = read_lunch_detail()
    summary = attach_abc_class(load_or_build_dish_summary(detail))
    rules = load_or_build_association_rules()
    preference = build_association_strength(summary, rules)
    preference["preference_weight_old"] = 0.6 * preference["weight_share"] + 0.4 * preference["order_penetration"]
    preference["preference_weight_new"] = preference["preference_weight_old"] + GAMMA * preference["association_strength_norm"]

    preference_output = preference[
        [
            "dish_name",
            "weight_share",
            "order_penetration",
            "abc_class",
            "association_strength",
            "association_strength_norm",
            "preference_weight_old",
            "preference_weight_new",
        ]
    ].copy()
    preference_output.to_csv(PREFERENCE_OUTPUT, index=False, encoding="utf-8-sig")

    dishes, avg_units_per_customer = build_candidate_dishes(detail, preference)
    pred = read_prediction_for_experiment()

    all_details: List[pd.DataFrame] = []
    summaries: List[Dict[str, float]] = []
    solvers: List[str] = []
    for _, pred_row in pred.iterrows():
        targets = base_q3.get_day_targets(pred_row, avg_units_per_customer)
        x, _, solver_name = solve_day_milp_with_abc(dishes, targets)
        day_detail, day_summary = summarize_day(pred_row["日期"], dishes, x)
        all_details.append(day_detail)
        summaries.append(day_summary)
        solvers.append(solver_name)

    plan = pd.concat(all_details, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    for frame in [plan, summary_df, preference_output]:
        float_cols = frame.select_dtypes(include=["float64", "float32"]).columns
        frame[float_cols] = frame[float_cols].round(6)

    plan.to_csv(PLAN_OUTPUT, index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")

    old_plan = load_old_plan()
    comparison = build_comparison(old_plan, plan, preference_output)
    comparison.to_csv(COMPARISON_OUTPUT, index=False, encoding="utf-8-sig")

    save_figures(preference_output, plan, summary_df, comparison)
    write_notes(comparison, old_plan is not None)

    print("补充实验完成。")
    print("使用求解器：", sorted(set(solvers)))
    print("输出文件：")
    for path in [
        PREFERENCE_OUTPUT,
        PLAN_OUTPUT,
        SUMMARY_OUTPUT,
        COMPARISON_OUTPUT,
        NOTES_OUTPUT,
        FIG_PREF,
        FIG_ABC,
        FIG_ASSOC,
    ]:
        print(" -", path)
    if FIG_COMPARE.exists():
        print(" -", FIG_COMPARE)


if __name__ == "__main__":
    main()
