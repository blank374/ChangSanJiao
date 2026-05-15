#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为问题三补充实验生成论文图表与 LaTeX 插图代码。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
FIG_DIR = ROOT / "第三题" / "figures"
PREFERENCE_PATH = ROOT / "q3_preference_weight_with_association.csv"
OLD_PLAN_PATH = ROOT / "第三题" / "problem3_lunch_menu_plan.csv"
NEW_PLAN_PATH = ROOT / "q3_lunch_plan_with_association.csv"
LATEX_OUTPUT = ROOT / "q3_association_revision_figures_latex.txt"

OUTPUT_CHANGE = FIG_DIR / "q3_preference_revision_compare.png"
OUTPUT_RATIO = FIG_DIR / "q3_related_dish_ratio_compare.png"
OUTPUT_ABC = FIG_DIR / "q3_abc_structure_compare.png"
OUTPUT_PROFIT = FIG_DIR / "q3_profit_compare.png"

METRICS = pd.DataFrame(
    {
        "指标": ["总利润", "平均消费偏好权重", "A类菜品占比", "关联菜品占比", "日均选择菜品数"],
        "原方案": [7725.5860, 0.0395, 0.7368, 0.6140, 34.2000],
        "修正方案": [7303.8383, 0.1832, 0.6000, 0.9474, 19.0000],
    }
)


def set_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10


def add_bar_labels(ax, bars, formatter) -> None:
    for bar in bars:
        value = bar.get_height()
        offset = 3 if value >= 0 else -12
        va = "bottom" if value >= 0 else "top"
        ax.annotate(
            formatter(value),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
        )


def draw_change_rate_chart() -> None:
    data = METRICS.copy()
    data["变化率"] = (data["修正方案"] - data["原方案"]) / data["原方案"] * 100
    colors = ["#D95F59" if x < 0 else "#2E8B8B" for x in data["变化率"]]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.bar(data["指标"], data["变化率"], color=colors, width=0.62)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_ylabel("变化率（%）")
    ax.set_title("消费偏好修正前后主要指标变化率", pad=14)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.set_ylim(min(data["变化率"].min() * 1.18, -55), max(data["变化率"].max() * 1.12, 380))
    ax.tick_params(axis="x", rotation=20, labelsize=10)
    add_bar_labels(ax, bars, lambda v: f"{v:+.1f}%")

    fig.tight_layout()
    fig.savefig(OUTPUT_CHANGE, dpi=320, bbox_inches="tight")
    plt.close(fig)


def draw_related_ratio_chart() -> None:
    labels = ["原方案", "修正方案"]
    values = [0.6140, 0.9474]
    colors = ["#6C8EBF", "#2E8B8B"]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    bars = ax.bar(labels, values, color=colors, width=0.52)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("关联菜品占比")
    ax.set_title("消费偏好修正前后关联菜品占比对比", pad=14)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    add_bar_labels(ax, bars, lambda v: f"{v:.2%}")

    fig.tight_layout()
    fig.savefig(OUTPUT_RATIO, dpi=320, bbox_inches="tight")
    plt.close(fig)


def load_abc_structure() -> pd.DataFrame:
    preference = pd.read_csv(PREFERENCE_PATH)[["dish_name", "abc_class"]]
    old_plan = pd.read_csv(OLD_PLAN_PATH).rename(columns={"菜品名称": "dish_name"})
    new_plan = pd.read_csv(NEW_PLAN_PATH)

    old = old_plan.merge(preference, on="dish_name", how="left")
    new = new_plan[["dish_name", "abc_class"]].copy()
    old_counts = old["abc_class"].value_counts(normalize=True).reindex(["A", "B", "C"], fill_value=0)
    new_counts = new["abc_class"].value_counts(normalize=True).reindex(["A", "B", "C"], fill_value=0)

    return pd.DataFrame({"原方案": old_counts, "修正方案": new_counts}).T


def draw_abc_structure_chart() -> None:
    structure = load_abc_structure()
    colors = {"A": "#2E8B8B", "B": "#6C8EBF", "C": "#D9A441"}

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    bottom = np.zeros(len(structure))
    for cls in ["A", "B", "C"]:
        values = structure[cls].to_numpy()
        bars = ax.bar(structure.index, values, bottom=bottom, label=f"{cls}类", color=colors[cls], width=0.54)
        for bar, value, base in zip(bars, values, bottom):
            if value >= 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    color="white" if cls in {"A", "B"} else "#333333",
                    fontsize=10,
                )
        bottom += values

    ax.set_ylim(0, 1.02)
    ax.set_ylabel("菜品结构占比")
    ax.set_title("消费偏好修正前后ABC类菜品结构对比", pad=14)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_ABC, dpi=320, bbox_inches="tight")
    plt.close(fig)


def draw_profit_chart() -> None:
    old_plan = pd.read_csv(OLD_PLAN_PATH).rename(columns={"日期": "date", "预计利润": "estimated_profit"})
    daily = old_plan.groupby("date", as_index=False)["estimated_profit"].sum()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["label"] = daily["date"].dt.strftime("%m-%d")
    avg_profit = daily["estimated_profit"].mean()

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    bars = ax.bar(daily["label"], daily["estimated_profit"], color="#6C8EBF", width=0.56, label="预计利润")
    ax.axhline(
        avg_profit,
        color="#D95F59",
        linestyle="--",
        linewidth=1.8,
        label=f"平均值 {avg_profit:.2f} 元",
    )
    ax.set_xlabel("日期")
    ax.set_ylabel("预计利润/元")
    ax.set_title("各工作日午餐预计利润分布", pad=14)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    add_bar_labels(ax, bars, lambda v: f"{v:.1f}")

    fig.tight_layout()
    fig.savefig(OUTPUT_PROFIT, dpi=320, bbox_inches="tight")
    plt.close(fig)


def write_latex() -> None:
    latex = r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.76\textwidth]{第三题/figures/q3_profit_compare.png}
    \caption{各工作日午餐预计利润分布}
    \label{fig:q3_profit_compare}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.88\textwidth]{第三题/figures/q3_preference_revision_compare.png}
    \caption{消费偏好修正前后主要指标变化率}
    \label{fig:q3_preference_revision_compare}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.68\textwidth]{第三题/figures/q3_related_dish_ratio_compare.png}
    \caption{消费偏好修正前后关联菜品占比对比}
    \label{fig:q3_related_dish_ratio_compare}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.72\textwidth]{第三题/figures/q3_abc_structure_compare.png}
    \caption{消费偏好修正前后ABC类菜品结构对比}
    \label{fig:q3_abc_structure_compare}
\end{figure}
"""
    LATEX_OUTPUT.write_text(latex, encoding="utf-8")


def main() -> None:
    required = [PREFERENCE_PATH, OLD_PLAN_PATH, NEW_PLAN_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少绘图所需文件：{missing}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    draw_profit_chart()
    draw_change_rate_chart()
    draw_related_ratio_chart()
    draw_abc_structure_chart()
    write_latex()

    print("已生成图表：")
    for path in [OUTPUT_PROFIT, OUTPUT_CHANGE, OUTPUT_RATIO, OUTPUT_ABC]:
        print(" -", path)
    print("已生成 LaTeX 代码：")
    print(" -", LATEX_OUTPUT)


if __name__ == "__main__":
    main()
