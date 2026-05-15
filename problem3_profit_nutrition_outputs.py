#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题三补充输出：
1. 各工作日午餐预计利润对比图；
2. 午餐备菜方案营养素满足情况表。

默认使用多目标整数规划主方案：problem3_lunch_menu_plan.xlsx
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/mpl_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("output/cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path("problem3_lunch_menu_plan.xlsx")
FIG_DIR = Path("figures/problem3")
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = Path("problem3_nutrition_satisfaction.csv")
OUT_LATEX = Path("problem3_nutrition_satisfaction_latex.txt")


def setup_style():
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
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10


def plot_profit(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    dates = pd.to_datetime(df["日期"]).dt.strftime("%m-%d")
    profits = df["预计利润"].astype(float)
    bars = ax.bar(dates, profits, color="#4C78A8", width=0.62, label="预计利润")
    avg = profits.mean()
    ax.axhline(avg, color="#D62728", linestyle="--", linewidth=1.6, label="平均值")
    for bar, val in zip(bars, profits):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + profits.max() * 0.012,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("问题三各工作日午餐预计利润对比")
    ax.set_xlabel("日期")
    ax.set_ylabel("预计利润/元")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "q3_profit_compare.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


def build_nutrition_satisfaction(df: pd.DataFrame) -> pd.DataFrame:
    mapping = [
        ("热量供给", "预测热量需求", "热量满足率/%"),
        ("碳水供给", "预测碳水需求", "碳水满足率/%"),
        ("蛋白质供给", "预测蛋白质需求", "蛋白质满足率/%"),
        ("脂肪供给", "预测脂肪需求", "脂肪满足率/%"),
        ("膳食纤维供给", "预测膳食纤维需求", "膳食纤维满足率/%"),
    ]
    out = pd.DataFrame({
        "日期": pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d"),
        "星期": df["星期"],
    })
    for supply, demand, name in mapping:
        out[name] = df[supply].astype(float) / df[demand].astype(float) * 100
    for col in out.columns:
        if col.endswith("/%"):
            out[col] = out[col].round(2)
    return out


def make_latex_table(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \scriptsize",
        r"    \caption{午餐备菜方案营养素满足情况}",
        r"    \label{tab:q3_nutrition_satisfaction}",
        r"    \resizebox{\textwidth}{!}{",
        r"    \begin{tabular}{ccccccc}",
        r"        \toprule",
        r"        日期 & 星期 & 热量满足率/\% & 碳水满足率/\% & 蛋白质满足率/\% & 脂肪满足率/\% & 膳食纤维满足率/\% \\",
        r"        \midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"        {row['日期']} & {row['星期']} & {row['热量满足率/%']:.2f} & "
            f"{row['碳水满足率/%']:.2f} & {row['蛋白质满足率/%']:.2f} & "
            f"{row['脂肪满足率/%']:.2f} & {row['膳食纤维满足率/%']:.2f} \\\\"
        )
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    setup_style()
    df = pd.read_excel(DATA_PATH, sheet_name="每日汇总")
    plot_profit(df)
    sat = build_nutrition_satisfaction(df)
    sat.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    latex = make_latex_table(sat)
    OUT_LATEX.write_text(latex, encoding="utf-8")
    print(OUT_CSV)
    print(OUT_LATEX)
    print("\n营养素满足情况：")
    print(sat.to_string(index=False))
    print("\nLaTeX 表格：")
    print(latex)


if __name__ == "__main__":
    main()
