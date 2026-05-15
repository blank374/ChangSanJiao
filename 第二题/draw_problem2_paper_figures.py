#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题二预测结果论文图绘制脚本。

只需修改 data_path 即可切换数据文件：
    data_path = "q2_prediction_results.csv"
或：
    data_path = "problem2_prediction_results.xlsx"
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
import matplotlib.dates as mdates


data_path = "problem2_prediction_results.xlsx"
fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)


def setup_style():
    """统一论文图风格。"""
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
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10


def read_prediction_data(path):
    """读取 csv/xlsx，并自动清理列名与日期列。"""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("仅支持 csv、xlsx 或 xls 文件")

    df.columns = [str(c).strip() for c in df.columns]
    date_col = None
    for col in df.columns:
        if "日期" in col or col.lower() in {"date", "day"}:
            date_col = col
            break
    if date_col is None:
        raise ValueError("未识别到日期列")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return df, date_col


def set_common_axis(ax):
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", rotation=35)


def set_y_margin(ax, values, margin_ratio=0.12):
    values = np.asarray(values, dtype=float)
    low, high = np.nanmin(values), np.nanmax(values)
    margin = (high - low) * margin_ratio if high > low else max(abs(high) * 0.05, 1)
    ax.set_ylim(low - margin, high + margin)


def save_fig(fig, filename):
    path = fig_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(path)


def plot_people(df, date_col):
    col = "预测就餐人数"
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(df[date_col], df[col], marker="o", markersize=5, linewidth=2.3,
            color="#1f77b4", label="预测值")
    avg = df[col].mean()
    ax.axhline(avg, linestyle="--", linewidth=1.8, color="#d62728", label="平均值")
    ax.set_title("2025年5月工作日就餐人数预测趋势")
    ax.set_xlabel("日期")
    ax.set_ylabel("预测就餐人数")
    set_y_margin(ax, df[col])
    set_common_axis(ax)
    ax.legend(frameon=False)
    save_fig(fig, "q2_people_prediction.png")


def plot_sales(df, date_col):
    col = "预测销售总额"
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(df[date_col], df[col], marker="o", markersize=5, linewidth=2.3,
            color="#2ca02c", label="预测值")
    avg = df[col].mean()
    ax.axhline(avg, linestyle="--", linewidth=1.8, color="#d62728", label="平均值")
    ax.set_title("2025年5月工作日销售总额预测趋势")
    ax.set_xlabel("日期")
    ax.set_ylabel("预测销售总额/元")
    set_y_margin(ax, df[col])
    set_common_axis(ax)
    ax.legend(frameon=False)
    save_fig(fig, "q2_sales_prediction.png")


def minmax_standardize(s):
    s = pd.to_numeric(s, errors="coerce")
    min_v, max_v = s.min(), s.max()
    if max_v == min_v:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - min_v) / (max_v - min_v)


def plot_nutrition(df, date_col):
    cols = ["预测热量需求", "预测碳水需求", "预测蛋白质需求", "预测脂肪需求", "预测膳食纤维需求"]
    labels = ["热量需求", "碳水需求", "蛋白质需求", "脂肪需求", "膳食纤维需求"]
    styles = [
        ("#1f77b4", "o", "-"),
        ("#ff7f0e", "s", "--"),
        ("#2ca02c", "^", "-."),
        ("#9467bd", "D", ":"),
        ("#8c564b", "v", "-"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for col, label, (color, marker, linestyle) in zip(cols, labels, styles):
        y = minmax_standardize(df[col])
        ax.plot(df[date_col], y, marker=marker, markersize=4.8, linewidth=2.1,
                linestyle=linestyle, color=color, label=label)
    ax.set_title("2025年5月工作日各类营养素需求标准化趋势")
    ax.set_xlabel("日期")
    ax.set_ylabel("标准化需求值")
    ax.set_ylim(-0.05, 1.05)
    set_common_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)
    save_fig(fig, "q2_nutrition_standardized.png")


def plot_nutrition_subplots(df, date_col):
    cols = ["预测热量需求", "预测碳水需求", "预测蛋白质需求", "预测脂肪需求", "预测膳食纤维需求"]
    labels = ["热量需求", "碳水需求", "蛋白质需求", "脂肪需求", "膳食纤维需求"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

    fig, axes = plt.subplots(3, 2, figsize=(9, 7.2), sharex=True)
    axes = axes.ravel()
    for ax, col, label, color in zip(axes, cols, labels, colors):
        ax.plot(df[date_col], df[col], marker="o", markersize=4.2, linewidth=2.0,
                color=color, label=label)
        ax.set_title(label, fontsize=12)
        ax.set_ylabel("预测需求量", fontsize=10)
        set_y_margin(ax, df[col], margin_ratio=0.15)
        set_common_axis(ax)
        ax.legend(frameon=False, loc="best", fontsize=9)

    axes[-1].axis("off")
    for ax in axes[-3:-1]:
        ax.set_xlabel("日期", fontsize=11)
    fig.suptitle("2025年5月工作日各类营养素需求分项预测趋势", fontsize=15, y=0.995)
    save_fig(fig, "q2_nutrition_subplots.png")


def plot_main_nutrition_standardized(df, date_col):
    """只展示三个主要营养指标，降低五线图的交叉干扰。"""
    cols = ["预测热量需求", "预测碳水需求", "预测蛋白质需求"]
    labels = ["热量需求", "碳水需求", "蛋白质需求"]
    styles = [
        ("#1f77b4", "o", "-"),
        ("#ff7f0e", "s", "--"),
        ("#2ca02c", "^", "-."),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for col, label, (color, marker, linestyle) in zip(cols, labels, styles):
        y = minmax_standardize(df[col])
        ax.plot(df[date_col], y, marker=marker, markersize=5, linewidth=2.3,
                linestyle=linestyle, color=color, label=label)
    ax.set_title("2025年5月工作日主要营养素需求标准化趋势")
    ax.set_xlabel("日期")
    ax.set_ylabel("标准化需求值")
    ax.set_ylim(-0.05, 1.05)
    set_common_axis(ax)
    ax.legend(loc="upper right", frameon=False)
    save_fig(fig, "q2_main_nutrition_standardized.png")


def plot_monte_carlo(df, date_col):
    col = "预测就餐人数"
    rng = np.random.default_rng(2026)
    sigma = 50.372
    base = df[col].to_numpy(dtype=float)
    sims = rng.normal(loc=base[None, :], scale=sigma, size=(1000, len(base)))
    lower = np.percentile(sims, 2.5, axis=0)
    upper = np.percentile(sims, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(df[date_col], base, marker="o", markersize=5, linewidth=2.4,
            color="#1f77b4", label="预测值")
    ax.fill_between(df[date_col], lower, upper, color="#1f77b4", alpha=0.18,
                    label="95%预测区间")
    ax.set_title("Monte Carlo模拟下就餐人数预测区间")
    ax.set_xlabel("日期")
    ax.set_ylabel("预测就餐人数")
    set_common_axis(ax)
    ax.legend(frameon=False)
    save_fig(fig, "q2_monte_carlo_interval.png")


def plot_monte_carlo_revised(df, date_col):
    """论文推荐版：同时显示 80% 与 95% 区间，弱化过宽的外层区间。"""
    col = "预测就餐人数"
    rng = np.random.default_rng(2026)
    sigma = 50.372
    base = df[col].to_numpy(dtype=float)
    sims = rng.normal(loc=base[None, :], scale=sigma, size=(1000, len(base)))
    lower95 = np.percentile(sims, 2.5, axis=0)
    upper95 = np.percentile(sims, 97.5, axis=0)
    lower80 = np.percentile(sims, 10, axis=0)
    upper80 = np.percentile(sims, 90, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.fill_between(df[date_col], lower95, upper95, color="#1f77b4", alpha=0.10,
                    label="95%预测区间")
    ax.fill_between(df[date_col], lower80, upper80, color="#1f77b4", alpha=0.22,
                    label="80%预测区间")
    ax.plot(df[date_col], base, marker="o", markersize=5.2, linewidth=2.5,
            color="#0f4c81", label="预测值")
    ax.set_title("就餐人数预测结果的不确定性分析")
    ax.set_xlabel("日期")
    ax.set_ylabel("预测就餐人数")
    ax.text(
        0.01, 0.97,
        r"基于验证集RMSE估计扰动标准差：$\sigma=50.372$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#444444",
    )
    y_min = max(0, np.nanmin(lower95) - 12)
    y_max = np.nanmax(upper95) + 12
    ax.set_ylim(y_min, y_max)
    set_common_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    save_fig(fig, "q2_monte_carlo_interval_revised.png")


def main():
    setup_style()
    df, date_col = read_prediction_data(data_path)

    required_cols = [
        "预测就餐人数",
        "预测销售总额",
        "预测热量需求",
        "预测碳水需求",
        "预测蛋白质需求",
        "预测脂肪需求",
        "预测膳食纤维需求",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段：{missing}")
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    plot_people(df, date_col)
    plot_sales(df, date_col)
    plot_nutrition(df, date_col)
    plot_nutrition_subplots(df, date_col)
    plot_main_nutrition_standardized(df, date_col)
    plot_monte_carlo(df, date_col)
    plot_monte_carlo_revised(df, date_col)


if __name__ == "__main__":
    main()
