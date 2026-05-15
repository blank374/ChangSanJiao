#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026 长三角高校数学建模竞赛 B 题 - 问题二
自助量贩餐厅每日需求预测：就餐人数、销售总额、营养素需求量。

运行方式：
    /Users/linjiamin/venv/bin/python problem2_all_methods.py

脚本特点：
1. 优先使用 daily_summary.csv，并自动检查列完整性；
2. 手写 Simple Exponential Smoothing 与 Croston；
3. 使用 sklearn 实现随机森林、MLP、线性回归、岭回归、梯度提升树；
4. 若 xgboost 不存在则自动跳过；
5. 构造日期、滞后、滚动统计特征，递推预测 2025 年 5 月工作日；
6. 输出 Excel、Markdown 说明和 figures 图像。
"""

from __future__ import annotations

import math
import os
import re
import sys
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.sax.saxutils import escape

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/mpl_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("output/cache").resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    print(f"[WARN] matplotlib 不可用，将跳过绘图：{exc}")

try:
    from sklearn.ensemble import (
        AdaBoostRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import SVR

    SKLEARN_AVAILABLE = True
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception as exc:  # pragma: no cover
    SKLEARN_AVAILABLE = False
    print(f"[WARN] sklearn 不可用，将只运行统计模型和基准模型：{exc}")

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except Exception:
    LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except Exception:
    CatBoostRegressor = None
    CATBOOST_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA

    STATSMODELS_AVAILABLE = True
except Exception:
    ARIMA = None
    STATSMODELS_AVAILABLE = False


ROOT = Path(".")
FIG_DIR = Path("figures")
OUT_DIR = Path("output")
RANDOM_STATE = 42
VALID_RATIO = 0.2
ALPHAS = np.round(np.arange(0.1, 1.0, 0.1), 1)
MAIN_NUTRIENT_KEYWORDS = [
    "热量",
    "能量",
    "碳水",
    "蛋白",
    "脂肪",
    "膳食纤维",
    "钠",
    "钙",
    "铁",
    "维生素",
]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def setup_chinese_font() -> None:
    if plt is None:
        return
    plt.rcParams["axes.unicode_minus"] = False
    candidates = [
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
    ]
    plt.rcParams["font.sans-serif"] = candidates + plt.rcParams.get("font.sans-serif", [])


def list_current_files() -> List[str]:
    return sorted([str(p) for p in ROOT.iterdir() if not p.name.startswith(".")])


def find_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        name = str(col).strip().lower()
        if name in {"date", "day", "日期", "销售日期", "订单日期"} or "日期" in str(col):
            return col
    best_col, best_valid = None, 0
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid = int(parsed.notna().sum())
        if valid > best_valid and valid >= max(3, len(df) * 0.5):
            best_col, best_valid = col, valid
    return best_col


def find_customer_column(df: pd.DataFrame) -> Optional[str]:
    patterns = ["就餐人数", "人数", "客流", "人次", "customer", "count"]
    for col in df.columns:
        low = str(col).lower()
        if any(p.lower() in low for p in patterns):
            return col
    return None


def find_sales_column(df: pd.DataFrame) -> Optional[str]:
    patterns = ["销售总额", "销售额", "营业额", "金额", "收入", "sales", "amount", "revenue"]
    for col in df.columns:
        low = str(col).lower()
        if any(p.lower() in low for p in patterns):
            return col
    return None


def identify_targets(df: pd.DataFrame, date_col: str) -> Tuple[List[str], Dict[str, str]]:
    customer_col = find_customer_column(df)
    sales_col = find_sales_column(df)
    excluded = {date_col}
    if customer_col:
        excluded.add(customer_col)
    if sales_col:
        excluded.add(sales_col)

    numeric_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= max(3, len(df) * 0.6):
            numeric_cols.append(col)

    nutrient_cols = []
    for col in numeric_cols:
        name = str(col)
        if "需求" in name or any(k in name for k in MAIN_NUTRIENT_KEYWORDS):
            nutrient_cols.append(col)

    # 如果列名没有营养素关键词，则保守地把除日期/人数/销售外的数值列作为营养素需求候选。
    if not nutrient_cols:
        nutrient_cols = numeric_cols

    targets = []
    if customer_col:
        targets.append(customer_col)
    if sales_col and sales_col not in targets:
        targets.append(sales_col)
    targets.extend([c for c in nutrient_cols if c not in targets])

    mapping = {
        "date": date_col or "",
        "customer": customer_col or "",
        "sales": sales_col or "",
        "nutrients": ", ".join(nutrient_cols),
    }
    return targets, mapping


def load_main_data() -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    files = list_current_files()
    daily_path = Path("daily_summary.csv")
    if not daily_path.exists():
        raise FileNotFoundError("未找到 daily_summary.csv；当前脚本按题目要求优先读取该文件。")

    df = pd.read_csv(daily_path)
    date_col = find_date_column(df)
    if date_col is None:
        raise ValueError("daily_summary.csv 未识别到日期列。")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy()
    targets, mapping = identify_targets(df, date_col)

    usable = bool(date_col and mapping["customer"] and mapping["sales"] and len(targets) >= 3)
    check = {
        "files": files,
        "columns": list(df.columns),
        "rows": int(len(df)),
        "date_col": date_col,
        "date_min": df[date_col].min(),
        "date_max": df[date_col].max(),
        "missing": df.isna().sum().to_dict(),
        "usable": usable,
        "target_cols": targets,
        "mapping": mapping,
    }
    if not usable:
        raise ValueError("daily_summary.csv 字段不完整；本脚本已检测到不可直接用于问题二建模。")

    df = df.sort_values(date_col).reset_index(drop=True)
    for col in targets:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=targets).reset_index(drop=True)
    return df, "daily_summary.csv", check


def add_date_features(date: pd.Timestamp, start_date: pd.Timestamp) -> Dict[str, float]:
    day = int(date.day)
    return {
        "date_ordinal": float((date - start_date).days),
        "dayofweek": float(date.dayofweek),
        "is_weekend": float(date.dayofweek >= 5),
        "is_workday": float(date.dayofweek < 5),
        "month": float(date.month),
        "is_month_start": float(day <= 10),
        "is_month_mid": float(11 <= day <= 20),
        "is_month_end": float(day >= 21),
    }


def make_feature_frame(df: pd.DataFrame, date_col: str, target: str) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col]), "y": pd.to_numeric(df[target], errors="coerce")})
    start_date = out["date"].min()
    date_feats = out["date"].apply(lambda d: pd.Series(add_date_features(d, start_date)))
    out = pd.concat([out, date_feats], axis=1)
    for lag in [1, 3, 7]:
        out[f"lag_{lag}"] = out["y"].shift(lag)
    shifted = out["y"].shift(1)
    for window in [3, 7]:
        out[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        out[f"rolling_std_{window}"] = shifted.rolling(window).std()
    return out.dropna().reset_index(drop=True)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-9
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calc_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    if SKLEARN_AVAILABLE:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    else:
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": safe_mape(y_true, y_pred)}


def ses_forecast(train: np.ndarray, steps: int, alpha: float) -> np.ndarray:
    train = np.asarray(train, dtype=float)
    level = train[0]
    for y in train[1:]:
        level = alpha * y + (1 - alpha) * level
    return np.repeat(level, steps)


def croston_forecast(train: np.ndarray, steps: int, alpha: float) -> np.ndarray:
    y = np.asarray(train, dtype=float)
    nonzero_idx = np.where(y > 0)[0]
    if len(nonzero_idx) == 0:
        return np.zeros(steps)
    first = nonzero_idx[0]
    z = y[first]
    p = max(first + 1, 1)
    last_nonzero = first
    for t in range(first + 1, len(y)):
        if y[t] > 0:
            interval = t - last_nonzero
            z = alpha * y[t] + (1 - alpha) * z
            p = alpha * interval + (1 - alpha) * p
            last_nonzero = t
    rate = z / max(p, 1e-9)
    return np.repeat(rate, steps)


def choose_alpha(model_name: str, train: np.ndarray, valid: np.ndarray) -> Tuple[float, np.ndarray, Dict[str, float]]:
    best = None
    for alpha in ALPHAS:
        if model_name == "Simple Exponential Smoothing":
            pred = ses_forecast(train, len(valid), alpha)
        else:
            pred = croston_forecast(train, len(valid), alpha)
        metrics = calc_metrics(valid, pred)
        key = metrics["RMSE"]
        if best is None or key < best[0]:
            best = (key, alpha, pred, metrics)
    assert best is not None
    return float(best[1]), np.asarray(best[2]), best[3]


def choose_arima(train: np.ndarray, valid: np.ndarray) -> Tuple[Tuple[int, int, int], np.ndarray, Dict[str, float]]:
    if not STATSMODELS_AVAILABLE or ARIMA is None:
        raise RuntimeError("statsmodels 不可用")
    orders = [(1, 0, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 0, 2)]
    best = None
    for order in orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(train, order=order).fit()
                pred = np.asarray(fit.forecast(steps=len(valid)), dtype=float)
            pred = np.maximum(pred, 0)
            metrics = calc_metrics(valid, pred)
            if best is None or metrics["RMSE"] < best[0]:
                best = (metrics["RMSE"], order, pred, metrics)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("ARIMA 所有候选阶数均训练失败")
    return best[1], np.asarray(best[2]), best[3]


def make_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    if not SKLEARN_AVAILABLE:
        return models
    models["Linear Regression"] = LinearRegression()
    models["Ridge Regression"] = Ridge(alpha=1.0)
    models["ElasticNet"] = Pipeline(
        [("scaler", StandardScaler()), ("elasticnet", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000))]
    )
    models["SVR"] = Pipeline(
        [("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1))]
    )
    models["KNN"] = Pipeline(
        [("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=7, weights="distance"))]
    )
    models["Gradient Boosting"] = GradientBoostingRegressor(random_state=RANDOM_STATE)
    models["AdaBoost"] = AdaBoostRegressor(n_estimators=100, random_state=RANDOM_STATE)
    models["Random Forest"] = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE
    )
    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE
    )
    models["MLP"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    activation="relu",
                    solver="adam",
                    learning_rate="constant",
                    alpha=0.0001,
                    early_stopping=True,
                ),
            ),
        ]
    )
    if XGBOOST_AVAILABLE and XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        )
    if LIGHTGBM_AVAILABLE and LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            objective="regression",
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
    if CATBOOST_AVAILABLE and CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            iterations=150,
            depth=5,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
    return models


def make_knn_pipeline(n_neighbors: int, weights: str, p: int) -> Any:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    p=p,
                ),
            ),
        ]
    )


def tune_knn(
    train_feat: pd.DataFrame, feature_cols: List[str], valid_feat: pd.DataFrame
) -> Tuple[Any, Dict[str, Any], np.ndarray, Dict[str, float]]:
    """用训练段末尾做内层时间验证选择 KNN 参数，再在外层验证集评价。"""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("sklearn 不可用")
    inner_split = max(int(len(train_feat) * 0.8), 20)
    inner_train = train_feat.iloc[:inner_split].copy()
    inner_valid = train_feat.iloc[inner_split:].copy()
    if len(inner_valid) < 5:
        raise RuntimeError("内层验证集样本不足，无法优化 KNN")

    candidate_ks = [3, 5, 7, 9, 11, 15, 21, 31]
    candidate_weights = ["uniform", "distance"]
    candidate_p = [1, 2]
    best = None
    for k in candidate_ks:
        if k >= len(inner_train):
            continue
        for weights in candidate_weights:
            for p in candidate_p:
                try:
                    model = make_knn_pipeline(k, weights, p)
                    model.fit(inner_train[feature_cols], inner_train["y"])
                    pred = np.maximum(model.predict(inner_valid[feature_cols]), 0)
                    metrics = calc_metrics(inner_valid["y"], pred)
                    if best is None or metrics["RMSE"] < best[0]:
                        best = (metrics["RMSE"], {"n_neighbors": k, "weights": weights, "p": p})
                except Exception:
                    continue
    if best is None:
        raise RuntimeError("KNN 网格搜索失败")

    params = best[1]
    outer_model = make_knn_pipeline(**params)
    outer_model.fit(train_feat[feature_cols], train_feat["y"])
    outer_pred = np.maximum(outer_model.predict(valid_feat[feature_cols]), 0)
    outer_metrics = calc_metrics(valid_feat["y"], outer_pred)
    return outer_model, params, np.asarray(outer_pred, dtype=float), outer_metrics


@dataclass
class TargetResult:
    target: str
    best_model: str
    best_params: Dict[str, Any]
    final_model: Any
    feature_cols: List[str]
    residual_std: float
    validation_plot_data: pd.DataFrame


def train_and_evaluate_target(
    df: pd.DataFrame, date_col: str, target: str
) -> Tuple[List[Dict[str, Any]], TargetResult]:
    series = pd.to_numeric(df[target], errors="coerce").dropna().to_numpy(dtype=float)
    split_idx = max(int(len(series) * (1 - VALID_RATIO)), 10)
    train_y, valid_y = series[:split_idx], series[split_idx:]
    eval_rows: List[Dict[str, Any]] = []
    valid_dates = pd.to_datetime(df[date_col]).iloc[split_idx:].reset_index(drop=True)
    plot_data = pd.DataFrame({"date": valid_dates, "actual": valid_y})

    mean_pred = np.repeat(np.mean(train_y), len(valid_y))
    eval_rows.append({"target": target, "model": "Historical Mean", **calc_metrics(valid_y, mean_pred)})
    plot_data["Historical Mean"] = mean_pred

    moving_value = float(pd.Series(train_y).tail(7).mean())
    mov_pred = np.repeat(moving_value, len(valid_y))
    eval_rows.append({"target": target, "model": "Moving Average", **calc_metrics(valid_y, mov_pred)})
    plot_data["Moving Average"] = mov_pred

    for stat_name in ["Simple Exponential Smoothing", "Croston"]:
        alpha, pred, metrics = choose_alpha(stat_name, train_y, valid_y)
        eval_rows.append({"target": target, "model": stat_name, "alpha": alpha, **metrics})
        plot_data[stat_name] = pred

    if STATSMODELS_AVAILABLE:
        try:
            order, pred, metrics = choose_arima(train_y, valid_y)
            eval_rows.append({"target": target, "model": "ARIMA", "order": str(order), **metrics})
            plot_data["ARIMA"] = pred
        except Exception as exc:
            eval_rows.append(
                {
                    "target": target,
                    "model": "ARIMA",
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "MAPE": np.nan,
                    "note": f"训练失败：{exc}",
                }
            )

    feature_df = make_feature_frame(df, date_col, target)
    feature_cols = [
        c for c in feature_df.columns if c not in {"date", "y"}
    ]
    split_date = pd.to_datetime(df[date_col]).iloc[split_idx]
    train_feat = feature_df[feature_df["date"] < split_date].copy()
    valid_feat = feature_df[feature_df["date"] >= split_date].copy()

    trained_models: Dict[str, Any] = {}
    final_model_candidates: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
    if len(train_feat) >= 20 and len(valid_feat) >= 3:
        X_train, y_train = train_feat[feature_cols], train_feat["y"]
        X_valid, y_valid = valid_feat[feature_cols], valid_feat["y"]
        try:
            tuned_model, tuned_params, tuned_pred, tuned_metrics = tune_knn(train_feat, feature_cols, valid_feat)
            eval_rows.append(
                {
                    "target": target,
                    "model": "KNN Optimized",
                    "n_neighbors": tuned_params["n_neighbors"],
                    "weights": tuned_params["weights"],
                    "p": tuned_params["p"],
                    **tuned_metrics,
                }
            )
            trained_models["KNN Optimized"] = tuned_model
            final_model_candidates["KNN Optimized"] = (tuned_model, tuned_params)
            aligned = pd.DataFrame({"date": valid_feat["date"].values, "KNN Optimized": tuned_pred})
            plot_data = plot_data.merge(aligned, on="date", how="left")
        except Exception as exc:
            eval_rows.append(
                {
                    "target": target,
                    "model": "KNN Optimized",
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "MAPE": np.nan,
                    "note": f"训练失败：{exc}",
                }
            )

        for name, model in make_models().items():
            try:
                model.fit(X_train, y_train)
                pred = np.asarray(model.predict(X_valid), dtype=float)
                pred = np.maximum(pred, 0)
                metrics = calc_metrics(y_valid, pred)
                eval_rows.append({"target": target, "model": name, **metrics})
                trained_models[name] = model
                aligned = pd.DataFrame({"date": valid_feat["date"].values, name: pred})
                plot_data = plot_data.merge(aligned, on="date", how="left")
            except Exception as exc:
                eval_rows.append(
                    {
                        "target": target,
                        "model": name,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "MAPE": np.nan,
                        "note": f"训练失败：{exc}",
                    }
                )

    eval_df = pd.DataFrame(eval_rows)
    best_row = eval_df.dropna(subset=["RMSE"]).sort_values(["RMSE", "MAE"]).iloc[0].to_dict()
    best_model = str(best_row["model"])
    best_params = {
        "alpha": best_row.get("alpha", np.nan),
        "order": best_row.get("order", np.nan),
        "n_neighbors": best_row.get("n_neighbors", np.nan),
        "weights": best_row.get("weights", np.nan),
        "p": best_row.get("p", np.nan),
    }

    final_model = None
    if best_model == "KNN Optimized" and len(feature_df) >= 20:
        params = {
            "n_neighbors": int(best_params["n_neighbors"]),
            "weights": str(best_params["weights"]),
            "p": int(best_params["p"]),
        }
        final_model = make_knn_pipeline(**params)
        final_model.fit(feature_df[feature_cols], feature_df["y"])
    elif best_model in make_models() and len(feature_df) >= 20:
        final_model = make_models()[best_model]
        final_model.fit(feature_df[feature_cols], feature_df["y"])

    if best_model == "Simple Exponential Smoothing":
        best_pred = ses_forecast(train_y, len(valid_y), float(best_params["alpha"]))
    elif best_model == "Croston":
        best_pred = croston_forecast(train_y, len(valid_y), float(best_params["alpha"]))
    elif best_model == "ARIMA" and best_model in plot_data.columns:
        best_params["order"] = best_row.get("order")
        best_pred = plot_data[best_model].ffill().bfill().to_numpy()
    elif best_model in plot_data.columns:
        best_pred = plot_data[best_model].ffill().bfill().to_numpy()
    elif best_model == "Historical Mean":
        best_pred = mean_pred
    else:
        best_pred = mov_pred
    residual_std = float(np.nanstd(valid_y - np.asarray(best_pred, dtype=float), ddof=1))
    if not np.isfinite(residual_std) or residual_std <= 0:
        residual_std = float(np.nanstd(valid_y, ddof=1) * 0.1)

    result = TargetResult(
        target=target,
        best_model=best_model,
        best_params=best_params,
        final_model=final_model,
        feature_cols=feature_cols,
        residual_std=residual_std,
        validation_plot_data=plot_data,
    )
    return eval_rows, result


def get_may_2025_workdays() -> pd.DatetimeIndex:
    all_days = pd.date_range("2025-05-01", "2025-05-31", freq="D")
    holidays = set(pd.date_range("2025-05-01", "2025-05-05", freq="D"))
    return pd.DatetimeIndex([d for d in all_days if d not in holidays and d.dayofweek < 5])


def make_future_feature_row(
    date: pd.Timestamp, start_date: pd.Timestamp, history_values: List[float]
) -> Dict[str, float]:
    row = add_date_features(date, start_date)
    s = pd.Series(history_values, dtype=float)
    for lag in [1, 3, 7]:
        row[f"lag_{lag}"] = float(s.iloc[-lag]) if len(s) >= lag else float(s.mean())
    for window in [3, 7]:
        tail = s.tail(window)
        row[f"rolling_mean_{window}"] = float(tail.mean())
        row[f"rolling_std_{window}"] = float(tail.std(ddof=1)) if len(tail) > 1 else 0.0
    return row


def recursive_forecast_target(
    df: pd.DataFrame,
    date_col: str,
    target: str,
    result: TargetResult,
    future_dates: pd.DatetimeIndex,
) -> Tuple[np.ndarray, pd.DataFrame]:
    y_hist = pd.to_numeric(df[target], errors="coerce").dropna().to_list()
    start_date = pd.to_datetime(df[date_col]).min()
    preds = []
    future_features = []
    for d in future_dates:
        row = make_future_feature_row(d, start_date, y_hist)
        feature_row = pd.DataFrame([row]).reindex(columns=result.feature_cols, fill_value=0.0)
        if result.best_model == "Historical Mean":
            pred = float(np.mean(y_hist))
        elif result.best_model == "Moving Average":
            pred = float(np.mean(y_hist[-7:]))
        elif result.best_model == "Simple Exponential Smoothing":
            alpha = float(result.best_params.get("alpha", 0.3))
            pred = float(ses_forecast(np.asarray(y_hist), 1, alpha)[0])
        elif result.best_model == "Croston":
            alpha = float(result.best_params.get("alpha", 0.3))
            pred = float(croston_forecast(np.asarray(y_hist), 1, alpha)[0])
        elif result.best_model == "ARIMA" and STATSMODELS_AVAILABLE and ARIMA is not None:
            raw_order = result.best_params.get("order")
            if isinstance(raw_order, str):
                nums = [int(x) for x in re.findall(r"\d+", raw_order)]
                order = tuple(nums[:3]) if len(nums) >= 3 else (1, 1, 1)
            else:
                order = (1, 1, 1)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = ARIMA(np.asarray(y_hist, dtype=float), order=order).fit()
                    pred = float(fit.forecast(steps=1)[0])
            except Exception:
                pred = float(np.mean(y_hist[-7:]))
        elif result.final_model is not None:
            pred = float(result.final_model.predict(feature_row)[0])
        else:
            pred = float(np.mean(y_hist[-7:]))
        pred = max(pred, 0.0)
        preds.append(pred)
        y_hist.append(pred)
        feature_row.insert(0, "date", d)
        future_features.append(feature_row)
    return np.asarray(preds, dtype=float), pd.concat(future_features, ignore_index=True)


def monte_carlo_interval(
    result: TargetResult,
    base_features: pd.DataFrame,
    point_pred: np.ndarray,
    n_sim: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    point_pred = np.asarray(point_pred, dtype=float)
    feature_matrix = base_features[result.feature_cols].to_numpy(dtype=float)
    if result.final_model is not None and result.best_model in {
        "XGBoost",
        "Random Forest",
        "ExtraTrees",
        "MLP",
        "Gradient Boosting",
        "AdaBoost",
        "LightGBM",
        "CatBoost",
        "SVR",
        "KNN",
        "KNN Optimized",
    }:
        scale = np.nanstd(feature_matrix, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale * 0.03, 0.01)
        sims = []
        columns = result.feature_cols
        for _ in range(n_sim):
            noise = rng.normal(0, scale, size=feature_matrix.shape)
            noisy = pd.DataFrame(feature_matrix + noise, columns=columns)
            pred = np.maximum(result.final_model.predict(noisy), 0)
            sims.append(pred)
        sim_arr = np.asarray(sims, dtype=float)
        mean = sim_arr.mean(axis=0)
        std = sim_arr.std(axis=0, ddof=1)
    else:
        std = np.repeat(max(result.residual_std, 1e-6), len(point_pred))
        mean = point_pred
    lower = np.maximum(mean - 1.96 * std, 0)
    upper = mean + 1.96 * std
    return mean, lower, upper


def write_xlsx_builtin(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    """无 openpyxl/xlsxwriter 时的简易 xlsx 写出器，支持多 sheet 和基础数据类型。"""

    def col_name(idx: int) -> str:
        name = ""
        idx += 1
        while idx:
            idx, rem = divmod(idx - 1, 26)
            name = chr(65 + rem) + name
        return name

    def cell_xml(value: Any, row: int, col: int) -> str:
        ref = f"{col_name(col)}{row}"
        if pd.isna(value):
            return f'<c r="{ref}"/>'
        if isinstance(value, pd.Timestamp):
            value = value.strftime("%Y-%m-%d")
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            return f'<c r="{ref}"><v>{float(value):.10g}</v></c>'
        text = escape(str(value))
        return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'

    def sheet_xml(df: pd.DataFrame) -> str:
        rows = []
        header = "".join(cell_xml(c, 1, j) for j, c in enumerate(df.columns))
        rows.append(f'<row r="1">{header}</row>')
        for i, (_, row) in enumerate(df.iterrows(), start=2):
            cells = "".join(cell_xml(row.iloc[j], i, j) for j in range(len(df.columns)))
            rows.append(f'<row r="{i}">{cells}</row>')
        dimension = f"A1:{col_name(max(len(df.columns) - 1, 0))}{max(len(df) + 1, 1)}"
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="{dimension}"/><sheetData>{"".join(rows)}</sheetData></worksheet>'
        )

    safe_names = []
    used = set()
    for name in sheets:
        clean = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(name))[:31] or "Sheet"
        base = clean
        n = 1
        while clean in used:
            suffix = f"_{n}"
            clean = (base[: 31 - len(suffix)] + suffix)[:31]
            n += 1
        used.add(clean)
        safe_names.append(clean)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        workbook_sheets = "".join(
            f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
            for i, name in enumerate(safe_names, start=1)
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{workbook_sheets}</sheets></workbook>",
        )
        rels = "".join(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(sheets) + 1)
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{rels}</Relationships>",
        )
        for i, df in enumerate(sheets.values(), start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(df.reset_index(drop=True)))


def save_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path) as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=str(name)[:31], index=False)
    except Exception as exc:
        print(f"[WARN] pandas ExcelWriter 不可用，启用内置 xlsx 写出器：{exc}")
        write_xlsx_builtin(path, sheets)


def plot_history(df: pd.DataFrame, date_col: str, customer_col: str, sales_col: str, nutrient_cols: List[str]) -> None:
    if plt is None:
        return
    for col, filename, title in [
        (customer_col, "daily_customer_count_history.png", "每日就餐人数历史趋势"),
        (sales_col, "daily_sales_amount_history.png", "每日销售总额历史趋势"),
    ]:
        plt.figure(figsize=(12, 5))
        plt.plot(df[date_col], df[col], linewidth=1.4)
        plt.title(title)
        plt.xlabel("日期")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=160)
        plt.close()

    if nutrient_cols:
        plt.figure(figsize=(12, 6))
        for col in nutrient_cols[:5]:
            values = pd.to_numeric(df[col], errors="coerce")
            normalized = values / values.max() if values.max() else values
            plt.plot(df[date_col], normalized, label=col, linewidth=1.2)
        plt.title("主要营养素需求历史趋势（归一化）")
        plt.xlabel("日期")
        plt.ylabel("归一化需求量")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "main_nutrient_demand_history.png", dpi=160)
        plt.close()


def plot_validation_and_metrics(eval_df: pd.DataFrame, target_results: Dict[str, TargetResult], key_targets: List[str]) -> None:
    if plt is None:
        return
    for target in key_targets[:3]:
        if target not in target_results:
            continue
        plot_data = target_results[target].validation_plot_data
        plt.figure(figsize=(12, 5))
        plt.plot(plot_data["date"], plot_data["actual"], label="真实值", color="black", linewidth=2)
        models_to_show = [c for c in plot_data.columns if c not in {"date", "actual"}][:6]
        for col in models_to_show:
            plt.plot(plot_data["date"], plot_data[col], label=col, alpha=0.8)
        plt.title(f"{target}：各模型验证集预测效果对比")
        plt.xlabel("日期")
        plt.ylabel(target)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"validation_prediction_compare_{target}.png", dpi=160)
        plt.close()

        sub = eval_df[eval_df["target"] == target].dropna(subset=["RMSE"]).sort_values("RMSE")
        plt.figure(figsize=(10, 5))
        plt.bar(sub["model"], sub["RMSE"])
        plt.title(f"{target}：各模型 RMSE 对比")
        plt.xlabel("模型")
        plt.ylabel("RMSE")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"model_rmse_compare_{target}.png", dpi=160)
        plt.close()


def plot_forecast(pred_df: pd.DataFrame, customer_col: str, sales_col: str, nutrient_cols: List[str]) -> None:
    if plt is None:
        return
    plt.figure(figsize=(12, 5))
    plt.plot(pred_df["日期"], pred_df[f"预测{customer_col}"], marker="o", label=customer_col)
    plt.title("2025年5月工作日就餐人数预测趋势")
    plt.xlabel("日期")
    plt.ylabel(customer_col)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_customer_forecast_trend.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(pred_df["日期"], pred_df[f"预测{sales_col}"], marker="o", color="#B45309", label=sales_col)
    plt.title("2025年5月工作日销售总额预测趋势")
    plt.xlabel("日期")
    plt.ylabel(sales_col)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_sales_forecast_trend.png", dpi=160)
    plt.close()

    if nutrient_cols:
        plt.figure(figsize=(12, 6))
        for col in nutrient_cols[:5]:
            y = pred_df[f"预测{col}"]
            normalized = y / y.max() if y.max() else y
            plt.plot(pred_df["日期"], normalized, marker="o", label=col)
        plt.title("2025年5月工作日主要营养素需求预测趋势（归一化）")
        plt.xlabel("日期")
        plt.ylabel("归一化预测量")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "final_nutrient_forecast_trend.png", dpi=160)
        plt.close()

    main = f"预测{customer_col}"
    low = f"{customer_col}_下界"
    high = f"{customer_col}_上界"
    if low in pred_df.columns and high in pred_df.columns:
        plt.figure(figsize=(12, 5))
        x = pd.to_datetime(pred_df["日期"])
        plt.plot(x, pred_df[main], marker="o", label="预测均值")
        plt.fill_between(x, pred_df[low], pred_df[high], alpha=0.25, label="95%预测区间")
        plt.title("Monte Carlo 不确定性分析：就餐人数预测区间")
        plt.xlabel("日期")
        plt.ylabel(customer_col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "monte_carlo_customer_interval.png", dpi=160)
        plt.close()


def format_weekday_cn(date: pd.Timestamp) -> str:
    names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return names[int(date.dayofweek)]


def make_summary_md(
    check: Dict[str, Any],
    main_file: str,
    target_cols: List[str],
    nutrient_cols: List[str],
    eval_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    skipped_models: List[str],
) -> str:
    def df_to_markdown(df: pd.DataFrame) -> str:
        show = df.copy()
        show.columns = [str(c) for c in show.columns]
        rows = [list(show.columns)]
        for _, row in show.iterrows():
            rows.append([row[c] for c in show.columns])

        def fmt(v: Any) -> str:
            if pd.isna(v):
                return ""
            if isinstance(v, (float, np.floating)):
                return f"{float(v):.3f}".rstrip("0").rstrip(".")
            return str(v)

        str_rows = [[fmt(v) for v in row] for row in rows]
        widths = [max(len(r[i]) for r in str_rows) for i in range(len(str_rows[0]))]
        lines = []
        header = "| " + " | ".join(str_rows[0][i].ljust(widths[i]) for i in range(len(widths))) + " |"
        sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
        lines.extend([header, sep])
        for row in str_rows[1:]:
            lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(widths))) + " |")
        return "\n".join(lines)

    top_eval = (
        eval_df.dropna(subset=["RMSE"])
        .sort_values(["target", "RMSE"])
        .groupby("target")
        .head(3)
        .copy()
    )
    eval_brief = df_to_markdown(top_eval[["target", "model", "MAE", "RMSE", "MAPE"]].round(3))
    selection_md = df_to_markdown(selection_df.round(3))
    table_cols = ["日期", "星期"] + [f"预测{c}" for c in target_cols]
    pred_table = pred_df[table_cols].copy()
    for col in pred_table.columns:
        if col.startswith("预测"):
            pred_table[col] = pred_table[col].map(lambda x: round(float(x), 2))
    pred_md = df_to_markdown(pred_table)
    skipped = "、".join(skipped_models) if skipped_models else "无"

    return f"""# 问题二：餐厅菜量需求预测与运营优化设计

## 一、问题二分析

问题二要求根据餐厅历史销售记录，对 2025 年 5 月份工作日的每日就餐人数、各类营养素需求量以及销售总额进行预测。该问题本质上是多目标时间序列预测问题：一方面需要刻画餐厅需求随星期、月份位置和近期消费水平变化的周期性，另一方面也要考虑餐饮销售中存在的随机波动和偶发性变化。

## 二、数据预处理

本研究按照题目给定的数据使用优先级，首先读取 `{main_file}` 作为主数据表。数据检查结果显示，该表共有 {check["rows"]} 行，日期范围为 {pd.to_datetime(check["date_min"]).date()} 至 {pd.to_datetime(check["date_max"]).date()}，列名为：{", ".join(map(str, check["columns"]))}。缺失值检查结果为：{check["missing"]}。表中已经包含日期、每日就餐人数、每日销售总额和每日营养素需求总量，因此可直接用于问题二预测建模，无需重新从原始 Excel 全量清洗。

## 三、预测指标构造

自动识别得到的预测目标为：{", ".join(target_cols)}。其中每日就餐人数由每日消费记录统计得到，销售总额为每日订单金额汇总，营养素需求量为菜品营养成分与销售量折算后的每日总需求。识别出的营养素需求字段为：{", ".join(nutrient_cols)}。

## 四、模型建立

参考餐饮需求预测研究《A comparative study of various statistical and machine learning models for predicting restaurant demand in Bangladesh》的建模思路，本文比较统计预测模型和机器学习模型。统计模型包括简单指数平滑法、Croston 间歇性需求预测法和 ARIMA；机器学习模型包括随机森林回归、MLP 多层感知机回归、XGBoost、LightGBM、CatBoost、SVR、KNN、KNN 参数优化、ExtraTrees 和 AdaBoost。为增强稳健性，本文还加入历史均值法、移动平均法、线性回归、岭回归、ElasticNet 和梯度提升树作为对照模型。

特征构造包括日期序号、星期几、是否周末、是否工作日、月份、是否月初、月中、月末，以及 lag_1、lag_3、lag_7 和 rolling_mean_3、rolling_mean_7、rolling_std_3、rolling_std_7 等历史序列特征。所有滞后和滚动特征均使用前一期及以前数据构造，避免使用未来信息。预测 2025 年 5 月工作日时，采用逐日递推方式更新历史序列。

环境中自动跳过的模型：{skipped}。

## 五、模型评价与选择

采用时间顺序划分训练集与验证集，最后约 {int(VALID_RATIO * 100)}% 的样本作为验证集，不进行随机打乱。评价指标包括 MAE、RMSE 和 MAPE，其中 MAPE 对真实值为 0 的样本进行了避零处理。每个预测目标分别选择验证集 RMSE 最小的模型作为最终预测模型。

各目标验证集排名靠前模型如下：

{eval_brief}

最终模型选择结果如下：

{selection_md}

## 六、预测结果可靠性分析

从时间顺序验证结果看，本文模型选择依据来自未来时段验证误差，能够更贴近真实预测场景。不同模型在多数指标上的误差存在差异，说明单一模型难以同时适配就餐人数、销售额和各类营养素需求，因此逐指标选模更合理。对机器学习模型，本文采用 Monte Carlo Simulation 对预测输入特征加入小幅高斯扰动，计算预测均值和 95% 区间；若最终模型为统计模型或简单基准模型，则使用验证集残差标准差近似给出预测区间。

需要注意的是，历史数据覆盖 2022 年 9 月至 2025 年 4 月，样本量能够支持日尺度预测，但餐厅消费仍会受节假日、学校教学安排、天气和临时活动影响。2025 年 5 月 1 日至 5 月 5 日被设定为劳动节假期，最终结果仅输出 5 月 6 日之后的普通工作日。因此预测结果适合作为经营计划和菜量备货的基准参考，实际运营中仍需结合临近日期销售反馈滚动修正。

## 七、预测结果表格

{pred_md}
"""


def main() -> None:
    ensure_dirs()
    setup_chinese_font()

    df, main_file, check = load_main_data()
    date_col = check["date_col"]
    target_cols = check["target_cols"]
    customer_col = check["mapping"]["customer"]
    sales_col = check["mapping"]["sales"]
    nutrient_cols = [c for c in target_cols if c not in {customer_col, sales_col}]

    print("========== 数据检查 ==========")
    print("当前目录文件列表：")
    for f in check["files"]:
        print(" -", f)
    print("daily_summary.csv 列名：", check["columns"])
    print("数据行数：", check["rows"])
    print("日期范围：", pd.to_datetime(check["date_min"]).date(), "至", pd.to_datetime(check["date_max"]).date())
    print("缺失值情况：", check["missing"])
    print("daily_summary.csv 是否可直接用于问题二建模：", "是" if check["usable"] else "否")
    print("自动识别预测目标列：", target_cols)

    plot_history(df, date_col, customer_col, sales_col, nutrient_cols)

    all_eval_rows: List[Dict[str, Any]] = []
    target_results: Dict[str, TargetResult] = {}
    for target in target_cols:
        if df[target].isna().all() or pd.to_numeric(df[target], errors="coerce").nunique() <= 1:
            print(f"[SKIP] {target} 全为空或无有效波动，跳过预测。")
            continue
        print(f"[INFO] 训练与评价目标：{target}")
        rows, result = train_and_evaluate_target(df, date_col, target)
        all_eval_rows.extend(rows)
        target_results[target] = result

    eval_df = pd.DataFrame(all_eval_rows)
    selection_df = pd.DataFrame(
        [
            {
                "预测指标": target,
                "最终模型": res.best_model,
                "验证残差标准差": res.residual_std,
            }
            for target, res in target_results.items()
        ]
    )

    future_dates = get_may_2025_workdays()
    pred_df = pd.DataFrame(
        {
            "日期": future_dates.strftime("%Y-%m-%d"),
            "星期": [format_weekday_cn(d) for d in future_dates],
        }
    )
    interval_sheets = {}
    for target, res in target_results.items():
        point, features = recursive_forecast_target(df, date_col, target, res, future_dates)
        mean, lower, upper = monte_carlo_interval(res, features, point, n_sim=1000)
        pred_df[f"预测{target}"] = mean
        pred_df[f"{target}_下界"] = lower
        pred_df[f"{target}_上界"] = upper
        interval_sheets[target[:20]] = pd.DataFrame(
            {
                "日期": future_dates.strftime("%Y-%m-%d"),
                "预测均值": mean,
                "预测下界": lower,
                "预测上界": upper,
                "最终模型": res.best_model,
            }
        )

    plot_validation_and_metrics(eval_df, target_results, [customer_col, sales_col] + nutrient_cols[:1])
    plot_forecast(pred_df, customer_col, sales_col, nutrient_cols)

    skipped_models = []
    if not XGBOOST_AVAILABLE:
        skipped_models.append("XGBoost（当前环境未安装 xgboost）")
    if not LIGHTGBM_AVAILABLE:
        skipped_models.append("LightGBM（当前环境未安装 lightgbm）")
    if not CATBOOST_AVAILABLE:
        skipped_models.append("CatBoost（当前环境未安装 catboost）")
    if not STATSMODELS_AVAILABLE:
        skipped_models.append("ARIMA（当前环境未安装 statsmodels）")
    if not SKLEARN_AVAILABLE:
        skipped_models.append("sklearn 系列模型（当前环境未安装 sklearn）")

    pred_path = Path("problem2_prediction_results.xlsx")
    eval_path = Path("problem2_model_evaluation.xlsx")
    summary_path = Path("problem2_summary.md")

    save_excel(
        pred_path,
        {
            "预测结果": pred_df,
            "模型选择": selection_df,
            **{f"区间_{k}": v for k, v in interval_sheets.items()},
        },
    )
    save_excel(
        eval_path,
        {
            "模型评价": eval_df,
            "最终模型选择": selection_df,
            "数据检查": pd.DataFrame(
                {
                    "项目": ["主数据文件", "数据行数", "日期范围", "目标列", "是否可直接建模"],
                    "结果": [
                        main_file,
                        check["rows"],
                        f"{pd.to_datetime(check['date_min']).date()} 至 {pd.to_datetime(check['date_max']).date()}",
                        ", ".join(target_cols),
                        "是" if check["usable"] else "否",
                    ],
                }
            ),
        },
    )
    summary_md = make_summary_md(
        check=check,
        main_file=main_file,
        target_cols=list(target_results.keys()),
        nutrient_cols=[c for c in nutrient_cols if c in target_results],
        eval_df=eval_df,
        selection_df=selection_df,
        pred_df=pred_df,
        skipped_models=skipped_models,
    )
    summary_path.write_text(summary_md, encoding="utf-8")

    required = [Path("problem2_all_methods.py"), pred_path, eval_path, summary_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"以下输出文件未生成：{missing}")

    print("\n========== 运行总结 ==========")
    print("使用主数据文件：", main_file)
    print("预测指标：", list(target_results.keys()))
    attempted = [
        "Historical Mean",
        "Moving Average",
        "Simple Exponential Smoothing",
        "Croston",
        "Linear Regression",
        "Ridge Regression",
        "ElasticNet",
        "SVR",
        "KNN",
        "KNN Optimized",
        "Gradient Boosting",
        "AdaBoost",
        "Random Forest",
        "ExtraTrees",
        "MLP",
    ]
    if XGBOOST_AVAILABLE:
        attempted.append("XGBoost")
    if LIGHTGBM_AVAILABLE:
        attempted.append("LightGBM")
    if CATBOOST_AVAILABLE:
        attempted.append("CatBoost")
    if STATSMODELS_AVAILABLE:
        attempted.append("ARIMA")
    print("尝试模型：", attempted)
    print("每个指标最终选择模型：")
    for _, row in selection_df.iterrows():
        print(f" - {row['预测指标']}: {row['最终模型']}")
    print("模型评价结果保存：", eval_path)
    print("预测结果保存：", pred_path)
    print("论文说明保存：", summary_path)
    print("图像输出目录：", FIG_DIR)
    print("2025 年 5 月工作日预测行数：", len(pred_df))


if __name__ == "__main__":
    main()
