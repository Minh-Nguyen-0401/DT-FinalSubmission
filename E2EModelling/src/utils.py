from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SRC_ROOT = Path(__file__).resolve().parent
E2E_ROOT = SRC_ROOT.parent
PROJECT_ROOT = E2E_ROOT.parent
RAW_DATA_DIR = PROJECT_ROOT / "datathon-2026-round-1"

ENGINEERED_DIR = SRC_ROOT / "data" / "engineered_feats"
FULL_FEATURE_DIR = ENGINEERED_DIR / "full"
FORECAST_FEATURE_DIR = ENGINEERED_DIR / "forecast"
FORECAST_OUTPUT_DIR = SRC_ROOT / "data" / "forecasts"
REPORTS_DIR = SRC_ROOT / "reports"
LOGS_DIR = SRC_ROOT / "logs"

FORECAST_GAP_DAYS = 549
LOCAL_TEST_START = pd.Timestamp("2021-07-01")
FORECAST_START = pd.Timestamp("2023-01-01")
FORECAST_END = pd.Timestamp("2024-07-01")
EXCLUDED_DEMAND_STATUSES = {"cancelled", "canceled"}


def ensure_dirs() -> None:
    for path in [
        FULL_FEATURE_DIR,
        FORECAST_FEATURE_DIR,
        FORECAST_OUTPUT_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def timestamp_seconds() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def raw_path(filename: str) -> Path:
    return RAW_DATA_DIR / filename


def read_raw_csv(filename: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(raw_path(filename), **kwargs)


def get_full_date_index() -> pd.DatetimeIndex:
    sales = read_raw_csv("sales.csv", usecols=["Date"], parse_dates=["Date"])
    return pd.date_range(sales["Date"].min(), FORECAST_END, freq="D")


def forecast_date_index() -> pd.DatetimeIndex:
    return pd.date_range(FORECAST_START, FORECAST_END, freq="D")


def valid_orders(orders: pd.DataFrame) -> pd.DataFrame:
    status = orders["order_status"].astype(str).str.lower()
    return orders.loc[~status.isin(EXCLUDED_DEMAND_STATUSES)].copy()


def clean_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").drop_duplicates("Date")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def save_feature_group(df: pd.DataFrame, group_name: str) -> tuple[Path, Path]:
    ensure_dirs()
    out = clean_feature_frame(df)
    full_path = FULL_FEATURE_DIR / f"{group_name}_features.csv"
    forecast_path = FORECAST_FEATURE_DIR / f"{group_name}_features.csv"

    out.set_index("Date").to_csv(full_path, index_label="Date")
    forecast = out[(out["Date"] >= FORECAST_START) & (out["Date"] <= FORECAST_END)]
    forecast.set_index("Date").to_csv(forecast_path, index_label="Date")
    return full_path, forecast_path


def make_daily_frame(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"Date": date_index})


def daily_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    date_index: pd.DatetimeIndex,
    agg: str = "sum",
) -> pd.Series:
    tmp = df[[date_col, value_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    grouped = tmp.groupby(date_col)[value_col].agg(agg)
    return grouped.reindex(date_index)


def safe_divide(numerator, denominator):
    numerator = pd.Series(numerator)
    denominator = pd.Series(denominator)
    return numerator.divide(denominator.replace(0, np.nan))


def shifted(series: pd.Series, gap_days: int = FORECAST_GAP_DAYS) -> pd.Series:
    return series.shift(gap_days)


def shifted_rolling_sum(
    series: pd.Series,
    window: int,
    gap_days: int = FORECAST_GAP_DAYS,
    min_periods: int = 1,
) -> pd.Series:
    return series.shift(gap_days).rolling(window, min_periods=min_periods).sum()


def shifted_rolling_mean(
    series: pd.Series,
    window: int,
    gap_days: int = FORECAST_GAP_DAYS,
    min_periods: int = 1,
) -> pd.Series:
    return series.shift(gap_days).rolling(window, min_periods=min_periods).mean()


def shifted_rolling_std(
    series: pd.Series,
    window: int,
    gap_days: int = FORECAST_GAP_DAYS,
    min_periods: int = 2,
) -> pd.Series:
    return series.shift(gap_days).rolling(window, min_periods=min_periods).std()


def shifted_rolling_quantile(
    series: pd.Series,
    window: int,
    quantile: float,
    gap_days: int = FORECAST_GAP_DAYS,
    min_periods: int = 1,
) -> pd.Series:
    return series.shift(gap_days).rolling(window, min_periods=min_periods).quantile(quantile)


def shifted_rolling_max(
    series: pd.Series,
    window: int,
    gap_days: int = FORECAST_GAP_DAYS,
    min_periods: int = 1,
) -> pd.Series:
    return series.shift(gap_days).rolling(window, min_periods=min_periods).max()


def shifted_rolling_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    window: int,
    gap_days: int = FORECAST_GAP_DAYS,
    min_periods: int = 1,
) -> pd.Series:
    num = numerator.shift(gap_days).rolling(window, min_periods=min_periods).sum()
    den = denominator.shift(gap_days).rolling(window, min_periods=min_periods).sum()
    return safe_divide(num, den)


def rolling_entropy_from_counts(
    counts: pd.DataFrame,
    window: int,
    gap_days: int = FORECAST_GAP_DAYS,
    min_periods: int = 1,
) -> pd.Series:
    rolled = counts.shift(gap_days).rolling(window, min_periods=min_periods).sum()
    shares = rolled.div(rolled.sum(axis=1).replace(0, np.nan), axis=0)
    entropy = -(shares * np.log(shares.replace(0, np.nan))).sum(axis=1)
    return entropy


def add_missing_columns(df: pd.DataFrame, columns: Iterable[str], fill_value: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = fill_value
    return out


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No feature frames provided.")
    merged = clean_feature_frame(frames[0])
    for frame in frames[1:]:
        merged = merged.merge(clean_feature_frame(frame), on="Date", how="left")
    return merged.replace([np.inf, -np.inf], np.nan)
