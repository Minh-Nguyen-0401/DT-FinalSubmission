from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from feature_engineering._cust_market import build_customer_market_features
from feature_engineering._ops import build_ops_features
from feature_engineering._prod import build_product_features
from feature_engineering._promo import build_promo_mkt_features
from utils import (
    FULL_FEATURE_DIR,
    FORECAST_FEATURE_DIR,
    clean_feature_frame,
    ensure_dirs,
    get_full_date_index,
    make_daily_frame,
    merge_feature_frames,
    read_raw_csv,
    save_feature_group,
)


def build_calendar_features():
    date_index = get_full_date_index()
    out = make_daily_frame(date_index)
    dt = out["Date"]
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["day_of_week"] = dt.dt.dayofweek
    out["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    out["quarter"] = dt.dt.quarter
    out["day_of_year"] = dt.dt.dayofyear
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    out["is_month_start"] = dt.dt.is_month_start.astype(int)
    out["is_month_end"] = dt.dt.is_month_end.astype(int)
    out["is_q3"] = (out["quarter"] == 3).astype(int)
    out["is_december"] = (out["month"] == 12).astype(int)
    out["sin_7"] = __import__("numpy").sin(2 * __import__("numpy").pi * out["day_of_week"] / 7)
    out["cos_7"] = __import__("numpy").cos(2 * __import__("numpy").pi * out["day_of_week"] / 7)
    out["sin_30_5"] = __import__("numpy").sin(2 * __import__("numpy").pi * out["day"] / 30.5)
    out["cos_30_5"] = __import__("numpy").cos(2 * __import__("numpy").pi * out["day"] / 30.5)
    out["sin_365_25"] = __import__("numpy").sin(2 * __import__("numpy").pi * out["day_of_year"] / 365.25)
    out["cos_365_25"] = __import__("numpy").cos(2 * __import__("numpy").pi * out["day_of_year"] / 365.25)
    out["sin_365"] = __import__("numpy").sin(2 * __import__("numpy").pi * out["day_of_year"] / 365.25)
    out["cos_365"] = __import__("numpy").cos(2 * __import__("numpy").pi * out["day_of_year"] / 365.25)
    return out


def build_sales_history_features():
    import numpy as np
    import pandas as pd
    from pmdarima.arima import decompose

    date_index = get_full_date_index()
    sales = read_raw_csv("sales.csv", parse_dates=["Date"]).set_index("Date").reindex(date_index)
    revenue = sales["Revenue"]
    cogs = sales["COGS"]
    margin_pct = (revenue - cogs).divide(revenue.replace(0, np.nan))

    from utils import FORECAST_GAP_DAYS, shifted_rolling_mean, shifted_rolling_std

    out = make_daily_frame(date_index)
    for lag in [1, 7, 14, 28, 56, 365, 366]:
        out[f"revenue_lag_{lag}"] = revenue.shift(FORECAST_GAP_DAYS + lag).values
    for lag in [7, 28, 365]:
        out[f"revenue_lag_{lag}_gap549"] = revenue.shift(FORECAST_GAP_DAYS + lag).values
    for lag in [7, 365]:
        out[f"cogs_lag_{lag}"] = cogs.shift(FORECAST_GAP_DAYS + lag).values
        out[f"cogs_lag_{lag}_gap549"] = cogs.shift(FORECAST_GAP_DAYS + lag).values

    for window in [7, 14, 28, 56, 91, 365]:
        out[f"revenue_roll_mean_{window}"] = shifted_rolling_mean(revenue, window).values
        out[f"revenue_roll_std_{window}"] = shifted_rolling_std(revenue, window).values
    for window in [28, 91, 365]:
        out[f"revenue_roll_mean_{window}d_gap549"] = shifted_rolling_mean(revenue, window).values
        out[f"revenue_roll_std_{window}d_gap549"] = shifted_rolling_std(revenue, window).values
        out[f"cogs_roll_mean_{window}d_gap549"] = shifted_rolling_mean(cogs, window).values
        out[f"cogs_roll_std_{window}d_gap549"] = shifted_rolling_std(cogs, window).values

    out["cogs_roll_mean_28"] = shifted_rolling_mean(cogs, 28).values
    out["gross_margin_lag_365"] = margin_pct.shift(FORECAST_GAP_DAYS + 365).values
    out["revenue_same_day_ly_365"] = revenue.shift(FORECAST_GAP_DAYS + 365).values
    out["revenue_same_day_ly_366"] = revenue.shift(FORECAST_GAP_DAYS + 366).values
    out["gross_margin_pct_lag_365_gap549"] = margin_pct.shift(FORECAST_GAP_DAYS + 365).values
    out["gross_margin_pct_roll_mean_91d_gap549"] = shifted_rolling_mean(margin_pct, 91).values

    def causal_decompose_features(
        series: pd.Series,
        prefix: str,
        m: int,
        window: int,
    ) -> dict[str, list[float]]:
        values = series.dropna()
        features = {
            f"{prefix}_decomp_trend_m{m}_gap549": [],
            f"{prefix}_decomp_seasonal_m{m}_gap549": [],
            f"{prefix}_decomp_random_m{m}_gap549": [],
        }
        min_obs = max(2 * m, m + 3)
        for target_date in date_index:
            as_of_date = target_date - pd.Timedelta(days=FORECAST_GAP_DAYS)
            history = values.loc[:as_of_date].tail(window)
            if len(history) < min_obs:
                for key in features:
                    features[key].append(np.nan)
                continue

            component = decompose(history.to_numpy(dtype=float), "additive", m=m)
            trend = pd.Series(component.trend).dropna()
            seasonal = pd.Series(component.seasonal).dropna()
            random = pd.Series(component.random).dropna()
            features[f"{prefix}_decomp_trend_m{m}_gap549"].append(
                float(trend.iloc[-1]) if not trend.empty else np.nan
            )
            features[f"{prefix}_decomp_seasonal_m{m}_gap549"].append(
                float(seasonal.iloc[-1]) if not seasonal.empty else np.nan
            )
            features[f"{prefix}_decomp_random_m{m}_gap549"].append(
                float(random.iloc[-1]) if not random.empty else np.nan
            )
        return features

    for feature, values in causal_decompose_features(revenue, "revenue", m=7, window=365).items():
        out[feature] = values
    for feature, values in causal_decompose_features(cogs, "cogs", m=7, window=365).items():
        out[feature] = values
    for feature, values in causal_decompose_features(revenue, "revenue", m=365, window=1095).items():
        out[feature] = values
    for feature, values in causal_decompose_features(cogs, "cogs", m=365, window=1095).items():
        out[feature] = values
    return out.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    ensure_dirs()
    frames = {
        "calendar": build_calendar_features(),
        "sales_history": build_sales_history_features(),
        "promo_mkt": build_promo_mkt_features(),
        "ops": build_ops_features(),
        "product": build_product_features(),
        "customer_market": build_customer_market_features(),
    }

    for name, frame in frames.items():
        full_path, forecast_path = save_feature_group(frame, name)
        print(f"Saved {name}: {full_path}")
        print(f"Saved {name} forecast: {forecast_path}")

    model_matrix = merge_feature_frames(list(frames.values()))
    model_matrix = clean_feature_frame(model_matrix)
    full_matrix = FULL_FEATURE_DIR / "model_matrix.csv"
    forecast_matrix = FORECAST_FEATURE_DIR / "model_matrix.csv"
    model_matrix.set_index("Date").to_csv(full_matrix, index_label="Date")
    forecast = model_matrix[
        (model_matrix["Date"] >= "2023-01-01") & (model_matrix["Date"] <= "2024-07-01")
    ]
    forecast.set_index("Date").to_csv(forecast_matrix, index_label="Date")
    print(f"Saved model matrix: {full_matrix} rows={len(model_matrix)} cols={model_matrix.shape[1]}")
    print(f"Saved forecast model matrix: {forecast_matrix} rows={len(forecast)}")


if __name__ == "__main__":
    main()
