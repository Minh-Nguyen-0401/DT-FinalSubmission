from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from utils import (
    FORECAST_GAP_DAYS,
    FORECAST_END,
    FORECAST_OUTPUT_DIR,
    FORECAST_START,
    FULL_FEATURE_DIR,
    raw_path,
    timestamp,
)


TARGETS = ["Revenue", "COGS"]


def _load_model_data() -> tuple[pd.DataFrame, list[str]]:
    matrix = pd.read_csv(FULL_FEATURE_DIR / "model_matrix.csv", parse_dates=["Date"])
    sales = pd.read_csv(raw_path("sales.csv"), parse_dates=["Date"])
    data = matrix.merge(sales, on="Date", how="left")
    feature_cols = [col for col in matrix.columns if col != "Date"]
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data, feature_cols


def main() -> None:
    FORECAST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, feature_cols = _load_model_data()
    first_usable_date = data.loc[data[TARGETS].notna().all(axis=1), "Date"].min() + pd.Timedelta(
        days=FORECAST_GAP_DAYS
    )
    train = data[(data[TARGETS].notna().all(axis=1)) & (data["Date"] >= first_usable_date)].copy()
    forecast = data[(data["Date"] >= FORECAST_START) & (data["Date"] <= FORECAST_END)].copy()

    if train.empty:
        raise ValueError("No labelled train rows available.")
    if len(forecast) != 548:
        raise ValueError(f"Forecast horizon must contain 548 rows, got {len(forecast)}.")

    output = pd.DataFrame({"Date": forecast["Date"].dt.strftime("%Y-%m-%d")})
    for target in TARGETS:
        model = LGBMRegressor(random_state=42, verbosity=-1)
        model.fit(train[feature_cols], train[target])
        output[target] = np.clip(model.predict(forecast[feature_cols]), a_min=0, a_max=None)

    output_path = FORECAST_OUTPUT_DIR / f"lightgbm_baseline_forecast_{timestamp()}.csv"
    output.to_csv(output_path, index=False)
    print(f"Saved forecast: {output_path}")
    print(output.head().to_string(index=False))


if __name__ == "__main__":
    main()
