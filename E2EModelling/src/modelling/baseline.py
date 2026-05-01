from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import FORECAST_GAP_DAYS, FULL_FEATURE_DIR, LOCAL_TEST_START, REPORTS_DIR, raw_path, timestamp


TARGETS = ["Revenue", "COGS"]


def _load_model_data() -> tuple[pd.DataFrame, list[str]]:
    matrix = pd.read_csv(FULL_FEATURE_DIR / "model_matrix.csv", parse_dates=["Date"])
    sales = pd.read_csv(raw_path("sales.csv"), parse_dates=["Date"])
    data = matrix.merge(sales, on="Date", how="left")
    feature_cols = [col for col in matrix.columns if col != "Date"]
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data, feature_cols


def _fit_model(X: pd.DataFrame, y: pd.Series) -> LGBMRegressor:
    model = LGBMRegressor(random_state=42, verbosity=-1)
    model.fit(X, y)
    return model


def _evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _importance_table(model: LGBMRegressor, feature_cols: list[str], top_n: int = 25) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )


def _write_report(
    report_path: Path,
    data: pd.DataFrame,
    feature_cols: list[str],
    metrics: dict[str, dict[str, float]],
    importances: dict[str, pd.DataFrame],
) -> None:
    train = data[(data["Date"] < LOCAL_TEST_START) & data[TARGETS].notna().all(axis=1)]
    test = data[(data["Date"] >= LOCAL_TEST_START) & data[TARGETS].notna().all(axis=1)]
    lines = [
        "# LightGBM Baseline Local Validation",
        "",
        f"- Run time: `{timestamp()}`",
        f"- Feature rows: `{len(data)}`",
        f"- Feature count: `{len(feature_cols)}`",
        f"- Train range: `{train['Date'].min().date()}` to `{train['Date'].max().date()}`",
        f"- Local test range: `{test['Date'].min().date()}` to `{test['Date'].max().date()}`",
        f"- Train rows: `{len(train)}`",
        f"- Local test rows: `{len(test)}`",
        "",
        "## Metrics",
        "",
        "| Target | MAE | RMSE | R2 |",
        "|---|---:|---:|---:|",
    ]
    for target, vals in metrics.items():
        lines.append(
            f"| {target} | {vals['MAE']:,.4f} | {vals['RMSE']:,.4f} | {vals['R2']:,.6f} |"
        )

    for target, table in importances.items():
        lines.extend(["", f"## Top Feature Importance - {target}", "", "| Feature | Importance |", "|---|---:|"])
        for row in table.itertuples(index=False):
            lines.append(f"| `{row.feature}` | {row.importance:,.0f} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data, feature_cols = _load_model_data()
    first_usable_date = data.loc[data[TARGETS].notna().all(axis=1), "Date"].min() + pd.Timedelta(
        days=FORECAST_GAP_DAYS
    )
    labelled = data[(data[TARGETS].notna().all(axis=1)) & (data["Date"] >= first_usable_date)].copy()
    train = labelled[labelled["Date"] < LOCAL_TEST_START]
    test = labelled[labelled["Date"] >= LOCAL_TEST_START]

    if train.empty or test.empty:
        raise ValueError("Train or local test split is empty.")

    metrics: dict[str, dict[str, float]] = {}
    importances: dict[str, pd.DataFrame] = {}
    for target in TARGETS:
        model = _fit_model(train[feature_cols], train[target])
        pred = model.predict(test[feature_cols])
        metrics[target] = _evaluate(test[target], pred)
        importances[target] = _importance_table(model, feature_cols)
        print(f"{target}: {metrics[target]}")

    report_path = REPORTS_DIR / f"baseline_{timestamp()}.md"
    _write_report(report_path, labelled, feature_cols, metrics, importances)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
