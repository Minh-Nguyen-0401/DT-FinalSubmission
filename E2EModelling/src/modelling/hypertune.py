from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.model_selection import TimeSeriesSplit

from utils import (
    FORECAST_END,
    FORECAST_GAP_DAYS,
    FORECAST_OUTPUT_DIR,
    FORECAST_START,
    FULL_FEATURE_DIR,
    REPORTS_DIR,
    raw_path,
    timestamp,
)


TARGETS = ["Revenue", "COGS"]
N_ITER = 30
N_SPLITS = 5
RANDOM_STATE = 42


SEARCH_SPACE = {
    "n_estimators": Integer(300, 1400),
    "learning_rate": Real(0.01, 0.10, prior="log-uniform"),
    "num_leaves": Integer(15, 127),
    "max_depth": Categorical([-1, 4, 6, 8, 10, 12]),
    "min_child_samples": Integer(10, 140),
    "subsample": Real(0.65, 1.0),
    "colsample_bytree": Real(0.65, 1.0),
    "reg_alpha": Real(1e-4, 5.0, prior="log-uniform"),
    "reg_lambda": Real(1e-4, 10.0, prior="log-uniform"),
}


def _load_model_data() -> tuple[pd.DataFrame, list[str]]:
    matrix = pd.read_csv(FULL_FEATURE_DIR / "model_matrix.csv", parse_dates=["Date"])
    sales = pd.read_csv(raw_path("sales.csv"), parse_dates=["Date"])
    data = matrix.merge(sales, on="Date", how="left")
    feature_cols = [col for col in matrix.columns if col != "Date"]
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data, feature_cols


def _train_rows(data: pd.DataFrame) -> pd.DataFrame:
    first_usable_date = data.loc[data[TARGETS].notna().all(axis=1), "Date"].min() + pd.Timedelta(
        days=FORECAST_GAP_DAYS
    )
    return data[(data[TARGETS].notna().all(axis=1)) & (data["Date"] >= first_usable_date)].copy()


def _forecast_rows(data: pd.DataFrame) -> pd.DataFrame:
    return data[(data["Date"] >= FORECAST_START) & (data["Date"] <= FORECAST_END)].copy()


def _tune_target(X: pd.DataFrame, y: pd.Series, target: str) -> BayesSearchCV:
    estimator = LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=SEARCH_SPACE,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=False,
    )
    print(
        f"Bayes tuning {target}: rows={len(X)}, features={X.shape[1]}, "
        f"cv_splits={N_SPLITS}, n_iter={N_ITER}"
    )
    search.fit(X, y)
    print(f"{target} best CV RMSE: {-search.best_score_:,.4f}")
    print(f"{target} best params: {search.best_params_}")
    return search


def _importance_table(model: LGBMRegressor, feature_cols: list[str], top_n: int = 30) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )


def _write_report(
    report_path: Path,
    train: pd.DataFrame,
    feature_cols: list[str],
    searches: dict[str, BayesSearchCV],
) -> None:
    lines = [
        "# LightGBM BayesSearchCV TimeSeriesSplit",
        "",
        f"- Run time: `{timestamp()}`",
        f"- Train range: `{train['Date'].min().date()}` to `{train['Date'].max().date()}`",
        f"- Train rows: `{len(train)}`",
        f"- Feature count: `{len(feature_cols)}`",
        f"- CV: `TimeSeriesSplit(n_splits={N_SPLITS})`",
        f"- Search: `BayesSearchCV(n_iter={N_ITER}, scoring='neg_root_mean_squared_error')`",
        "",
        "## Best CV Scores",
        "",
        "| Target | Best CV RMSE |",
        "|---|---:|",
    ]
    for target, search in searches.items():
        lines.append(f"| {target} | {-search.best_score_:,.4f} |")

    for target, search in searches.items():
        lines.extend(["", f"## Best Params - {target}", "", "```text"])
        for key, value in sorted(search.best_params_.items()):
            lines.append(f"{key}: {value}")
        lines.extend(["```", "", f"## Top Feature Importance - {target}", "", "| Feature | Importance |", "|---|---:|"])
        table = _importance_table(search.best_estimator_, feature_cols)
        for row in table.itertuples(index=False):
            lines.append(f"| `{row.feature}` | {row.importance:,.0f} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FORECAST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data, feature_cols = _load_model_data()
    train = _train_rows(data)
    forecast = _forecast_rows(data)

    if train.empty:
        raise ValueError("No labelled train rows available after 549-day cutoff.")
    if len(forecast) != 548:
        raise ValueError(f"Forecast horizon must contain 548 rows, got {len(forecast)}.")

    X = train[feature_cols]
    searches: dict[str, BayesSearchCV] = {}
    output = pd.DataFrame({"Date": forecast["Date"].dt.strftime("%Y-%m-%d")})

    for target in TARGETS:
        search = _tune_target(X, train[target], target)
        searches[target] = search
        output[target] = np.clip(search.best_estimator_.predict(forecast[feature_cols]), 0, None)

    run_ts = timestamp()
    forecast_path = FORECAST_OUTPUT_DIR / f"lightgbm_tuned_forecast_{run_ts}.csv"
    report_path = REPORTS_DIR / f"hypertune_{run_ts}.md"
    output.to_csv(forecast_path, index=False)
    _write_report(report_path, train, feature_cols, searches)

    print(f"Saved tuned forecast: {forecast_path}")
    print(f"Saved hypertune report: {report_path}")
    print(output.head().to_string(index=False))


if __name__ == "__main__":
    main()
