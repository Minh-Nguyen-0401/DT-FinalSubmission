from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "PyTorch is required for modelling/lstm.py. Run it from the conda base "
        "environment that has torch installed, e.g. `conda activate base; python modelling/lstm.py`."
    ) from exc

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "statsmodels is required for seasonal_decompose features. Install statsmodels "
        "in the active environment or run from conda base."
    ) from exc

from utils import (
    FORECAST_END,
    FORECAST_GAP_DAYS,
    FORECAST_OUTPUT_DIR,
    FORECAST_START,
    FULL_FEATURE_DIR,
    LOCAL_TEST_START,
    REPORTS_DIR,
    raw_path,
    timestamp,
)


TARGETS = ["Revenue", "COGS"]
RANDOM_STATE = 42
CORE_FEATURES = {
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "sin_7",
    "cos_7",
    "sin_30_5",
    "cos_30_5",
    "sin_365_25",
    "cos_365_25",
    "revenue_lag_1",
    "revenue_lag_7",
    "revenue_lag_28",
    "revenue_lag_365",
    "revenue_roll_mean_28d_gap549",
    "revenue_roll_std_28d_gap549",
}


@dataclass
class FeatureSelectionResult:
    selected: list[str]
    vif_input: list[str]
    dropped_vif: list[tuple[str, float]]
    dropped_corr: list[tuple[str, float]]
    corr_scores: dict[str, float]


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict[str, float]]
    best_epoch: int


class SalesLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int = len(TARGETS),
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a gap-safe PyTorch LSTM forecast model.")
    parser.add_argument("--seq-len", type=int, default=56)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--vif-threshold", type=float, default=12.0)
    parser.add_argument("--corr-threshold", type=float, default=0.02)
    parser.add_argument("--min-features", type=int, default=40)
    parser.add_argument("--max-vif-drops", type=int, default=90)
    parser.add_argument("--seasonal-periods", type=int, nargs="+", default=[7, 365])
    parser.add_argument(
        "--valid-start",
        type=str,
        default=str(LOCAL_TEST_START.date()),
        help="First validation date for early stopping runs. Example: 2022-01-01.",
    )
    parser.add_argument("--no-refit", action="store_true", help="Forecast with the local-validation model.")
    parser.add_argument(
        "--no-valid",
        action="store_true",
        help="Skip validation and train one final model on all labelled rows through 2022.",
    )
    return parser.parse_args()


def _validation_start(args: argparse.Namespace) -> pd.Timestamp:
    return pd.Timestamp(args.valid_start)


def _set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_data() -> tuple[pd.DataFrame, list[str]]:
    matrix = pd.read_csv(FULL_FEATURE_DIR / "model_matrix.csv", parse_dates=["Date"])
    sales = pd.read_csv(raw_path("sales.csv"), parse_dates=["Date"])
    data = matrix.merge(sales, on="Date", how="left").sort_values("Date").reset_index(drop=True)
    feature_cols = [col for col in matrix.columns if col != "Date"]
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data, feature_cols


def _is_candidate_feature(col: str) -> bool:
    if col in CORE_FEATURES:
        return True
    if col.startswith(("sin_", "cos_")):
        return True
    if col in {
        "year",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "quarter",
        "day_of_year",
        "is_month_start",
        "is_month_end",
        "is_q3",
        "is_december",
    }:
        return True
    temporal_markers = ("promo_day", "days_to_next", "days_since", "progress_ratio")
    history_markers = ("revenue", "cogs", "sales", "gross_margin", "lag", "roll_", "decomp")
    group_prefixes = (
        "mkt_",
        "expected_",
        "exp_",
        "ops_",
        "prod_",
        "cm_",
        "lane_",
        "lead_",
    )
    aggregate_markers = (
        "_91d",
        "_28d",
        "_365d",
        "_7d",
        "share",
        "rate",
        "ratio",
        "mean",
        "std",
        "sum",
        "avg",
        "p90",
        "p95",
        "entropy",
        "idx",
        "per_",
        "customers",
        "orders",
    )
    if col.startswith(group_prefixes) and any(marker in col for marker in aggregate_markers):
        return True
    if any(marker in col for marker in temporal_markers):
        return True
    if any(marker in col for marker in history_markers):
        return True
    if col.startswith(("active_", "has_", "is_any_")):
        return True
    return False


def _candidate_features(feature_cols: list[str]) -> list[str]:
    candidates = [col for col in feature_cols if _is_candidate_feature(col)]
    return [col for col in feature_cols if col in set(candidates)]


def _numeric_train_matrix(train: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = train[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    X = X.dropna(axis=1, how="all")
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique > 1]
    medians = X.median(numeric_only=True).fillna(0.0)
    return X.fillna(medians)


def _target_correlation_scores(train: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for col in feature_cols:
        values = pd.to_numeric(train[col], errors="coerce")
        target_scores = []
        for target in TARGETS:
            corr = values.corr(train[target])
            if pd.notna(corr):
                target_scores.append(abs(float(corr)))
        scores[col] = max(target_scores) if target_scores else 0.0
    return scores


def _standardized_array(X: pd.DataFrame) -> np.ndarray:
    arr = X.to_numpy(dtype=np.float64)
    std = arr.std(axis=0)
    keep = std > 0
    arr[:, keep] = (arr[:, keep] - arr[:, keep].mean(axis=0)) / std[keep]
    arr[:, ~keep] = 0.0
    return arr


def _corr_matrix(arr: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _drop_near_duplicates(
    X: pd.DataFrame,
    corr_scores: dict[str, float],
    threshold: float = 0.999,
) -> tuple[pd.DataFrame, list[tuple[str, float]]]:
    cols = list(X.columns)
    if len(cols) < 2:
        return X, []

    corr = np.abs(_corr_matrix(_standardized_array(X)))
    dropped: list[tuple[str, float]] = []
    keep = np.ones(len(cols), dtype=bool)
    for i, left in enumerate(cols):
        if not keep[i]:
            continue
        for j in range(i + 1, len(cols)):
            right = cols[j]
            if not keep[j] or corr[i, j] < threshold:
                continue
            if left in CORE_FEATURES and right not in CORE_FEATURES:
                drop_idx = j
            elif right in CORE_FEATURES and left not in CORE_FEATURES:
                drop_idx = i
            else:
                left_score = corr_scores.get(left, 0.0)
                right_score = corr_scores.get(right, 0.0)
                drop_idx = i if left_score < right_score else j
            keep[drop_idx] = False
            dropped.append((cols[drop_idx], float(corr[i, j])))
            if drop_idx == i:
                break
    kept_cols = [col for col, col_keep in zip(cols, keep) if col_keep]
    return X[kept_cols], dropped


def _vif_values(X: pd.DataFrame) -> pd.Series:
    cols = list(X.columns)
    if len(cols) < 2:
        return pd.Series(1.0, index=cols)
    corr = _corr_matrix(_standardized_array(X))
    ridge = 1e-6
    try:
        inv_corr = np.linalg.inv(corr + np.eye(len(cols)) * ridge)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr + np.eye(len(cols)) * ridge)
    vif = np.diag(inv_corr)
    vif = np.nan_to_num(vif, nan=np.inf, posinf=np.inf, neginf=np.inf)
    return pd.Series(vif, index=cols)


def _filter_by_vif(
    train: pd.DataFrame,
    feature_cols: list[str],
    max_vif: float,
    min_features: int,
    max_drops: int,
) -> tuple[list[str], list[tuple[str, float]], dict[str, float]]:
    X = _numeric_train_matrix(train, feature_cols)
    corr_scores = _target_correlation_scores(train, list(X.columns))
    X, duplicate_drops = _drop_near_duplicates(X, corr_scores)

    dropped: list[tuple[str, float]] = [(name, value) for name, value in duplicate_drops]
    for _ in range(max_drops):
        if X.shape[1] <= min_features:
            break
        vif = _vif_values(X)
        removable = vif.drop(labels=[col for col in CORE_FEATURES if col in vif.index], errors="ignore")
        over_limit = removable[removable > max_vif]
        if over_limit.empty:
            break
        worst_vif = over_limit.max()
        tied = over_limit[over_limit >= worst_vif * 0.98].index.tolist()
        drop_col = min(tied, key=lambda col: corr_scores.get(col, 0.0))
        dropped.append((drop_col, float(vif[drop_col])))
        X = X.drop(columns=[drop_col])
    return list(X.columns), dropped, corr_scores


def _filter_by_target_correlation(
    train: pd.DataFrame,
    feature_cols: list[str],
    corr_scores: dict[str, float],
    threshold: float,
    min_features: int,
) -> tuple[list[str], list[tuple[str, float]]]:
    scores = {col: corr_scores.get(col, 0.0) for col in feature_cols}
    selected = [col for col in feature_cols if scores[col] >= threshold or col in CORE_FEATURES]
    if len(selected) < min_features:
        selected = [
            col
            for col, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:min_features]
        ]
        selected = sorted(set(selected) | (CORE_FEATURES & set(feature_cols)), key=feature_cols.index)
    dropped = [(col, scores[col]) for col in feature_cols if col not in set(selected)]
    return selected, dropped


def _select_features(
    data: pd.DataFrame,
    feature_cols: list[str],
    max_vif: float,
    corr_threshold: float,
    min_features: int,
    max_vif_drops: int,
    valid_start: pd.Timestamp,
    use_all_labelled: bool = False,
) -> FeatureSelectionResult:
    labelled = data[data[TARGETS].notna().all(axis=1)].copy()
    train = labelled.copy() if use_all_labelled else labelled[labelled["Date"] < valid_start].copy()
    candidates = _candidate_features(feature_cols)
    vif_cols, dropped_vif, corr_scores = _filter_by_vif(
        train=train,
        feature_cols=candidates,
        max_vif=max_vif,
        min_features=min_features,
        max_drops=max_vif_drops,
    )
    selected, dropped_corr = _filter_by_target_correlation(
        train=train,
        feature_cols=vif_cols,
        corr_scores=corr_scores,
        threshold=corr_threshold,
        min_features=min_features,
    )
    return FeatureSelectionResult(
        selected=selected,
        vif_input=candidates,
        dropped_vif=dropped_vif,
        dropped_corr=dropped_corr,
        corr_scores=corr_scores,
    )


def _last_valid(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def _seasonal_window(period: int) -> int:
    if period >= 365:
        return 3 * period
    return max(8 * period, 365)


def _decompose_one_history(history: pd.Series, period: int) -> tuple[float, float, float, float]:
    min_obs = 2 * period
    if len(history) < min_obs:
        return np.nan, np.nan, np.nan, np.nan
    result = seasonal_decompose(
        history.to_numpy(dtype=float),
        model="additive",
        period=period,
        extrapolate_trend="freq",
    )
    trend = pd.Series(result.trend)
    seasonal = pd.Series(result.seasonal)
    resid = pd.Series(result.resid)
    seasonal_lag = float(seasonal.iloc[-period]) if len(seasonal.dropna()) >= period else np.nan
    return _last_valid(trend), _last_valid(seasonal), _last_valid(resid), seasonal_lag


def _add_seasonal_decompose_features(data: pd.DataFrame, periods: list[int]) -> tuple[pd.DataFrame, list[str]]:
    out = data.copy()
    sales = pd.read_csv(raw_path("sales.csv"), parse_dates=["Date"]).sort_values("Date")
    sales = sales.set_index("Date")[TARGETS].asfreq("D")
    feature_names: list[str] = []

    for target in TARGETS:
        series = sales[target].astype(float)
        values = series.dropna()
        prefix = target.lower()
        for period in periods:
            names = [
                f"{prefix}_sd_trend_p{period}_gap{FORECAST_GAP_DAYS}",
                f"{prefix}_sd_seasonal_p{period}_gap{FORECAST_GAP_DAYS}",
                f"{prefix}_sd_resid_p{period}_gap{FORECAST_GAP_DAYS}",
                f"{prefix}_sd_seasonal_lag_p{period}_gap{FORECAST_GAP_DAYS}",
            ]
            feature_names.extend(names)
            rows: list[tuple[float, float, float, float]] = []
            window = _seasonal_window(period)
            print(f"Building seasonal_decompose features: target={target}, period={period}, window={window}")
            for date in out["Date"]:
                as_of_date = date - pd.Timedelta(days=FORECAST_GAP_DAYS)
                history = values.loc[:as_of_date].tail(window)
                rows.append(_decompose_one_history(history, period))
            values_frame = pd.DataFrame(rows, columns=names, index=out.index)
            out[names] = values_frame

    return out.replace([np.inf, -np.inf], np.nan), feature_names


def _first_usable_date(data: pd.DataFrame) -> pd.Timestamp:
    return data.loc[data[TARGETS].notna().all(axis=1), "Date"].min() + pd.Timedelta(
        days=FORECAST_GAP_DAYS
    )


def _prepare_feature_arrays(
    data: pd.DataFrame,
    feature_cols: list[str],
    fit_end_date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler, StandardScaler]:
    train_mask = (data["Date"] <= fit_end_date) & data[TARGETS].notna().all(axis=1)
    imputer = SimpleImputer(strategy="median")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    train_features = data.loc[train_mask, feature_cols].replace([np.inf, -np.inf], np.nan)
    imputer.fit(train_features)
    x_scaler.fit(imputer.transform(train_features))
    y_scaler.fit(data.loc[train_mask, TARGETS])

    features = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_all = x_scaler.transform(imputer.transform(features)).astype(np.float32)
    y_all = y_scaler.transform(data[TARGETS].fillna(0.0)).astype(np.float32)
    return X_all, y_all, imputer, x_scaler, y_scaler


def _make_sequences(
    X_all: np.ndarray,
    y_all: np.ndarray,
    end_indices: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_end_indices = end_indices[end_indices >= seq_len - 1]
    X_seq = np.stack([X_all[idx - seq_len + 1 : idx + 1] for idx in valid_end_indices]).astype(np.float32)
    y_seq = y_all[valid_end_indices].astype(np.float32)
    return X_seq, y_seq, valid_end_indices


def _train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> TrainResult:
    if len(X_train) < 20:
        raise ValueError("Not enough training sequences for LSTM.")
    if len(X_valid) < 1:
        raise ValueError("Validation sequence split is empty.")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_x = torch.from_numpy(X_valid).to(device)
    val_y = torch.from_numpy(y_valid).to(device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)

    model = SalesLSTM(
        input_size=X_train.shape[2],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss()

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y).detach().cpu())
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"Epoch {epoch:03d}/{args.epochs} - "
            f"train_loss={train_loss:.6f} - valid_loss={val_loss:.6f}"
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    return TrainResult(model=model, history=history, best_epoch=best_epoch)


def _fit_lstm_fixed_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
) -> TrainResult:
    if len(X_train) < 20:
        raise ValueError("Not enough training sequences for final LSTM refit.")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    model = SalesLSTM(
        input_size=X_train.shape[2],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": np.nan})
        print(f"Final refit epoch {epoch:03d}/{epochs} - train_loss={train_loss:.6f}")
    return TrainResult(model=model, history=history, best_epoch=epochs)


def _predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            preds.append(model(batch_x.to(device)).detach().cpu().numpy())
    return np.vstack(preds)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for i, target in enumerate(TARGETS):
        metrics[target] = {
            "MAE": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "RMSE": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "R2": float(r2_score(y_true[:, i], y_pred[:, i])),
        }
    return metrics


def _write_report(
    report_path: Path,
    data: pd.DataFrame,
    selection: FeatureSelectionResult,
    seasonal_cols: list[str],
    final_feature_cols: list[str],
    metrics: dict[str, dict[str, float]],
    train_result: TrainResult,
    args: argparse.Namespace,
    device: torch.device,
    forecast_path: Path,
    valid_start: pd.Timestamp,
) -> None:
    labelled = data[(data["Date"] >= _first_usable_date(data)) & data[TARGETS].notna().all(axis=1)]
    if args.no_valid:
        train = labelled
        valid = pd.DataFrame(columns=labelled.columns)
    else:
        train = labelled[labelled["Date"] < valid_start]
        valid = labelled[labelled["Date"] >= valid_start]

    lines = [
        "# PyTorch LSTM Forecast",
        "",
        f"- Run time: `{timestamp()}`",
        f"- Device: `{device}`",
        f"- Training mode: `{'full_train_no_validation' if args.no_valid else 'train_with_validation_early_stop'}`",
        f"- Train range: `{train['Date'].min().date()}` to `{train['Date'].max().date()}`",
        f"- Train rows: `{len(train)}`",
        f"- Sequence length: `{args.seq_len}`",
        f"- Max epochs: `{args.epochs}`",
        f"- Early stopping patience: `{args.patience}`",
        f"- Validation start: `{valid_start.date() if not args.no_valid else 'not used'}`",
        f"- Candidate features from model_matrix: `{len(selection.vif_input)}`",
        f"- Dropped by VIF/near-duplicate: `{len(selection.dropped_vif)}`",
        f"- Dropped by near-zero target correlation: `{len(selection.dropped_corr)}`",
        f"- seasonal_decompose features added after filters: `{len(seasonal_cols)}`",
        f"- Final feature count: `{len(final_feature_cols)}`",
        f"- Best epoch: `{train_result.best_epoch}`",
        f"- Forecast file: `{forecast_path}`",
        "",
    ]
    if args.no_valid:
        lines.extend(["- Validation range: `not used`", "- Validation rows: `0`"])
    else:
        lines.extend(
            [
                f"- Validation range: `{valid['Date'].min().date()}` to `{valid['Date'].max().date()}`",
                f"- Validation rows: `{len(valid)}`",
            ]
        )

    if metrics:
        lines.extend(["", "## Validation Metrics", "", "| Target | MAE | RMSE | R2 |", "|---|---:|---:|---:|"])
        for target, vals in metrics.items():
            lines.append(
                f"| {target} | {vals['MAE']:,.4f} | {vals['RMSE']:,.4f} | {vals['R2']:,.6f} |"
            )
    else:
        lines.extend(["", "## Validation Metrics", "", "Validation was skipped for this full-train run."])

    lines.extend(
        [
            "",
            "## Training",
            "",
            "| Epoch | Train Loss | Validation Loss |",
            "|---:|---:|---:|",
        ]
    )
    for row in train_result.history:
        lines.append(f"| {row['epoch']:.0f} | {row['train_loss']:.6f} | {row['val_loss']:.6f} |")

    lines.extend(["", "## Added seasonal_decompose Features", ""])
    for col in seasonal_cols:
        lines.append(f"- `{col}`")

    lines.extend(["", "## Selected Features", ""])
    top_features = sorted(
        final_feature_cols,
        key=lambda col: selection.corr_scores.get(col, 0.0),
        reverse=True,
    )
    for col in top_features[:80]:
        lines.append(f"- `{col}`")

    if selection.dropped_vif:
        lines.extend(["", "## Top VIF Drops", "", "| Feature | VIF/Corr |", "|---|---:|"])
        for col, value in selection.dropped_vif[:40]:
            lines.append(f"| `{col}` | {value:,.4f} |")

    if selection.dropped_corr:
        lines.extend(["", "## Near-Zero Correlation Drops", "", "| Feature | Max Abs Corr |", "|---|---:|"])
        for col, value in selection.dropped_corr[:40]:
            lines.append(f"| `{col}` | {value:,.6f} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    valid_start = _validation_start(args)
    _set_seed()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data, matrix_feature_cols = _load_model_data()
    first_usable_date = _first_usable_date(data)
    forecast_mask = (data["Date"] >= FORECAST_START) & (data["Date"] <= FORECAST_END)
    if forecast_mask.sum() != 548:
        raise ValueError(f"Forecast horizon must contain 548 rows, got {forecast_mask.sum()}.")

    selection = _select_features(
        data=data,
        feature_cols=matrix_feature_cols,
        max_vif=args.vif_threshold,
        corr_threshold=args.corr_threshold,
        min_features=args.min_features,
        max_vif_drops=args.max_vif_drops,
        valid_start=valid_start,
        use_all_labelled=args.no_valid,
    )
    print(
        "Feature filters: "
        f"candidates={len(selection.vif_input)}, "
        f"after_vif={len(selection.vif_input) - len(selection.dropped_vif)}, "
        f"after_corr={len(selection.selected)}"
    )

    data, seasonal_cols = _add_seasonal_decompose_features(data, args.seasonal_periods)
    final_feature_cols = selection.selected + seasonal_cols
    final_feature_cols = [col for col in final_feature_cols if col in data.columns]
    labelled_mask = (data["Date"] >= first_usable_date) & data[TARGETS].notna().all(axis=1)
    labelled_indices = np.flatnonzero(labelled_mask.to_numpy())

    if args.no_valid:
        all_fit_end = data.loc[data[TARGETS].notna().all(axis=1), "Date"].max()
        X_all, y_all, _, _, y_scaler = _prepare_feature_arrays(data, final_feature_cols, all_fit_end)
        X_seq, y_seq, _ = _make_sequences(X_all, y_all, labelled_indices, args.seq_len)
        print(f"Training final model on all labelled rows through {all_fit_end.date()} for {args.epochs} epochs.")
        train_result = _fit_lstm_fixed_epochs(X_seq, y_seq, args, device, args.epochs)
        metrics: dict[str, dict[str, float]] = {}
        forecast_model = train_result.model
        forecast_y_scaler = y_scaler
        forecast_X_all = X_all
    else:
        local_fit_end = valid_start - pd.Timedelta(days=1)
        X_all, y_all, _, _, y_scaler = _prepare_feature_arrays(data, final_feature_cols, local_fit_end)
        X_seq, y_seq, end_indices = _make_sequences(X_all, y_all, labelled_indices, args.seq_len)
        end_dates = data.loc[end_indices, "Date"].reset_index(drop=True)

        train_mask = end_dates < valid_start
        valid_mask = end_dates >= valid_start
        if not train_mask.any() or not valid_mask.any():
            raise ValueError("Train or validation sequence split is empty.")

        train_result = _train_lstm(
            X_seq[train_mask],
            y_seq[train_mask],
            X_seq[valid_mask],
            y_seq[valid_mask],
            args,
            device,
        )
        pred_scaled = _predict(train_result.model, X_seq[valid_mask], device, args.batch_size)
        pred = np.clip(y_scaler.inverse_transform(pred_scaled), 0, None)
        truth = y_scaler.inverse_transform(y_seq[valid_mask])
        metrics = _evaluate(truth, pred)
        for target, vals in metrics.items():
            print(f"{target}: {vals}")

        if args.no_refit:
            forecast_model = train_result.model
            forecast_y_scaler = y_scaler
            forecast_X_all = X_all
        else:
            all_fit_end = data.loc[data[TARGETS].notna().all(axis=1), "Date"].max()
            forecast_X_all, forecast_y_all, _, _, forecast_y_scaler = _prepare_feature_arrays(
                data, final_feature_cols, all_fit_end
            )
            all_labelled_indices = np.flatnonzero(labelled_mask.to_numpy())
            all_X_seq, all_y_seq, _ = _make_sequences(
                forecast_X_all, forecast_y_all, all_labelled_indices, args.seq_len
            )
            final_epochs = max(1, train_result.best_epoch)
            print(f"Refitting final forecast model on all labelled rows for {final_epochs} epochs.")
            forecast_train_result = _fit_lstm_fixed_epochs(all_X_seq, all_y_seq, args, device, final_epochs)
            forecast_model = forecast_train_result.model
            forecast_X_all = forecast_X_all

    forecast_indices = np.flatnonzero(forecast_mask.to_numpy())
    forecast_X_seq, _, forecast_end_indices = _make_sequences(
        forecast_X_all,
        np.zeros((len(data), len(TARGETS)), dtype=np.float32),
        forecast_indices,
        args.seq_len,
    )
    forecast_scaled = _predict(forecast_model, forecast_X_seq, device, args.batch_size)
    forecast_values = np.clip(forecast_y_scaler.inverse_transform(forecast_scaled), 0, None)

    output = pd.DataFrame({"Date": data.loc[forecast_end_indices, "Date"].dt.strftime("%Y-%m-%d")})
    for i, target in enumerate(TARGETS):
        output[target] = forecast_values[:, i]

    run_ts = timestamp()
    forecast_path = FORECAST_OUTPUT_DIR / f"lstm_forecast_{run_ts}.csv"
    report_path = REPORTS_DIR / f"lstm_{run_ts}.md"
    output.to_csv(forecast_path, index=False)
    _write_report(
        report_path=report_path,
        data=data,
        selection=selection,
        seasonal_cols=seasonal_cols,
        final_feature_cols=final_feature_cols,
        metrics=metrics,
        train_result=train_result,
        args=args,
        device=device,
        forecast_path=forecast_path,
        valid_start=valid_start,
    )

    print(f"Saved LSTM forecast: {forecast_path}")
    print(f"Saved LSTM report: {report_path}")
    print(output.head().to_string(index=False))


if __name__ == "__main__":
    main()
