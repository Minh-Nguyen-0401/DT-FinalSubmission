from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from modelling import lstm
from utils import LOCAL_TEST_START, REPORTS_DIR, timestamp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for the best LSTM run.")
    parser.add_argument("--valid-start", type=str, default=str(LOCAL_TEST_START.date()))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--background-size", type=int, default=64)
    parser.add_argument("--explain-size", type=int, default=96)
    parser.add_argument("--target", choices=lstm.TARGETS, default="Revenue")
    return parser.parse_args()


class TargetModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, target_index: int) -> None:
        super().__init__()
        self.model = model
        self.target_index = target_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[:, self.target_index : self.target_index + 1]


def _make_lstm_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        seq_len=56,
        epochs=args.epochs,
        batch_size=64,
        hidden_size=96,
        num_layers=2,
        dropout=0.20,
        lr=1e-3,
        weight_decay=1e-5,
        patience=args.patience,
        vif_threshold=12.0,
        corr_threshold=0.02,
        min_features=40,
        max_vif_drops=90,
        seasonal_periods=[7, 365],
        no_refit=True,
        no_valid=False,
        valid_start=args.valid_start,
    )


def _top_shap_table(
    shap_values: np.ndarray,
    feature_cols: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    # shap_values shape: samples x sequence_length x features
    mean_abs = np.abs(shap_values).mean(axis=(0, 1))
    signed_mean = shap_values.mean(axis=(0, 1))
    table = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "mean_abs_shap": mean_abs,
                "mean_shap": signed_mean,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    table["rank"] = np.arange(1, len(table) + 1)
    return table[["rank", "feature", "mean_abs_shap", "mean_shap"]]


def _aggregate_sequence_shap(shap_values: np.ndarray) -> np.ndarray:
    # Collapse the 56 sequence steps into one contribution per logical feature.
    return shap_values.sum(axis=1)


def _plot_top_features(table: pd.DataFrame, path: Path, target: str) -> None:
    plot_table = table.sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(7.0, 5.0))
    plt.barh(plot_table["feature"], plot_table["mean_abs_shap"], color="#2f6f73")
    plt.title(f"LSTM SHAP Feature Importance - {target}")
    plt.xlabel("Mean |SHAP| across validation samples and sequence steps")
    plt.ylabel("")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_beeswarm(
    shap_values_2d: np.ndarray,
    feature_values_2d: pd.DataFrame,
    path: Path,
    target: str,
    max_display: int = 15,
) -> None:
    plt.figure(figsize=(9.0, 6.2))
    shap.summary_plot(
        shap_values_2d,
        feature_values_2d,
        feature_names=list(feature_values_2d.columns),
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title(f"LSTM SHAP Beeswarm - {target}", fontsize=12)
    plt.xlabel("SHAP value aggregated across 56-day sequence")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = _parse_args()
    run_args = _make_lstm_args(args)
    valid_start = lstm._validation_start(run_args)
    lstm._set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, matrix_feature_cols = lstm._load_model_data()
    first_usable_date = lstm._first_usable_date(data)

    selection = lstm._select_features(
        data=data,
        feature_cols=matrix_feature_cols,
        max_vif=run_args.vif_threshold,
        corr_threshold=run_args.corr_threshold,
        min_features=run_args.min_features,
        max_vif_drops=run_args.max_vif_drops,
        valid_start=valid_start,
        use_all_labelled=False,
    )
    data, seasonal_cols = lstm._add_seasonal_decompose_features(data, run_args.seasonal_periods)
    final_feature_cols = [col for col in selection.selected + seasonal_cols if col in data.columns]

    fit_end = valid_start - pd.Timedelta(days=1)
    X_all, y_all, _, _, y_scaler = lstm._prepare_feature_arrays(data, final_feature_cols, fit_end)
    labelled_mask = (data["Date"] >= first_usable_date) & data[lstm.TARGETS].notna().all(axis=1)
    labelled_indices = np.flatnonzero(labelled_mask.to_numpy())
    X_seq, y_seq, end_indices = lstm._make_sequences(X_all, y_all, labelled_indices, run_args.seq_len)
    end_dates = data.loc[end_indices, "Date"].reset_index(drop=True)

    train_mask = end_dates < valid_start
    valid_mask = end_dates >= valid_start
    train_result = lstm._train_lstm(
        X_seq[train_mask],
        y_seq[train_mask],
        X_seq[valid_mask],
        y_seq[valid_mask],
        run_args,
        device,
    )

    pred_scaled = lstm._predict(train_result.model, X_seq[valid_mask], device, run_args.batch_size)
    pred = np.clip(y_scaler.inverse_transform(pred_scaled), 0, None)
    truth = y_scaler.inverse_transform(y_seq[valid_mask])
    metrics = lstm._evaluate(truth, pred)

    train_result.model.eval()
    target_index = lstm.TARGETS.index(args.target)
    target_model = TargetModel(train_result.model.to("cpu"), target_index).eval()

    rng = np.random.default_rng(lstm.RANDOM_STATE)
    train_positions = np.flatnonzero(train_mask)
    valid_positions = np.flatnonzero(valid_mask)
    background_positions = rng.choice(
        train_positions,
        size=min(args.background_size, len(train_positions)),
        replace=False,
    )
    explain_positions = np.linspace(0, len(valid_positions) - 1, min(args.explain_size, len(valid_positions)))
    explain_positions = valid_positions[np.unique(explain_positions.astype(int))]

    background = torch.from_numpy(X_seq[background_positions]).float()
    explain = torch.from_numpy(X_seq[explain_positions]).float()
    explainer = shap.GradientExplainer(target_model, background)
    shap_values = explainer.shap_values(explain)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 4:
        shap_values = shap_values[..., 0]

    table = _top_shap_table(shap_values, final_feature_cols)
    shap_values_2d = _aggregate_sequence_shap(shap_values)
    raw_feature_values = data.loc[end_indices[explain_positions], final_feature_cols].copy()
    raw_feature_values = raw_feature_values.replace([np.inf, -np.inf], np.nan)
    raw_feature_values = raw_feature_values.fillna(raw_feature_values.median(numeric_only=True)).fillna(0.0)

    run_ts = timestamp()
    figure_path = REPORTS_DIR / "figures" / f"lstm_shap_{args.target.lower()}_{run_ts}.png"
    beeswarm_path = REPORTS_DIR / "figures" / f"lstm_shap_beeswarm_{args.target.lower()}_{run_ts}.png"
    table_path = REPORTS_DIR / f"lstm_shap_{args.target.lower()}_{run_ts}.csv"
    _plot_top_features(table, figure_path, args.target)
    _plot_beeswarm(shap_values_2d, raw_feature_values, beeswarm_path, args.target)
    table.to_csv(table_path, index=False)

    print(f"Validation start: {valid_start.date()}")
    print(f"Best epoch: {train_result.best_epoch}")
    for target, vals in metrics.items():
        print(f"{target}: {vals}")
    print(f"Saved SHAP table: {table_path}")
    print(f"Saved SHAP figure: {figure_path}")
    print(f"Saved SHAP beeswarm: {beeswarm_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
