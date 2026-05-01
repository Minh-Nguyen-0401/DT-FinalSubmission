"""Helper trực quan hóa dùng chung cho EDA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from EDA.constants.config import FIGURES_DIR


def setup_vietnamese_style() -> None:
    """Thiết lập style biểu đồ hỗ trợ tiếng Việt có dấu."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def money_formatter(value: float, _: int | None = None) -> str:
    """Định dạng tiền theo đơn vị triệu/tỷ để trục dễ đọc."""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f} tỷ"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f} tr"
    if abs_value >= 1_000:
        return f"{value / 1_000:.0f} nghìn"
    return f"{value:.0f}"


def percent_formatter(value: float, _: int | None = None) -> str:
    """Định dạng tỷ lệ phần trăm."""
    return f"{value:.0%}"


def format_money_axis(axis) -> None:
    """Áp dụng định dạng tiền cho trục y."""
    axis.yaxis.set_major_formatter(FuncFormatter(money_formatter))


def format_percent_axis(axis) -> None:
    """Áp dụng định dạng phần trăm cho trục y."""
    axis.yaxis.set_major_formatter(FuncFormatter(percent_formatter))


def save_figure(name: str, fig=None, folder: Path = FIGURES_DIR) -> Path:
    """Lưu biểu đồ hiện tại vào thư mục figures."""
    folder.mkdir(parents=True, exist_ok=True)
    figure = fig or plt.gcf()
    path = folder / f"{name}.png"
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    return path


def annotate_bars(axis, fmt: str = "{:.0f}", rotation: int = 0) -> None:
    """Gắn nhãn số lên bar chart."""
    for container in axis.containers:
        labels = []
        for value in container.datavalues:
            if pd.isna(value):
                labels.append("")
            elif "%" in fmt:
                labels.append(fmt.format(value))
            else:
                labels.append(fmt.format(value))
        axis.bar_label(container, labels=labels, padding=3, fontsize=8, rotation=rotation)

