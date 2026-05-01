"""Hàm đọc dữ liệu và tạo bảng tóm tắt."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from EDA.constants.config import DATA_DIR, DATE_COLUMNS, TABLE_FILES


def table_name_from_file(file_name: str) -> str:
    """Chuyển tên file CSV thành tên bảng nội bộ."""
    return file_name.removesuffix(".csv")


def load_table(name: str, *, low_memory: bool = False) -> pd.DataFrame:
    """Đọc một bảng CSV và parse các cột ngày đã biết."""
    table_name = table_name_from_file(name)
    file_name = TABLE_FILES.get(table_name, name if name.endswith(".csv") else f"{name}.csv")
    parse_dates = DATE_COLUMNS.get(table_name, [])
    return pd.read_csv(DATA_DIR / file_name, parse_dates=parse_dates, low_memory=low_memory)


def load_tables(names: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
    """Đọc nhiều bảng CSV, trả về dict theo tên bảng."""
    selected = list(names) if names is not None else list(TABLE_FILES)
    return {table_name_from_file(name): load_table(name) for name in selected}


def summarize_dataframe(df: pd.DataFrame, table_name: str) -> dict[str, object]:
    """Tóm tắt nhanh số dòng, số cột, missing và duplicate."""
    return {
        "bảng": table_name,
        "số dòng": len(df),
        "số cột": df.shape[1],
        "ô thiếu": int(df.isna().sum().sum()),
        "dòng trùng": int(df.duplicated().sum()),
        "dung lượng MB": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }


def build_order_item_metrics(order_items: pd.DataFrame) -> pd.DataFrame:
    """Tính metric doanh thu ở cấp dòng sản phẩm."""
    fact = order_items.copy()
    fact["gross_sales"] = fact["quantity"] * fact["unit_price"]
    fact["net_sales"] = fact["gross_sales"] - fact["discount_amount"]
    fact["discount_rate"] = fact["discount_amount"] / fact["gross_sales"].replace(0, pd.NA)
    fact["has_promo"] = fact["promo_id"].notna()
    fact["has_second_promo"] = fact["promo_id_2"].notna()
    return fact


def build_order_metrics(order_items: pd.DataFrame) -> pd.DataFrame:
    """Gom các metric dòng sản phẩm về cấp đơn hàng."""
    item_fact = build_order_item_metrics(order_items)
    return (
        item_fact.groupby("order_id", as_index=False)
        .agg(
            gross_sales=("gross_sales", "sum"),
            net_sales=("net_sales", "sum"),
            discount_amount=("discount_amount", "sum"),
            units=("quantity", "sum"),
            item_lines=("product_id", "count"),
            promo_lines=("has_promo", "sum"),
        )
        .assign(
            discount_rate=lambda d: d["discount_amount"] / d["gross_sales"].replace(0, pd.NA),
            has_promo=lambda d: d["promo_lines"] > 0,
        )
    )

