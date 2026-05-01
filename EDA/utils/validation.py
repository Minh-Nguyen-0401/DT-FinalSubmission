"""Kiểm tra chất lượng dữ liệu và quan hệ khóa."""

from __future__ import annotations

import pandas as pd

from EDA.utils.io import summarize_dataframe


def quality_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tạo bảng chất lượng dữ liệu cơ bản cho nhiều bảng."""
    return pd.DataFrame(
        [summarize_dataframe(df, name) for name, df in tables.items()]
    ).sort_values("bảng")


def primary_key_report(df: pd.DataFrame, key: str | list[str], table_name: str) -> dict[str, object]:
    """Kiểm tra độ duy nhất của khóa chính."""
    keys = [key] if isinstance(key, str) else key
    duplicated = int(df.duplicated(keys).sum())
    return {
        "bảng": table_name,
        "khóa": ", ".join(keys),
        "số dòng": len(df),
        "số khóa duy nhất": int(df.drop_duplicates(keys).shape[0]),
        "dòng trùng khóa": duplicated,
        "hợp lệ": duplicated == 0,
    }


def foreign_key_report(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_key: str,
    right_key: str,
    left_name: str,
    right_name: str,
) -> dict[str, object]:
    """Kiểm tra số khóa ở bảng trái không có trong bảng phải."""
    missing_keys = set(left[left_key].dropna().unique()) - set(right[right_key].dropna().unique())
    return {
        "quan hệ": f"{left_name}.{left_key} → {right_name}.{right_key}",
        "số khóa thiếu": len(missing_keys),
        "tỷ lệ dòng bị ảnh hưởng": round(float(left[left_key].isin(missing_keys).mean()), 6),
        "hợp lệ": len(missing_keys) == 0,
    }

