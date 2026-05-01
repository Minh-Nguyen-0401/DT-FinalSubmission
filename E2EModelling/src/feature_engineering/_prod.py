from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (
    add_missing_columns,
    get_full_date_index,
    make_daily_frame,
    read_raw_csv,
    rolling_entropy_from_counts,
    safe_divide,
    shifted_rolling_ratio,
    shifted_rolling_sum,
    valid_orders,
)


WINDOW = 91


def _prepare_product_fact() -> pd.DataFrame:
    orders = valid_orders(read_raw_csv("orders.csv", parse_dates=["order_date"]))
    items = read_raw_csv("order_items.csv")
    products = read_raw_csv("products.csv")

    price_cutoff = products["price"].quantile(0.75)
    products = products.assign(is_high_price=(products["price"] >= price_cutoff).astype(int))

    fact = items.merge(orders[["order_id", "order_date"]], on="order_id", how="inner")
    fact = fact.merge(
        products[["product_id", "category", "segment", "price", "cogs", "is_high_price"]],
        on="product_id",
        how="left",
    )
    fact["discount_amount"] = fact["discount_amount"].fillna(0)
    fact["gross_sales"] = fact["quantity"] * fact["unit_price"]
    fact["net_revenue"] = fact["gross_sales"] - fact["discount_amount"]
    fact["cogs_total"] = fact["quantity"] * fact["cogs"]
    fact["gross_margin"] = fact["net_revenue"] - fact["cogs_total"]
    return fact


def build_product_features() -> pd.DataFrame:
    date_index = get_full_date_index()
    fact = _prepare_product_fact()
    fact["order_date"] = pd.to_datetime(fact["order_date"])

    fact["high_price_revenue"] = fact["net_revenue"] * fact["is_high_price"].fillna(0)
    fact["high_price_units"] = fact["quantity"] * fact["is_high_price"].fillna(0)

    category_revenue = (
        fact.pivot_table(index="order_date", columns="category", values="net_revenue", aggfunc="sum")
        .reindex(date_index)
        .fillna(0)
    )
    category_revenue = add_missing_columns(
        category_revenue, ["Casual", "GenZ", "Outdoor", "Streetwear"]
    )
    segment_revenue = (
        fact.pivot_table(index="order_date", columns="segment", values="net_revenue", aggfunc="sum")
        .reindex(date_index)
        .fillna(0)
    )
    segment_revenue = add_missing_columns(
        segment_revenue, ["Balanced", "Everyday", "Performance", "Premium"]
    )

    daily = fact.groupby("order_date").agg(
        prod_revenue=("net_revenue", "sum"),
        prod_gross_sales=("gross_sales", "sum"),
        prod_discounts=("discount_amount", "sum"),
        prod_units=("quantity", "sum"),
        prod_orders=("order_id", "nunique"),
        prod_margin=("gross_margin", "sum"),
        prod_high_price_revenue=("high_price_revenue", "sum"),
        prod_high_price_units=("high_price_units", "sum"),
    )
    daily = daily.reindex(date_index).fillna(0)

    out = make_daily_frame(date_index)
    out["prod_avg_unit_price_91d_gap549"] = safe_divide(
        shifted_rolling_sum(daily["prod_revenue"], WINDOW),
        shifted_rolling_sum(daily["prod_units"], WINDOW),
    ).values
    out["prod_units_per_order_91d_gap549"] = safe_divide(
        shifted_rolling_sum(daily["prod_units"], WINDOW),
        shifted_rolling_sum(daily["prod_orders"], WINDOW),
    ).values
    out["prod_margin_pct_91d_gap549"] = safe_divide(
        shifted_rolling_sum(daily["prod_margin"], WINDOW),
        shifted_rolling_sum(daily["prod_revenue"], WINDOW),
    ).values
    out["prod_discount_rate_91d_gap549"] = safe_divide(
        shifted_rolling_sum(daily["prod_discounts"], WINDOW),
        shifted_rolling_sum(daily["prod_gross_sales"], WINDOW),
    ).values
    out["prod_high_price_revenue_share_91d_gap549"] = shifted_rolling_ratio(
        daily["prod_high_price_revenue"], daily["prod_revenue"], WINDOW
    ).values
    out["prod_high_price_unit_share_91d_gap549"] = shifted_rolling_ratio(
        daily["prod_high_price_units"], daily["prod_units"], WINDOW
    ).values

    for category in ["Casual", "GenZ", "Outdoor", "Streetwear"]:
        out[f"prod_{category.lower()}_revenue_share_91d_gap549"] = shifted_rolling_ratio(
            category_revenue[category], daily["prod_revenue"], WINDOW
        ).values
        out[f"exp_category_share_{category.lower()}"] = shifted_rolling_ratio(
            category_revenue[category], daily["prod_revenue"], WINDOW
        ).values

    for segment in ["Balanced", "Everyday", "Performance", "Premium"]:
        out[f"prod_{segment.lower()}_segment_revenue_share_91d_gap549"] = shifted_rolling_ratio(
            segment_revenue[segment], daily["prod_revenue"], WINDOW
        ).values

    out["prod_category_entropy_91d_gap549"] = rolling_entropy_from_counts(
        category_revenue[["Casual", "GenZ", "Outdoor", "Streetwear"]], WINDOW
    ).values
    return out.replace([np.inf, -np.inf], np.nan)
