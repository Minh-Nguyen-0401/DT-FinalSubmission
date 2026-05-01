from __future__ import annotations

from collections import Counter, deque

import numpy as np
import pandas as pd

from utils import (
    FORECAST_GAP_DAYS,
    add_missing_columns,
    get_full_date_index,
    make_daily_frame,
    read_raw_csv,
    rolling_entropy_from_counts,
    safe_divide,
    shifted_rolling_mean,
    shifted_rolling_ratio,
    shifted_rolling_sum,
    valid_orders,
)


WINDOW = 91


def _rolling_unique_count(daily_sets: pd.Series, window: int) -> pd.Series:
    counts: Counter = Counter()
    queue: deque[set] = deque()
    values = []
    for customers in daily_sets:
        customers = set() if customers is np.nan else set(customers)
        queue.append(customers)
        counts.update(customers)
        if len(queue) > window:
            old = queue.popleft()
            for customer_id in old:
                counts[customer_id] -= 1
                if counts[customer_id] <= 0:
                    del counts[customer_id]
        values.append(len(counts))
    return pd.Series(values, index=daily_sets.index)


def _prepare_order_fact() -> pd.DataFrame:
    orders = read_raw_csv("orders.csv", parse_dates=["order_date"])
    customers = read_raw_csv("customers.csv", parse_dates=["signup_date"])
    geography = read_raw_csv("geography.csv")
    items = read_raw_csv("order_items.csv")

    valid = valid_orders(orders)
    item_revenue = items.assign(
        discount_amount=items["discount_amount"].fillna(0),
        gross_sales=items["quantity"] * items["unit_price"],
    )
    item_revenue["net_revenue"] = item_revenue["gross_sales"] - item_revenue["discount_amount"]
    order_revenue = item_revenue.groupby("order_id", as_index=False)["net_revenue"].sum()

    fact = valid.merge(order_revenue, on="order_id", how="left")
    fact = fact.merge(
        customers[
            [
                "customer_id",
                "gender",
                "age_group",
                "acquisition_channel",
                "signup_date",
            ]
        ],
        on="customer_id",
        how="left",
    )
    fact = fact.merge(geography[["zip", "region"]], on="zip", how="left")
    fact = fact.sort_values(["customer_id", "order_date", "order_id"])
    fact["customer_order_no"] = fact.groupby("customer_id").cumcount() + 1
    fact["previous_order_date"] = fact.groupby("customer_id")["order_date"].shift(1)
    fact["inter_order_gap_days"] = (fact["order_date"] - fact["previous_order_date"]).dt.days
    fact["first_order_date"] = fact.groupby("customer_id")["order_date"].transform("min")
    fact["customer_tenure_days"] = (fact["order_date"] - fact["first_order_date"]).dt.days
    fact["is_new_customer_order"] = (fact["customer_order_no"] == 1).astype(int)
    fact["is_repeat_order"] = (fact["customer_order_no"] > 1).astype(int)
    fact["is_loyal_order"] = (fact["customer_order_no"] >= 3).astype(int)
    fact["is_repeat_within_365d"] = (fact["inter_order_gap_days"] <= 365).fillna(False).astype(int)
    fact["age_18_34"] = fact["age_group"].isin(["18-24", "25-34"]).astype(int)
    fact["age_35_54"] = fact["age_group"].isin(["35-44", "45-54"]).astype(int)
    fact["age_55_plus"] = (fact["age_group"] == "55+").astype(int)
    fact["is_female"] = (fact["gender"] == "Female").astype(int)
    fact["is_mobile"] = (fact["device_type"] == "mobile").astype(int)
    fact["is_paid_search_acq"] = (fact["acquisition_channel"] == "paid_search").astype(int)
    fact["is_organic_search_acq"] = (fact["acquisition_channel"] == "organic_search").astype(int)
    fact["is_east_region"] = (fact["region"] == "East").astype(int)
    return fact


def build_customer_market_features() -> pd.DataFrame:
    date_index = get_full_date_index()
    fact = _prepare_order_fact()
    fact["order_date"] = pd.to_datetime(fact["order_date"])

    daily = fact.groupby("order_date").agg(
        cm_orders=("order_id", "nunique"),
        cm_revenue=("net_revenue", "sum"),
        cm_new_orders=("is_new_customer_order", "sum"),
        cm_repeat_orders=("is_repeat_order", "sum"),
        cm_loyal_orders=("is_loyal_order", "sum"),
        cm_repeat_365_orders=("is_repeat_within_365d", "sum"),
        cm_age_18_34_orders=("age_18_34", "sum"),
        cm_age_35_54_orders=("age_35_54", "sum"),
        cm_age_55_plus_orders=("age_55_plus", "sum"),
        cm_female_orders=("is_female", "sum"),
        cm_mobile_orders=("is_mobile", "sum"),
        cm_paid_search_acq_orders=("is_paid_search_acq", "sum"),
        cm_organic_search_acq_orders=("is_organic_search_acq", "sum"),
        cm_east_region_orders=("is_east_region", "sum"),
        cm_median_inter_order_gap=("inter_order_gap_days", "median"),
        cm_median_customer_tenure=("customer_tenure_days", "median"),
    )
    daily = daily.reindex(date_index).fillna(
        {
            "cm_orders": 0,
            "cm_revenue": 0,
            "cm_new_orders": 0,
            "cm_repeat_orders": 0,
            "cm_loyal_orders": 0,
            "cm_repeat_365_orders": 0,
            "cm_age_18_34_orders": 0,
            "cm_age_35_54_orders": 0,
            "cm_age_55_plus_orders": 0,
            "cm_female_orders": 0,
            "cm_mobile_orders": 0,
            "cm_paid_search_acq_orders": 0,
            "cm_organic_search_acq_orders": 0,
            "cm_east_region_orders": 0,
        }
    )

    daily_sets = fact.groupby("order_date")["customer_id"].agg(lambda x: set(x)).reindex(date_index)
    daily_sets = daily_sets.apply(lambda x: set() if not isinstance(x, set) else x)
    active_28 = _rolling_unique_count(daily_sets, 28)
    active_91 = _rolling_unique_count(daily_sets, WINDOW)

    out = make_daily_frame(date_index)
    out["cm_active_customers_28d_gap549"] = active_28.shift(FORECAST_GAP_DAYS).values
    out["cm_active_customers_91d_gap549"] = active_91.shift(FORECAST_GAP_DAYS).values
    out["cm_orders_per_active_customer_91d_gap549"] = safe_divide(
        shifted_rolling_sum(daily["cm_orders"], WINDOW), active_91.shift(FORECAST_GAP_DAYS)
    ).values
    out["cm_revenue_per_active_customer_91d_gap549"] = safe_divide(
        shifted_rolling_sum(daily["cm_revenue"], WINDOW), active_91.shift(FORECAST_GAP_DAYS)
    ).values
    out["cm_new_customer_order_share_91d_gap549"] = shifted_rolling_ratio(
        daily["cm_new_orders"], daily["cm_orders"], WINDOW
    ).values
    out["cm_repeat_customer_order_share_91d_gap549"] = shifted_rolling_ratio(
        daily["cm_repeat_orders"], daily["cm_orders"], WINDOW
    ).values
    out["cm_loyal_customer_order_share_91d_gap549"] = shifted_rolling_ratio(
        daily["cm_loyal_orders"], daily["cm_orders"], WINDOW
    ).values
    out["cm_repeat_within_365d_order_share_91d_gap549"] = shifted_rolling_ratio(
        daily["cm_repeat_365_orders"], daily["cm_orders"], WINDOW
    ).values
    out["cm_median_inter_order_gap_91d_gap549"] = shifted_rolling_mean(
        daily["cm_median_inter_order_gap"], WINDOW
    ).values
    out["cm_median_customer_tenure_days_91d_gap549"] = shifted_rolling_mean(
        daily["cm_median_customer_tenure"], WINDOW
    ).values

    share_specs = {
        "cm_age_18_34_order_share_91d_gap549": "cm_age_18_34_orders",
        "cm_age_35_54_order_share_91d_gap549": "cm_age_35_54_orders",
        "cm_age_55_plus_order_share_91d_gap549": "cm_age_55_plus_orders",
        "cm_female_order_share_91d_gap549": "cm_female_orders",
        "cm_mobile_order_share_91d_gap549": "cm_mobile_orders",
        "cm_paid_search_acq_order_share_91d_gap549": "cm_paid_search_acq_orders",
        "cm_organic_search_acq_order_share_91d_gap549": "cm_organic_search_acq_orders",
        "cm_east_region_order_share_91d_gap549": "cm_east_region_orders",
    }
    for feature, source_col in share_specs.items():
        out[feature] = shifted_rolling_ratio(daily[source_col], daily["cm_orders"], WINDOW).values

    region_counts = pd.crosstab(fact["order_date"], fact["region"]).reindex(date_index).fillna(0)
    region_counts = add_missing_columns(region_counts, ["Central", "East", "West"])
    acq_counts = (
        pd.crosstab(fact["order_date"], fact["acquisition_channel"]).reindex(date_index).fillna(0)
    )
    acq_counts = add_missing_columns(
        acq_counts,
        ["direct", "email_campaign", "organic_search", "paid_search", "referral", "social_media"],
    )
    out["cm_region_entropy_91d_gap549"] = rolling_entropy_from_counts(region_counts, WINDOW).values
    out["cm_acquisition_entropy_91d_gap549"] = rolling_entropy_from_counts(acq_counts, WINDOW).values
    return out
