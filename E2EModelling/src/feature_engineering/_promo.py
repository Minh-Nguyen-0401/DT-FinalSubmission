from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (
    FORECAST_GAP_DAYS,
    add_missing_columns,
    get_full_date_index,
    make_daily_frame,
    read_raw_csv,
    safe_divide,
    shifted,
    shifted_rolling_mean,
    shifted_rolling_ratio,
    shifted_rolling_sum,
    valid_orders,
)


WINDOW = 91


def _has_promo(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return text.notna() & text.str.strip().ne("") & text.str.lower().ne("nan")


def _build_order_promo_daily(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    orders = valid_orders(read_raw_csv("orders.csv", parse_dates=["order_date"]))
    items = read_raw_csv("order_items.csv")
    fact = items.merge(orders[["order_id", "order_date"]], on="order_id", how="inner")
    fact["discount_amount"] = fact["discount_amount"].fillna(0)
    fact["gross_sales"] = fact["quantity"] * fact["unit_price"]
    fact["net_revenue"] = fact["gross_sales"] - fact["discount_amount"]
    fact["has_promo"] = _has_promo(fact["promo_id"]) | _has_promo(fact["promo_id_2"])
    fact["promo_revenue"] = fact["net_revenue"].where(fact["has_promo"], 0)
    fact["promo_line"] = fact["has_promo"].astype(int)
    daily = fact.groupby("order_date").agg(
        mkt_lines=("order_id", "size"),
        mkt_promo_lines=("promo_line", "sum"),
        mkt_gross_sales=("gross_sales", "sum"),
        mkt_discount=("discount_amount", "sum"),
        mkt_revenue=("net_revenue", "sum"),
        mkt_promo_revenue=("promo_revenue", "sum"),
    )
    return daily.reindex(date_index).fillna(0)


def _build_promo_calendar(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    promos = read_raw_csv("promotions.csv", parse_dates=["start_date", "end_date"])
    rows = []
    for promo in promos.itertuples(index=False):
        dates = pd.date_range(promo.start_date, promo.end_date, freq="D")
        duration = max((promo.end_date - promo.start_date).days + 1, 1)
        for date in dates:
            name = str(promo.promo_name).lower()
            channel = str(promo.promo_channel).lower()
            category = str(promo.applicable_category).lower()
            promo_type = str(promo.promo_type).lower()
            day_index = (date - promo.start_date).days + 1
            rows.append(
                {
                    "Date": date,
                    "active_promo_count": 1,
                    "is_any_promo_active": 1,
                    "active_discount_avg": promo.discount_value,
                    "active_discount_max": promo.discount_value,
                    "active_discount_sum": promo.discount_value,
                    "active_percentage_promo_count": int(promo_type == "percentage"),
                    "active_fixed_promo_count": int(promo_type == "fixed"),
                    "has_fixed_discount_promo": int(promo_type == "fixed"),
                    "active_stackable_promo_count": int(promo.stackable_flag == 1),
                    "has_stackable_promo": int(promo.stackable_flag == 1),
                    "active_min_order_promo_count": int(float(promo.min_order_value) > 0),
                    "active_min_order_avg": promo.min_order_value,
                    "sitewide_promo_count": int(pd.isna(promo.applicable_category)),
                    "category_promo_count": int(pd.notna(promo.applicable_category)),
                    "has_streetwear_promo": int("streetwear" in category),
                    "has_outdoor_promo": int("outdoor" in category),
                    "email_promo_count": int(channel == "email"),
                    "online_promo_count": int(channel == "online"),
                    "all_channels_promo_count": int(channel == "all_channels"),
                    "social_promo_count": int(channel == "social_media"),
                    "in_store_promo_count": int(channel == "in_store"),
                    "is_spring_sale_active": int("spring sale" in name),
                    "is_mid_year_sale_active": int("mid-year sale" in name),
                    "is_fall_launch_active": int("fall launch" in name),
                    "is_year_end_sale_active": int("year-end sale" in name),
                    "is_urban_blowout_active": int("urban blowout" in name),
                    "is_rural_special_active": int("rural special" in name),
                    "promo_day_index": day_index,
                    "promo_days_remaining": (promo.end_date - date).days,
                    "promo_progress_ratio": day_index / duration,
                }
            )
    if not rows:
        return pd.DataFrame(index=date_index)
    daily = pd.DataFrame(rows).groupby("Date").agg(
        active_promo_count=("active_promo_count", "sum"),
        is_any_promo_active=("is_any_promo_active", "max"),
        active_discount_avg=("active_discount_avg", "mean"),
        active_discount_max=("active_discount_max", "max"),
        active_discount_sum=("active_discount_sum", "sum"),
        active_percentage_promo_count=("active_percentage_promo_count", "sum"),
        active_fixed_promo_count=("active_fixed_promo_count", "sum"),
        has_fixed_discount_promo=("has_fixed_discount_promo", "max"),
        active_stackable_promo_count=("active_stackable_promo_count", "sum"),
        has_stackable_promo=("has_stackable_promo", "max"),
        active_min_order_promo_count=("active_min_order_promo_count", "sum"),
        active_min_order_avg=("active_min_order_avg", "mean"),
        email_promo_count=("email_promo_count", "sum"),
        online_promo_count=("online_promo_count", "sum"),
        all_channels_promo_count=("all_channels_promo_count", "sum"),
        social_promo_count=("social_promo_count", "sum"),
        in_store_promo_count=("in_store_promo_count", "sum"),
        sitewide_promo_count=("sitewide_promo_count", "sum"),
        category_promo_count=("category_promo_count", "sum"),
        has_streetwear_promo=("has_streetwear_promo", "max"),
        has_outdoor_promo=("has_outdoor_promo", "max"),
        is_spring_sale_active=("is_spring_sale_active", "max"),
        is_mid_year_sale_active=("is_mid_year_sale_active", "max"),
        is_fall_launch_active=("is_fall_launch_active", "max"),
        is_year_end_sale_active=("is_year_end_sale_active", "max"),
        is_urban_blowout_active=("is_urban_blowout_active", "max"),
        is_rural_special_active=("is_rural_special_active", "max"),
        promo_day_index=("promo_day_index", "mean"),
        promo_days_remaining=("promo_days_remaining", "min"),
        promo_progress_ratio=("promo_progress_ratio", "mean"),
    )
    daily = daily.reindex(date_index).fillna(0)
    starts = sorted(promos["start_date"].dropna())
    ends = sorted(promos["end_date"].dropna())
    daily["days_to_next_promo_start"] = [
        min([(start - date).days for start in starts if start > date], default=60)
        for date in date_index
    ]
    daily["days_since_last_promo_end"] = [
        min([(date - end).days for end in ends if end < date], default=60)
        for date in date_index
    ]
    return daily


def _build_web_daily(date_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    web = read_raw_csv("web_traffic.csv", parse_dates=["date"])
    web["weighted_bounce"] = web["sessions"] * web["bounce_rate"]
    web["weighted_duration"] = web["sessions"] * web["avg_session_duration_sec"]
    daily = web.groupby("date").agg(
        web_sessions=("sessions", "sum"),
        web_unique_visitors=("unique_visitors", "sum"),
        web_page_views=("page_views", "sum"),
        web_weighted_bounce=("weighted_bounce", "sum"),
        web_weighted_duration=("weighted_duration", "sum"),
    )
    source_sessions = (
        web.pivot_table(index="date", columns="traffic_source", values="sessions", aggfunc="sum")
        .reindex(date_index)
        .fillna(0)
    )
    source_weighted_bounce = (
        web.pivot_table(index="date", columns="traffic_source", values="weighted_bounce", aggfunc="sum")
        .reindex(date_index)
        .fillna(0)
    )
    source_sessions = add_missing_columns(
        source_sessions,
        ["direct", "email_campaign", "organic_search", "paid_search", "referral", "social_media"],
    )
    source_weighted_bounce = add_missing_columns(
        source_weighted_bounce,
        ["direct", "email_campaign", "organic_search", "paid_search", "referral", "social_media"],
    )
    return daily.reindex(date_index).fillna(0), source_sessions, source_weighted_bounce


def build_promo_mkt_features() -> pd.DataFrame:
    date_index = get_full_date_index()
    daily = _build_order_promo_daily(date_index)
    promo_calendar = _build_promo_calendar(date_index)
    web_daily, web_source, web_source_bounce = _build_web_daily(date_index)

    out = make_daily_frame(date_index)
    out["mkt_promo_line_share_91d_gap549"] = shifted_rolling_ratio(
        daily["mkt_promo_lines"], daily["mkt_lines"], WINDOW
    ).values
    out["daily_promo_line_share_lag_365"] = shifted_rolling_ratio(
        daily["mkt_promo_lines"], daily["mkt_lines"], 1, gap_days=FORECAST_GAP_DAYS + 365
    ).values
    out["mkt_discount_rate_91d_gap549"] = safe_divide(
        shifted_rolling_sum(daily["mkt_discount"], WINDOW),
        shifted_rolling_sum(daily["mkt_gross_sales"], WINDOW),
    ).values
    out["daily_discount_rate_lag_365"] = safe_divide(
        shifted_rolling_sum(daily["mkt_discount"], 1, gap_days=FORECAST_GAP_DAYS + 365),
        shifted_rolling_sum(daily["mkt_gross_sales"], 1, gap_days=FORECAST_GAP_DAYS + 365),
    ).values
    out["mkt_promo_revenue_share_91d_gap549"] = shifted_rolling_ratio(
        daily["mkt_promo_revenue"], daily["mkt_revenue"], WINDOW
    ).values
    out["mkt_discount_amount_sum_91d_gap549"] = shifted_rolling_sum(
        daily["mkt_discount"], WINDOW
    ).values

    promo_exact_cols = [
        "active_promo_count",
        "is_any_promo_active",
        "active_discount_avg",
        "active_discount_max",
        "active_discount_sum",
        "active_percentage_promo_count",
        "active_fixed_promo_count",
        "has_fixed_discount_promo",
        "active_stackable_promo_count",
        "has_stackable_promo",
        "active_min_order_promo_count",
        "active_min_order_avg",
        "sitewide_promo_count",
        "category_promo_count",
        "has_streetwear_promo",
        "has_outdoor_promo",
        "email_promo_count",
        "online_promo_count",
        "all_channels_promo_count",
        "social_promo_count",
        "in_store_promo_count",
        "is_spring_sale_active",
        "is_mid_year_sale_active",
        "is_fall_launch_active",
        "is_year_end_sale_active",
        "is_urban_blowout_active",
        "is_rural_special_active",
        "promo_day_index",
        "promo_days_remaining",
        "promo_progress_ratio",
        "days_to_next_promo_start",
        "days_since_last_promo_end",
    ]
    for col in promo_exact_cols:
        out[col] = shifted(promo_calendar[col]).values
    promo_aliases = {
        "active_email_promo_count": "email_promo_count",
        "active_online_promo_count": "online_promo_count",
        "active_all_channels_promo_count": "all_channels_promo_count",
        "active_social_promo_count": "social_promo_count",
        "active_in_store_promo_count": "in_store_promo_count",
        "active_sitewide_promo_count": "sitewide_promo_count",
        "active_category_promo_count": "category_promo_count",
    }
    for alias, source_col in promo_aliases.items():
        out[alias] = shifted(promo_calendar[source_col]).values

    for col in [
        "active_promo_count",
        "email_promo_count",
        "online_promo_count",
        "sitewide_promo_count",
        "category_promo_count",
        "active_discount_avg",
    ]:
        out[f"mkt_{col}_gap549"] = shifted(promo_calendar[col]).values
        out[f"mkt_{col}_roll_mean_28d_gap549"] = shifted_rolling_mean(
            promo_calendar[col], 28
        ).values
    out["mkt_avg_discount_value_gap549"] = shifted(promo_calendar["active_discount_avg"]).values
    out["mkt_avg_discount_value_roll_mean_28d_gap549"] = shifted_rolling_mean(
        promo_calendar["active_discount_avg"], 28
    ).values

    out["expected_promo_line_share_month"] = shifted_rolling_ratio(
        daily["mkt_promo_lines"], daily["mkt_lines"], 365
    ).values
    out["expected_discount_rate_month"] = safe_divide(
        shifted_rolling_sum(daily["mkt_discount"], 365),
        shifted_rolling_sum(daily["mkt_gross_sales"], 365),
    ).values
    out["expected_promo_revenue_share_month"] = shifted_rolling_ratio(
        daily["mkt_promo_revenue"], daily["mkt_revenue"], 365
    ).values
    out["expected_active_promo_count_month"] = shifted_rolling_mean(
        promo_calendar["active_promo_count"], 365
    ).values
    out["expected_email_promo_month"] = shifted_rolling_mean(
        (promo_calendar["email_promo_count"] > 0).astype(int), 365
    ).values
    out["expected_online_promo_month"] = shifted_rolling_mean(
        (promo_calendar["online_promo_count"] > 0).astype(int), 365
    ).values
    out["expected_sitewide_promo_month"] = shifted_rolling_mean(
        (promo_calendar["sitewide_promo_count"] > 0).astype(int), 365
    ).values
    out["expected_category_promo_month"] = shifted_rolling_mean(
        (promo_calendar["category_promo_count"] > 0).astype(int), 365
    ).values

    out["mkt_web_sessions_91d_gap549"] = shifted_rolling_sum(
        web_daily["web_sessions"], WINDOW
    ).values
    out["mkt_web_pageviews_per_session_91d_gap549"] = safe_divide(
        shifted_rolling_sum(web_daily["web_page_views"], WINDOW),
        shifted_rolling_sum(web_daily["web_sessions"], WINDOW),
    ).values
    out["mkt_web_bounce_rate_91d_gap549"] = safe_divide(
        shifted_rolling_sum(web_daily["web_weighted_bounce"], WINDOW),
        shifted_rolling_sum(web_daily["web_sessions"], WINDOW),
    ).values
    out["mkt_web_avg_session_duration_91d_gap549"] = safe_divide(
        shifted_rolling_sum(web_daily["web_weighted_duration"], WINDOW),
        shifted_rolling_sum(web_daily["web_sessions"], WINDOW),
    ).values

    total_sessions = web_source.sum(axis=1)
    for source in ["direct", "email_campaign", "organic_search", "paid_search", "referral", "social_media"]:
        out[f"exp_sessions_{source}"] = shifted_rolling_mean(web_source[source], 365).values
        out[f"exp_bounce_rate_{source}"] = safe_divide(
            shifted_rolling_sum(web_source_bounce[source], 365),
            shifted_rolling_sum(web_source[source], 365),
        ).values
    for source in ["direct", "email_campaign", "organic_search", "paid_search", "social_media"]:
        out[f"mkt_{source}_session_share_91d_gap549"] = shifted_rolling_ratio(
            web_source[source], total_sessions, WINDOW
        ).values
    return out.replace([np.inf, -np.inf], np.nan)
