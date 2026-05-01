from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (
    get_full_date_index,
    make_daily_frame,
    read_raw_csv,
    safe_divide,
    shifted,
    shifted_rolling_mean,
    shifted_rolling_max,
    shifted_rolling_quantile,
    shifted_rolling_ratio,
    shifted_rolling_sum,
)


WINDOW = 91


def _build_shipment_daily(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    shipments = read_raw_csv("shipments.csv", parse_dates=["ship_date", "delivery_date"])
    orders = read_raw_csv("orders.csv", parse_dates=["order_date"])
    fact = shipments.merge(orders[["order_id", "order_date", "order_status"]], on="order_id", how="left")
    fact["lead_time_days"] = (fact["delivery_date"] - fact["order_date"]).dt.days
    fact["late_flag"] = (fact["lead_time_days"] > 7).astype(int)
    fact["lead_8_10_flag"] = fact["lead_time_days"].between(8, 10).astype(int)
    fact["lead_gt10_flag"] = (fact["lead_time_days"] > 10).astype(int)
    fact["sla_breach_days"] = (fact["lead_time_days"] - 7).clip(lower=0)
    fact["delivery_date"] = pd.to_datetime(fact["delivery_date"])
    daily = fact.groupby("delivery_date").agg(
        ops_delivered_orders=("order_id", "nunique"),
        ops_lead_time_mean=("lead_time_days", "mean"),
        ops_late_orders=("late_flag", "sum"),
        ops_8_10_orders=("lead_8_10_flag", "sum"),
        ops_gt10_orders=("lead_gt10_flag", "sum"),
        ops_sla_breach_days=("sla_breach_days", "sum"),
        ops_shipping_fee_sum=("shipping_fee", "sum"),
    )
    p90 = fact.groupby("delivery_date")["lead_time_days"].quantile(0.90).rename("ops_lead_time_p90")
    p95 = fact.groupby("delivery_date")["lead_time_days"].quantile(0.95).rename("ops_lead_time_p95")
    daily = daily.join([p90, p95], how="left")
    return daily.reindex(date_index).fillna(
        {
            "ops_delivered_orders": 0,
            "ops_late_orders": 0,
            "ops_8_10_orders": 0,
            "ops_gt10_orders": 0,
            "ops_sla_breach_days": 0,
            "ops_shipping_fee_sum": 0,
        }
    )


def _build_order_daily(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    orders = read_raw_csv("orders.csv", parse_dates=["order_date"])
    shipments = read_raw_csv("shipments.csv", parse_dates=["ship_date"])
    fact = orders.merge(shipments[["order_id", "ship_date"]], on="order_id", how="left")
    status = fact["order_status"].astype(str).str.lower()
    fact["cancelled"] = status.isin(["cancelled", "canceled"]).astype(int)
    fact["pre_ship_cancel"] = ((fact["cancelled"] == 1) & fact["ship_date"].isna()).astype(int)
    fact["post_ship_cancel"] = ((fact["cancelled"] == 1) & fact["ship_date"].notna()).astype(int)
    fact["cod"] = (fact["payment_method"].astype(str).str.lower() == "cod").astype(int)
    return (
        fact.groupby("order_date")
        .agg(
            ops_total_orders=("order_id", "nunique"),
            ops_cancel_orders=("cancelled", "sum"),
            ops_pre_ship_cancel_orders=("pre_ship_cancel", "sum"),
            ops_post_ship_cancel_orders=("post_ship_cancel", "sum"),
            ops_cod_orders=("cod", "sum"),
        )
        .reindex(date_index)
        .fillna(0)
    )


def _build_revenue_daily(date_index: pd.DatetimeIndex) -> pd.Series:
    orders = read_raw_csv("orders.csv", parse_dates=["order_date"])
    items = read_raw_csv("order_items.csv")
    fact = items.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    status = orders.set_index("order_id")["order_status"].astype(str).str.lower()
    fact["order_status"] = fact["order_id"].map(status)
    fact = fact[~fact["order_status"].isin(["cancelled", "canceled"])].copy()
    fact["net_revenue"] = fact["quantity"] * fact["unit_price"] - fact["discount_amount"].fillna(0)
    return fact.groupby("order_date")["net_revenue"].sum().reindex(date_index).fillna(0)


def _build_return_daily(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    returns = read_raw_csv("returns.csv", parse_dates=["return_date"])
    daily = returns.groupby("return_date").agg(
        ops_return_orders=("order_id", "nunique"),
        ops_return_qty=("return_quantity", "sum"),
        ops_refund_amount=("refund_amount", "sum"),
    )
    reason_counts = pd.crosstab(returns["return_date"], returns["return_reason"]).reindex(date_index).fillna(0)
    for col in ["wrong_size", "defective", "not_as_described", "late_delivery", "changed_mind"]:
        if col not in reason_counts.columns:
            reason_counts[col] = 0
    daily = daily.reindex(date_index).fillna(0)
    return daily.join(reason_counts[["wrong_size", "defective", "not_as_described", "late_delivery", "changed_mind"]])


def _build_review_daily(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    reviews = read_raw_csv("reviews.csv", parse_dates=["review_date"])
    orders = read_raw_csv("orders.csv", parse_dates=["order_date"])
    shipments = read_raw_csv("shipments.csv", parse_dates=["delivery_date"])
    late_orders = shipments.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    late_orders["late_flag"] = ((late_orders["delivery_date"] - late_orders["order_date"]).dt.days > 7).astype(int)
    reviews = reviews.merge(late_orders[["order_id", "late_flag"]], on="order_id", how="left")
    reviews["low_rating"] = (reviews["rating"] <= 2).astype(int)
    reviews["late_review"] = reviews["late_flag"].fillna(0).astype(int)
    reviews["low_after_late"] = ((reviews["low_rating"] == 1) & (reviews["late_review"] == 1)).astype(int)
    daily = reviews.groupby("review_date").agg(
        ops_review_count=("review_id", "count"),
        ops_rating_sum=("rating", "sum"),
        ops_low_rating_count=("low_rating", "sum"),
        ops_late_review_count=("late_review", "sum"),
        ops_low_after_late_count=("low_after_late", "sum"),
    )
    return daily.reindex(date_index).fillna(0)


def _build_inventory_daily(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    inventory = read_raw_csv("inventory.csv", parse_dates=["snapshot_date"])
    items = read_raw_csv("order_items.csv")
    items["net_revenue"] = items["quantity"] * items["unit_price"] - items["discount_amount"].fillna(0)
    product_revenue = items.groupby("product_id")["net_revenue"].sum().rename("product_revenue")
    hero_products = set(product_revenue.sort_values(ascending=False).head(50).index)
    inventory = inventory.merge(product_revenue, on="product_id", how="left")
    inventory["product_revenue"] = inventory["product_revenue"].fillna(0)
    inventory["low_stock_flag"] = (inventory["reorder_flag"].fillna(0) == 1).astype(int)
    inventory["stale_stock_flag"] = (inventory["days_of_supply"] > 180).astype(int)
    inventory["inv_turnover_proxy_row"] = inventory["units_sold"].divide(
        inventory["stock_on_hand"].replace(0, np.nan)
    )
    inventory["weighted_fill_numerator"] = inventory["fill_rate"] * inventory["product_revenue"]
    inventory["hero_stockout_flag"] = inventory["product_id"].isin(hero_products).astype(int) * inventory["stockout_flag"]
    inventory["hero_sku_flag"] = inventory["product_id"].isin(hero_products).astype(int)
    daily = inventory.groupby("snapshot_date").agg(
        ops_fill_rate=("fill_rate", "mean"),
        ops_stockout_rate=("stockout_flag", "mean"),
        ops_overstock_rate=("overstock_flag", "mean"),
        ops_sell_through=("sell_through_rate", "mean"),
        ops_days_of_supply=("days_of_supply", "mean"),
        ops_days_of_supply_p90=("days_of_supply", lambda x: x.quantile(0.90)),
        ops_days_of_supply_p95=("days_of_supply", lambda x: x.quantile(0.95)),
        ops_low_stock_share=("low_stock_flag", "mean"),
        ops_stale_stock_share=("stale_stock_flag", "mean"),
        ops_inv_turnover_proxy=("inv_turnover_proxy_row", "mean"),
        ops_weighted_fill_numerator=("weighted_fill_numerator", "sum"),
        ops_product_revenue_weight=("product_revenue", "sum"),
        ops_hero_stockout_count=("hero_stockout_flag", "sum"),
        ops_hero_sku_count=("hero_sku_flag", "sum"),
    )
    return daily.reindex(date_index).ffill()


def build_ops_features() -> pd.DataFrame:
    date_index = get_full_date_index()
    shipments = _build_shipment_daily(date_index)
    orders = _build_order_daily(date_index)
    revenue = _build_revenue_daily(date_index)
    returns = _build_return_daily(date_index)
    reviews = _build_review_daily(date_index)
    inventory = _build_inventory_daily(date_index)

    out = make_daily_frame(date_index)
    out["lead_time_mean_7d"] = shifted_rolling_mean(shipments["ops_lead_time_mean"], 7).values
    out["lead_time_p90_7d"] = shifted_rolling_mean(shipments["ops_lead_time_p90"], 7).values
    out["lead_time_p95_7d"] = shifted_rolling_mean(shipments["ops_lead_time_p95"], 7).values
    out["late_rate_7d"] = shifted_rolling_ratio(
        shipments["ops_late_orders"], shipments["ops_delivered_orders"], 7
    ).values
    out["ontime_rate_7d"] = 1 - out["late_rate_7d"]
    out["orders_8_10_days_share"] = shifted_rolling_ratio(
        shipments["ops_8_10_orders"], shipments["ops_delivered_orders"], 7
    ).values
    out["orders_gt10_days_share"] = shifted_rolling_ratio(
        shipments["ops_gt10_orders"], shipments["ops_delivered_orders"], 7
    ).values
    out["shipping_fee_avg_7d"] = safe_divide(
        shifted_rolling_sum(shipments["ops_shipping_fee_sum"], 7),
        shifted_rolling_sum(shipments["ops_delivered_orders"], 7),
    ).values
    out["cancel_rate_7d"] = shifted_rolling_ratio(
        orders["ops_cancel_orders"], orders["ops_total_orders"], 7
    ).values
    out["pre_ship_cancel_rate_7d"] = shifted_rolling_ratio(
        orders["ops_pre_ship_cancel_orders"], orders["ops_total_orders"], 7
    ).values
    out["post_ship_cancel_rate_7d"] = shifted_rolling_ratio(
        orders["ops_post_ship_cancel_orders"], orders["ops_total_orders"], 7
    ).values
    out["cod_share_7d"] = shifted_rolling_ratio(
        orders["ops_cod_orders"], orders["ops_total_orders"], 7
    ).values

    out["ops_lead_time_mean_91d_gap549"] = shifted_rolling_mean(
        shipments["ops_lead_time_mean"], WINDOW
    ).values
    out["ops_lead_time_p90_91d_gap549"] = shifted_rolling_mean(
        shipments["ops_lead_time_p90"], WINDOW
    ).values
    out["ops_late_rate_91d_gap549"] = shifted_rolling_ratio(
        shipments["ops_late_orders"], shipments["ops_delivered_orders"], WINDOW
    ).values
    out["ops_shipping_fee_per_order_91d_gap549"] = safe_divide(
        shifted_rolling_sum(shipments["ops_shipping_fee_sum"], WINDOW),
        shifted_rolling_sum(shipments["ops_delivered_orders"], WINDOW),
    ).values

    out["ops_return_order_rate_91d_gap549"] = shifted_rolling_ratio(
        returns["ops_return_orders"], shipments["ops_delivered_orders"], WINDOW
    ).values
    out["return_rate_7d"] = shifted_rolling_ratio(
        returns["ops_return_orders"], shipments["ops_delivered_orders"], 7
    ).values
    out["refund_to_revenue_7d"] = safe_divide(
        shifted_rolling_sum(returns["ops_refund_amount"], 7),
        shifted_rolling_sum(revenue, 7),
    ).values
    out["ops_return_qty_sum_91d_gap549"] = shifted_rolling_sum(
        returns["ops_return_qty"], WINDOW
    ).values
    out["refund_amount_7d"] = shifted_rolling_sum(returns["ops_refund_amount"], 7).values
    out["ops_refund_amount_sum_91d_gap549"] = shifted_rolling_sum(
        returns["ops_refund_amount"], WINDOW
    ).values
    for reason in ["wrong_size", "defective", "not_as_described", "late_delivery"]:
        out[f"ops_{reason}_return_share_91d_gap549"] = shifted_rolling_ratio(
            returns[reason], returns["ops_return_orders"], WINDOW
        ).values
    out["wrong_size_share_7d"] = shifted_rolling_ratio(
        returns["wrong_size"], returns["ops_return_orders"], 7
    ).values
    out["defective_share_7d"] = shifted_rolling_ratio(
        returns["defective"], returns["ops_return_orders"], 7
    ).values
    out["not_as_described_share_7d"] = shifted_rolling_ratio(
        returns["not_as_described"], returns["ops_return_orders"], 7
    ).values
    out["late_delivery_refund_share_7d"] = shifted_rolling_ratio(
        returns["late_delivery"], returns["ops_return_orders"], 7
    ).values
    out["exchange_share_7d"] = shifted_rolling_ratio(
        returns["changed_mind"], returns["ops_return_orders"], 7
    ).values

    out["ops_avg_rating_91d_gap549"] = safe_divide(
        shifted_rolling_sum(reviews["ops_rating_sum"], WINDOW),
        shifted_rolling_sum(reviews["ops_review_count"], WINDOW),
    ).values
    out["ops_low_rating_share_91d_gap549"] = shifted_rolling_ratio(
        reviews["ops_low_rating_count"], reviews["ops_review_count"], WINDOW
    ).values
    out["rating_leq2_share_7d"] = shifted_rolling_ratio(
        reviews["ops_low_rating_count"], reviews["ops_review_count"], 7
    ).values
    out["low_rating_after_late_share_7d"] = shifted_rolling_ratio(
        reviews["ops_low_after_late_count"], reviews["ops_late_review_count"], 7
    ).values

    for col in [
        "ops_fill_rate",
        "ops_stockout_rate",
        "ops_overstock_rate",
        "ops_sell_through",
        "ops_days_of_supply",
    ]:
        out[f"{col}_gap549"] = shifted(inventory[col]).values
        out[f"{col}_roll_mean_91d_gap549"] = shifted_rolling_mean(inventory[col], WINDOW).values

    out["stockout_flag_rate_7d"] = shifted_rolling_mean(inventory["ops_stockout_rate"], 7).values
    out["overstock_flag_rate_7d"] = shifted_rolling_mean(inventory["ops_overstock_rate"], 7).values
    out["avg_dos_7d"] = shifted_rolling_mean(inventory["ops_days_of_supply"], 7).values
    out["p90_dos_28d"] = shifted_rolling_mean(inventory["ops_days_of_supply_p90"], 28).values
    out["p95_dos_28d"] = shifted_rolling_mean(inventory["ops_days_of_supply_p95"], 28).values
    out["dos_tail_ratio"] = safe_divide(out["p95_dos_28d"], out["avg_dos_7d"]).values
    out["hero_sku_stockout_rate"] = safe_divide(
        shifted_rolling_sum(inventory["ops_hero_stockout_count"], 7),
        shifted_rolling_sum(inventory["ops_hero_sku_count"], 7),
    ).values
    out["fill_rate_weighted_rev"] = safe_divide(
        shifted_rolling_sum(inventory["ops_weighted_fill_numerator"], 7),
        shifted_rolling_sum(inventory["ops_product_revenue_weight"], 7),
    ).values
    out["low_stock_sku_share"] = shifted_rolling_mean(inventory["ops_low_stock_share"], 7).values
    out["stale_stock_share_180d"] = shifted_rolling_mean(inventory["ops_stale_stock_share"], 180).values
    out["inventory_imbalance_idx"] = out["stockout_flag_rate_7d"] + out["overstock_flag_rate_7d"]
    out["inv_turnover_proxy"] = shifted_rolling_mean(inventory["ops_inv_turnover_proxy"], 7).values
    out["dos_acceleration_28d"] = out["avg_dos_7d"] - shifted_rolling_mean(
        inventory["ops_days_of_supply"], 28
    ).values
    out["stockout_streak_max_14d"] = shifted_rolling_max(inventory["ops_stockout_rate"], 14).values
    out["overstock_age_pressure_idx"] = out["overstock_flag_rate_7d"] * out["dos_tail_ratio"]

    late_rate = pd.Series(out["late_rate_7d"], index=date_index)
    out["late_rate_zscore_28d"] = safe_divide(
        late_rate - late_rate.rolling(28, min_periods=7).mean(),
        late_rate.rolling(28, min_periods=7).std(),
    ).values
    out["late_persistence_3d"] = late_rate.rolling(3, min_periods=1).mean().values
    out["sla_breach_severity_7d"] = safe_divide(
        shifted_rolling_sum(shipments["ops_sla_breach_days"], 7),
        shifted_rolling_sum(shipments["ops_delivered_orders"], 7),
    ).values
    out["eta_tail_concentration_7d"] = shifted_rolling_ratio(
        shipments["ops_gt10_orders"], shipments["ops_late_orders"], 7
    ).values

    out["lane_late_rate_7d"] = out["late_rate_7d"]
    out["lane_refund_to_rev_7d"] = out["refund_to_revenue_7d"]
    out["lane_cancel_rate_7d"] = out["cancel_rate_7d"]
    out["lane_stockout_rate_7d"] = out["stockout_flag_rate_7d"]
    out["lane_cod_share_7d"] = out["cod_share_7d"]
    out["lane_profit_leakage_proxy"] = (
        out["lane_late_rate_7d"] + out["lane_refund_to_rev_7d"] + out["lane_stockout_rate_7d"]
    )
    reason_shares = pd.DataFrame(
        {
            "wrong_size": out["wrong_size_share_7d"],
            "defective": out["defective_share_7d"],
            "not_as_described": out["not_as_described_share_7d"],
            "late_delivery": out["late_delivery_refund_share_7d"],
            "changed_mind": out["exchange_share_7d"],
        }
    )
    out["lane_refund_hhi_7d"] = (reason_shares.fillna(0) ** 2).sum(axis=1)
    out["lane_tail_eta_refund_coupling"] = out["orders_gt10_days_share"] * out["refund_to_revenue_7d"]
    out["lane_inventory_stress_idx"] = out["stockout_flag_rate_7d"] + out["overstock_flag_rate_7d"]
    out["refund_reason_entropy_7d"] = -(
        reason_shares.replace(0, np.nan) * np.log(reason_shares.replace(0, np.nan))
    ).sum(axis=1)
    out["wrong_size_to_defect_ratio_7d"] = safe_divide(
        out["wrong_size_share_7d"], out["defective_share_7d"]
    ).values
    refund_amount = pd.Series(out["refund_amount_7d"], index=date_index)
    out["refund_acceleration_7d"] = refund_amount - refund_amount.shift(7)
    out["late_x_refund"] = out["late_rate_7d"] * out["refund_to_revenue_7d"]
    out["late_tail_x_low_rating"] = out["orders_gt10_days_share"] * out["rating_leq2_share_7d"]
    out["stockout_x_late"] = out["stockout_flag_rate_7d"] * out["late_rate_7d"]
    out["overstock_x_refund"] = out["overstock_flag_rate_7d"] * out["refund_to_revenue_7d"]
    target_dates = pd.Series(date_index)
    out["q3_x_overstock"] = (target_dates.dt.quarter.eq(3).astype(int).values * out["overstock_flag_rate_7d"])
    out["dec_x_dos_tail"] = (target_dates.dt.month.eq(12).astype(int).values * out["dos_tail_ratio"])
    out["cod_x_cancel"] = out["cod_share_7d"] * out["cancel_rate_7d"]
    out["lane_late_x_lane_refund"] = out["lane_late_rate_7d"] * out["lane_refund_to_rev_7d"]
    return out.replace([np.inf, -np.inf], np.nan)
