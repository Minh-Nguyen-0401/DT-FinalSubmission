from pathlib import Path
import pandas as pd


def export_powerbi_tables(base_path: str = ".", out_dir: str = "powerbi_exports"):
    base = Path(base_path)
    out = base / out_dir
    out.mkdir(parents=True, exist_ok=True)

    orders = pd.read_csv(base / "orders.csv")
    order_items = pd.read_csv(base / "order_items.csv")
    products = pd.read_csv(base / "products.csv")
    shipments = pd.read_csv(base / "shipments.csv")
    returns = pd.read_csv(base / "returns.csv")
    reviews = pd.read_csv(base / "reviews.csv")
    inventory = pd.read_csv(base / "inventory.csv")
    customers = pd.read_csv(base / "customers.csv")
    geography = pd.read_csv(base / "geography.csv")

    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    shipments["ship_date"] = pd.to_datetime(shipments["ship_date"], errors="coerce")
    shipments["delivery_date"] = pd.to_datetime(shipments["delivery_date"], errors="coerce")

    items = order_items.merge(
        products[["product_id", "product_name", "category", "segment", "cogs"]],
        on="product_id",
        how="left",
    )
    items["gross_revenue"] = items["quantity"] * items["unit_price"]
    items["net_revenue"] = items["gross_revenue"] - items["discount_amount"].fillna(0)
    items["total_cogs"] = items["quantity"] * items["cogs"].fillna(0)
    items["gross_margin"] = items["net_revenue"] - items["total_cogs"]

    order_fin = items.groupby("order_id", as_index=False).agg(
        order_revenue=("net_revenue", "sum"),
        order_cogs=("total_cogs", "sum"),
        order_margin=("gross_margin", "sum"),
        item_count=("product_id", "count"),
    )

    order_core = (
        orders.merge(shipments, on="order_id", how="left")
        .merge(order_fin, on="order_id", how="left")
        .merge(customers[["customer_id", "city", "age_group", "acquisition_channel"]], on="customer_id", how="left")
        .merge(geography[["zip", "region", "district"]], on="zip", how="left")
    )

    order_core["delivery_lead_days"] = (order_core["delivery_date"] - order_core["order_date"]).dt.days
    delay_threshold = order_core["delivery_lead_days"].quantile(0.75)
    order_core["late_delivery_flag"] = (order_core["delivery_lead_days"] > delay_threshold).astype(int)

    ret_by_order = returns.groupby("order_id", as_index=False).agg(
        return_lines=("return_id", "count"),
        refund_amount=("refund_amount", "sum"),
    )
    order_core = order_core.merge(ret_by_order, on="order_id", how="left")
    order_core["return_lines"] = order_core["return_lines"].fillna(0)
    order_core["refund_amount"] = order_core["refund_amount"].fillna(0)
    order_core["returned_flag"] = (order_core["return_lines"] > 0).astype(int)

    review_order = reviews.groupby("order_id", as_index=False).agg(avg_rating=("rating", "mean"))
    order_core = order_core.merge(review_order, on="order_id", how="left")
    order_core["low_rating_flag"] = (order_core["avg_rating"] < 3).astype(int)
    order_core["order_month"] = order_core["order_date"].dt.to_period("M").astype(str)

    prod_dim = products[["product_id", "category", "segment"]].drop_duplicates("product_id")
    inv_prod = inventory.merge(prod_dim, on="product_id", how="left", suffixes=("", "_prod"))
    for col in ["category", "segment"]:
        prod_col = f"{col}_prod"
        if col in inv_prod.columns and prod_col in inv_prod.columns:
            inv_prod[col] = inv_prod[col].combine_first(inv_prod[prod_col])
            inv_prod = inv_prod.drop(columns=[prod_col])
        elif col not in inv_prod.columns and prod_col in inv_prod.columns:
            inv_prod = inv_prod.rename(columns={prod_col: col})

    inv_kpi = inv_prod.groupby(["category", "segment"], as_index=False).agg(
        sku_snapshots=("product_id", "count"),
        avg_fill_rate=("fill_rate", "mean"),
        avg_stockout_days=("stockout_days", "mean"),
        stockout_flag_rate=("stockout_flag", "mean"),
        overstock_flag_rate=("overstock_flag", "mean"),
        avg_days_of_supply=("days_of_supply", "mean"),
        avg_sell_through=("sell_through_rate", "mean"),
    )

    lane_kpi = order_core.groupby(["region", "order_source"], as_index=False).agg(
        orders=("order_id", "nunique"),
        revenue=("order_revenue", "sum"),
        late_rate=("late_delivery_flag", "mean"),
        return_rate=("returned_flag", "mean"),
        low_rating_rate=("low_rating_flag", "mean"),
        refund=("refund_amount", "sum"),
    )

    reason_kpi = returns.groupby("return_reason", as_index=False).agg(
        return_lines=("return_id", "count"),
        refund=("refund_amount", "sum"),
    )

    order_core.to_csv(out / "fact_orders_ops_quality.csv", index=False)
    items.to_csv(out / "fact_order_items_ops_quality.csv", index=False)
    lane_kpi.to_csv(out / "agg_lane_kpi.csv", index=False)
    inv_kpi.to_csv(out / "agg_inventory_kpi.csv", index=False)
    reason_kpi.to_csv(out / "agg_return_reason_kpi.csv", index=False)

    print(f"Exported files to: {out.resolve()}")


if __name__ == "__main__":
    export_powerbi_tables()
