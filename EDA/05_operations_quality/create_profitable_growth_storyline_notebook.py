from pathlib import Path
from textwrap import dedent

import nbformat as nbf


OUT = Path(__file__).with_name("profitable_growth_storyline.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(
        """
        # From Revenue Decline to Profit Leakage

        ## Storyline trung tâm

        Doanh nghiệp không chỉ mất doanh thu sau 2016, mà đang mất khả năng tạo ra **doanh thu tốt**.

        Doanh thu giảm không phải vì khách mua rẻ hơn. AOV vẫn tăng và traffic vẫn còn. Vấn đề nằm ở chỗ hệ thống tạo doanh thu đã yếu đi: doanh nghiệp kéo được người dùng vào phễu nhưng chuyển đổi kém hơn, thu hút ít khách mới hơn, giữ khách kém hơn, rồi phải dựa nhiều hơn vào promo và vận hành thiếu nhịp để giữ volume.

        **Revenue engine yếu đi ở phía khách hàng, còn profit engine bị rò rỉ ở phía promo và operations.**

        Notebook này dựng đầy đủ các biểu đồ trong storyline:

        1. Revenue bridge và customer engine
        2. Acquisition, retention, loyalty
        3. Funnel leakage và channel quality
        4. Promo economics
        5. Operations leakage
        6. Driver tree và action plan
        """
    ),
    code(
        """
        from pathlib import Path
        import os
        import warnings

        warnings.filterwarnings("ignore")

        BASE = Path.cwd()
        if not (BASE / "orders.csv").exists():
            BASE = Path("datathon-2026-round-1")

        os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mplconfig"))
        (BASE / ".mplconfig").mkdir(exist_ok=True)

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        import seaborn as sns
        from IPython.display import display, Markdown

        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_theme(style="whitegrid", context="notebook")

        COLORS = {
            "navy": "#17324D",
            "blue": "#2F80ED",
            "teal": "#00A6A6",
            "green": "#2E7D32",
            "yellow": "#F2C94C",
            "orange": "#F2994A",
            "red": "#D64545",
            "purple": "#7B61FF",
            "gray": "#6B7280",
            "light": "#EEF2F7",
            "dark": "#111827",
        }

        def money_vnd(x, pos=None):
            x = float(x)
            if abs(x) >= 1e9:
                return f"{x/1e9:.1f}B"
            if abs(x) >= 1e6:
                return f"{x/1e6:.0f}M"
            if abs(x) >= 1e3:
                return f"{x/1e3:.0f}K"
            return f"{x:.0f}"

        def pct_label(x):
            return f"{x * 100:.1f}%"

        def add_bar_labels(ax, fmt="{:.1f}", dy=3, fontsize=9):
            for patch in ax.patches:
                h = patch.get_height()
                if pd.isna(h):
                    continue
                ax.annotate(
                    fmt.format(h),
                    (patch.get_x() + patch.get_width() / 2, h),
                    ha="center",
                    va="bottom" if h >= 0 else "top",
                    xytext=(0, dy if h >= 0 else -dy),
                    textcoords="offset points",
                    fontsize=fontsize,
                    fontweight="bold",
                )

        def annotate_last(ax, x, y, label, color):
            ax.annotate(
                label,
                xy=(x.iloc[-1], y.iloc[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color=color,
                fontweight="bold",
            )

        def section_note(text):
            display(Markdown(f"> {text}"))
        """
    ),
    md("## 0. Load data và chuẩn hóa fact table"),
    code(
        """
        orders = pd.read_csv(BASE / "orders.csv", parse_dates=["order_date"])
        order_items = pd.read_csv(BASE / "order_items.csv")
        products = pd.read_csv(BASE / "products.csv")
        customers = pd.read_csv(BASE / "customers.csv", parse_dates=["signup_date"])
        web_traffic = pd.read_csv(BASE / "web_traffic.csv", parse_dates=["date"])
        promotions = pd.read_csv(BASE / "promotions.csv", parse_dates=["start_date", "end_date"])
        returns = pd.read_csv(BASE / "returns.csv", parse_dates=["return_date"])
        shipments = pd.read_csv(BASE / "shipments.csv", parse_dates=["ship_date", "delivery_date"])
        reviews = pd.read_csv(BASE / "reviews.csv", parse_dates=["review_date"])
        inventory = pd.read_csv(BASE / "inventory.csv", parse_dates=["snapshot_date"])
        geography = pd.read_csv(BASE / "geography.csv")

        order_items = order_items.drop(columns=["promo_id_2"], errors="ignore")

        items = (
            order_items
            .merge(products, on="product_id", how="left")
            .merge(orders, on="order_id", how="left")
            .merge(promotions[["promo_id", "promo_name", "promo_type", "discount_value"]], on="promo_id", how="left")
        )
        items["year"] = items["order_date"].dt.year
        items["month"] = items["order_date"].dt.month
        items["quarter"] = items["order_date"].dt.quarter
        items["gross_revenue"] = items["quantity"] * items["unit_price"]
        items["discount_amount"] = items["discount_amount"].fillna(0)
        items["net_revenue"] = items["gross_revenue"] - items["discount_amount"]
        items["total_cogs"] = items["quantity"] * items["cogs"].fillna(0)
        items["gross_profit"] = items["net_revenue"] - items["total_cogs"]
        items["promo_flag"] = items["promo_id"].notna()

        order_fin = items.groupby("order_id", as_index=False).agg(
            order_revenue=("net_revenue", "sum"),
            gross_revenue=("gross_revenue", "sum"),
            order_discount=("discount_amount", "sum"),
            order_cogs=("total_cogs", "sum"),
            order_gross_profit=("gross_profit", "sum"),
            units=("quantity", "sum"),
            promo_order=("promo_flag", "max"),
        )

        ret_order = returns.groupby("order_id", as_index=False).agg(
            refund_amount=("refund_amount", "sum"),
            return_lines=("return_id", "count"),
        )
        rev_order = reviews.groupby("order_id", as_index=False).agg(avg_rating=("rating", "mean"))

        fact_orders = (
            orders
            .merge(order_fin, on="order_id", how="left")
            .merge(shipments, on="order_id", how="left")
            .merge(ret_order, on="order_id", how="left")
            .merge(rev_order, on="order_id", how="left")
            .merge(customers[["customer_id", "acquisition_channel", "age_group", "gender"]], on="customer_id", how="left")
            .merge(geography[["zip", "region", "district"]], on="zip", how="left")
        )
        fact_orders[["refund_amount", "return_lines"]] = fact_orders[["refund_amount", "return_lines"]].fillna(0)
        fact_orders["returned_flag"] = fact_orders["return_lines"].gt(0)
        fact_orders["profit_after_refund"] = fact_orders["order_gross_profit"].fillna(0) - fact_orders["refund_amount"]
        fact_orders["delivery_lead_days"] = (fact_orders["delivery_date"] - fact_orders["order_date"]).dt.days
        fact_orders["year"] = fact_orders["order_date"].dt.year
        fact_orders["month"] = fact_orders["order_date"].dt.month
        fact_orders["year_month"] = fact_orders["order_date"].dt.to_period("M").astype(str)

        year_min, year_max = int(fact_orders["year"].min()), int(fact_orders["year"].max())
        display(pd.DataFrame({
            "table": ["orders", "order_items", "products", "customers", "web_traffic", "returns", "shipments", "reviews", "inventory"],
            "rows": [len(orders), len(order_items), len(products), len(customers), len(web_traffic), len(returns), len(shipments), len(reviews), len(inventory)]
        }))
        print(f"Data window: {year_min} - {year_max}")
        """
    ),
    md(
        """
        ## 1. Doanh thu giảm không phải vì khách mua ít tiền hơn, mà vì customer engine co lại

        **Câu hỏi:** Doanh thu giảm do yếu tố nào?

        **Chart cần đọc:** Revenue bridge từ 2016 đến 2022, tách tác động của active customers, orders/customer và AOV.
        """
    ),
    code(
        """
        annual = fact_orders.groupby("year", as_index=False).agg(
            net_sales=("order_revenue", "sum"),
            active_customers=("customer_id", "nunique"),
            orders=("order_id", "nunique"),
        )
        annual["orders_per_customer"] = annual["orders"] / annual["active_customers"]
        annual["aov"] = annual["net_sales"] / annual["orders"]
        annual["sales_per_customer"] = annual["net_sales"] / annual["active_customers"]

        start_year = 2016 if 2016 in annual["year"].values else int(annual["year"].min())
        end_year = 2022 if 2022 in annual["year"].values else int(annual["year"].max())
        base = annual.set_index("year").loc[start_year]
        end = annual.set_index("year").loc[end_year]

        rev0 = base["active_customers"] * base["orders_per_customer"] * base["aov"]
        after_customers = end["active_customers"] * base["orders_per_customer"] * base["aov"]
        after_repeat = end["active_customers"] * end["orders_per_customer"] * base["aov"]
        rev1 = end["active_customers"] * end["orders_per_customer"] * end["aov"]

        bridge = pd.DataFrame({
            "step": [
                f"Net sales {start_year}",
                "Active customers",
                "Orders / customer",
                "AOV",
                f"Net sales {end_year}",
            ],
            "delta": [
                rev0,
                after_customers - rev0,
                after_repeat - after_customers,
                rev1 - after_repeat,
                rev1,
            ],
            "type": ["total", "delta", "delta", "delta", "total"],
        })

        running = [0]
        for i in range(1, len(bridge) - 1):
            running.append(running[-1] + bridge.loc[i - 1, "delta"])
        running.append(0)
        bridge["base"] = running

        fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [1.1, 1]})

        ax = axes[0]
        for i, row in bridge.iterrows():
            color = COLORS["blue"] if row["type"] == "total" else (COLORS["green"] if row["delta"] >= 0 else COLORS["red"])
            bottom = 0 if row["type"] == "total" else row["base"]
            ax.bar(row["step"], row["delta"], bottom=bottom, color=color, width=0.62)
            label_y = bottom + row["delta"]
            ax.text(i, label_y, money_vnd(row["delta"]), ha="center", va="bottom" if row["delta"] >= 0 else "top", fontweight="bold")

        ax.axhline(0, color=COLORS["dark"], linewidth=1)
        ax.set_title(f"Revenue bridge: {start_year} -> {end_year}", loc="left", fontweight="bold")
        ax.set_ylabel("Net sales")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_vnd))
        ax.tick_params(axis="x", rotation=18)

        ax = axes[1]
        ax2 = ax.twinx()
        ax.plot(annual["year"], annual["net_sales"], marker="o", color=COLORS["blue"], linewidth=2.5, label="Net sales")
        ax.plot(annual["year"], annual["active_customers"], marker="o", color=COLORS["red"], linewidth=2.5, label="Active customers")
        ax2.plot(annual["year"], annual["aov"], marker="o", color=COLORS["green"], linewidth=2.5, label="AOV")
        ax.set_title("Net sales vs Active customers vs AOV", loc="left", fontweight="bold")
        ax.set_ylabel("Net sales / Active customers")
        ax2.set_ylabel("AOV")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_vnd))
        ax2.yaxis.set_major_formatter(mtick.FuncFormatter(money_vnd))
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=True, loc="best")

        plt.tight_layout()
        plt.show()

        sales_change = end["net_sales"] / base["net_sales"] - 1
        cust_change = end["active_customers"] / base["active_customers"] - 1
        aov_change = end["aov"] / base["aov"] - 1
        section_note(
            f"Từ {start_year} đến {end_year}, net sales đổi {sales_change:.1%}, active customers đổi {cust_change:.1%}, "
            f"trong khi AOV đổi {aov_change:.1%}. Nếu AOV tăng nhưng sales giảm, gốc vấn đề nằm ở customer base và repeat behavior."
        )
        display(annual.round(2))
        """
    ),
    md(
        """
        ## 2. Customer engine gãy ở ba điểm: acquisition yếu, retention giảm, loyalty bị bào mòn

        **Câu hỏi:** Vì sao customer base co lại?

        **Chart cần đọc:** New customers theo năm, retention/cohort repeat, mix tần suất mua, và tỷ trọng doanh thu từ khách 5+ đơn.
        """
    ),
    code(
        """
        cust_orders = fact_orders.sort_values("order_date")[["customer_id", "order_id", "order_date", "year", "order_revenue"]].copy()
        first_order = cust_orders.groupby("customer_id", as_index=False).agg(first_order_date=("order_date", "min"))
        first_order["cohort_year"] = first_order["first_order_date"].dt.year

        new_customers = first_order.groupby("cohort_year", as_index=False).agg(new_customers=("customer_id", "nunique"))

        # Retention sang nam sau: active customer nam Y co tiep tuc mua trong nam Y+1 hay khong.
        active_pairs = cust_orders[["customer_id", "year"]].drop_duplicates()
        ret_rows = []
        for y in sorted(active_pairs["year"].unique()):
            cur = set(active_pairs.loc[active_pairs["year"].eq(y), "customer_id"])
            nxt = set(active_pairs.loc[active_pairs["year"].eq(y + 1), "customer_id"])
            if cur:
                ret_rows.append({"year": y, "next_year_retention": len(cur & nxt) / len(cur), "active_customers": len(cur)})
        retention = pd.DataFrame(ret_rows)

        # Repeat trong 365 ngay cho cohort khach moi. Chi tinh cohort co du cua so 365 ngay.
        max_date = cust_orders["order_date"].max()
        cust_with_first = cust_orders.merge(first_order, on="customer_id", how="left")
        cust_with_first["days_since_first"] = (cust_with_first["order_date"] - cust_with_first["first_order_date"]).dt.days
        repeat_365 = (
            cust_with_first[cust_with_first["days_since_first"].between(0, 365)]
            .groupby(["customer_id", "cohort_year"], as_index=False)
            .agg(orders_365=("order_id", "nunique"))
        )
        repeat_365["repeat_365_flag"] = repeat_365["orders_365"].ge(2)
        repeat_365 = repeat_365.merge(first_order[["customer_id", "first_order_date"]], on="customer_id", how="left")
        repeat_365 = repeat_365[repeat_365["first_order_date"].le(max_date - pd.Timedelta(days=365))]
        repeat_365_summary = repeat_365.groupby("cohort_year", as_index=False).agg(repeat_365_rate=("repeat_365_flag", "mean"))

        cust_year_orders = cust_orders.groupby(["year", "customer_id"], as_index=False).agg(
            orders=("order_id", "nunique"),
            revenue=("order_revenue", "sum"),
        )
        cust_year_orders["frequency_group"] = pd.cut(
            cust_year_orders["orders"],
            bins=[0, 1, 4, np.inf],
            labels=["1 order", "2-4 orders", "5+ orders"],
        )
        freq_mix = (
            cust_year_orders.groupby(["year", "frequency_group"], observed=False, as_index=False)
            .agg(customers=("customer_id", "nunique"), revenue=("revenue", "sum"))
        )
        freq_mix["customer_share"] = freq_mix["customers"] / freq_mix.groupby("year")["customers"].transform("sum")
        freq_mix["revenue_share"] = freq_mix["revenue"] / freq_mix.groupby("year")["revenue"].transform("sum")
        freq_pivot = freq_mix.pivot(index="year", columns="frequency_group", values="customer_share").fillna(0)
        rev5 = freq_mix[freq_mix["frequency_group"].eq("5+ orders")][["year", "revenue_share"]]

        fig, axes = plt.subplots(2, 2, figsize=(18, 11))

        ax = axes[0, 0]
        sns.barplot(data=new_customers, x="cohort_year", y="new_customers", color=COLORS["blue"], ax=ax)
        ax.set_title("New customers theo năm", loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("New customers")
        add_bar_labels(ax, "{:.0f}", fontsize=8)

        ax = axes[0, 1]
        sns.lineplot(data=retention, x="year", y="next_year_retention", marker="o", linewidth=2.5, color=COLORS["red"], ax=ax, label="Next-year retention")
        sns.lineplot(data=repeat_365_summary, x="cohort_year", y="repeat_365_rate", marker="o", linewidth=2.5, color=COLORS["purple"], ax=ax, label="New cohort repeat within 365d")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Retention và repeat rate", loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Rate")
        ax.legend(frameon=True)

        ax = axes[1, 0]
        freq_pivot[["1 order", "2-4 orders", "5+ orders"]].plot(
            kind="bar",
            stacked=True,
            color=[COLORS["red"], COLORS["yellow"], COLORS["green"]],
            ax=ax,
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mix khách theo tần suất mua", loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Share of active customers")
        ax.legend(title="")

        ax = axes[1, 1]
        sns.lineplot(data=rev5, x="year", y="revenue_share", marker="o", linewidth=3, color=COLORS["green"], ax=ax)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Tỷ trọng doanh thu từ khách 5+ đơn", loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Revenue share")

        plt.tight_layout()
        plt.show()

        start_new = new_customers.loc[new_customers["cohort_year"].eq(start_year), "new_customers"]
        end_new = new_customers.loc[new_customers["cohort_year"].eq(end_year), "new_customers"]
        if len(start_new) and len(end_new):
            msg = f"New customers đổi {end_new.iloc[0] / start_new.iloc[0] - 1:.1%} từ {start_year} đến {end_year}."
        else:
            msg = "New customers giảm mạnh ở giai đoạn cuối."
        section_note(
            msg + " Khi acquisition yếu, repeat giảm và nhóm 5+ đơn co lại, doanh nghiệp buộc phải bù bằng traffic mới hoặc promo, thường đắt hơn và kém bền vững hơn."
        )
        """
    ),
    md(
        """
        ## 3. Traffic vẫn có, nhưng chất lượng chuyển đổi và chất lượng channel suy giảm

        **Câu hỏi:** Phễu đang rò ở đâu?

        **Chart cần đọc:** Sessions vs orders/1000 sessions theo năm và channel quality matrix.
        """
    ),
    code(
        """
        traffic_year = web_traffic.groupby(web_traffic["date"].dt.year, as_index=False).agg(
            sessions=("sessions", "sum"),
            unique_visitors=("unique_visitors", "sum"),
            page_views=("page_views", "sum"),
        ).rename(columns={"date": "year"})
        traffic_year["year"] = traffic_year["year"].astype(int)

        funnel_year = annual.merge(traffic_year, on="year", how="left")
        funnel_year["orders_per_1000_sessions"] = funnel_year["orders"] / funnel_year["sessions"] * 1000
        funnel_year["revenue_per_session"] = funnel_year["net_sales"] / funnel_year["sessions"]

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        ax = axes[0]
        ax2 = ax.twinx()
        ax.plot(funnel_year["year"], funnel_year["sessions"], color=COLORS["blue"], marker="o", linewidth=2.5, label="Sessions")
        ax2.plot(funnel_year["year"], funnel_year["orders_per_1000_sessions"], color=COLORS["red"], marker="o", linewidth=2.5, label="Orders / 1000 sessions")
        ax.set_title("Traffic vẫn còn, nhưng conversion yếu đi", loc="left", fontweight="bold")
        ax.set_ylabel("Sessions")
        ax2.set_ylabel("Orders / 1000 sessions")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_vnd))
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=True, loc="best")

        ax = axes[1]
        ax.plot(funnel_year["year"], funnel_year["revenue_per_session"], color=COLORS["purple"], marker="o", linewidth=2.8)
        ax.set_title("Revenue / session giảm", loc="left", fontweight="bold")
        ax.set_ylabel("Revenue / session")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_vnd))
        ax.set_xlabel("")

        plt.tight_layout()
        plt.show()

        traffic_channel = web_traffic.groupby("traffic_source", as_index=False).agg(sessions=("sessions", "sum"))
        channel_orders = fact_orders.groupby("order_source", as_index=False).agg(
            orders=("order_id", "nunique"),
            revenue=("order_revenue", "sum"),
            customers=("customer_id", "nunique"),
            refund=("refund_amount", "sum"),
            avg_rating=("avg_rating", "mean"),
        ).rename(columns={"order_source": "traffic_source"})
        channel_cust_orders = fact_orders.groupby(["order_source", "customer_id"], as_index=False).agg(orders=("order_id", "nunique"))
        channel_repeat = (
            channel_cust_orders.assign(repeat_flag=lambda d: d["orders"].ge(2))
            .groupby("order_source", as_index=False)
            .agg(repeat_rate=("repeat_flag", "mean"))
            .rename(columns={"order_source": "traffic_source"})
        )
        channel = (
            channel_orders
            .merge(traffic_channel, on="traffic_source", how="left")
            .merge(channel_repeat, on="traffic_source", how="left")
        )
        channel["conversion_rate"] = channel["orders"] / channel["sessions"]
        channel["revenue_per_session"] = channel["revenue"] / channel["sessions"]
        channel["refund_rate"] = channel["refund"] / channel["revenue"]
        for col in ["revenue_per_session", "conversion_rate", "repeat_rate"]:
            mn, mx = channel[col].min(), channel[col].max()
            channel[f"{col}_score"] = 0 if mx == mn else (channel[col] - mn) / (mx - mn)
        channel["quality_score"] = (
            0.45 * channel["revenue_per_session_score"]
            + 0.35 * channel["repeat_rate_score"]
            + 0.20 * (1 - channel["refund_rate"].rank(pct=True))
        )

        fig, ax = plt.subplots(figsize=(13, 8))
        sizes = 250 + 2600 * channel["sessions"] / channel["sessions"].max()
        sns.scatterplot(
            data=channel,
            x="revenue_per_session",
            y="repeat_rate",
            size=sizes,
            sizes=(300, 2800),
            hue="traffic_source",
            palette="tab10",
            alpha=0.78,
            edgecolor="white",
            linewidth=1.2,
            legend=False,
            ax=ax,
        )
        for _, r in channel.iterrows():
            ax.annotate(
                r["traffic_source"],
                (r["revenue_per_session"], r["repeat_rate"]),
                xytext=(8, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_title("Channel quality matrix: scale, efficiency, quality", loc="left", fontweight="bold")
        ax.set_xlabel("Revenue / session")
        ax.set_ylabel("Repeat rate")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(money_vnd))
        plt.tight_layout()
        plt.show()

        display(channel.sort_values("quality_score", ascending=False).round(4))
        section_note(
            "Không thể đánh giá channel bằng traffic hoặc revenue tổng. Channel cần được đo theo ba lớp: scale, efficiency và quality sau đơn hàng."
        )
        """
    ),
    md(
        """
        ## 4. Promo đang mua thêm volume nhưng không tạo ra lợi nhuận tương xứng

        **Câu hỏi:** Doanh thu có đang được mua bằng margin không?

        **Chart cần đọc:** Promo vs non-promo comparison và campaign profitability scatter.
        """
    ),
    code(
        """
        promo_line = items.groupby("promo_flag", as_index=False).agg(
            lines=("product_id", "count"),
            orders=("order_id", "nunique"),
            revenue=("net_revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            units=("quantity", "sum"),
        )
        promo_line["label"] = promo_line["promo_flag"].map({True: "Promo line", False: "Non-promo line"})
        promo_line["revenue_share"] = promo_line["revenue"] / promo_line["revenue"].sum()
        promo_line["gross_profit_share"] = promo_line["gross_profit"] / promo_line["gross_profit"].sum()
        promo_line["margin"] = promo_line["gross_profit"] / promo_line["revenue"]

        promo_order = fact_orders.groupby("promo_order", as_index=False).agg(
            orders=("order_id", "nunique"),
            revenue=("order_revenue", "sum"),
            units=("units", "sum"),
            gross_profit=("order_gross_profit", "sum"),
        )
        promo_order["label"] = promo_order["promo_order"].map({True: "Promo order", False: "Non-promo order"})
        promo_order["aov"] = promo_order["revenue"] / promo_order["orders"]
        promo_order["units_per_order"] = promo_order["units"] / promo_order["orders"]

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        comp = promo_line.melt(
            id_vars="label",
            value_vars=["revenue_share", "gross_profit_share", "margin"],
            var_name="metric",
            value_name="value",
        )
        ax = axes[0]
        sns.barplot(data=comp, x="metric", y="value", hue="label", palette=[COLORS["orange"], COLORS["blue"]], ax=ax)
        ax.set_title("Promo vs Non-promo: revenue share không đồng nghĩa profit", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(title="")
        add_bar_labels(ax, "{:.1%}", fontsize=8)

        order_comp = promo_order.melt(
            id_vars="label",
            value_vars=["aov", "units_per_order"],
            var_name="metric",
            value_name="value",
        )
        ax = axes[1]
        sns.barplot(data=order_comp, x="metric", y="value", hue="label", palette=[COLORS["orange"], COLORS["blue"]], ax=ax)
        ax.set_title("Promo order kéo units nhưng không chắc kéo AOV tốt", loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(title="")
        for p in ax.patches:
            h = p.get_height()
            if np.isfinite(h):
                ax.annotate(f"{h:.1f}", (p.get_x() + p.get_width()/2, h), ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.show()

        campaign = items[items["promo_flag"]].groupby("promo_name", as_index=False).agg(
            revenue=("net_revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            orders=("order_id", "nunique"),
            units=("quantity", "sum"),
        )
        campaign["revenue_share"] = campaign["revenue"] / campaign["revenue"].sum()
        campaign["margin"] = campaign["gross_profit"] / campaign["revenue"]
        campaign = campaign.sort_values("revenue", ascending=False)

        fig, ax = plt.subplots(figsize=(14, 8))
        sizes = 250 + 2600 * campaign["orders"] / campaign["orders"].max()
        ax.scatter(
            campaign["revenue_share"],
            campaign["margin"],
            s=sizes,
            c=np.where(campaign["margin"] < 0, COLORS["red"], COLORS["green"]),
            alpha=0.70,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.axhline(0, color=COLORS["dark"], linewidth=1.2)
        ax.axvline(campaign["revenue_share"].median(), color=COLORS["gray"], linestyle="--", linewidth=1)
        highlight = campaign.head(10)
        for _, r in highlight.iterrows():
            ax.annotate(
                r["promo_name"],
                (r["revenue_share"], r["margin"]),
                xytext=(7, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_title("Campaign profitability scatter", loc="left", fontweight="bold")
        ax.set_xlabel("Campaign revenue share")
        ax.set_ylabel("Gross margin")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.tight_layout()
        plt.show()

        display(campaign.head(12).round(4))
        promo_share = promo_line.loc[promo_line["promo_flag"], "revenue_share"].iloc[0]
        promo_gp_share = promo_line.loc[promo_line["promo_flag"], "gross_profit_share"].iloc[0]
        promo_margin = promo_line.loc[promo_line["promo_flag"], "margin"].iloc[0]
        non_margin = promo_line.loc[~promo_line["promo_flag"], "margin"].iloc[0]
        section_note(
            f"Promo đóng góp {promo_share:.1%} revenue nhưng chỉ {promo_gp_share:.1%} gross profit. "
            f"Margin promo là {promo_margin:.1%} so với non-promo {non_margin:.1%}; promo phải được quản trị như portfolio, không đánh giá một màu bằng revenue."
        )
        """
    ),
    md(
        """
        ## 5. Operations biến một phần doanh thu thành “doanh thu xấu”

        **Câu hỏi:** Sau khi có đơn, lợi nhuận thất thoát ở đâu?

        **Chart cần đọc:** Pareto refund reasons, delivery bucket quality, inventory risk matrix, và monthly margin/markdown.
        """
    ),
    code(
        """
        reason = returns.groupby("return_reason", as_index=False).agg(
            refund_amount=("refund_amount", "sum"),
            return_lines=("return_id", "count"),
        ).sort_values("refund_amount", ascending=False)
        reason["share"] = reason["refund_amount"] / reason["refund_amount"].sum()
        reason["cum_share"] = reason["share"].cumsum()

        fig, axes = plt.subplots(2, 2, figsize=(19, 12))

        ax = axes[0, 0]
        sns.barplot(data=reason, x="return_reason", y="refund_amount", color=COLORS["red"], ax=ax)
        ax2 = ax.twinx()
        ax2.plot(reason["return_reason"], reason["cum_share"], color=COLORS["navy"], marker="o", linewidth=2.5)
        ax.set_title("Pareto refund reasons", loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Refund amount")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(money_vnd))
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.tick_params(axis="x", rotation=25)
        for i, r in reason.iterrows():
            ax.text(i, r["refund_amount"], f"{r['share']:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        delivery = fact_orders.dropna(subset=["delivery_lead_days"]).copy()
        delivery["delivery_bucket"] = pd.cut(
            delivery["delivery_lead_days"],
            bins=[-np.inf, 3, 7, 10, np.inf],
            labels=["1-3 days", "4-7 days", "8-10 days", ">10 days"],
        )
        delivery_bucket = delivery.groupby("delivery_bucket", observed=False, as_index=False).agg(
            orders=("order_id", "nunique"),
            refund_rate=("returned_flag", "mean"),
            avg_rating=("avg_rating", "mean"),
            refund_amount=("refund_amount", "sum"),
        )
        delivery_bucket["order_share"] = delivery_bucket["orders"] / delivery_bucket["orders"].sum()

        ax = axes[0, 1]
        x = np.arange(len(delivery_bucket))
        width = 0.28
        ax.bar(x - width, delivery_bucket["refund_rate"], width, label="Return rate", color=COLORS["red"])
        ax.bar(x, delivery_bucket["order_share"], width, label="Order share", color=COLORS["blue"])
        ax2 = ax.twinx()
        ax2.plot(x + width, delivery_bucket["avg_rating"], color=COLORS["green"], marker="o", linewidth=2.5, label="Avg rating")
        ax.set_xticks(x)
        ax.set_xticklabels(delivery_bucket["delivery_bucket"])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Delivery bucket: tail 8-10 ngày tạo rủi ro", loc="left", fontweight="bold")
        ax.set_ylabel("Rate / share")
        ax2.set_ylabel("Avg rating")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=True, loc="best")

        inv_prod = inventory.merge(
            items.groupby("product_id", as_index=False).agg(revenue=("net_revenue", "sum"), units_sold_fact=("quantity", "sum")),
            on="product_id",
            how="left",
        )
        inv_sku = inv_prod.groupby(["product_id", "product_name", "category", "segment"], as_index=False).agg(
            avg_dos=("days_of_supply", "mean"),
            avg_sell_through=("sell_through_rate", "mean"),
            stockout_rate=("stockout_flag", "mean"),
            overstock_rate=("overstock_flag", "mean"),
            revenue=("revenue", "max"),
        )
        inv_sku["revenue"] = inv_sku["revenue"].fillna(0)
        inv_sku["inventory_risk"] = np.select(
            [
                (inv_sku["stockout_rate"] >= inv_sku["stockout_rate"].quantile(0.75)) & (inv_sku["revenue"] >= inv_sku["revenue"].quantile(0.75)),
                (inv_sku["avg_dos"] >= inv_sku["avg_dos"].quantile(0.75)) | (inv_sku["overstock_rate"] >= inv_sku["overstock_rate"].quantile(0.75)),
                (inv_sku["avg_dos"] <= inv_sku["avg_dos"].quantile(0.50)) & (inv_sku["stockout_rate"] <= inv_sku["stockout_rate"].quantile(0.50)),
            ],
            ["Hero SKU stockout", "Tail SKU overstock", "Healthy inventory"],
            default="Slow-moving risk",
        )

        ax = axes[1, 0]
        plot_inv = inv_sku.sample(min(len(inv_sku), 1200), random_state=42)
        sns.scatterplot(
            data=plot_inv,
            x="avg_dos",
            y="avg_sell_through",
            hue="inventory_risk",
            size="revenue",
            sizes=(30, 900),
            alpha=0.62,
            palette={
                "Hero SKU stockout": COLORS["red"],
                "Tail SKU overstock": COLORS["orange"],
                "Healthy inventory": COLORS["green"],
                "Slow-moving risk": COLORS["gray"],
            },
            ax=ax,
        )
        ax.axvline(inv_sku["avg_dos"].quantile(0.75), color=COLORS["gray"], linestyle="--", linewidth=1)
        ax.axhline(inv_sku["avg_sell_through"].quantile(0.50), color=COLORS["gray"], linestyle="--", linewidth=1)
        ax.set_title("Inventory risk matrix", loc="left", fontweight="bold")
        ax.set_xlabel("Average days of supply")
        ax.set_ylabel("Sell-through rate")
        ax.legend(frameon=True, bbox_to_anchor=(1.02, 1), loc="upper left")

        monthly = items.groupby(["year", "month"], as_index=False).agg(
            gross_revenue=("gross_revenue", "sum"),
            net_revenue=("net_revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            discount=("discount_amount", "sum"),
        )
        monthly["date"] = pd.to_datetime(monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01")
        monthly["margin_rate"] = monthly["gross_profit"] / monthly["net_revenue"]
        monthly["markdown_rate"] = monthly["discount"] / monthly["gross_revenue"]

        ax = axes[1, 1]
        ax2 = ax.twinx()
        ax.plot(monthly["date"], monthly["margin_rate"], color=COLORS["red"], linewidth=2.3, label="Margin rate")
        ax2.plot(monthly["date"], monthly["markdown_rate"], color=COLORS["purple"], linewidth=2.3, label="Markdown rate")
        ax.axhline(0, color=COLORS["dark"], linewidth=1)
        for _, row in monthly[monthly["month"].isin([8, 12])].iterrows():
            ax.axvspan(row["date"] - pd.Timedelta(days=15), row["date"] + pd.Timedelta(days=15), color=COLORS["orange"], alpha=0.10)
        ax.set_title("Monthly margin vs markdown: highlight T8/T12", loc="left", fontweight="bold")
        ax.set_ylabel("Margin rate")
        ax2.set_ylabel("Markdown rate")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=True, loc="best")

        plt.tight_layout()
        plt.show()

        wrong_size = reason.loc[reason["return_reason"].eq("wrong_size"), "share"]
        late_refund = reason.loc[reason["return_reason"].eq("late_delivery"), "refund_amount"]
        wrong_text = f"wrong_size chiếm {wrong_size.iloc[0]:.1%} refund value" if len(wrong_size) else "wrong_size là nhóm refund lớn"
        late_text = f"late_delivery refund = {money_vnd(late_refund.iloc[0])}" if len(late_refund) else "late_delivery tạo refund đáng kể"
        section_note(
            f"{wrong_text}; {late_text}. Operations leakage làm doanh thu sau đơn hàng biến thành refund, markdown và rating thấp."
        )
        display(reason.round(3))
        """
    ),
    md(
        """
        ## 6. Final synthesis: From Revenue Decline to Profit Leakage

        Driver tree này dùng để đặt ở slide tổng kết: từ revenue decline sau 2016 đến profitable growth suy yếu.
        """
    ),
    code(
        """
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.axis("off")

        nodes = [
            ("Revenue decline after 2016", 0.04, 0.55, COLORS["navy"]),
            ("Active customers, new customers, repeat giảm", 0.24, 0.75, COLORS["red"]),
            ("Traffic còn nhưng conversion yếu", 0.24, 0.35, COLORS["red"]),
            ("Promo được dùng để giữ volume", 0.47, 0.55, COLORS["orange"]),
            ("Promo tạo revenue nhưng margin mỏng", 0.67, 0.70, COLORS["orange"]),
            ("Operations rò profit: refund, late, markdown", 0.67, 0.40, COLORS["purple"]),
            ("Profitable growth suy yếu", 0.87, 0.55, COLORS["dark"]),
        ]

        for text, x, y, color in nodes:
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.15", facecolor=color, edgecolor="none"),
                transform=ax.transAxes,
            )

        arrows = [
            (0.12, 0.60, 0.20, 0.73),
            (0.12, 0.50, 0.20, 0.37),
            (0.32, 0.73, 0.41, 0.58),
            (0.32, 0.37, 0.41, 0.52),
            (0.55, 0.58, 0.61, 0.68),
            (0.55, 0.52, 0.61, 0.42),
            (0.75, 0.68, 0.82, 0.58),
            (0.75, 0.42, 0.82, 0.52),
        ]
        for x1, y1, x2, y2 in arrows:
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=2, color=COLORS["gray"]),
            )

        ax.set_title("Driver tree: From Revenue Decline to Profit Leakage", loc="left", fontsize=16, fontweight="bold")
        plt.show()

        final_text = '''
        **Synthesis:** Doanh nghiệp không chỉ giảm doanh thu sau 2016, mà còn suy giảm khả năng tạo ra doanh thu có chất lượng. Ở tầng customer, khách mới giảm, khách mới quay lại kém hơn và nhóm khách trung thành bị bào mòn. Ở tầng funnel/channel, traffic vẫn còn nhưng conversion và revenue/session giảm mạnh. Ở tầng promo, doanh nghiệp tạo được volume nhưng không chuyển hóa tương xứng thành gross profit. Ở tầng operations, sai size, giao hàng trễ và tồn kho lệch nhịp tiếp tục biến doanh thu thành refund, markdown và rating thấp.

        **Kết luận:** Tăng trưởng tiếp theo không nên được đo chỉ bằng revenue. Doanh nghiệp cần chuyển từ volume growth sang profitable growth.
        '''
        display(Markdown(final_text))
        """
    ),
    md("## 7. Action plan: ưu tiên, hành động, KPI"),
    code(
        """
        action_plan = pd.DataFrame([
            {
                "Priority": "Fix customer engine",
                "Action": "Tối ưu acquisition quality, lifecycle CRM cho khách mới, win-back nhóm từng mua nhiều lần",
                "KPI": "New customers, 90-day repeat, 365-day repeat, 5+ customer share",
            },
            {
                "Priority": "Fix channel quality",
                "Action": "Scale Organic Search có kiểm soát, dùng Email cho retention, bóc tách Direct attribution",
                "KPI": "Conversion rate, revenue/session, repeat rate, refund rate by channel",
            },
            {
                "Priority": "Fix promo economics",
                "Action": "Tách growth promo và clearance promo, siết campaign margin âm, ưu tiên promo margin dương",
                "KPI": "Promo margin, gross profit uplift, incremental revenue, sell-through",
            },
            {
                "Priority": "Fix operations leakage",
                "Action": "Size recommendation, exchange-first, ETA gating, replenish hero SKU, freeze tail SKU DOS cao",
                "KPI": "Wrong_size refund, late_delivery refund, exchange rate, p90 DOS, markdown rate",
            },
            {
                "Priority": "Fix seasonality risk",
                "Action": "Không đẩy volume mù vào Q3/T12 nếu margin và inventory chưa sẵn sàng",
                "KPI": "Monthly margin, markdown rate, stockout rate, contribution profit",
            },
        ])

        fig, ax = plt.subplots(figsize=(18, 5.2))
        ax.axis("off")
        tbl = ax.table(
            cellText=action_plan.values,
            colLabels=action_plan.columns,
            loc="center",
            cellLoc="left",
            colLoc="left",
            colWidths=[0.18, 0.46, 0.36],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2.0)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#D1D5DB")
            if row == 0:
                cell.set_facecolor(COLORS["navy"])
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            elif col == 0:
                cell.set_facecolor("#EAF2FF")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("white")
        ax.set_title("Action plan: chuyển từ volume growth sang profitable growth", loc="left", fontsize=15, fontweight="bold")
        plt.show()

        display(action_plan)
        display(Markdown(
            "> **Slide cuối:** The business does not simply have a revenue problem. It has a profitable growth problem. "
            "After 2016, the customer engine weakened, channel and promo quality became uneven, and operational leakage turned part of the revenue into refunds, markdowns and negative margin. "
            "The priority is therefore not to push more volume into the same system, but to rebuild the engine that creates good revenue: better customers, better channels, better promo economics and lower operational leakage."
        ))
        """
    ),
]


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

OUT.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Wrote {OUT}")
