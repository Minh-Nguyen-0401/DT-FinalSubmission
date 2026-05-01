import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import streamlit as st
from datetime import datetime
from pathlib import Path
from plotly.subplots import make_subplots

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

st.set_page_config(page_title="Ops & Quality Dashboard", layout="wide")

DEFAULT_AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

INSIGHT_DEPTH_CONFIG = {
    "Deep Executive": {
        "temperature": 0.08,
        "style": (
            "Tập trung executive: ít nhưng sắc, ưu tiên profit pool, margin leakage, refund drag, "
            "return root cause, và quyết định vận hành có thể làm ngay."
        ),
    },
    "Forensic Profit": {
        "temperature": 0.05,
        "style": (
            "Tập trung forensic: đào cơ chế bên trong. Với mỗi insight phải đi qua chuỗi "
            "metric -> pattern -> cơ chế kinh doanh -> giả thuyết nguyên nhân -> hành động kiểm chứng -> tác động profit."
        ),
    },
}


def build_insight_system_prompt(depth_mode: str) -> str:
    depth_style = INSIGHT_DEPTH_CONFIG.get(depth_mode, INSIGHT_DEPTH_CONFIG["Deep Executive"])["style"]
    return (
        "Bạn là senior profit intelligence agent cho thương mại điện tử. "
        "Mục tiêu không phải mô tả dashboard, mà là tìm bản chất kinh doanh nằm sau số liệu. "
        f"Chế độ phân tích: {depth_style} "
        "BẮT BUỘC trả lời bằng tiếng Việt, có cấu trúc, sắc bén và có bằng chứng số liệu từ context. "
        "Luật chống insight chung chung: không được viết các câu kiểu 'cần cải thiện trải nghiệm', "
        "'tối ưu vận hành', 'ảnh hưởng doanh thu/lợi nhuận' nếu không gắn với metric cụ thể, cơ chế cụ thể và hành động cụ thể. "
        "Luật bám dữ liệu: chỉ dùng số có trong context; nếu thiếu số để kết luận nhân quả thì nói là giả thuyết và nêu cách kiểm chứng. "
        "Luật ưu tiên: profit_after_refund và profit_rate đứng trước revenue; revenue lớn nhưng profit thấp phải được xem là rủi ro, không phải thành tích. "
        "Luật phản biện: mỗi insight mạnh phải chỉ ra một dấu hiệu cần kiểm tra thêm hoặc một rủi ro nếu hành động sai."
    )


def build_insight_user_prompt(section_name: str, time_grain: str, global_context: str, context_text: str) -> str:
    return (
        f"PHẦN PHÂN TÍCH: {section_name}\n"
        f"Độ phân giải thời gian hiện tại: {time_grain}.\n\n"
        "Hãy tạo insight theo format sau:\n"
        "1) Executive thesis: 2-3 câu kết luận bản chất đang xảy ra, ưu tiên profit/margin/refund/return.\n"
        "2) Profit diagnosis: 3-5 insight sâu nhất. Mỗi insight phải có:\n"
        "   - Phát hiện: pattern không hiển nhiên từ số liệu.\n"
        "   - Bằng chứng: trích metric cụ thể từ context.\n"
        "   - Cơ chế bên trong: vì sao pattern này có thể làm profit thay đổi.\n"
        "   - Hành động: việc cụ thể nên làm, không nói chung chung.\n"
        "   - Kiểm chứng: metric cần theo dõi để xác nhận giả thuyết.\n"
        "3) Contradictions / traps: chỉ ra chỗ dễ đọc sai chart, ví dụ revenue tăng nhưng profit_rate giảm, refund kéo margin âm, hoặc sample thiếu.\n"
        "4) Priority moves: top 5 hành động xếp theo tác động profit_after_refund, ghi rõ owner gợi ý và KPI theo dõi.\n"
        "5) What not to do: 2-3 hành động không nên làm vì có thể tăng revenue nhưng phá margin.\n\n"
        f"GLOBAL CONTEXT (đa năm + model):\n{global_context}\n\n"
        f"DATA CONTEXT:\n{context_text}"
    )


def normalize_chat_temperature(model: str, temperature: float) -> float:
    model_id = str(model or "").lower()
    if model_id.startswith("gpt-5"):
        return 1
    return temperature


def inject_ui_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }
        .main > div {
            padding-top: 1.2rem;
            padding-bottom: 1rem;
        }
        .dashboard-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #2563eb 100%);
            color: white;
            border-radius: 22px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }
        .hero-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.65rem;
        }
        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .brand-logo {
            width: 48px;
            height: 48px;
            border-radius: 14px;
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.24);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            letter-spacing: 0.04em;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.2);
        }
        .brand-title {
            margin: 0;
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 0.01em;
        }
        .brand-subtitle {
            margin: 0.1rem 0 0 0;
            opacity: 0.84;
            font-size: 0.92rem;
        }
        .hero-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.18);
            color: white;
            border-radius: 999px;
            padding: 0.38rem 0.75rem;
            font-size: 0.82rem;
            white-space: nowrap;
        }
        .dashboard-hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.2;
        }
        .dashboard-hero p {
            margin: 0.35rem 0 0 0;
            opacity: 0.9;
            font-size: 0.98rem;
        }
        .section-card {
            background: rgba(255,255,255,0.75);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin: 0.6rem 0 1rem 0;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .section-card h3 {
            margin-bottom: 0.25rem;
        }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(148,163,184,0.18);
            padding: 14px 16px;
            border-radius: 18px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.86rem;
            color: #475569;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 700;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255,255,255,0.68);
            padding: 8px;
            border-radius: 16px;
            border: 1px solid rgba(148,163,184,0.15);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 10px 14px;
            background: transparent;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: #1d4ed8 !important;
            color: white !important;
        }
        .stButton > button {
            border-radius: 12px;
            padding: 0.55rem 0.9rem;
            border: 1px solid rgba(37,99,235,0.25);
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            font-weight: 600;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(29, 78, 216, 0.22);
        }
        .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid rgba(100,116,139,0.2);
            background: white;
            color: #0f172a;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border-right: 1px solid rgba(148,163,184,0.18);
        }
        .insight-callout {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #0f172a;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            margin: 0.5rem 0 0.9rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header():
    st.markdown(
        """
        <div class="dashboard-hero">
            <div class="hero-top">
                <div class="brand-wrap">
                    <div class="brand-logo">H</div>
                    <div>
                        <p class="brand-title">HuyTr · Ops & Quality Command Center</p>
                        <p class="brand-subtitle">Revenue-first analytics · multi-year insight · AI copilot</p>
                    </div>
                </div>
                <div class="hero-pill">Dashboard ready for handoff</div>
            </div>
            <p>Deep-dive revenue, operational risk, customer voice, and structural patterns across years.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_card(title: str, description: str):
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_note(title: str, body: str):
    st.markdown(
        f"""
        <div class="insight-callout">
            <strong>{title}</strong><br/>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data(base_path: str = "."):
    orders = pd.read_csv(f"{base_path}/orders.csv")
    order_items = pd.read_csv(f"{base_path}/order_items.csv")
    products = pd.read_csv(f"{base_path}/products.csv")
    shipments = pd.read_csv(f"{base_path}/shipments.csv")
    returns = pd.read_csv(f"{base_path}/returns.csv")
    reviews = pd.read_csv(f"{base_path}/reviews.csv")
    inventory = pd.read_csv(f"{base_path}/inventory.csv")
    customers = pd.read_csv(f"{base_path}/customers.csv")
    geography = pd.read_csv(f"{base_path}/geography.csv")

    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    shipments["ship_date"] = pd.to_datetime(shipments["ship_date"], errors="coerce")
    shipments["delivery_date"] = pd.to_datetime(shipments["delivery_date"], errors="coerce")
    returns["return_date"] = pd.to_datetime(returns["return_date"], errors="coerce")
    reviews["review_date"] = pd.to_datetime(reviews["review_date"], errors="coerce")
    inventory["snapshot_date"] = pd.to_datetime(inventory["snapshot_date"], errors="coerce")

    items_enriched = order_items.merge(
        products[["product_id", "product_name", "category", "segment", "cogs"]],
        on="product_id",
        how="left",
    )
    items_enriched["gross_revenue"] = items_enriched["quantity"] * items_enriched["unit_price"]
    items_enriched["net_revenue"] = items_enriched["gross_revenue"] - items_enriched["discount_amount"].fillna(0)
    items_enriched["total_cogs"] = items_enriched["quantity"] * items_enriched["cogs"].fillna(0)
    items_enriched["gross_margin"] = items_enriched["net_revenue"] - items_enriched["total_cogs"]

    order_fin = items_enriched.groupby("order_id", as_index=False).agg(
        order_revenue=("net_revenue", "sum"),
        order_cogs=("total_cogs", "sum"),
        order_margin=("gross_margin", "sum"),
        item_count=("product_id", "count"),
        qty_sum=("quantity", "sum"),
    )

    order_core = (
        orders.merge(shipments, on="order_id", how="left")
        .merge(order_fin, on="order_id", how="left")
        .merge(customers[["customer_id", "city", "age_group", "acquisition_channel"]], on="customer_id", how="left")
        .merge(geography[["zip", "region", "district"]], on="zip", how="left")
    )

    order_core["ship_lead_days"] = (order_core["ship_date"] - order_core["order_date"]).dt.days
    order_core["delivery_lead_days"] = (order_core["delivery_date"] - order_core["order_date"]).dt.days
    order_core["post_ship_days"] = (order_core["delivery_date"] - order_core["ship_date"]).dt.days

    delay_threshold = order_core["delivery_lead_days"].quantile(0.75)
    order_core["late_delivery_flag"] = np.where(
        order_core["delivery_lead_days"].notna(),
        (order_core["delivery_lead_days"] > delay_threshold).astype(int),
        np.nan,
    )

    ret_by_order = returns.groupby("order_id", as_index=False).agg(
        return_lines=("return_id", "count"),
        refund_amount=("refund_amount", "sum"),
    )
    order_core = order_core.merge(ret_by_order, on="order_id", how="left")
    order_core["return_lines"] = order_core["return_lines"].fillna(0)
    order_core["refund_amount"] = order_core["refund_amount"].fillna(0)
    order_core["returned_flag"] = (order_core["return_lines"] > 0).astype(int)
    order_core["profit_after_refund"] = order_core["order_margin"].fillna(0) - order_core["refund_amount"].fillna(0)
    order_core["gross_margin_rate"] = np.where(
        order_core["order_revenue"].fillna(0) > 0,
        order_core["order_margin"].fillna(0) / order_core["order_revenue"].fillna(0),
        np.nan,
    )
    order_core["profit_rate"] = np.where(
        order_core["order_revenue"].fillna(0) > 0,
        order_core["profit_after_refund"] / order_core["order_revenue"].fillna(0),
        np.nan,
    )
    order_core["refund_rate"] = np.where(
        order_core["order_revenue"].fillna(0) > 0,
        order_core["refund_amount"].fillna(0) / order_core["order_revenue"].fillna(0),
        np.nan,
    )

    rev_by_order = reviews.groupby("order_id", as_index=False).agg(avg_rating=("rating", "mean"))
    order_core = order_core.merge(rev_by_order, on="order_id", how="left")
    order_core["low_rating_flag"] = np.where(
        order_core["avg_rating"].notna(),
        (order_core["avg_rating"] < 3).astype(int),
        np.nan,
    )
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

    line_quality = items_enriched.merge(
        returns[["order_id", "product_id", "return_quantity", "refund_amount", "return_reason"]],
        on=["order_id", "product_id"],
        how="left",
    )
    line_quality["return_quantity"] = line_quality["return_quantity"].fillna(0)
    line_quality["refund_amount"] = line_quality["refund_amount"].fillna(0)
    line_quality["is_returned_line"] = (line_quality["return_quantity"] > 0).astype(int)
    line_quality["profit_after_refund"] = line_quality["gross_margin"].fillna(0) - line_quality["refund_amount"].fillna(0)

    return {
        "orders": orders,
        "order_core": order_core,
        "items_enriched": items_enriched,
        "line_quality": line_quality,
        "returns": returns,
        "reviews": reviews,
        "inventory": inventory,
        "inv_prod": inv_prod,
        "delay_threshold": delay_threshold,
    }


def kpi_metrics(df: pd.DataFrame):
    delivered_mask = df["delivery_date"].notna()
    rating_mask = df["avg_rating"].notna()

    total_orders = int(df["order_id"].nunique())
    delivered_orders = int(delivered_mask.sum())
    avg_delivery = float(df.loc[delivered_mask, "delivery_lead_days"].mean()) if delivered_orders else np.nan
    p90_delivery = float(df.loc[delivered_mask, "delivery_lead_days"].quantile(0.9)) if delivered_orders else np.nan
    late_rate = float(df.loc[delivered_mask, "late_delivery_flag"].mean()) if delivered_orders else np.nan
    return_rate = float(df["returned_flag"].mean()) if total_orders else np.nan

    revenue_sum = float(df["order_revenue"].fillna(0).sum())
    margin_sum = float(df["order_margin"].fillna(0).sum())
    refund_sum = float(df["refund_amount"].fillna(0).sum())
    refund_ratio = refund_sum / revenue_sum if revenue_sum else np.nan
    profit_after_refund = margin_sum - refund_sum
    gross_margin_rate = margin_sum / revenue_sum if revenue_sum else np.nan
    profit_rate = profit_after_refund / revenue_sum if revenue_sum else np.nan
    avg_order_value = revenue_sum / total_orders if total_orders else np.nan
    revenue_at_risk = float(df.loc[(df["late_delivery_flag"] == 1) | (df["returned_flag"] == 1), "order_revenue"].fillna(0).sum())
    profit_at_risk = float(df.loc[(df["late_delivery_flag"] == 1) | (df["returned_flag"] == 1), "profit_after_refund"].fillna(0).sum())
    at_risk_share = revenue_at_risk / revenue_sum if revenue_sum else np.nan
    profit_at_risk_share = profit_at_risk / profit_after_refund if profit_after_refund else np.nan
    low_rating_rate = float(df.loc[rating_mask, "low_rating_flag"].mean()) if rating_mask.any() else np.nan

    return {
        "total_orders": total_orders,
        "delivered_orders": delivered_orders,
        "avg_delivery": avg_delivery,
        "p90_delivery": p90_delivery,
        "late_rate": late_rate,
        "return_rate": return_rate,
        "revenue_sum": revenue_sum,
        "margin_sum": margin_sum,
        "refund_sum": refund_sum,
        "profit_after_refund": profit_after_refund,
        "gross_margin_rate": gross_margin_rate,
        "profit_rate": profit_rate,
        "avg_order_value": avg_order_value,
        "revenue_at_risk": revenue_at_risk,
        "profit_at_risk": profit_at_risk,
        "at_risk_share": at_risk_share,
        "profit_at_risk_share": profit_at_risk_share,
        "refund_ratio": refund_ratio,
        "low_rating_rate": low_rating_rate,
    }


def safe_pct(value: float):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def safe_num(value: float, suffix: str = ""):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}{suffix}"


def compute_time_series(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    data = df.copy()
    if data.empty:
        return pd.DataFrame(columns=["time_label", "orders", "revenue", "margin", "profit_after_refund", "late_rate", "return_rate", "low_rating_rate", "refund"])

    if grain == "Ngày":
        data["time_sort"] = data["order_date"].dt.floor("D")
        data["time_label"] = data["time_sort"].dt.strftime("%Y-%m-%d")
    elif grain == "Tuần":
        data["time_sort"] = data["order_date"].dt.to_period("W").apply(lambda r: r.start_time)
        data["time_label"] = data["time_sort"].dt.strftime("W%U-%Y")
    elif grain == "Tháng":
        data["time_sort"] = data["order_date"].dt.to_period("M").dt.to_timestamp()
        data["time_label"] = data["time_sort"].dt.strftime("%Y-%m")
    elif grain == "Quý":
        data["time_sort"] = data["order_date"].dt.to_period("Q").dt.to_timestamp()
        data["time_label"] = data["order_date"].dt.to_period("Q").astype(str)
    elif grain == "Năm":
        data["time_sort"] = data["order_date"].dt.to_period("Y").dt.to_timestamp()
        data["time_label"] = data["time_sort"].dt.strftime("%Y")
    else:
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        data["time_label"] = pd.Categorical(data["order_date"].dt.day_name(), categories=weekday_order, ordered=True)
        data["time_sort"] = data["time_label"].cat.codes

    ts = (
        data.groupby(["time_sort", "time_label"], as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            revenue=("order_revenue", "sum"),
            margin=("order_margin", "sum"),
            profit_after_refund=("profit_after_refund", "sum"),
            late_rate=("late_delivery_flag", "mean"),
            return_rate=("returned_flag", "mean"),
            low_rating_rate=("low_rating_flag", "mean"),
            refund=("refund_amount", "sum"),
        )
        .sort_values("time_sort")
    )
    ts["gross_margin_rate"] = np.where(ts["revenue"] > 0, ts["margin"] / ts["revenue"], np.nan)
    ts["profit_rate"] = np.where(ts["revenue"] > 0, ts["profit_after_refund"] / ts["revenue"], np.nan)
    ts["refund_rate"] = np.where(ts["revenue"] > 0, ts["refund"] / ts["revenue"], np.nan)
    return ts


def compute_revenue_time_series(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    data = df.copy()
    if data.empty:
        return pd.DataFrame(columns=["time_label", "revenue", "margin", "profit_after_refund", "refund", "net_after_refund", "orders", "aov"])

    if grain == "Ngày":
        data["time_sort"] = data["order_date"].dt.floor("D")
        data["time_label"] = data["time_sort"].dt.strftime("%Y-%m-%d")
    elif grain == "Tuần":
        data["time_sort"] = data["order_date"].dt.to_period("W").apply(lambda r: r.start_time)
        data["time_label"] = data["time_sort"].dt.strftime("W%U-%Y")
    elif grain == "Tháng":
        data["time_sort"] = data["order_date"].dt.to_period("M").dt.to_timestamp()
        data["time_label"] = data["time_sort"].dt.strftime("%Y-%m")
    elif grain == "Quý":
        data["time_sort"] = data["order_date"].dt.to_period("Q").dt.to_timestamp()
        data["time_label"] = data["order_date"].dt.to_period("Q").astype(str)
    elif grain == "Năm":
        data["time_sort"] = data["order_date"].dt.to_period("Y").dt.to_timestamp()
        data["time_label"] = data["time_sort"].dt.strftime("%Y")
    else:
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        data["time_label"] = pd.Categorical(data["order_date"].dt.day_name(), categories=weekday_order, ordered=True)
        data["time_sort"] = data["time_label"].cat.codes

    ts = (
        data.groupby(["time_sort", "time_label"], as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            revenue=("order_revenue", "sum"),
            margin=("order_margin", "sum"),
            profit_after_refund=("profit_after_refund", "sum"),
            refund=("refund_amount", "sum"),
        )
        .sort_values("time_sort")
    )
    ts["net_after_refund"] = ts["revenue"] - ts["refund"]
    ts["aov"] = np.where(ts["orders"] > 0, ts["revenue"] / ts["orders"], 0)
    ts["gross_margin_rate"] = np.where(ts["revenue"] > 0, ts["margin"] / ts["revenue"], np.nan)
    ts["profit_rate"] = np.where(ts["revenue"] > 0, ts["profit_after_refund"] / ts["revenue"], np.nan)
    ts["refund_rate"] = np.where(ts["revenue"] > 0, ts["refund"] / ts["revenue"], np.nan)
    return ts


def build_revenue_factor_impact(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["factor", "group_a", "group_b", "avg_rev_a", "avg_rev_b", "avg_profit_a", "avg_profit_b", "delta_per_order", "delta_profit_per_order", "affected_orders", "est_revenue_uplift", "est_profit_uplift"])

    def binary_impact(flag_col: str, label: str):
        grp = df[["order_revenue", "profit_after_refund", flag_col]].dropna()
        if grp.empty:
            return None
        rev_flag = grp.loc[grp[flag_col] == 1, "order_revenue"]
        rev_ok = grp.loc[grp[flag_col] == 0, "order_revenue"]
        prof_flag = grp.loc[grp[flag_col] == 1, "profit_after_refund"]
        prof_ok = grp.loc[grp[flag_col] == 0, "profit_after_refund"]
        if len(rev_flag) == 0 or len(rev_ok) == 0:
            return None
        avg_a = float(rev_ok.mean())
        avg_b = float(rev_flag.mean())
        avg_profit_a = float(prof_ok.mean())
        avg_profit_b = float(prof_flag.mean())
        delta = avg_a - avg_b
        delta_profit = avg_profit_a - avg_profit_b
        affected = int((grp[flag_col] == 1).sum())
        uplift = max(delta, 0) * affected
        profit_uplift = max(delta_profit, 0) * affected
        return {
            "factor": label,
            "group_a": "Không bị ảnh hưởng",
            "group_b": "Bị ảnh hưởng",
            "avg_rev_a": avg_a,
            "avg_rev_b": avg_b,
            "avg_profit_a": avg_profit_a,
            "avg_profit_b": avg_profit_b,
            "delta_per_order": delta,
            "delta_profit_per_order": delta_profit,
            "affected_orders": affected,
            "est_revenue_uplift": uplift,
            "est_profit_uplift": profit_uplift,
        }

    impacts = []
    for col, label in [
        ("late_delivery_flag", "Giao trễ"),
        ("returned_flag", "Có trả hàng"),
        ("low_rating_flag", "Rating thấp (<3)"),
    ]:
        rec = binary_impact(col, label)
        if rec:
            impacts.append(rec)

    if not impacts:
        return pd.DataFrame(columns=["factor", "group_a", "group_b", "avg_rev_a", "avg_rev_b", "avg_profit_a", "avg_profit_b", "delta_per_order", "delta_profit_per_order", "affected_orders", "est_revenue_uplift", "est_profit_uplift"])
    return pd.DataFrame(impacts).sort_values("est_profit_uplift", ascending=False)


def compute_multiyear_revenue_patterns(df: pd.DataFrame):
    base_cols = [
        "order_date",
        "order_id",
        "order_revenue",
        "order_margin",
        "profit_after_refund",
        "refund_amount",
    ]
    data = df[[c for c in base_cols if c in df.columns]].dropna(subset=["order_date"]).copy()
    if data.empty:
        empty_annual = pd.DataFrame(columns=["year", "orders", "revenue", "margin", "profit_after_refund", "refund", "aov", "margin_rate", "profit_rate", "refund_rate", "yoy_revenue", "yoy_profit"]) 
        empty_quarter = pd.DataFrame(columns=["quarter", "avg_revenue_share", "avg_profit_share", "avg_margin_rate", "avg_refund_rate", "std_revenue_share", "top_quarter_years"]) 
        empty_month = pd.DataFrame(columns=["month", "avg_revenue_share", "avg_profit_share", "avg_margin_rate", "avg_refund_rate", "std_revenue_share"]) 
        empty_dow = pd.DataFrame(columns=["dow_name", "avg_daily_revenue", "median_daily_revenue"]) 
        return {
            "annual": empty_annual,
            "quarter": empty_quarter,
            "month": empty_month,
            "weekday": empty_dow,
            "summary": {
                "avg_yoy_growth": np.nan,
                "best_quarter": None,
                "worst_quarter": None,
                "most_stable_quarter": None,
            },
        }

    data["year"] = data["order_date"].dt.year
    data["quarter"] = data["order_date"].dt.quarter
    data["month"] = data["order_date"].dt.month
    data["dow"] = data["order_date"].dt.dayofweek

    annual = data.groupby("year", as_index=False).agg(
        orders=("order_id", "nunique"),
        revenue=("order_revenue", "sum"),
        margin=("order_margin", "sum"),
        profit_after_refund=("profit_after_refund", "sum"),
        refund=("refund_amount", "sum"),
    )
    annual = annual.sort_values("year")
    annual["aov"] = np.where(annual["orders"] > 0, annual["revenue"] / annual["orders"], 0)
    annual["margin_rate"] = np.where(annual["revenue"] > 0, annual["margin"] / annual["revenue"], 0)
    annual["profit_rate"] = np.where(annual["revenue"] > 0, annual["profit_after_refund"] / annual["revenue"], 0)
    annual["refund_rate"] = np.where(annual["revenue"] > 0, annual["refund"] / annual["revenue"], 0)
    annual["yoy_revenue"] = annual["revenue"].pct_change()
    annual["yoy_profit"] = annual["profit_after_refund"].pct_change()

    quarter_share = data.groupby(["year", "quarter"], as_index=False).agg(
        revenue=("order_revenue", "sum"),
        margin=("order_margin", "sum"),
        profit_after_refund=("profit_after_refund", "sum"),
        refund=("refund_amount", "sum"),
    )
    quarter_total = quarter_share.groupby("year", as_index=False).agg(
        year_revenue=("revenue", "sum"),
        year_profit=("profit_after_refund", "sum"),
    )
    quarter_share = quarter_share.merge(quarter_total, on="year", how="left")
    quarter_share["revenue_share"] = np.where(quarter_share["year_revenue"] > 0, quarter_share["revenue"] / quarter_share["year_revenue"], 0)
    quarter_share["profit_share"] = np.where(quarter_share["year_profit"] != 0, quarter_share["profit_after_refund"] / quarter_share["year_profit"], 0)
    quarter_share["margin_rate"] = np.where(quarter_share["revenue"] > 0, quarter_share["margin"] / quarter_share["revenue"], 0)
    quarter_share["refund_rate"] = np.where(quarter_share["revenue"] > 0, quarter_share["refund"] / quarter_share["revenue"], 0)

    q_top_per_year = quarter_share.loc[quarter_share.groupby("year")["revenue_share"].idxmax(), ["year", "quarter"]]
    q_top_count = q_top_per_year.groupby("quarter", as_index=False).agg(top_quarter_years=("year", "count"))
    quarter_pattern = quarter_share.groupby("quarter", as_index=False).agg(
        avg_revenue_share=("revenue_share", "mean"),
        avg_profit_share=("profit_share", "mean"),
        avg_margin_rate=("margin_rate", "mean"),
        avg_refund_rate=("refund_rate", "mean"),
        std_revenue_share=("revenue_share", "std"),
    )
    quarter_pattern = quarter_pattern.merge(q_top_count, on="quarter", how="left")
    quarter_pattern["top_quarter_years"] = quarter_pattern["top_quarter_years"].fillna(0).astype(int)
    quarter_pattern = quarter_pattern.sort_values("avg_revenue_share", ascending=False)

    month_share = data.groupby(["year", "month"], as_index=False).agg(
        revenue=("order_revenue", "sum"),
        margin=("order_margin", "sum"),
        profit_after_refund=("profit_after_refund", "sum"),
        refund=("refund_amount", "sum"),
    )
    month_total = month_share.groupby("year", as_index=False).agg(
        year_revenue=("revenue", "sum"),
        year_profit=("profit_after_refund", "sum"),
    )
    month_share = month_share.merge(month_total, on="year", how="left")
    month_share["revenue_share"] = np.where(month_share["year_revenue"] > 0, month_share["revenue"] / month_share["year_revenue"], 0)
    month_share["profit_share"] = np.where(month_share["year_profit"] != 0, month_share["profit_after_refund"] / month_share["year_profit"], 0)
    month_share["margin_rate"] = np.where(month_share["revenue"] > 0, month_share["margin"] / month_share["revenue"], 0)
    month_share["refund_rate"] = np.where(month_share["revenue"] > 0, month_share["refund"] / month_share["revenue"], 0)
    month_pattern = month_share.groupby("month", as_index=False).agg(
        avg_revenue_share=("revenue_share", "mean"),
        avg_profit_share=("profit_share", "mean"),
        avg_margin_rate=("margin_rate", "mean"),
        avg_refund_rate=("refund_rate", "mean"),
        std_revenue_share=("revenue_share", "std"),
    ).sort_values("avg_revenue_share", ascending=False)

    weekday_pattern = data.groupby("dow", as_index=False).agg(
        avg_daily_revenue=("order_revenue", "mean"),
        median_daily_revenue=("order_revenue", "median"),
    )
    weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    weekday_pattern["dow_name"] = weekday_pattern["dow"].map(weekday_map)
    weekday_pattern = weekday_pattern.sort_values("avg_daily_revenue", ascending=False)

    avg_yoy_growth = annual["yoy_revenue"].dropna().mean() if len(annual) > 1 else np.nan
    best_quarter = int(quarter_pattern.iloc[0]["quarter"]) if len(quarter_pattern) else None
    worst_quarter = int(quarter_pattern.sort_values("avg_revenue_share").iloc[0]["quarter"]) if len(quarter_pattern) else None
    stable_quarter = int(quarter_pattern.sort_values("std_revenue_share").iloc[0]["quarter"]) if len(quarter_pattern) else None

    return {
        "annual": annual,
        "quarter": quarter_pattern,
        "month": month_pattern,
        "weekday": weekday_pattern,
        "summary": {
            "avg_yoy_growth": avg_yoy_growth,
            "best_quarter": best_quarter,
            "worst_quarter": worst_quarter,
            "most_stable_quarter": stable_quarter,
        },
    }


def compute_revenue_decline_story(df: pd.DataFrame):
    base_cols = [
        "order_date",
        "order_id",
        "order_revenue",
        "order_margin",
        "profit_after_refund",
        "refund_amount",
        "acquisition_channel",
        "region",
    ]
    data = df[[c for c in base_cols if c in df.columns]].dropna(subset=["order_date"]).copy()
    if data.empty:
        empty_annual = pd.DataFrame(columns=["year", "orders", "revenue", "profit_after_refund", "refund", "aov", "profit_rate", "refund_rate", "yoy_revenue", "yoy_orders", "yoy_aov", "yoy_profit"])
        empty_mix = pd.DataFrame(columns=["group", "revenue_base", "revenue_latest", "share_base", "share_latest", "delta_share", "orders_base", "orders_latest", "profit_base", "profit_latest"])
        return {
            "annual": empty_annual,
            "bridge": pd.DataFrame(columns=["step", "measure", "value"]),
            "channel_compare": empty_mix.copy(),
            "region_compare": empty_mix.copy(),
            "summary": {
                "base_year": None,
                "latest_year": None,
                "revenue_delta": np.nan,
                "orders_delta": np.nan,
                "aov_delta": np.nan,
                "order_effect": np.nan,
                "aov_effect": np.nan,
            },
        }

    data["year"] = data["order_date"].dt.year
    annual = data.groupby("year", as_index=False).agg(
        orders=("order_id", "nunique"),
        revenue=("order_revenue", "sum"),
        profit_after_refund=("profit_after_refund", "sum"),
        refund=("refund_amount", "sum"),
    ).sort_values("year")
    annual["aov"] = np.where(annual["orders"] > 0, annual["revenue"] / annual["orders"], np.nan)
    annual["profit_rate"] = np.where(annual["revenue"] > 0, annual["profit_after_refund"] / annual["revenue"], np.nan)
    annual["refund_rate"] = np.where(annual["revenue"] > 0, annual["refund"] / annual["revenue"], np.nan)
    annual["yoy_revenue"] = annual["revenue"].pct_change()
    annual["yoy_orders"] = annual["orders"].pct_change()
    annual["yoy_aov"] = annual["aov"].pct_change()
    annual["yoy_profit"] = annual["profit_after_refund"].pct_change()

    base_year = 2018 if (annual["year"] == 2018).any() else int(annual.iloc[0]["year"])
    latest_year = int(annual.iloc[-1]["year"])
    base_row = annual.loc[annual["year"] == base_year].iloc[0]
    latest_row = annual.iloc[-1]

    revenue_delta = float(latest_row["revenue"] - base_row["revenue"])
    orders_delta = float(latest_row["orders"] - base_row["orders"])
    aov_delta = float(latest_row["aov"] - base_row["aov"])
    order_effect = float(orders_delta * base_row["aov"])
    aov_effect = float(latest_row["orders"] * aov_delta)

    bridge = pd.DataFrame(
        [
            {"step": f"{base_year} revenue", "measure": "absolute", "value": float(base_row["revenue"])},
            {"step": "Order volume effect", "measure": "relative", "value": order_effect},
            {"step": "AOV effect", "measure": "relative", "value": aov_effect},
            {"step": f"{latest_year} revenue", "measure": "absolute", "value": float(latest_row["revenue"])},
        ]
    )

    def build_mix_compare(group_col: str):
        mix = data.groupby(["year", group_col], as_index=False).agg(
            orders=("order_id", "nunique"),
            revenue=("order_revenue", "sum"),
            profit=("profit_after_refund", "sum"),
        )
        mix["year_total"] = mix.groupby("year")["revenue"].transform("sum")
        mix["share"] = np.where(mix["year_total"] > 0, mix["revenue"] / mix["year_total"], 0)
        base_mix = mix.loc[mix["year"] == base_year, [group_col, "orders", "revenue", "profit", "share"]].rename(
            columns={"orders": "orders_base", "revenue": "revenue_base", "profit": "profit_base", "share": "share_base"}
        )
        latest_mix = mix.loc[mix["year"] == latest_year, [group_col, "orders", "revenue", "profit", "share"]].rename(
            columns={"orders": "orders_latest", "revenue": "revenue_latest", "profit": "profit_latest", "share": "share_latest"}
        )
        compare = base_mix.merge(latest_mix, on=group_col, how="outer").fillna(0)
        compare["delta_share"] = compare["share_latest"] - compare["share_base"]
        compare = compare.rename(columns={group_col: "group"}).sort_values("revenue_latest", ascending=False)
        return compare

    return {
        "annual": annual,
        "bridge": bridge,
        "channel_compare": build_mix_compare("acquisition_channel") if "acquisition_channel" in data.columns else pd.DataFrame(),
        "region_compare": build_mix_compare("region") if "region" in data.columns else pd.DataFrame(),
        "summary": {
            "base_year": base_year,
            "latest_year": latest_year,
            "revenue_delta": revenue_delta,
            "orders_delta": orders_delta,
            "aov_delta": aov_delta,
            "order_effect": order_effect,
            "aov_effect": aov_effect,
        },
    }


def fit_revenue_driver_model(df: pd.DataFrame):
    needed = ["order_date", "order_revenue", "late_delivery_flag", "returned_flag", "low_rating_flag", "order_source", "region"]
    data = df[[c for c in needed if c in df.columns]].dropna(subset=["order_date", "order_revenue"]).copy()
    if len(data) < 120:
        empty = pd.DataFrame(columns=["group", "baseline_mape", "mape_without_group", "delta_mape", "delta_r2"])
        return {"mape": np.nan, "r2": np.nan, "n_train": 0, "n_test": 0}, empty

    data = data.sort_values("order_date")
    data["quarter"] = data["order_date"].dt.quarter.astype(str)
    data["month"] = data["order_date"].dt.month.astype(str)
    data["dow"] = data["order_date"].dt.dayofweek.astype(str)
    min_date = data["order_date"].min()
    data["trend_day"] = (data["order_date"] - min_date).dt.days

    model_df = pd.get_dummies(
        data[["trend_day", "quarter", "month", "dow", "late_delivery_flag", "returned_flag", "low_rating_flag", "order_source", "region"]],
        columns=["quarter", "month", "dow", "order_source", "region"],
        drop_first=True,
    )
    model_df.insert(0, "intercept", 1.0)
    model_df = model_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    y = data["order_revenue"].astype(float).values

    split_idx = int(len(model_df) * 0.8)
    X_train = model_df.iloc[:split_idx].values
    X_test = model_df.iloc[split_idx:].values
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    if len(y_test) == 0 or len(y_train) == 0:
        empty = pd.DataFrame(columns=["group", "baseline_mape", "mape_without_group", "delta_mape", "delta_r2"])
        return {"mape": np.nan, "r2": np.nan, "n_train": len(y_train), "n_test": len(y_test)}, empty

    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    pred = X_test @ coef
    denom = np.where(np.abs(y_test) < 1e-9, 1.0, np.abs(y_test))
    baseline_mape = float(np.mean(np.abs(y_test - pred) / denom))
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    baseline_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    feature_names = model_df.columns.tolist()
    group_map = {
        "time": [c for c in feature_names if c.startswith("trend_day") or c.startswith("quarter_") or c.startswith("month_") or c.startswith("dow_")],
        "ops_quality": [c for c in feature_names if c in ["late_delivery_flag", "returned_flag", "low_rating_flag"]],
        "channel_geo": [c for c in feature_names if c.startswith("order_source_") or c.startswith("region_")],
    }

    X_test_df = model_df.iloc[split_idx:].copy()
    impacts = []
    for group, cols in group_map.items():
        if not cols:
            continue
        Xg = X_test_df.copy()
        Xg[cols] = 0.0
        pred_g = Xg.values @ coef
        mape_g = float(np.mean(np.abs(y_test - pred_g) / denom))
        ss_res_g = float(np.sum((y_test - pred_g) ** 2))
        r2_g = 1 - ss_res_g / ss_tot if ss_tot > 0 else np.nan
        impacts.append(
            {
                "group": group,
                "baseline_mape": baseline_mape,
                "mape_without_group": mape_g,
                "delta_mape": mape_g - baseline_mape,
                "delta_r2": baseline_r2 - r2_g if pd.notna(baseline_r2) and pd.notna(r2_g) else np.nan,
            }
        )

    impact_df = pd.DataFrame(impacts).sort_values("delta_mape", ascending=False) if impacts else pd.DataFrame(columns=["group", "baseline_mape", "mape_without_group", "delta_mape", "delta_r2"])
    metrics = {"mape": baseline_mape, "r2": baseline_r2, "n_train": len(y_train), "n_test": len(y_test)}
    return metrics, impact_df


def df_to_compact_csv(df: pd.DataFrame, cols: list[str], top_n: int = 8) -> str:
    if df.empty:
        return "No rows"
    available_cols = [col for col in cols if col in df.columns]
    return df[available_cols].head(top_n).to_csv(index=False)


def generate_revenue_decline_insights(api_key: str, decline_story: dict, annual_multi: pd.DataFrame, model: str = DEFAULT_AI_MODEL) -> tuple[str, str]:
    """
    Generate AI insights about why revenue declined after 2018.
    Analyzes across all business domains: promotion, operational friction, quality, inventory, etc.
    Returns (insights_text, error_message)
    """
    if decline_story.get("annual", pd.DataFrame()).empty:
        return "", "Không có dữ liệu giảm doanh thu để phân tích."
    
    try:
        decline_annual = decline_story.get("annual", pd.DataFrame())
        decline_bridge = decline_story.get("bridge", pd.DataFrame())
        decline_channel = decline_story.get("channel_compare", pd.DataFrame())
        decline_region = decline_story.get("region_compare", pd.DataFrame())
        decline_summary = decline_story.get("summary", {})
        
        # Build detailed context
        decline_context = f"""
DỮ LIỆU DECLINE (2018 🔻 → Latest year):

Annual Trend:
{df_to_compact_csv(decline_annual, ['year', 'orders', 'revenue', 'aov', 'profit_after_refund', 'profit_rate', 'refund_rate', 'yoy_orders', 'yoy_revenue', 'yoy_aov', 'yoy_profit'], top_n=20)}

Revenue Bridge Waterfall:
{df_to_compact_csv(decline_bridge, ['step', 'measure', 'value'], top_n=10)}

Channel Comparison 2018 vs Latest:
{df_to_compact_csv(decline_channel, ['group', 'orders_base', 'orders_latest', 'revenue_base', 'revenue_latest', 'share_base', 'share_latest', 'delta_share', 'profit_base', 'profit_latest'], top_n=20)}

Region Comparison 2018 vs Latest:
{df_to_compact_csv(decline_region, ['group', 'orders_base', 'orders_latest', 'revenue_base', 'revenue_latest', 'share_base', 'share_latest', 'delta_share', 'profit_base', 'profit_latest'], top_n=20)}

Multi-year context (wider view):
{df_to_compact_csv(annual_multi, ['year', 'orders', 'revenue', 'margin', 'profit_after_refund', 'margin_rate', 'profit_rate', 'refund_rate', 'yoy_revenue', 'yoy_profit'], top_n=20)}

Base Year: {decline_summary.get('base_year', 2018)}
Latest Year: {decline_summary.get('latest_year', 'N/A')}
Revenue Delta: {decline_summary.get('revenue_delta', 'N/A')}
Orders Delta: {decline_summary.get('orders_delta', 'N/A')}
AOV Delta: {decline_summary.get('aov_delta', 'N/A')}
Order Effect: {decline_summary.get('order_effect', 'N/A')}
AOV Effect: {decline_summary.get('aov_effect', 'N/A')}
"""
        
        system_prompt = build_insight_system_prompt("Deep Executive")
        
        user_prompt = f"""
PHÂN TÍCH NGUYÊN NHÂN GIẢ CÓ: Tại sao doanh thu giảm sau 2018?

Bạn là CEO/CFO phân tích một vụ rơi doanh thu 40% trong 1-2 năm rồi recovery chậm. Nhiệm vụ:

1) **Root Cause Diagnosis (cross-domain)**:
   - Đây có phải là vấn đề chiếm thị phần (market shrink, competitor, market trend)?
   - Hay là vấn đề của chúng ta cụ thể (acquisition bỏ, retention xấu, COGS tăng, promo không hiệu)?
   - Hay operational/quality (delivery chậm, return cao, inventory out)?
   - Hay business model (chỗ được margin tốt bị cut, chỗ được volume được thay bằng chỗ margin xấu)?
   
2) **Profit vs Volume separation**:
   - Volume rơi bao nhiêu %, AOV thay đổi ra sao, margin rate rơi không?
   - Refund/return có tăng khi rơi doanh thu không (dấu hiệu quality)?
   - Profit rate rơi mức nào - có phải do revenue xấu hay do cost tăng?
   
3) **Channel/region stability check**:
   - Các channel/region bỏng đi không đều không (dấu hiệu loss of competitive edge)?
   - Hay toàn bộ bị decline tương tự (dấu hiệu market-wide)?
   
4) **Recovery potential**:
   - Sau 2018, revenue có recovery không, speed bao nhiêu?
   - Nếu recovery chậm, bottleneck là gì (can't acquire, can't retain, structure xấu)?
   - Chi phí/cơ cấu lợi nhuận có thay đổi trong recovery không?

5) **Next questions to ask** (nếu thiếu dữ liệu):
   - Nên check acquisition cost, retention rate, COGS, promo budget, ops SLA
   - Nên drill down: category nào bị xóa, SKU nào discontinued, customer cohort nào mất

Format trả lời:
- **Executive Summary**: 1-2 câu kết luận chính.
- **5 Root Cause Hypotheses**: Xếp theo likelihood, mỗi cái ghi (likelihood %, evidence, check method).
- **Profit Impact Story**: Cách thay đổi volume/AOV/margin/refund dẫn tới profit như vậy.
- **Channel/Region Stability**: Có bằng chứng là market shrink hay company-specific problem?
- **Top 5 Priorities to investigate**: Cụ thể metrics, data sources, owner gợi ý.
- **Red Flags & Risks**: Cái gì trong dữ liệu không bình thường (ví dụ refund tăng, margin collapse, channel divergence).

DATA CONTEXT:
{decline_context}
"""
        
        content, err = call_openai_chat(api_key, model, system_prompt, user_prompt, temperature=0.15)
        return content or "", err or ""
    
    except Exception as e:
        return "", f"Lỗi tạo insight: {str(e)}"


def call_openai_chat(api_key: str, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2):
    if not api_key:
        return None, "Thiếu OpenAI API key."
    if OpenAI is None:
        return None, "Thiếu package openai. Hãy cài: pip install openai"

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=normalize_chat_temperature(model, temperature),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content, None
    except Exception as exc:
        return None, str(exc)


def call_openai_chat_with_history(api_key: str, model: str, system_prompt: str, messages: list[dict], temperature: float = 0.2):
    if not api_key:
        return None, "Thiếu OpenAI API key."
    if OpenAI is None:
        return None, "Thiếu package openai. Hãy cài: pip install openai"

    try:
        client = OpenAI(api_key=api_key)
        payload = [{"role": "system", "content": system_prompt}] + messages
        resp = client.chat.completions.create(
            model=model,
            temperature=normalize_chat_temperature(model, temperature),
            messages=payload,
        )
        return resp.choices[0].message.content, None
    except Exception as exc:
        return None, str(exc)


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", str(text).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "insight"


def get_insight_dir(base_path: str = ".") -> Path:
    insight_dir = Path(base_path) / "insight_exports"
    insight_dir.mkdir(parents=True, exist_ok=True)
    return insight_dir


def save_insight_file(
    base_path: str,
    section_key: str,
    section_name: str,
    model: str,
    time_grain: str,
    start_date,
    end_date,
    content: str,
):
    insight_dir = get_insight_dir(base_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{slugify(section_key)}_{slugify(time_grain)}.md"
    fpath = insight_dir / filename

    payload = (
        f"# {section_name}\n\n"
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}\n"
        f"- Model: {model}\n"
        f"- Time grain: {time_grain}\n"
        f"- Date range: {pd.to_datetime(start_date).date()} -> {pd.to_datetime(end_date).date()}\n"
        f"- Section key: {section_key}\n\n"
        "---\n\n"
        f"{content}\n"
    )
    fpath.write_text(payload, encoding="utf-8")
    return fpath


def list_saved_insight_files(base_path: str = ".", limit: int = 200):
    insight_dir = get_insight_dir(base_path)
    files = sorted(insight_dir.glob("*.md"), reverse=True)
    return files[:limit]


def final_synthesis_path(base_path: str = ".") -> Path:
    return get_insight_dir(base_path) / "_FINAL_INSIGHT_SYNTHESIS.md"


def list_source_insight_files(base_path: str = ".", limit: int = 300):
    final_path = final_synthesis_path(base_path).resolve()
    files = [
        fpath for fpath in list_saved_insight_files(base_path, limit=limit + 10)
        if fpath.resolve() != final_path
    ]
    return files[:limit]


def load_latest_saved_insight(base_path: str, section_key: str, time_grain: str):
    insight_dir = get_insight_dir(base_path)
    section_slug = slugify(section_key)
    grain_slug = slugify(time_grain)
    matches = sorted(insight_dir.glob(f"*_{section_slug}_{grain_slug}.md"), reverse=True)
    if not matches:
        matches = sorted(insight_dir.glob(f"*_{section_slug}_*.md"), reverse=True)
    if not matches:
        return None, ""
    latest = matches[0]
    return latest, latest.read_text(encoding="utf-8")


def final_synthesis_is_current(base_path: str = "."):
    final_path = final_synthesis_path(base_path)
    source_files = list_source_insight_files(base_path)
    if not final_path.exists() or not source_files:
        return False
    newest_source_mtime = max(fpath.stat().st_mtime for fpath in source_files)
    return final_path.stat().st_mtime >= newest_source_mtime


def build_final_synthesis_context(base_path: str = ".", max_chars_per_file: int = 7000, max_total_chars: int = 52000):
    chunks = []
    total_chars = 0
    for fpath in list_source_insight_files(base_path):
        content = fpath.read_text(encoding="utf-8")
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file] + "\n\n[Truncated for synthesis]"
        chunk = f"\n\n===== SOURCE FILE: {fpath.name} =====\n{content}"
        if total_chars + len(chunk) > max_total_chars:
            chunks.append("\n\n[Stopped: source context reached synthesis limit]")
            break
        chunks.append(chunk)
        total_chars += len(chunk)
    return "".join(chunks).strip()


def generate_final_synthesis(
    api_key: str,
    model: str,
    depth_mode: str,
    base_path: str,
    force: bool = False,
):
    final_path = final_synthesis_path(base_path)
    source_files = list_source_insight_files(base_path)
    if not source_files:
        return None, "Chưa có file insight nguồn để tổng hợp.", False
    if final_path.exists() and final_synthesis_is_current(base_path) and not force:
        return final_path, None, False

    context_text = build_final_synthesis_context(base_path)
    if not context_text:
        return None, "Không đọc được nội dung insight nguồn.", False

    system_prompt = build_insight_system_prompt(depth_mode)
    user_prompt = (
        "Hãy tổng hợp TOÀN BỘ các file insight nguồn thành một bản final executive synthesis. "
        "Không lặp lại từng file. Hãy gom thành một câu chuyện kinh doanh thống nhất, ưu tiên profit_after_refund, profit_rate, "
        "refund/return leakage, lane/category/source nào đáng xử lý trước, và các quyết định không nên làm. "
        "Bắt buộc có các phần:\n"
        "1) Executive conclusion: kết luận bản chất trong 5-7 bullet.\n"
        "2) Profit pool & profit leak: profit đến từ đâu, leak ở đâu, bằng chứng số liệu.\n"
        "3) Root-cause hypotheses: giả thuyết nguyên nhân sâu, kèm metric cần kiểm chứng.\n"
        "4) Priority action portfolio: top hành động 30/60/90 ngày, owner gợi ý, KPI theo dõi.\n"
        "5) Contradictions / traps: những điểm dễ đọc sai giữa revenue, margin, refund, return, ops risk.\n"
        "6) Final recommendation: 3 quyết định nên làm ngay và 3 quyết định không nên làm.\n\n"
        f"ALL SAVED INSIGHT FILES:\n{context_text}"
    )
    temperature = INSIGHT_DEPTH_CONFIG.get(depth_mode, INSIGHT_DEPTH_CONFIG["Deep Executive"])["temperature"]
    ai_text, ai_err = call_openai_chat(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
    )
    if ai_err:
        return None, ai_err, False

    now_text = datetime.now().isoformat(timespec="seconds")
    payload = (
        "# Final Insight Synthesis\n\n"
        f"- Up to date at: {now_text}\n"
        f"- Model: {model}\n"
        f"- Insight depth: {depth_mode}\n"
        f"- Source insight files: {len(source_files)}\n"
        f"- Newest source file: {max(source_files, key=lambda p: p.stat().st_mtime).name}\n\n"
        "---\n\n"
        f"{ai_text}\n"
    )
    final_path.write_text(payload, encoding="utf-8")
    return final_path, None, True


def render_section_intro(section_title: str, what_is_it: str, analyzing: str):
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{section_title}</h3>
            <div class="insight-callout"><strong>Phần này là gì?</strong> {what_is_it}</div>
            <div class="insight-callout"><strong>Đang phân tích gì?</strong> {analyzing}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tab_ai_insight(
    section_key: str,
    section_name: str,
    context_text: str,
    api_key: str,
    model: str,
    depth_mode: str,
    base_path: str,
    time_grain: str,
    start_date,
    end_date,
    global_context: str = "",
):
    st.markdown("#### AI Insight cho phần này")
    grain_slug = slugify(time_grain)
    output_key = f"ai_output_{section_key}_{grain_slug}"
    saved_path_key = f"ai_saved_path_{section_key}_{grain_slug}"
    if output_key not in st.session_state:
        saved_path, saved_content = load_latest_saved_insight(base_path, section_key, time_grain)
        st.session_state[output_key] = saved_content
        st.session_state[saved_path_key] = str(saved_path) if saved_path else ""

    if st.session_state[output_key]:
        if st.session_state.get(saved_path_key):
            st.caption(f"Đang hiển thị insight đã lưu: {Path(st.session_state[saved_path_key]).name}")
        st.markdown(st.session_state[output_key])
    else:
        if st.button(f"Gen AI Insight ({section_name})", key=f"gen_ai_{section_key}", use_container_width=True):
            if not api_key:
                st.warning("Cần thiết lập biến môi trường OPENAI_API_KEY để chạy AI.")
            else:
                with st.spinner("AI đang phân tích phần này..."):
                    system_prompt = build_insight_system_prompt(depth_mode)
                    user_prompt = build_insight_user_prompt(section_name, time_grain, global_context, context_text)
                    temperature = INSIGHT_DEPTH_CONFIG.get(depth_mode, INSIGHT_DEPTH_CONFIG["Deep Executive"])["temperature"]
                    ai_text, ai_err = call_openai_chat(
                        api_key=api_key,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                    )
                    st.session_state[output_key] = ai_text if not ai_err else f"AI error: {ai_err}"
                    if not ai_err and ai_text:
                        saved_path = save_insight_file(
                            base_path=base_path,
                            section_key=section_key,
                            section_name=section_name,
                            model=model,
                            time_grain=time_grain,
                            start_date=start_date,
                            end_date=end_date,
                            content=ai_text,
                        )
                        st.success(f"Đã lưu insight: {saved_path}")
                        st.session_state[saved_path_key] = str(saved_path)
            if st.session_state[output_key]:
                st.markdown(st.session_state[output_key])
        else:
            st.caption("Chưa có insight đã lưu. Bấm Gen AI Insight để tạo insight chuyên sâu cho riêng phần này.")


def generate_and_save_section_insight(
    section_key: str,
    section_name: str,
    context_text: str,
    api_key: str,
    model: str,
    depth_mode: str,
    base_path: str,
    time_grain: str,
    start_date,
    end_date,
    global_context: str = "",
):
    system_prompt = build_insight_system_prompt(depth_mode)
    user_prompt = build_insight_user_prompt(section_name, time_grain, global_context, context_text)
    temperature = INSIGHT_DEPTH_CONFIG.get(depth_mode, INSIGHT_DEPTH_CONFIG["Deep Executive"])["temperature"]
    ai_text, ai_err = call_openai_chat(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
    )
    if ai_err:
        return None, None, ai_err
    saved_path = save_insight_file(
        base_path=base_path,
        section_key=section_key,
        section_name=section_name,
        model=model,
        time_grain=time_grain,
        start_date=start_date,
        end_date=end_date,
        content=ai_text,
    )
    return saved_path, ai_text, None


def main():
    inject_ui_styles()
    render_app_header()

    data = load_data(".")
    order_core = data["order_core"].copy()
    line_quality = data["line_quality"].copy()
    inv_prod = data["inv_prod"].copy()
    delay_threshold = data["delay_threshold"]

    order_core["order_weekday"] = order_core["order_date"].dt.day_name()
    order_core["order_hour"] = 0
    order_core["delay_bucket"] = pd.cut(
        order_core["delivery_lead_days"],
        bins=[-1, 2, 4, 7, 10, 999],
        labels=["<=2", "3-4", "5-7", "8-10", ">10"],
    )

    date_min = order_core["order_date"].min()
    date_max = order_core["order_date"].max()

    with st.sidebar:
        st.subheader("Bộ lọc phân tích")
        region_opts = sorted([x for x in order_core["region"].dropna().unique().tolist()])
        source_opts = sorted([x for x in order_core["order_source"].dropna().unique().tolist()])
        status_opts = sorted([x for x in order_core["order_status"].dropna().unique().tolist()])
        category_opts = sorted([x for x in line_quality["category"].dropna().unique().tolist()])
        segment_opts = sorted([x for x in inv_prod["segment"].dropna().unique().tolist()])

        sel_date = st.date_input("Khoảng thời gian order", value=(date_min, date_max), min_value=date_min, max_value=date_max)
        sel_regions = st.multiselect("Region", options=region_opts, default=region_opts)
        sel_sources = st.multiselect("Order source", options=source_opts, default=source_opts)
        sel_status = st.multiselect("Order status", options=status_opts, default=status_opts)
        sel_categories = st.multiselect("Product category", options=category_opts, default=category_opts)
        sel_segments = st.multiselect("Inventory segment", options=segment_opts, default=segment_opts)
        time_grain = st.selectbox(
            "Độ phân giải thời gian",
            options=["Ngày", "Tuần", "Tháng", "Quý", "Năm", "Thứ trong tuần"],
            index=2,
        )
        min_lane_orders = st.slider("Min đơn cho lane analysis", min_value=10, max_value=200, value=30, step=10)

        st.markdown("---")
        st.subheader("AI Settings")
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model_presets = [
            DEFAULT_AI_MODEL,
            "gpt-5",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
        ]
        model_presets = list(dict.fromkeys([m for m in model_presets if m]))
        saved_model = st.session_state.get("ai_model", DEFAULT_AI_MODEL)
        if saved_model not in model_presets:
            model_presets.insert(0, saved_model)
        ai_model = st.selectbox("OpenAI Model preset", options=model_presets, index=model_presets.index(saved_model))
        custom_model = st.text_input("Custom model id", value="" if ai_model in model_presets else ai_model)
        if custom_model.strip():
            ai_model = custom_model.strip()
        insight_depth = st.selectbox(
            "Insight depth",
            options=list(INSIGHT_DEPTH_CONFIG.keys()),
            index=1,
            help="Forensic Profit đào sâu cơ chế profit/refund/return hơn, ít mô tả chart hơn.",
        )
        st.session_state["ai_model"] = ai_model
        st.session_state["insight_depth"] = insight_depth
        if openai_api_key:
            st.success("OPENAI_API_KEY đã được nạp từ biến môi trường.")
        else:
            st.info("AI sẽ tắt cho đến khi bạn set biến môi trường OPENAI_API_KEY.")

    if isinstance(sel_date, tuple) and len(sel_date) == 2:
        start_date, end_date = pd.to_datetime(sel_date[0]), pd.to_datetime(sel_date[1])
    else:
        start_date, end_date = date_min, date_max

    filtered = order_core[
        order_core["region"].isin(sel_regions)
        & order_core["order_source"].isin(sel_sources)
        & order_core["order_status"].isin(sel_status)
        & order_core["order_date"].between(start_date, end_date)
    ].copy()

    filtered_order_ids = filtered["order_id"].dropna().unique().tolist()
    line_filtered = line_quality[
        line_quality["order_id"].isin(filtered_order_ids)
        & line_quality["category"].isin(sel_categories)
    ].copy()

    inv_filtered = inv_prod[inv_prod["segment"].isin(sel_segments)].copy()
    time_series = compute_time_series(filtered, time_grain)
    revenue_series = compute_revenue_time_series(filtered, time_grain)
    revenue_factor_impact = build_revenue_factor_impact(filtered)

    multiyear_scope = order_core[
        order_core["region"].isin(sel_regions)
        & order_core["order_source"].isin(sel_sources)
        & order_core["order_status"].isin(sel_status)
    ].copy()
    multiyear_patterns = compute_multiyear_revenue_patterns(multiyear_scope)
    revenue_decline_story = compute_revenue_decline_story(multiyear_scope)
    driver_model_metrics, driver_group_impact = fit_revenue_driver_model(multiyear_scope)

    annual_multi = multiyear_patterns["annual"]
    quarter_multi = multiyear_patterns["quarter"]
    month_multi = multiyear_patterns["month"]
    weekday_multi = multiyear_patterns["weekday"]
    multi_summary = multiyear_patterns["summary"]
    revenue_decline_annual = revenue_decline_story["annual"]
    revenue_decline_bridge = revenue_decline_story["bridge"]
    revenue_decline_channel = revenue_decline_story["channel_compare"]
    revenue_decline_region = revenue_decline_story["region_compare"]
    revenue_decline_summary = revenue_decline_story["summary"]

    shared_ai_context = f"""
Multi-year annual summary:
{df_to_compact_csv(annual_multi, ['year', 'orders', 'revenue', 'margin', 'profit_after_refund', 'refund', 'aov', 'margin_rate', 'profit_rate', 'refund_rate', 'yoy_revenue', 'yoy_profit'], top_n=20)}

Revenue decline story (base year vs latest year):
{df_to_compact_csv(revenue_decline_annual, ['year', 'orders', 'revenue', 'profit_after_refund', 'refund', 'aov', 'profit_rate', 'refund_rate', 'yoy_revenue', 'yoy_orders', 'yoy_aov', 'yoy_profit'], top_n=20)}

Revenue bridge:
{df_to_compact_csv(revenue_decline_bridge, ['step', 'measure', 'value'], top_n=10)}

Channel compare base vs latest:
{df_to_compact_csv(revenue_decline_channel, ['group', 'revenue_base', 'revenue_latest', 'share_base', 'share_latest', 'delta_share', 'orders_base', 'orders_latest', 'profit_base', 'profit_latest'], top_n=20)}

Region compare base vs latest:
{df_to_compact_csv(revenue_decline_region, ['group', 'revenue_base', 'revenue_latest', 'share_base', 'share_latest', 'delta_share', 'orders_base', 'orders_latest', 'profit_base', 'profit_latest'], top_n=20)}

Quarter structural pattern:
{df_to_compact_csv(quarter_multi, ['quarter', 'avg_revenue_share', 'avg_profit_share', 'avg_margin_rate', 'avg_refund_rate', 'std_revenue_share', 'top_quarter_years'], top_n=10)}

Month structural pattern:
{df_to_compact_csv(month_multi, ['month', 'avg_revenue_share', 'avg_profit_share', 'avg_margin_rate', 'avg_refund_rate', 'std_revenue_share'], top_n=12)}

Weekday pattern:
{df_to_compact_csv(weekday_multi, ['dow_name', 'avg_daily_revenue', 'median_daily_revenue'], top_n=10)}

Key multi-year summary:
- avg_yoy_growth={multi_summary.get('avg_yoy_growth')}
- best_quarter={multi_summary.get('best_quarter')}
- worst_quarter={multi_summary.get('worst_quarter')}
- most_stable_quarter={multi_summary.get('most_stable_quarter')}

Revenue driver model (time split):
- mape={driver_model_metrics.get('mape')}
- r2={driver_model_metrics.get('r2')}
- n_train={driver_model_metrics.get('n_train')}
- n_test={driver_model_metrics.get('n_test')}

Driver group impact:
{df_to_compact_csv(driver_group_impact, ['group', 'baseline_mape', 'mape_without_group', 'delta_mape', 'delta_r2'], top_n=10)}
"""

    kpi = kpi_metrics(filtered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders", f"{kpi['total_orders']:,}")
    c2.metric("Late Delivery Rate", safe_pct(kpi["late_rate"]))
    c3.metric("Return Rate", safe_pct(kpi["return_rate"]))
    c4.metric("Refund / Revenue", safe_pct(kpi["refund_ratio"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Delivered Orders", f"{kpi['delivered_orders']:,}")
    c6.metric("Avg Delivery Lead", safe_num(kpi["avg_delivery"], " days"))
    c7.metric("P90 Delivery Lead", safe_num(kpi["p90_delivery"], " days"))
    c8.metric("Low Rating Rate (<3)", safe_pct(kpi["low_rating_rate"]))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Revenue", safe_num(kpi["revenue_sum"]))
    c10.metric("Gross Margin", safe_num(kpi["margin_sum"]))
    c11.metric("Profit After Refund", safe_num(kpi["profit_after_refund"]))
    c12.metric("Profit Rate", safe_pct(kpi["profit_rate"]))

    c13, c14, c15, c16 = st.columns(4)
    c13.metric("Gross Margin Rate", safe_pct(kpi["gross_margin_rate"]))
    c14.metric("AOV", safe_num(kpi["avg_order_value"]))
    c15.metric("Revenue At Risk", f"{safe_num(kpi['revenue_at_risk'])} ({safe_pct(kpi['at_risk_share'])})")
    c16.metric("Profit At Risk", f"{safe_num(kpi['profit_at_risk'])} ({safe_pct(kpi['profit_at_risk_share'])})")

    st.caption(f"Late threshold hiện tại (P75 delivery lead): {delay_threshold:.2f} ngày")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Executive",
        "Delivery Deep Dive",
        "Returns & Quality",
        "Customer Voice",
        "Inventory Control",
        "Geo/Channel Lens",
        "Revenue Impact",
        "Auto Insights",
        "Insight 2018+",
    ])

    with tab1:
        render_section_intro(
            "Executive Overview",
            "Góc nhìn điều hành tổng quan về profit pool, thất thoát refund và rủi ro vận hành.",
            f"Đọc theo {time_grain.lower()} để xem doanh thu có chuyển thành lợi nhuận hay bị bào mòn bởi refund, giao trễ, hoàn trả và rating thấp.",
        )

        fig_finance = px.line(
            time_series,
            x="time_label",
            y=["revenue", "margin", "profit_after_refund"],
            markers=True,
            title=f"Executive profit pool theo {time_grain.lower()}: Revenue -> Gross Margin -> Profit sau refund",
        )
        st.plotly_chart(fig_finance, use_container_width=True)

        fig_rates = px.line(
            time_series,
            x="time_label",
            y=["gross_margin_rate", "profit_rate", "refund_rate", "late_rate", "return_rate"],
            markers=True,
            title=f"Margin leakage rates theo {time_grain.lower()}",
        )
        fig_rates.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_rates, use_container_width=True)

        bridge_data = pd.DataFrame(
            {
                "stage": ["Revenue", "COGS", "Refund", "Profit after refund"],
                "value": [
                    kpi["revenue_sum"],
                    -(kpi["revenue_sum"] - kpi["margin_sum"]),
                    -kpi["refund_sum"],
                    kpi["profit_after_refund"],
                ],
            }
        )
        fig_bridge = go.Figure(
            go.Waterfall(
                name="Profit bridge",
                orientation="v",
                measure=["absolute", "relative", "relative", "total"],
                x=bridge_data["stage"],
                y=bridge_data["value"],
                text=[f"{v:,.0f}" for v in bridge_data["value"]],
                connector={"line": {"color": "#94a3b8"}},
            )
        )
        fig_bridge.update_layout(title="Profit bridge: Revenue trừ COGS và refund", yaxis_title="Amount")
        st.plotly_chart(fig_bridge, use_container_width=True)

        if len(time_series):
            peak_profit = time_series.sort_values("profit_after_refund", ascending=False).iloc[0]
            worst_profit_rate = time_series.sort_values("profit_rate", ascending=True).iloc[0]
            st.success(
                "**Insight nhanh:** "
                f"Mốc tạo profit cao nhất là **{peak_profit['time_label']}** ({peak_profit['profit_after_refund']:,.0f}). "
                f"Mốc cần soi lại margin là **{worst_profit_rate['time_label']}** (profit rate {worst_profit_rate['profit_rate']:.2%}, refund rate {worst_profit_rate['refund_rate']:.2%})."
            )

        tab1_context = f"""
    Biểu đồ 1 - Line (Executive profit pool theo thời gian: Revenue, Gross Margin, Profit after refund):
    {df_to_compact_csv(time_series, ['time_label', 'orders', 'revenue', 'margin', 'profit_after_refund'], top_n=20)}

    Biểu đồ 2 - Line (Margin leakage rates: gross_margin_rate, profit_rate, refund_rate, late_rate, return_rate):
    {df_to_compact_csv(time_series, ['time_label', 'gross_margin_rate', 'profit_rate', 'refund_rate', 'late_rate', 'return_rate'], top_n=20)}

    Biểu đồ 3 - Waterfall (Profit bridge):
    {bridge_data.to_csv(index=False)}

    KPI tổng quan:
    total_orders={kpi['total_orders']}, delivered_orders={kpi['delivered_orders']}, avg_delivery={kpi['avg_delivery']},
    p90_delivery={kpi['p90_delivery']}, late_rate={kpi['late_rate']}, return_rate={kpi['return_rate']},
    revenue_sum={kpi['revenue_sum']}, margin_sum={kpi['margin_sum']}, profit_after_refund={kpi['profit_after_refund']},
    gross_margin_rate={kpi['gross_margin_rate']}, profit_rate={kpi['profit_rate']}, refund_ratio={kpi['refund_ratio']},
    profit_at_risk={kpi['profit_at_risk']}, profit_at_risk_share={kpi['profit_at_risk_share']}, low_rating_rate={kpi['low_rating_rate']}
    """
        render_tab_ai_insight("tab1_exec", "Executive Overview", tab1_context, openai_api_key, ai_model, insight_depth, ".", time_grain, start_date, end_date, shared_ai_context)

    with tab2:
        render_section_intro(
            "Delivery Deep Dive",
            "Phân tích sâu hiệu suất giao vận theo từng lane (region x order_source).",
            "Tập trung xác định lane nào trễ nhiều, phân phối lead-time và điểm nghẽn theo ngày trong tuần.",
        )

        lane = (
            filtered.groupby(["region", "order_source"], as_index=False)
            .agg(
                orders=("order_id", "nunique"),
                avg_delivery_days=("delivery_lead_days", "mean"),
                p90_delivery_days=("delivery_lead_days", lambda s: s.quantile(0.9)),
                late_rate=("late_delivery_flag", "mean"),
                return_rate=("returned_flag", "mean"),
            )
            .query("orders >= @min_lane_orders")
            .sort_values("late_rate", ascending=False)
        )
        st.dataframe(lane.head(25), use_container_width=True)
        fig_lane = px.bar(lane.head(15), x="late_rate", y="region", color="order_source", orientation="h")
        fig_lane.update_layout(title="Top lane có late rate cao", xaxis_tickformat=".0%")
        st.plotly_chart(fig_lane, use_container_width=True)

        fig_box = px.box(
            filtered[filtered["delivery_lead_days"].notna()],
            x="order_source",
            y="delivery_lead_days",
            color="order_source",
            title="Phân phối delivery lead theo source",
        )
        st.plotly_chart(fig_box, use_container_width=True)

        delay_heat = (
            filtered.groupby(["region", "order_weekday"], as_index=False)
            .agg(late_rate=("late_delivery_flag", "mean"))
            .pivot(index="region", columns="order_weekday", values="late_rate")
            .fillna(0)
        )
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        delay_heat = delay_heat.reindex(columns=[c for c in weekday_order if c in delay_heat.columns])
        if not delay_heat.empty:
            fig_heat = px.imshow(
                delay_heat,
                aspect="auto",
                color_continuous_scale="Reds",
                title="Heatmap late rate theo Region x Weekday",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        if len(lane):
            worst_lane = lane.iloc[0]
            st.warning(
                "**Insight nhanh:** "
                f"Lane rủi ro nhất hiện tại là **{worst_lane['region']} - {worst_lane['order_source']}** "
                f"với late rate **{worst_lane['late_rate']:.2%}** trên **{int(worst_lane['orders'])}** đơn."
            )

        tab2_context = f"""
    Biểu đồ 1 - Bar top lane late rate:
    {df_to_compact_csv(lane.head(15), ['region', 'order_source', 'orders', 'late_rate', 'avg_delivery_days', 'p90_delivery_days'], top_n=15)}

    Biểu đồ 2 - Box delivery lead by source (thống kê):
    {df_to_compact_csv(filtered.groupby('order_source', as_index=False).agg(avg_delivery=('delivery_lead_days', 'mean'), p90_delivery=('delivery_lead_days', lambda s: s.quantile(0.9))), ['order_source', 'avg_delivery', 'p90_delivery'], top_n=10)}

    Biểu đồ 3 - Heatmap region x weekday late rate:
    {delay_heat.reset_index().to_csv(index=False) if not delay_heat.empty else 'No heatmap data'}
    """
        render_tab_ai_insight("tab2_delivery", "Delivery Deep Dive", tab2_context, openai_api_key, ai_model, insight_depth, ".", time_grain, start_date, end_date, shared_ai_context)

    with tab3:
        render_section_intro(
            "Returns & Quality",
            "Theo dõi thất thoát do hoàn trả và chất lượng theo category/segment.",
            "Xác định lý do trả hàng chính, khu vực sản phẩm có tỷ lệ hoàn và refund cao để ưu tiên xử lý.",
        )

        reason = (
            line_filtered[line_filtered["is_returned_line"] == 1]
            .groupby("return_reason", as_index=False)
            .agg(return_lines=("is_returned_line", "sum"), refund=("refund_amount", "sum"))
            .sort_values("refund", ascending=False)
        )
        if len(reason):
            reason["refund_share"] = reason["refund"] / reason["refund"].sum()
        else:
            reason["refund_share"] = []
        st.dataframe(reason, use_container_width=True)

        fig_reason = px.bar(
            reason.head(10),
            x="refund",
            y="return_reason",
            text="refund_share",
            orientation="h",
            title="Top return reasons bào mòn lợi nhuận (theo refund)",
        )
        fig_reason.update_layout(xaxis_title="Refund amount", yaxis_title="Return reason")
        st.plotly_chart(fig_reason, use_container_width=True)

        cat_quality = (
            line_filtered.groupby(["category", "segment"], as_index=False)
            .agg(
                lines=("order_id", "count"),
                revenue=("net_revenue", "sum"),
                margin=("gross_margin", "sum"),
                profit_after_refund=("profit_after_refund", "sum"),
                returned_lines=("is_returned_line", "sum"),
                refund=("refund_amount", "sum"),
            )
            .query("lines >= 50")
        )
        if len(cat_quality):
            cat_quality["return_line_rate"] = cat_quality["returned_lines"] / cat_quality["lines"]
            cat_quality["refund_to_revenue"] = np.where(cat_quality["revenue"] > 0, cat_quality["refund"] / cat_quality["revenue"], 0)
            cat_quality["margin_rate"] = np.where(cat_quality["revenue"] > 0, cat_quality["margin"] / cat_quality["revenue"], 0)
            cat_quality["profit_rate"] = np.where(cat_quality["revenue"] > 0, cat_quality["profit_after_refund"] / cat_quality["revenue"], 0)
            fig_scatter = px.scatter(
                cat_quality,
                x="return_line_rate",
                y="profit_rate",
                size="revenue",
                color="segment",
                hover_data=["category", "margin_rate", "refund_to_revenue", "profit_after_refund"],
                title="Category profit risk map: Return rate vs Profit rate (size = revenue)",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            profit_leakage = cat_quality.sort_values("refund", ascending=False).head(15)
            fig_leakage = px.bar(
                profit_leakage,
                x="refund",
                y="category",
                color="segment",
                orientation="h",
                hover_data=["revenue", "profit_after_refund", "profit_rate", "return_line_rate"],
                title="Top category thất thoát profit do refund",
            )
            st.plotly_chart(fig_leakage, use_container_width=True)

        if len(reason):
            top_reason = reason.iloc[0]
            st.warning(
                "**Insight nhanh:** "
                f"Lý do trả hàng gây thiệt hại cao nhất là **{top_reason['return_reason']}** "
                f"với refund **{top_reason['refund']:,.0f}** "
                f"(chiếm **{top_reason['refund_share']:.2%}** tổng refund)."
            )

        tab3_context = f"""
    Biểu đồ 1 - Bar top return reasons theo refund:
    {df_to_compact_csv(reason, ['return_reason', 'return_lines', 'refund', 'refund_share'], top_n=20)}

    Biểu đồ 2 - Scatter category profit risk map:
    {df_to_compact_csv(cat_quality, ['category', 'segment', 'lines', 'return_line_rate', 'refund_to_revenue', 'revenue', 'margin', 'profit_after_refund', 'margin_rate', 'profit_rate'], top_n=25)}

    Biểu đồ 3 - Bar top category thất thoát profit do refund:
    {df_to_compact_csv(cat_quality.sort_values('refund', ascending=False) if len(cat_quality) else pd.DataFrame(), ['segment', 'category', 'revenue', 'margin', 'profit_after_refund', 'refund', 'profit_rate', 'refund_to_revenue'], top_n=25)}
    """
        render_tab_ai_insight("tab3_returns", "Returns & Quality", tab3_context, openai_api_key, ai_model, insight_depth, ".", time_grain, start_date, end_date, shared_ai_context)

    with tab4:
        render_section_intro(
            "Customer Voice",
            "Liên kết chất lượng vận hành với cảm nhận khách hàng qua rating.",
            "Đánh giá rating theo delivery bucket và theo các kịch bản trễ/trả hàng để thấy tác động trải nghiệm.",
        )

        review_join = filtered[filtered["avg_rating"].notna()].copy()
        bucket = review_join.groupby("delay_bucket", as_index=False).agg(
            avg_rating=("avg_rating", "mean"), orders=("order_id", "nunique")
        )
        fig_rating = px.bar(bucket, x="delay_bucket", y="avg_rating", title="Average rating theo delivery bucket")
        st.plotly_chart(fig_rating, use_container_width=True)

        rv_impact = review_join.groupby(["late_delivery_flag", "returned_flag"], as_index=False).agg(
            orders=("order_id", "nunique"),
            avg_rating=("avg_rating", "mean"),
        )
        rv_impact["scenario"] = (
            "Late=" + rv_impact["late_delivery_flag"].astype(int).astype(str)
            + " | Returned=" + rv_impact["returned_flag"].astype(int).astype(str)
        )
        fig_impact = px.bar(rv_impact, x="scenario", y="avg_rating", color="orders", title="Rating theo scenario vận hành")
        st.plotly_chart(fig_impact, use_container_width=True)

        top_bad = review_join.sort_values(["avg_rating", "delivery_lead_days"], ascending=[True, False]).head(20)
        st.dataframe(
            top_bad[["order_id", "region", "order_source", "delivery_lead_days", "returned_flag", "avg_rating"]],
            use_container_width=True,
        )

        if len(bucket):
            worst_bucket = bucket.sort_values("avg_rating", ascending=True).iloc[0]
            st.error(
                "**Insight nhanh:** "
                f"Nhóm lead-time ảnh hưởng xấu nhất đến rating là **{worst_bucket['delay_bucket']}** "
                f"với rating trung bình **{worst_bucket['avg_rating']:.2f}**."
            )

        tab4_context = f"""
    Biểu đồ 1 - Bar rating theo delay bucket:
    {df_to_compact_csv(bucket, ['delay_bucket', 'avg_rating', 'orders'], top_n=10)}

    Biểu đồ 2 - Bar rating theo scenario vận hành:
    {df_to_compact_csv(rv_impact, ['scenario', 'orders', 'avg_rating'], top_n=10)}

    Bảng top case rating thấp:
    {df_to_compact_csv(top_bad, ['order_id', 'region', 'order_source', 'delivery_lead_days', 'returned_flag', 'avg_rating'], top_n=20)}
    """
        render_tab_ai_insight("tab4_voice", "Customer Voice", tab4_context, openai_api_key, ai_model, insight_depth, ".", time_grain, start_date, end_date, shared_ai_context)

    with tab5:
        render_section_intro(
            "Inventory Control",
            "Theo dõi rủi ro tồn kho để cân bằng giữa thiếu hàng và dư hàng.",
            "Phân tích stockout, overstock, fill-rate và days of supply theo category/segment.",
        )

        inv_kpi = (
            inv_filtered.groupby(["category", "segment"], as_index=False)
            .agg(
                sku_snapshots=("product_id", "count"),
                avg_fill_rate=("fill_rate", "mean"),
                avg_stockout_days=("stockout_days", "mean"),
                stockout_flag_rate=("stockout_flag", "mean"),
                overstock_flag_rate=("overstock_flag", "mean"),
                avg_days_of_supply=("days_of_supply", "mean"),
            )
            .sort_values("stockout_flag_rate", ascending=False)
        )
        st.dataframe(inv_kpi.head(20), use_container_width=True)
        fig_inv = px.scatter(
            inv_kpi,
            x="stockout_flag_rate",
            y="overstock_flag_rate",
            size="sku_snapshots",
            color="segment",
            hover_data=["category", "avg_fill_rate", "avg_stockout_days", "avg_days_of_supply"],
            title="Risk map: stockout vs overstock",
        )
        st.plotly_chart(fig_inv, use_container_width=True)

        fig_dos = px.histogram(
            inv_filtered,
            x="days_of_supply",
            nbins=40,
            color="segment",
            opacity=0.65,
            title="Distribution of days_of_supply",
        )
        st.plotly_chart(fig_dos, use_container_width=True)

        if len(inv_kpi):
            inv_hotspot = inv_kpi.iloc[0]
            st.warning(
                "**Insight nhanh:** "
                f"Điểm nóng tồn kho là **{inv_hotspot['category']} - {inv_hotspot['segment']}** "
                f"(stockout {inv_hotspot['stockout_flag_rate']:.2%}, overstock {inv_hotspot['overstock_flag_rate']:.2%})."
            )

        tab5_context = f"""
    Biểu đồ 1 - Scatter stockout vs overstock:
    {df_to_compact_csv(inv_kpi, ['category', 'segment', 'sku_snapshots', 'stockout_flag_rate', 'overstock_flag_rate', 'avg_fill_rate', 'avg_days_of_supply'], top_n=25)}

    Biểu đồ 2 - Histogram days_of_supply (thống kê):
    {df_to_compact_csv(inv_filtered.groupby('segment', as_index=False).agg(avg_dos=('days_of_supply', 'mean'), p90_dos=('days_of_supply', lambda s: s.quantile(0.9))), ['segment', 'avg_dos', 'p90_dos'], top_n=20)}
    """
        render_tab_ai_insight("tab5_inventory", "Inventory Control", tab5_context, openai_api_key, ai_model, insight_depth, ".", time_grain, start_date, end_date, shared_ai_context)

    with tab6:
        render_section_intro(
            "Geo/Channel Lens",
            "Bóc tách hiệu suất theo địa lý và kênh bán để tìm cụm rủi ro.",
            "So sánh volume, revenue, late rate, return rate theo region/district/order source.",
        )

        geo_perf = (
            filtered.groupby(["region", "district", "order_source"], as_index=False)
            .agg(
                orders=("order_id", "nunique"),
                revenue=("order_revenue", "sum"),
                late_rate=("late_delivery_flag", "mean"),
                return_rate=("returned_flag", "mean"),
            )
            .query("orders >= 20")
        )
        st.dataframe(geo_perf.sort_values("late_rate", ascending=False).head(30), use_container_width=True)

        fig_geo = px.sunburst(
            geo_perf,
            path=["region", "district", "order_source"],
            values="orders",
            color="late_rate",
            color_continuous_scale="Reds",
            title="Sunburst volume + late risk theo geo/channel",
        )
        st.plotly_chart(fig_geo, use_container_width=True)

        if len(geo_perf):
            geo_hot = geo_perf.sort_values("late_rate", ascending=False).iloc[0]
            st.warning(
                "**Insight nhanh:** "
                f"Cụm geo/channel rủi ro cao nhất là **{geo_hot['region']} / {geo_hot['district']} / {geo_hot['order_source']}** "
                f"với late rate **{geo_hot['late_rate']:.2%}** trên **{int(geo_hot['orders'])}** đơn."
            )

        tab6_context = f"""
    Biểu đồ 1 - Sunburst geo/channel performance:
    {df_to_compact_csv(geo_perf.sort_values('late_rate', ascending=False), ['region', 'district', 'order_source', 'orders', 'revenue', 'late_rate', 'return_rate'], top_n=30)}
    """
        render_tab_ai_insight("tab6_geo", "Geo/Channel Lens", tab6_context, openai_api_key, ai_model, insight_depth, ".", time_grain, start_date, end_date, shared_ai_context)

    with tab7:
        render_section_intro(
            "Revenue Impact",
            "Góc nhìn chuyên sâu về profit impact: lợi nhuận tạo ra từ đâu, bị leak ở đâu, và yếu tố nào nên xử lý trước.",
            f"Phân tích trend revenue/profit theo {time_grain.lower()}, cơ cấu đóng góp theo region/source/category/segment, margin & refund, và ước tính profit uplift nếu giảm các yếu tố rủi ro.",
        )

        st.markdown("#### Pattern doanh thu đa năm (structural)")
        if len(annual_multi):
            fig_annual = px.line(
                annual_multi,
                x="year",
                y=["revenue", "margin", "profit_after_refund"],
                markers=True,
                title="Revenue, Gross Margin & Profit sau refund theo năm",
            )
            st.plotly_chart(fig_annual, use_container_width=True)

        st.markdown("#### Vì sao doanh thu giảm từ 2018?")
        if len(revenue_decline_annual):
            base_year = int(revenue_decline_summary.get("base_year") or revenue_decline_annual.iloc[0]["year"])
            latest_year = int(revenue_decline_summary.get("latest_year") or revenue_decline_annual.iloc[-1]["year"])
            base_row = revenue_decline_annual.loc[revenue_decline_annual["year"] == base_year].iloc[0]
            latest_row = revenue_decline_annual.iloc[-1]
            revenue_delta = revenue_decline_summary.get("revenue_delta", np.nan)
            orders_delta = revenue_decline_summary.get("orders_delta", np.nan)
            aov_delta = revenue_decline_summary.get("aov_delta", np.nan)
            order_effect = revenue_decline_summary.get("order_effect", np.nan)
            aov_effect = revenue_decline_summary.get("aov_effect", np.nan)

            d1, d2, d3, d4 = st.columns(4)
            d1.metric(f"Revenue {base_year}", safe_num(base_row["revenue"]))
            d2.metric(f"Revenue {latest_year}", safe_num(latest_row["revenue"]))
            d3.metric("Change vs 2018", safe_num(revenue_delta))
            d4.metric("Orders change", f"{orders_delta:,.0f}")

            fig_decline = make_subplots(specs=[[{"secondary_y": True}]])
            fig_decline.add_trace(
                go.Scatter(
                    x=revenue_decline_annual["year"],
                    y=revenue_decline_annual["revenue"],
                    name="Revenue",
                    mode="lines+markers",
                    line=dict(color="#1d4ed8", width=3),
                ),
                secondary_y=False,
            )
            fig_decline.add_trace(
                go.Scatter(
                    x=revenue_decline_annual["year"],
                    y=revenue_decline_annual["orders"],
                    name="Orders",
                    mode="lines+markers",
                    line=dict(color="#f97316", width=3, dash="dot"),
                ),
                secondary_y=True,
            )
            fig_decline.update_layout(
                title="Revenue vs order volume theo năm",
                hovermode="x unified",
                legend=dict(orientation="h"),
            )
            fig_decline.update_yaxes(title_text="Revenue", secondary_y=False)
            fig_decline.update_yaxes(title_text="Orders", secondary_y=True)
            st.plotly_chart(fig_decline, use_container_width=True)

            fig_quality = px.line(
                revenue_decline_annual,
                x="year",
                y=["aov", "profit_rate", "refund_rate"],
                markers=True,
                title="AOV vs profit quality after 2018",
            )
            fig_quality.update_layout(yaxis_tickformat=",.0f")
            fig_quality.update_yaxes(title_text="AOV / Rate")
            st.plotly_chart(fig_quality, use_container_width=True)

            bridge_fig = go.Figure(
                go.Waterfall(
                    name="Revenue bridge",
                    orientation="v",
                    measure=revenue_decline_bridge["measure"].tolist(),
                    x=revenue_decline_bridge["step"].tolist(),
                    y=revenue_decline_bridge["value"].tolist(),
                    connector={"line": {"color": "rgba(71,85,105,0.5)"}},
                )
            )
            bridge_fig.update_layout(title=f"Revenue bridge: {base_year} -> {latest_year}", showlegend=False)
            st.plotly_chart(bridge_fig, use_container_width=True)

            order_effect_share = (order_effect / revenue_delta) if revenue_delta else np.nan
            aov_effect_share = (aov_effect / revenue_delta) if revenue_delta else np.nan
            decline_message = (
                f"Từ {base_year} -> {latest_year}, revenue đổi {safe_num(revenue_delta)}. "
                f"Đóng góp từ giảm số đơn là {safe_num(order_effect)} ({safe_pct(order_effect_share)} của biến động), "
                f"trong khi AOV bù lại {safe_num(aov_effect)} ({safe_pct(aov_effect_share)}). "
                f"Kết luận nhanh: đây chủ yếu là bài toán mất volume, không phải AOV giảm."
            )
            st.info(decline_message)

            st.markdown("---")
            st.markdown("#### 🤖 AI Insights: Tại sao giảm? (Phân tích toàn diện)")
            col_insight_1, col_insight_2 = st.columns([3, 1])
            with col_insight_2:
                btn_get_insight = st.button("📊 Get Insights", key="btn_decline_insights_v2")
            
            if btn_get_insight:
                with st.spinner("🔍 Phân tích nguyên nhân giảm doanh thu..."):
                    decline_insights_text, decline_insights_err = generate_revenue_decline_insights(
                        openai_api_key,
                        revenue_decline_story,
                        annual_multi,
                        ai_model
                    )
                    if decline_insights_err:
                        st.error(f"❌ Lỗi: {decline_insights_err}")
                    elif decline_insights_text:
                        st.markdown("### Insight từ AI:")
                        st.markdown(decline_insights_text)
                        with st.expander("📥 Save insight"):
                            insight_title = st.text_input("Tên insight:", value=f"revenue_decline_{base_year}_{latest_year}")
                            if st.button("💾 Save"):
                                saved_path = save_insight_file(insight_title, decline_insights_text, ".")
                                st.success(f"✅ Saved: {saved_path}")
                    else:
                        st.warning("Không có insight từ AI.")

            if len(revenue_decline_channel):
                st.markdown("##### Channel mix: 2018 vs latest year")
                channel_view = revenue_decline_channel.copy()
                channel_view["share_base"] = channel_view["share_base"].map(lambda x: f"{x:.1%}")
                channel_view["share_latest"] = channel_view["share_latest"].map(lambda x: f"{x:.1%}")
                channel_view["delta_share"] = channel_view["delta_share"].map(lambda x: f"{x:+.1%}")
                st.dataframe(channel_view[["group", "revenue_base", "revenue_latest", "share_base", "share_latest", "delta_share", "orders_base", "orders_latest", "profit_base", "profit_latest"]].head(8), use_container_width=True)

            if len(revenue_decline_region):
                st.markdown("##### Region mix: 2018 vs latest year")
                region_view = revenue_decline_region.copy()
                region_view["share_base"] = region_view["share_base"].map(lambda x: f"{x:.1%}")
                region_view["share_latest"] = region_view["share_latest"].map(lambda x: f"{x:.1%}")
                region_view["delta_share"] = region_view["delta_share"].map(lambda x: f"{x:+.1%}")
                st.dataframe(region_view[["group", "revenue_base", "revenue_latest", "share_base", "share_latest", "delta_share", "orders_base", "orders_latest", "profit_base", "profit_latest"]].head(8), use_container_width=True)

            item_decline = data["items_enriched"].merge(order_core[["order_id", "order_date"]], on="order_id", how="left").copy()
            item_decline["year"] = item_decline["order_date"].dt.year
            category_decline = (
                item_decline.groupby(["year", "category"], as_index=False)
                .agg(revenue=("net_revenue", "sum"), margin=("gross_margin", "sum"))
            )
            if len(category_decline):
                latest_category_year = int(category_decline["year"].max())
                base_category_year = 2018 if (category_decline["year"] == 2018).any() else int(category_decline["year"].min())
                cat_compare = (
                    category_decline.loc[category_decline["year"].isin([base_category_year, latest_category_year])]
                    .pivot_table(index="category", columns="year", values="revenue", aggfunc="sum", fill_value=0)
                    .reset_index()
                )
                cat_compare["delta_revenue"] = cat_compare.get(latest_category_year, 0) - cat_compare.get(base_category_year, 0)
                cat_compare = cat_compare.sort_values("delta_revenue", ascending=True)
                st.markdown("##### Category shift: revenue change by category")
                st.dataframe(cat_compare.head(8), use_container_width=True)

        c_multi_1, c_multi_2 = st.columns(2)
        with c_multi_1:
            if len(quarter_multi):
                fig_q_share = px.bar(
                    quarter_multi.sort_values("quarter"),
                    x="quarter",
                    y=["avg_revenue_share", "avg_profit_share"],
                    text="top_quarter_years",
                    title="Tỷ trọng Revenue vs Profit trung bình theo Quý (qua nhiều năm)",
                )
                fig_q_share.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig_q_share, use_container_width=True)
        with c_multi_2:
            if len(month_multi):
                fig_m_share = px.line(
                    month_multi.sort_values("month"),
                    x="month",
                    y=["avg_revenue_share", "avg_profit_share"],
                    markers=True,
                    title="Tỷ trọng Revenue vs Profit trung bình theo Tháng (qua nhiều năm)",
                )
                fig_m_share.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig_m_share, use_container_width=True)

        if len(driver_group_impact):
            st.markdown("#### Mức ảnh hưởng nhóm yếu tố lên dự báo doanh thu (model-based)")
            st.dataframe(driver_group_impact, use_container_width=True)
            fig_driver = px.bar(
                driver_group_impact,
                x="group",
                y="delta_mape",
                text="delta_mape",
                title="Độ suy giảm hiệu năng mô hình khi loại từng nhóm yếu tố (cao hơn = ảnh hưởng mạnh hơn)",
            )
            st.plotly_chart(fig_driver, use_container_width=True)

        fig_rev_trend = px.line(
            revenue_series,
            x="time_label",
            y=["revenue", "margin", "profit_after_refund"],
            markers=True,
            title=f"Revenue / Gross Margin / Profit after Refund theo {time_grain.lower()}",
        )
        st.plotly_chart(fig_rev_trend, use_container_width=True)

        fig_profit_rate = px.line(
            revenue_series,
            x="time_label",
            y=["gross_margin_rate", "profit_rate", "refund_rate"],
            markers=True,
            title=f"Profit quality rates theo {time_grain.lower()}",
        )
        fig_profit_rate.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_profit_rate, use_container_width=True)

        lane_rev = (
            filtered.groupby(["region", "order_source"], as_index=False)
            .agg(
                orders=("order_id", "nunique"),
                revenue=("order_revenue", "sum"),
                margin=("order_margin", "sum"),
                refund=("refund_amount", "sum"),
                profit_after_refund=("profit_after_refund", "sum"),
                late_rate=("late_delivery_flag", "mean"),
                return_rate=("returned_flag", "mean"),
            )
            .query("orders >= @min_lane_orders")
        )
        lane_rev["margin_rate"] = np.where(lane_rev["revenue"] > 0, lane_rev["margin"] / lane_rev["revenue"], 0)
        lane_rev["profit_rate"] = np.where(lane_rev["revenue"] > 0, lane_rev["profit_after_refund"] / lane_rev["revenue"], 0)
        lane_rev["refund_rate"] = np.where(lane_rev["revenue"] > 0, lane_rev["refund"] / lane_rev["revenue"], 0)
        lane_rev = lane_rev.sort_values("profit_after_refund", ascending=False)

        st.dataframe(lane_rev.head(25), use_container_width=True)

        fig_lane_rev = px.scatter(
            lane_rev,
            x="refund_rate",
            y="profit_rate",
            size="revenue",
            color="order_source",
            hover_data=["region", "orders", "margin_rate", "return_rate", "late_rate", "profit_after_refund"],
            title="Lane profit risk map: Refund rate vs Profit rate (size = revenue)",
        )
        st.plotly_chart(fig_lane_rev, use_container_width=True)

        rev_by_cat = (
            line_filtered.groupby(["segment", "category"], as_index=False)
            .agg(
                revenue=("net_revenue", "sum"),
                margin=("gross_margin", "sum"),
                profit_after_refund=("profit_after_refund", "sum"),
                refund=("refund_amount", "sum"),
                lines=("order_id", "count"),
            )
            .sort_values("profit_after_refund", ascending=False)
        )
        if len(rev_by_cat):
            rev_by_cat["margin_rate"] = np.where(rev_by_cat["revenue"] > 0, rev_by_cat["margin"] / rev_by_cat["revenue"], 0)
            rev_by_cat["profit_rate"] = np.where(rev_by_cat["revenue"] > 0, rev_by_cat["profit_after_refund"] / rev_by_cat["revenue"], 0)
            rev_by_cat["refund_rate"] = np.where(rev_by_cat["revenue"] > 0, rev_by_cat["refund"] / rev_by_cat["revenue"], 0)

            fig_tree_rev = px.scatter(
                rev_by_cat,
                x="refund_rate",
                y="profit_rate",
                size="revenue",
                color="segment",
                hover_data=["category", "profit_after_refund", "margin_rate"],
                title="Category profit map: Refund rate vs Profit rate (size = revenue)",
            )
            st.plotly_chart(fig_tree_rev, use_container_width=True)

        st.markdown("#### Phân tích ảnh hưởng yếu tố đến revenue")
        if len(revenue_factor_impact):
            st.dataframe(revenue_factor_impact, use_container_width=True)
            fig_factor = px.bar(
                revenue_factor_impact,
                x="factor",
                y="est_profit_uplift",
                text="est_profit_uplift",
                title="Estimated profit uplift nếu xử lý yếu tố rủi ro",
            )
            st.plotly_chart(fig_factor, use_container_width=True)

            top_factor = revenue_factor_impact.iloc[0]
            st.success(
                "**Insight nhanh (Profit):** "
                f"Yếu tố ưu tiên xử lý trước là **{top_factor['factor']}** với profit uplift ước tính **{top_factor['est_profit_uplift']:,.0f}** "
                f"(delta profit/order {top_factor['delta_profit_per_order']:,.0f}, affected orders {int(top_factor['affected_orders'])})."
            )

        tab7_context = f"""
KPI revenue/profit:
revenue_sum={kpi['revenue_sum']}, margin_sum={kpi['margin_sum']}, profit_after_refund={kpi['profit_after_refund']}, gross_margin_rate={kpi['gross_margin_rate']}, profit_rate={kpi['profit_rate']}, aov={kpi['avg_order_value']}, revenue_at_risk={kpi['revenue_at_risk']}, at_risk_share={kpi['at_risk_share']}, profit_at_risk={kpi['profit_at_risk']}, profit_at_risk_share={kpi['profit_at_risk_share']}

Biểu đồ trend revenue/gross margin/profit after refund:
{df_to_compact_csv(revenue_series, ['time_label', 'orders', 'revenue', 'margin', 'profit_after_refund', 'refund', 'gross_margin_rate', 'profit_rate', 'refund_rate', 'aov'], top_n=30)}

Bảng lane revenue risk:
{df_to_compact_csv(lane_rev, ['region', 'order_source', 'orders', 'revenue', 'margin', 'profit_after_refund', 'refund', 'margin_rate', 'profit_rate', 'refund_rate', 'late_rate', 'return_rate'], top_n=30)}

Revenue by category/segment:
{df_to_compact_csv(rev_by_cat if 'rev_by_cat' in locals() else pd.DataFrame(), ['segment', 'category', 'revenue', 'margin', 'profit_after_refund', 'refund', 'margin_rate', 'profit_rate', 'refund_rate'], top_n=30)}

Factor impact:
{df_to_compact_csv(revenue_factor_impact, ['factor', 'avg_rev_a', 'avg_rev_b', 'avg_profit_a', 'avg_profit_b', 'delta_per_order', 'delta_profit_per_order', 'affected_orders', 'est_revenue_uplift', 'est_profit_uplift'], top_n=10)}
"""
        render_tab_ai_insight("tab7_revenue", "Revenue Impact", tab7_context, openai_api_key, ai_model, insight_depth, ".", time_grain, start_date, end_date, shared_ai_context)

    with tab8:
        render_section_intro(
            "Auto Insights & AI Copilot",
            "Trung tâm ưu tiên hành động tổng hợp từ toàn bộ dashboard.",
            "Xếp hạng lane theo risk score, xem tương quan chỉ số và dùng AI để đề xuất chiến lược xử lý.",
        )

        priority = (
            filtered.groupby(["region", "order_source"], as_index=False)
            .agg(
                orders=("order_id", "nunique"),
                revenue=("order_revenue", "sum"),
                profit_after_refund=("profit_after_refund", "sum"),
                late_rate=("late_delivery_flag", "mean"),
                return_rate=("returned_flag", "mean"),
                low_rating_rate=("low_rating_flag", "mean"),
                refund=("refund_amount", "sum"),
            )
            .query("orders >= @min_lane_orders")
        )
        priority["refund_rate"] = np.where(priority["revenue"] > 0, priority["refund"] / priority["revenue"], 0)
        priority["profit_rate"] = np.where(priority["revenue"] > 0, priority["profit_after_refund"] / priority["revenue"], 0)
        priority["profit_leakage"] = priority["revenue"] * (priority["profit_rate"].max() - priority["profit_rate"])

        for c in ["late_rate", "return_rate", "low_rating_rate", "refund_rate", "profit_leakage"]:
            std = priority[c].std()
            priority[f"{c}_z"] = (priority[c] - priority[c].mean()) / std if std and std > 0 else 0

        priority["risk_score"] = (
            0.40 * priority["profit_leakage_z"]
            + 0.25 * priority["refund_rate_z"]
            + 0.20 * priority["return_rate_z"]
            + 0.10 * priority["late_rate_z"]
            + 0.05 * priority["low_rating_rate_z"]
        )
        priority = priority.sort_values("risk_score", ascending=False)
        st.dataframe(priority.head(15), use_container_width=True)

        corr_cols = [
            "delivery_lead_days",
            "late_delivery_flag",
            "returned_flag",
            "refund_amount",
            "avg_rating",
            "order_revenue",
            "order_margin",
        ]
        corr_df = filtered[corr_cols].copy()
        corr_matrix = corr_df.corr(numeric_only=True).fillna(0)
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Correlation matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

        if len(priority):
            top = priority.iloc[0]
            worst_reason = reason.iloc[0] if len(reason) else None
            worst_inv = inv_kpi.sort_values(["stockout_flag_rate", "overstock_flag_rate"], ascending=False).iloc[0] if len(inv_kpi) else None
            st.markdown(
                f"""
                ### Insight tự động (Top ưu tiên)
                - Điểm nghẽn chính: **{top['region']} - {top['order_source']}**
                - `Profit rate`: **{top['profit_rate']:.2%}** | `Refund rate`: **{top['refund_rate']:.2%}**
                - `Return rate`: **{top['return_rate']:.2%}** | `Late rate`: **{top['late_rate']:.2%}** | `Refund`: **{top['refund']:,.0f}**
                - Hành động: rà soát refund/return root cause và SLA giao vận cho lane này trước vì điểm rủi ro đã ưu tiên theo profit leakage.
                """
            )

            if worst_reason is not None:
                st.info(
                    f"Return reason ảnh hưởng lớn nhất: {worst_reason['return_reason']} "
                    f"(refund {worst_reason['refund']:,.0f}, share {worst_reason['refund_share']:.2%})."
                )

            if worst_inv is not None:
                st.warning(
                    f"Inventory hotspot: {worst_inv['category']} - {worst_inv['segment']} | "
                    f"stockout {worst_inv['stockout_flag_rate']:.2%} | overstock {worst_inv['overstock_flag_rate']:.2%}."
                )

        st.markdown("---")
        st.subheader("AI Analyst — Insight tự động & Hỏi đáp")

        ai_context = f"""
Khoảng thời gian: {start_date.date()} đến {end_date.date()}
Độ phân giải thời gian: {time_grain}

KPI tổng quan:
- Total Orders: {kpi['total_orders']}
- Delivered Orders: {kpi['delivered_orders']}
- Revenue: {kpi['revenue_sum']}
- Margin: {kpi['margin_sum']}
- Profit After Refund: {kpi['profit_after_refund']}
- Profit Rate: {kpi['profit_rate']}
- AOV: {kpi['avg_order_value']}
- Revenue At Risk: {kpi['revenue_at_risk']} (share {kpi['at_risk_share']})
- Profit At Risk: {kpi['profit_at_risk']} (share {kpi['profit_at_risk_share']})
- Avg Delivery Lead: {kpi['avg_delivery']}
- P90 Delivery Lead: {kpi['p90_delivery']}
- Late Rate: {kpi['late_rate']}
- Return Rate: {kpi['return_rate']}
- Refund Ratio: {kpi['refund_ratio']}
- Low Rating Rate: {kpi['low_rating_rate']}

Time Series Snapshot:
{df_to_compact_csv(time_series, ['time_label', 'orders', 'revenue', 'margin', 'profit_after_refund', 'profit_rate', 'refund_rate', 'late_rate', 'return_rate', 'low_rating_rate', 'refund'], top_n=10)}

Top Priority Lanes:
{df_to_compact_csv(priority.sort_values('risk_score', ascending=False), ['region', 'order_source', 'orders', 'revenue', 'profit_after_refund', 'profit_rate', 'refund_rate', 'late_rate', 'return_rate', 'low_rating_rate', 'refund', 'profit_leakage', 'risk_score'], top_n=10)}

Top Return Reasons:
{df_to_compact_csv(reason.sort_values('refund', ascending=False), ['return_reason', 'return_lines', 'refund', 'refund_share'], top_n=8)}

Inventory Risk Snapshot:
{df_to_compact_csv(inv_kpi.sort_values('stockout_flag_rate', ascending=False), ['category', 'segment', 'stockout_flag_rate', 'overstock_flag_rate', 'avg_fill_rate', 'avg_days_of_supply'], top_n=8)}

Revenue Time Snapshot:
{df_to_compact_csv(revenue_series, ['time_label', 'revenue', 'margin', 'profit_after_refund', 'profit_rate', 'refund_rate', 'refund', 'aov'], top_n=12)}

Revenue Factor Impact:
{df_to_compact_csv(revenue_factor_impact, ['factor', 'delta_profit_per_order', 'affected_orders', 'est_profit_uplift', 'delta_per_order', 'est_revenue_uplift'], top_n=8)}

Global Multi-year + Driver Model:
{shared_ai_context}
"""

        ai_system_prompt = build_insight_system_prompt(insight_depth)
        ai_temperature = INSIGHT_DEPTH_CONFIG.get(insight_depth, INSIGHT_DEPTH_CONFIG["Deep Executive"])["temperature"]

        st.markdown("#### Batch generate")
        st.caption("Bấm một lần để sinh và lưu toàn bộ insight theo đúng `Độ phân giải thời gian` đang chọn ở sidebar.")
        if st.button("Generate ALL Insights & Save", use_container_width=True):
            if not openai_api_key:
                st.warning("Cần thiết lập biến môi trường OPENAI_API_KEY để chạy batch AI.")
            else:
                batch_sections = [
                    ("tab1_exec", "Executive Overview", tab1_context),
                    ("tab2_delivery", "Delivery Deep Dive", tab2_context),
                    ("tab3_returns", "Returns & Quality", tab3_context),
                    ("tab4_voice", "Customer Voice", tab4_context),
                    ("tab5_inventory", "Inventory Control", tab5_context),
                    ("tab6_geo", "Geo/Channel Lens", tab6_context),
                    ("tab7_revenue", "Revenue Impact", tab7_context),
                    ("tab8_strategic", "Auto Insights - Strategic", ai_context),
                ]
                progress = st.progress(0)
                saved_paths = []
                errors = []
                with st.spinner("Đang sinh toàn bộ insight và lưu file..."):
                    for idx, (section_key, section_name, context_text) in enumerate(batch_sections, start=1):
                        saved_path, ai_text, err = generate_and_save_section_insight(
                            section_key=section_key,
                            section_name=section_name,
                            context_text=context_text,
                            api_key=openai_api_key,
                            model=ai_model,
                            depth_mode=insight_depth,
                            base_path=".",
                            time_grain=time_grain,
                            start_date=start_date,
                            end_date=end_date,
                            global_context=shared_ai_context,
                        )
                        if err:
                            errors.append(f"{section_name}: {err}")
                        else:
                            saved_paths.append(saved_path)
                            grain_slug = slugify(time_grain)
                            st.session_state[f"ai_output_{section_key}_{grain_slug}"] = ai_text
                            st.session_state[f"ai_saved_path_{section_key}_{grain_slug}"] = str(saved_path)
                        progress.progress(idx / len(batch_sections))
                if saved_paths:
                    st.success(f"Đã lưu {len(saved_paths)} file insight vào insight_exports.")
                    for saved_path in saved_paths:
                        st.write(f"- {saved_path.name}")
                if errors:
                    st.error("Một số section bị lỗi:")
                    for err in errors:
                        st.write(f"- {err}")

        col_ai_1, col_ai_2 = st.columns([1, 1])
        with col_ai_1:
            if st.button("Generate AI Strategic Insight", use_container_width=True):
                if not openai_api_key:
                    st.warning("Cần thiết lập biến môi trường OPENAI_API_KEY để chạy AI.")
                else:
                    with st.spinner("AI đang phân tích dữ liệu..."):
                        prompt = (
                            "Hãy tạo bản strategic insight sâu cho toàn dashboard. "
                            "Không tóm tắt chart. Hãy tìm bản chất: profit pool nằm ở đâu, profit leak nằm ở đâu, "
                            "cơ chế nào làm revenue không chuyển thành profit, và can thiệp nào đáng làm trước. "
                            "Mỗi insight phải có: phát hiện -> bằng chứng số liệu -> cơ chế bên trong -> tác động profit/margin/risk -> hành động -> metric kiểm chứng. "
                            "Sau cùng cho roadmap 30/60/90 ngày, 5 câu hỏi cần drill-down, và 3 quyết định không nên làm vì có thể tăng revenue nhưng phá margin.\n\n"
                            f"DATA CONTEXT:\n{ai_context}"
                        )
                        ai_text, ai_err = call_openai_chat(
                            api_key=openai_api_key,
                            model=ai_model,
                            system_prompt=ai_system_prompt,
                            user_prompt=prompt,
                            temperature=ai_temperature,
                        )
                    if ai_err:
                        st.error(f"AI error: {ai_err}")
                    else:
                        st.markdown(ai_text)
                        saved_path = save_insight_file(
                            base_path=".",
                            section_key="tab7_strategic",
                            section_name="Auto Insights - Strategic",
                            model=ai_model,
                            time_grain=time_grain,
                            start_date=start_date,
                            end_date=end_date,
                            content=ai_text,
                        )
                        st.success(f"Đã lưu strategic insight: {saved_path}")

        with col_ai_2:
            if st.button("Reset AI Chat", use_container_width=True):
                st.session_state["ai_chat_history"] = []
                st.success("Đã reset lịch sử chat AI.")

        st.caption("Bạn có thể hỏi AI trực tiếp về dữ liệu đã lọc theo bộ lọc ở sidebar.")
        if "ai_chat_history" not in st.session_state:
            st.session_state["ai_chat_history"] = []

        for msg in st.session_state["ai_chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_q = st.chat_input("Hỏi AI về nguyên nhân, xu hướng, dự báo rủi ro, ưu tiên hành động...")
        if user_q:
            st.session_state["ai_chat_history"].append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            if not openai_api_key:
                assistant_text = "Cần thiết lập biến môi trường OPENAI_API_KEY trước khi chat."
            else:
                with st.spinner("AI đang trả lời..."):
                    recent_history = st.session_state["ai_chat_history"][-8:]
                    chat_messages = []
                    chat_messages.append({"role": "user", "content": f"DATA CONTEXT:\n{ai_context}"})
                    chat_messages.extend(recent_history)
                    assistant_text, ai_err = call_openai_chat_with_history(
                        api_key=openai_api_key,
                        model=ai_model,
                        system_prompt=ai_system_prompt,
                        messages=chat_messages,
                        temperature=ai_temperature,
                    )
                    if ai_err:
                        assistant_text = f"AI error: {ai_err}"

            st.session_state["ai_chat_history"].append({"role": "assistant", "content": assistant_text})
            with st.chat_message("assistant"):
                st.markdown(assistant_text)

            if assistant_text and not assistant_text.startswith("AI error"):
                save_insight_file(
                    base_path=".",
                    section_key="tab7_chat",
                    section_name=f"AI Chat - {user_q[:60]}",
                    model=ai_model,
                    time_grain=time_grain,
                    start_date=start_date,
                    end_date=end_date,
                    content=assistant_text,
                )

        st.markdown("---")
        st.subheader("Tổng hợp insight đã lưu (folder)")
        insight_dir = get_insight_dir(".")
        st.caption(f"Thư mục lưu insight: {insight_dir.resolve()}")
        saved_files = list_saved_insight_files(".", limit=300)
        st.write(f"Tổng số file insight: **{len(saved_files)}**")

        if saved_files:
            latest_file = saved_files[0]
            st.info(f"File mới nhất: {latest_file.name}")
            for fpath in saved_files[:30]:
                with st.expander(fpath.name):
                    content = fpath.read_text(encoding="utf-8")
                    st.markdown(content)
                    st.download_button(
                        label=f"Download {fpath.name}",
                        data=content.encode("utf-8"),
                        file_name=fpath.name,
                        mime="text/markdown",
                        key=f"dl_{fpath.name}",
                    )
        else:
            st.warning("Chưa có insight nào được lưu. Hãy dùng nút Generate ALL Insights & Save trong tab này để tạo một lượt.")

        st.markdown("---")
        st.subheader("Final synthesis")
        final_path = final_synthesis_path(".")
        source_files_for_final = list_source_insight_files(".", limit=300)
        if final_path.exists():
            status_text = "đã up-to-date" if final_synthesis_is_current(".") else "có insight nguồn mới hơn, nên cập nhật"
            st.caption(f"File tổng hợp cuối: {final_path.name} ({status_text})")
        else:
            st.caption("Chưa có file tổng hợp cuối.")

        if st.button("Generate / Update Final Synthesis", use_container_width=True):
            if not openai_api_key:
                st.warning("Cần thiết lập biến môi trường OPENAI_API_KEY để tổng hợp final synthesis.")
            elif not source_files_for_final:
                st.warning("Chưa có file insight nguồn để tổng hợp.")
            else:
                with st.spinner("Đang tổng hợp toàn bộ insight đã lưu..."):
                    final_saved_path, final_err, did_generate = generate_final_synthesis(
                        api_key=openai_api_key,
                        model=ai_model,
                        depth_mode=insight_depth,
                        base_path=".",
                    )
                if final_err:
                    st.error(final_err)
                elif did_generate:
                    st.success(f"Đã tạo/cập nhật final synthesis: {final_saved_path.name}")
                else:
                    st.info(f"Final synthesis hiện tại đã up-to-date, giữ nguyên: {final_saved_path.name}")

        if final_path.exists():
            final_content = final_path.read_text(encoding="utf-8")
            with st.expander("Xem Final Insight Synthesis", expanded=True):
                st.markdown(final_content)
                st.download_button(
                    label=f"Download {final_path.name}",
                    data=final_content.encode("utf-8"),
                    file_name=final_path.name,
                    mime="text/markdown",
                    key="dl_final_synthesis",
                )

        export_cols = [
            "order_id",
            "order_date",
            "region",
            "district",
            "order_source",
            "delivery_lead_days",
            "late_delivery_flag",
            "returned_flag",
            "refund_amount",
            "avg_rating",
            "order_revenue",
            "order_margin",
        ]
        export_df = filtered[export_cols].copy()
        st.download_button(
            "Download filtered orders CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_orders_ops_quality.csv",
            mime="text/csv",
        )

    with tab9:
        render_section_intro(
            "Insight 2018+",
            "Tab tổng hợp toàn bộ câu chuyện sau 2018: doanh thu giảm vì volume, AOV tăng nhưng không cứu được profit, và có dấu hiệu dịch chuyển theo vùng/danh mục.",
            "Mục tiêu của tab này là đọc như một bản điều tra: mỗi biểu đồ có note giải thích ngay bên dưới để người xem hiểu cơ chế, không chỉ đọc số.",
        )

        if len(revenue_decline_annual):
            base_year = int(revenue_decline_summary.get("base_year") or revenue_decline_annual.iloc[0]["year"])
            latest_year = int(revenue_decline_summary.get("latest_year") or revenue_decline_annual.iloc[-1]["year"])
            base_row = revenue_decline_annual.loc[revenue_decline_annual["year"] == base_year].iloc[0]
            latest_row = revenue_decline_annual.iloc[-1]

            st.markdown("### Executive thesis")
            render_insight_note(
                "Kết luận nhanh",
                (
                    f"Từ {base_year} đến {latest_year}, số đơn giảm mạnh hơn revenue, trong khi AOV tăng rõ. "
                    f"Điều đó nói rằng đây là <b>mất volume</b>, không phải vấn đề basket value. "
                    f"Refund/return/late rate không đổi đủ lớn để giải thích cú rơi; rủi ro nằm ở demand, acquisition, mix và cấu trúc chi phí."
                ),
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Revenue {base_year}", safe_num(base_row["revenue"]))
            m2.metric(f"Revenue {latest_year}", safe_num(latest_row["revenue"]))
            m3.metric("Orders change", safe_num(revenue_decline_summary.get("orders_delta", np.nan)))
            m4.metric("AOV change", safe_num(revenue_decline_summary.get("aov_delta", np.nan)))

            fig_insight_1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig_insight_1.add_trace(
                go.Scatter(
                    x=revenue_decline_annual["year"],
                    y=revenue_decline_annual["revenue"],
                    name="Revenue",
                    mode="lines+markers",
                    line=dict(color="#1d4ed8", width=3),
                ),
                secondary_y=False,
            )
            fig_insight_1.add_trace(
                go.Scatter(
                    x=revenue_decline_annual["year"],
                    y=revenue_decline_annual["orders"],
                    name="Orders",
                    mode="lines+markers",
                    line=dict(color="#f97316", width=3, dash="dot"),
                ),
                secondary_y=True,
            )
            fig_insight_1.update_layout(title=f"Revenue vs Orders theo năm ({base_year} -> {latest_year})", hovermode="x unified", legend=dict(orientation="h"))
            fig_insight_1.update_yaxes(title_text="Revenue", secondary_y=False)
            fig_insight_1.update_yaxes(title_text="Orders", secondary_y=True)
            st.plotly_chart(fig_insight_1, use_container_width=True, key="tab9_insight_1")
            render_insight_note(
                "Biểu đồ 1 - Revenue vs Orders",
                "Orders đi xuống song song với revenue, nhưng tốc độ giảm của orders sâu hơn. Đây là dấu hiệu điển hình của demand/traffic loss hoặc mất khách, chứ không phải chỉ là thay đổi giá trị mỗi đơn.",
            )

            fig_bridge = go.Figure(
                go.Waterfall(
                    name="Revenue bridge",
                    orientation="v",
                    measure=revenue_decline_bridge["measure"].tolist(),
                    x=revenue_decline_bridge["step"].tolist(),
                    y=revenue_decline_bridge["value"].tolist(),
                    connector={"line": {"color": "rgba(71,85,105,0.5)"}},
                )
            )
            fig_bridge.update_layout(title=f"Revenue bridge: {base_year} -> {latest_year}", showlegend=False)
            st.plotly_chart(fig_bridge, use_container_width=True, key="tab9_bridge")
            render_insight_note(
                "Biểu đồ 2 - Revenue bridge",
                "Phần âm của bridge chủ yếu đến từ hụt số đơn; phần dương từ AOV chỉ bù một phần nhỏ. Kết luận: có tăng giá trị đơn, nhưng không đủ để bù mất volume.",
            )

            fig_quality = px.line(
                revenue_decline_annual,
                x="year",
                y=["aov", "profit_rate", "refund_rate"],
                markers=True,
                title="AOV vs Profit Rate vs Refund Rate",
            )
            fig_quality.update_layout(yaxis_tickformat=",.0f")
            fig_quality.update_yaxes(title_text="AOV / Rate")
            st.plotly_chart(fig_quality, use_container_width=True, key="tab9_quality")
            render_insight_note(
                "Biểu đồ 3 - Quality guardrail",
                "AOV đi lên trong khi profit_rate không đi lên tương ứng. Refund_rate gần như phẳng, nên quality không phải thủ phạm chính; vấn đề nghiêng về cost structure, mix hoặc volume loss.",
            )

            if len(revenue_decline_channel):
                channel_view = revenue_decline_channel.copy().sort_values("share_latest", ascending=False)
                fig_channel = px.bar(
                    channel_view,
                    x="group",
                    y=["share_base", "share_latest"],
                    barmode="group",
                    title="Channel mix: 2018 vs latest year",
                    labels={"value": "Share", "group": "Channel"},
                )
                fig_channel.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_channel, use_container_width=True, key="tab9_channel")
                render_insight_note(
                    "Biểu đồ 4 - Channel mix",
                    "Các kênh giữ tỷ trọng khá ổn định, không có channel nào sụp riêng biệt đủ lớn để giải thích toàn bộ decline. Điều này ủng hộ giả thuyết system-wide hơn là channel-specific.",
                )

            if len(revenue_decline_region):
                region_view = revenue_decline_region.copy().sort_values("delta_share")
                fig_region = px.bar(
                    region_view,
                    x="group",
                    y="delta_share",
                    title="Region share change: latest vs 2018",
                    color="delta_share",
                    color_continuous_scale="RdBu",
                )
                fig_region.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_region, use_container_width=True, key="tab9_region")
                render_insight_note(
                    "Biểu đồ 5 - Region shift",
                    "West giảm share rõ nhất trong khi East/Central tăng nhẹ share. Nghĩa là decline không hoàn toàn đồng đều; có khu vực bị kéo xuống mạnh hơn và đáng drill-down tiếp.",
                )

            item_decline = data["items_enriched"].merge(order_core[["order_id", "order_date"]], on="order_id", how="left").copy()
            item_decline["year"] = item_decline["order_date"].dt.year
            category_decline = (
                item_decline.groupby(["year", "category"], as_index=False)
                .agg(revenue=("net_revenue", "sum"), margin=("gross_margin", "sum"))
            )
            if len(category_decline):
                latest_category_year = int(category_decline["year"].max())
                base_category_year = 2018 if (category_decline["year"] == 2018).any() else int(category_decline["year"].min())
                cat_compare = (
                    category_decline.loc[category_decline["year"].isin([base_category_year, latest_category_year])]
                    .pivot_table(index="category", columns="year", values="revenue", aggfunc="sum", fill_value=0)
                    .reset_index()
                )
                cat_compare["delta_revenue"] = cat_compare.get(latest_category_year, 0) - cat_compare.get(base_category_year, 0)
                cat_compare = cat_compare.sort_values("delta_revenue", ascending=True)

                fig_cat = px.bar(
                    cat_compare,
                    x="category",
                    y="delta_revenue",
                    color="delta_revenue",
                    color_continuous_scale="RdBu",
                    title="Category revenue change: latest vs 2018",
                )
                st.plotly_chart(fig_cat, use_container_width=True, key="tab9_category")
                render_insight_note(
                    "Biểu đồ 6 - Category shift",
                    "Streetwear là khối giảm mạnh nhất về absolute revenue, nên đây là category-level driver chính. Các category khác cũng giảm nhưng nhỏ hơn, cho thấy decline đến từ core mix chứ không chỉ một ngách phụ.",
                )

                st.markdown("#### Bảng kiểm chứng nhanh")
                st.dataframe(cat_compare.head(10), use_container_width=True)

            st.markdown("#### Giả thuyết cần kiểm chứng tiếp")
            st.markdown(
                """
                - **Systemic demand shock**: traffic / acquisition / market size giảm sau 2019.
                - **Cost structure**: COGS, shipping fee, promo rate tăng làm profit không theo kịp AOV.
                - **Regional drag**: West mất share nhanh hơn các vùng khác.
                - **Category concentration**: Streetwear và một số category lõi bị rơi mạnh.
                - **Không phải quality issue chính**: refund / return / late rate không tăng tương ứng với cú rơi doanh thu.
                """
            )

        tab9_context = f"""
Executive thesis:
- Revenue fell after 2018 while orders fell more sharply; AOV rose.
- Refund / return / late metrics are broadly stable, so quality is not the main explanation.

Annual decline snapshot:
{df_to_compact_csv(revenue_decline_annual, ['year', 'orders', 'revenue', 'aov', 'profit_after_refund', 'profit_rate', 'refund_rate', 'yoy_orders', 'yoy_revenue', 'yoy_aov', 'yoy_profit'], top_n=20)}

Revenue bridge:
{df_to_compact_csv(revenue_decline_bridge, ['step', 'measure', 'value'], top_n=10)}

Channel compare:
{df_to_compact_csv(revenue_decline_channel, ['group', 'revenue_base', 'revenue_latest', 'share_base', 'share_latest', 'delta_share', 'orders_base', 'orders_latest', 'profit_base', 'profit_latest'], top_n=20)}

Region compare:
{df_to_compact_csv(revenue_decline_region, ['group', 'revenue_base', 'revenue_latest', 'share_base', 'share_latest', 'delta_share', 'orders_base', 'orders_latest', 'profit_base', 'profit_latest'], top_n=20)}

Category compare:
{df_to_compact_csv(cat_compare if 'cat_compare' in locals() else pd.DataFrame(), ['category', 'revenue', 'delta_revenue'], top_n=20)}

Quality trend:
{df_to_compact_csv(annual_multi, ['year', 'orders', 'revenue', 'margin', 'profit_after_refund', 'profit_rate', 'refund_rate'], top_n=20)}

Key multi-year summary:
- avg_yoy_growth={multi_summary.get('avg_yoy_growth')}
- best_quarter={multi_summary.get('best_quarter')}
- worst_quarter={multi_summary.get('worst_quarter')}
- most_stable_quarter={multi_summary.get('most_stable_quarter')}

Global context:
{shared_ai_context}
"""
        render_tab_ai_insight("tab9_2018_plus", "Insight 2018+", tab9_context, openai_api_key, ai_model, insight_depth, ".", time_grain, start_date, end_date, shared_ai_context)


if __name__ == "__main__":
    main()
