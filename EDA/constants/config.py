"""Cấu hình dùng chung cho toàn bộ phần EDA."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Tìm thư mục gốc dự án dựa trên folder dữ liệu chính."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "datathon-2026-round-1").exists():
            return candidate
    raise FileNotFoundError("Không tìm thấy thư mục datathon-2026-round-1.")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "datathon-2026-round-1"
EDA_DIR = PROJECT_ROOT / "EDA"
FIGURES_DIR = EDA_DIR / "figures"
PDF_PATH = PROJECT_ROOT / "De_thi_vong_1.pdf"

DATE_COLUMNS = {
    "sales": ["Date"],
    "sample_submission": ["Date"],
    "orders": ["order_date"],
    "customers": ["signup_date"],
    "promotions": ["start_date", "end_date"],
    "shipments": ["ship_date", "delivery_date"],
    "returns": ["return_date"],
    "reviews": ["review_date"],
    "inventory": ["snapshot_date"],
    "web_traffic": ["date"],
}

TABLE_FILES = {
    "products": "products.csv",
    "customers": "customers.csv",
    "promotions": "promotions.csv",
    "geography": "geography.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
    "payments": "payments.csv",
    "shipments": "shipments.csv",
    "returns": "returns.csv",
    "reviews": "reviews.csv",
    "sales": "sales.csv",
    "sample_submission": "sample_submission.csv",
    "inventory": "inventory.csv",
    "web_traffic": "web_traffic.csv",
}

AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55+"]
SIZE_ORDER = ["S", "M", "L", "XL"]
MONTH_ORDER = list(range(1, 13))
WEEKDAY_ORDER = [
    "Thứ hai",
    "Thứ ba",
    "Thứ tư",
    "Thứ năm",
    "Thứ sáu",
    "Thứ bảy",
    "Chủ nhật",
]

SOURCE_ORDER = [
    "organic_search",
    "paid_search",
    "social_media",
    "email_campaign",
    "referral",
    "direct",
]

STATUS_ORDER = ["created", "paid", "shipped", "delivered", "returned", "cancelled"]

RANDOM_SEED = 2026

