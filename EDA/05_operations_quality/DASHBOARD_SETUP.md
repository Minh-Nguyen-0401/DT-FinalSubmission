# Ops & Quality Dashboard Setup

## 1) Tạo virtual environment (venv)

Tại thư mục gốc `DATATHON2026`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Cài package

Tại thư mục `datathon-2026-round-1`, chạy:

```bash
pip install --upgrade pip
pip install streamlit plotly pandas numpy openai
```

## 3) Chạy dashboard tương tác local

Tại thư mục `datathon-2026-round-1`, chạy:

```bash
streamlit run ops_quality_dashboard.py
```

Dashboard có các tab:
- Overview
- Delivery
- Returns & Quality
- Inventory
- Auto Insights

Tính năng mới:
- Bộ lọc thời gian đa mức: Ngày / Tuần / Tháng / Quý / Năm / Thứ trong tuần
- AI Strategic Insight (phân tích tự động từ dữ liệu đang lọc)
- AI Chat Q&A (người dùng hỏi trực tiếp theo context dữ liệu hiện tại)

## 4) Thiết lập OpenAI Key

Bạn có 2 cách:

1. Nhập trực tiếp key trong sidebar của dashboard (`AI Settings`)  
2. Hoặc export biến môi trường trước khi chạy:

```bash
export OPENAI_API_KEY="your_openai_key_here"
streamlit run ops_quality_dashboard.py
```

Model mặc định hiện tại: `gpt-4o-mini` (có thể đổi trong sidebar)

## 5) Xuất bảng cho Power BI

```bash
python powerbi_export_ops_quality.py
```

Sau khi chạy, thư mục `powerbi_exports/` sẽ có:
- `fact_orders_ops_quality.csv`
- `fact_order_items_ops_quality.csv`
- `agg_lane_kpi.csv`
- `agg_inventory_kpi.csv`
- `agg_return_reason_kpi.csv`

## 6) Kết nối Power BI Desktop

1. Mở Power BI Desktop  
2. `Get Data` -> `Text/CSV` -> import các file trong `powerbi_exports/`  
3. Tạo relationship chính:
   - `fact_orders_ops_quality[order_id]` -> `fact_order_items_ops_quality[order_id]`
4. Build visuals gợi ý:
   - Card: Late rate, Return rate, Refund/Revenue, Low rating rate
   - Line: trend theo `order_month`
   - Bar: top `region x order_source` theo `late_rate`
   - Scatter: `stockout_flag_rate` vs `overstock_flag_rate`

## 7) Insight nhanh nên theo dõi

- Lane nào (region + source) có late rate cao và volume lớn
- Return reason nào chiếm refund lớn nhất
- Category/segment nào vừa stockout cao vừa overstock cao
- Delivery lead-time bucket nào làm rating giảm mạnh
