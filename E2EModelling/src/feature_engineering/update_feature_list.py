from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURE_LIST_PATH = Path(__file__).resolve().parents[2] / "Feature list.csv"


CUSTOMER_MARKET_FEATURES = [
    (
        "cm_active_customers_28d_gap549",
        "Quy mô active base ngắn hạn đã lùi 549 ngày",
        "Số khách hàng unique có đơn trong cửa sổ 28 ngày kết thúc tại as_of_date = Date - 549 ngày",
        "customers.csv + orders.csv",
        "Date",
        "Đo độ rộng customer base gần nhất mà forecast được phép nhìn thấy",
    ),
    (
        "cm_active_customers_91d_gap549",
        "Quy mô active base trung hạn đã lùi 549 ngày",
        "Số khách hàng unique có đơn trong cửa sổ 91 ngày kết thúc tại as_of_date = Date - 549 ngày",
        "customers.csv + orders.csv",
        "Date",
        "Proxy cho sức khỏe nền khách mua hàng",
    ),
    (
        "cm_orders_per_active_customer_91d_gap549",
        "Tần suất mua trên mỗi active customer",
        "SUM(orders_91d_asof) / active_customers_91d_asof",
        "orders.csv + customers.csv",
        "Date",
        "Bắt tín hiệu khách còn mua nhiều hay chỉ còn mua thưa",
    ),
    (
        "cm_revenue_per_active_customer_91d_gap549",
        "Doanh thu trên mỗi active customer",
        "SUM(revenue_91d_asof) / active_customers_91d_asof",
        "orders.csv + order_items.csv + customers.csv",
        "Date",
        "Tách tác động giá trị khách khỏi quy mô active base",
    ),
    (
        "cm_new_customer_order_share_91d_gap549",
        "Tỷ trọng đơn từ khách mới",
        "SUM(new_customer_orders_91d_asof) / SUM(all_orders_91d_asof)",
        "orders.csv + customers.csv",
        "Date",
        "Đo sức kéo acquisition ở tầng order",
    ),
    (
        "cm_repeat_customer_order_share_91d_gap549",
        "Tỷ trọng đơn từ khách quay lại",
        "SUM(repeat_orders_91d_asof) / SUM(all_orders_91d_asof)",
        "orders.csv + customers.csv",
        "Date",
        "Đo sức khỏe retention/repeat behavior",
    ),
    (
        "cm_loyal_customer_order_share_91d_gap549",
        "Tỷ trọng đơn từ khách đã mua ít nhất 3 lần",
        "SUM(customer_order_no >= 3 trong 91d_asof) / SUM(all_orders_91d_asof)",
        "orders.csv + customers.csv",
        "Date",
        "Theo dõi loyal base có co lại hay không",
    ),
    (
        "cm_repeat_within_365d_order_share_91d_gap549",
        "Tỷ trọng đơn repeat có gap <=365 ngày",
        "SUM(repeat orders có previous_order_gap <=365 trong 91d_asof) / SUM(all_orders_91d_asof)",
        "orders.csv + customers.csv",
        "Date",
        "Đo repeat còn nằm trong chu kỳ mua hợp lý",
    ),
    (
        "cm_median_inter_order_gap_91d_gap549",
        "Khoảng cách giữa hai đơn của khách repeat",
        "Rolling mean 91 ngày của median(previous_order_gap), sau shift 549 ngày",
        "orders.csv + customers.csv",
        "Date",
        "Gap tăng là tín hiệu lifecycle chậm lại",
    ),
    (
        "cm_median_customer_tenure_days_91d_gap549",
        "Tuổi đời khách hàng tại thời điểm mua",
        "Rolling mean 91 ngày của median(order_date - first_order_date), sau shift 549 ngày",
        "orders.csv + customers.csv",
        "Date",
        "Phân biệt base đang phụ thuộc khách cũ hay khách mới",
    ),
    (
        "cm_age_18_34_order_share_91d_gap549",
        "Tỷ trọng đơn từ nhóm tuổi 18-34",
        "SUM(orders age_group in 18-24,25-34 trong 91d_asof) / SUM(all_orders_91d_asof)",
        "customers.csv + orders.csv",
        "Date",
        "Bắt thay đổi chân dung khách trẻ",
    ),
    (
        "cm_age_35_54_order_share_91d_gap549",
        "Tỷ trọng đơn từ nhóm tuổi 35-54",
        "SUM(orders age_group in 35-44,45-54 trong 91d_asof) / SUM(all_orders_91d_asof)",
        "customers.csv + orders.csv",
        "Date",
        "Bắt thay đổi chân dung khách trung niên",
    ),
    (
        "cm_age_55_plus_order_share_91d_gap549",
        "Tỷ trọng đơn từ nhóm tuổi 55+",
        "SUM(orders age_group = 55+ trong 91d_asof) / SUM(all_orders_91d_asof)",
        "customers.csv + orders.csv",
        "Date",
        "Đo dịch chuyển sang nhóm tuổi lớn",
    ),
    (
        "cm_female_order_share_91d_gap549",
        "Tỷ trọng đơn từ khách nữ",
        "SUM(orders gender = Female trong 91d_asof) / SUM(all_orders_91d_asof)",
        "customers.csv + orders.csv",
        "Date",
        "Bắt mix giới tính ở tầng đơn hàng",
    ),
    (
        "cm_mobile_order_share_91d_gap549",
        "Tỷ trọng đơn đặt từ mobile",
        "SUM(orders device_type = mobile trong 91d_asof) / SUM(all_orders_91d_asof)",
        "orders.csv",
        "Date",
        "Đo thay đổi hành vi mua qua thiết bị",
    ),
    (
        "cm_paid_search_acq_order_share_91d_gap549",
        "Tỷ trọng đơn từ khách có kênh acquisition paid search",
        "SUM(orders của paid_search customers trong 91d_asof) / SUM(all_orders_91d_asof)",
        "customers.csv + orders.csv",
        "Date",
        "Bắt phụ thuộc vào acquisition trả phí",
    ),
    (
        "cm_organic_search_acq_order_share_91d_gap549",
        "Tỷ trọng đơn từ khách có kênh acquisition organic search",
        "SUM(orders của organic_search customers trong 91d_asof) / SUM(all_orders_91d_asof)",
        "customers.csv + orders.csv",
        "Date",
        "Bắt sức kéo organic demand",
    ),
    (
        "cm_east_region_order_share_91d_gap549",
        "Tỷ trọng đơn từ region East",
        "SUM(orders region = East trong 91d_asof) / SUM(all_orders_91d_asof)",
        "geography.csv + orders.csv",
        "Date",
        "Bắt dịch chuyển thị trường địa lý chính",
    ),
    (
        "cm_region_entropy_91d_gap549",
        "Độ phân tán đơn hàng theo region",
        "-SUM(region_share * log(region_share)) trong 91d_asof",
        "geography.csv + orders.csv",
        "Date",
        "Entropy thấp hơn cho thấy thị trường cô đặc hơn",
    ),
    (
        "cm_acquisition_entropy_91d_gap549",
        "Độ phân tán đơn hàng theo acquisition channel của khách",
        "-SUM(channel_share * log(channel_share)) trong 91d_asof",
        "customers.csv + orders.csv",
        "Date",
        "Entropy thấp hơn cho thấy acquisition mix phụ thuộc vào ít kênh hơn",
    ),
]


DECOMPOSE_FEATURES = [
    (
        "revenue_decomp_trend_m7_gap549",
        "Trend ngắn hạn weekly của Revenue đã lùi 549 ngày",
        "Với target Date, lấy lịch sử Revenue đến as_of_date = Date - 549 ngày; gọi pmdarima.arima.decompose(type='additive', m=7) trên trailing window 365 ngày; lấy trend khả dụng mới nhất",
        "sales.csv[Revenue, Date] + pmdarima.arima.decompose",
        "Date",
        "Bắt xu hướng weekly causal của doanh thu mà không nhìn qua tương lai forecast",
    ),
    (
        "revenue_decomp_seasonal_m7_gap549",
        "Seasonality weekly của Revenue đã lùi 549 ngày",
        "Với target Date, lấy lịch sử Revenue đến as_of_date = Date - 549 ngày; gọi pmdarima.arima.decompose(type='additive', m=7) trên trailing window 365 ngày; lấy seasonal mới nhất",
        "sales.csv[Revenue, Date] + pmdarima.arima.decompose",
        "Date",
        "Bắt chu kỳ tuần còn quan sát được tại thời điểm forecast hợp lệ",
    ),
    (
        "revenue_decomp_random_m7_gap549",
        "Residual weekly của Revenue đã lùi 549 ngày",
        "Với target Date, lấy lịch sử Revenue đến as_of_date = Date - 549 ngày; gọi pmdarima.arima.decompose(type='additive', m=7) trên trailing window 365 ngày; lấy random/residual khả dụng mới nhất",
        "sales.csv[Revenue, Date] + pmdarima.arima.decompose",
        "Date",
        "Đo nhiễu doanh thu sau khi tách trend và weekly seasonality",
    ),
    (
        "revenue_decomp_trend_m365_gap549",
        "Trend dài hạn yearly của Revenue đã lùi 549 ngày",
        "Với target Date, lấy lịch sử Revenue đến as_of_date = Date - 549 ngày; gọi pmdarima.arima.decompose(type='additive', m=365) trên trailing window 1095 ngày; lấy trend khả dụng mới nhất",
        "sales.csv[Revenue, Date] + pmdarima.arima.decompose",
        "Date",
        "Bắt xu hướng dài hạn của doanh thu trước forecast horizon",
    ),
    (
        "revenue_decomp_seasonal_m365_gap549",
        "Seasonality yearly của Revenue đã lùi 549 ngày",
        "Với target Date, lấy lịch sử Revenue đến as_of_date = Date - 549 ngày; gọi pmdarima.arima.decompose(type='additive', m=365) trên trailing window 1095 ngày; lấy seasonal mới nhất",
        "sales.csv[Revenue, Date] + pmdarima.arima.decompose",
        "Date",
        "Bắt mùa vụ năm của doanh thu từ lịch sử hợp lệ",
    ),
    (
        "revenue_decomp_random_m365_gap549",
        "Residual yearly của Revenue đã lùi 549 ngày",
        "Với target Date, lấy lịch sử Revenue đến as_of_date = Date - 549 ngày; gọi pmdarima.arima.decompose(type='additive', m=365) trên trailing window 1095 ngày; lấy random/residual khả dụng mới nhất",
        "sales.csv[Revenue, Date] + pmdarima.arima.decompose",
        "Date",
        "Đo phần biến động doanh thu không giải thích bởi trend và yearly seasonality",
    ),
    (
        "cogs_decomp_trend_m7_gap549",
        "Trend ngắn hạn weekly của COGS đã lùi 549 ngày",
        "Tương tự Revenue nhưng áp dụng trên COGS với pmdarima.arima.decompose(type='additive', m=7)",
        "sales.csv[COGS, Date] + pmdarima.arima.decompose",
        "Date",
        "Bắt xu hướng weekly causal của COGS",
    ),
    (
        "cogs_decomp_seasonal_m7_gap549",
        "Seasonality weekly của COGS đã lùi 549 ngày",
        "Tương tự Revenue nhưng áp dụng trên COGS với pmdarima.arima.decompose(type='additive', m=7)",
        "sales.csv[COGS, Date] + pmdarima.arima.decompose",
        "Date",
        "Bắt chu kỳ tuần của COGS",
    ),
    (
        "cogs_decomp_random_m7_gap549",
        "Residual weekly của COGS đã lùi 549 ngày",
        "Tương tự Revenue nhưng áp dụng trên COGS với pmdarima.arima.decompose(type='additive', m=7)",
        "sales.csv[COGS, Date] + pmdarima.arima.decompose",
        "Date",
        "Đo nhiễu COGS sau khi tách trend và weekly seasonality",
    ),
    (
        "cogs_decomp_trend_m365_gap549",
        "Trend dài hạn yearly của COGS đã lùi 549 ngày",
        "Tương tự Revenue nhưng áp dụng trên COGS với pmdarima.arima.decompose(type='additive', m=365)",
        "sales.csv[COGS, Date] + pmdarima.arima.decompose",
        "Date",
        "Bắt xu hướng dài hạn của COGS trước forecast horizon",
    ),
    (
        "cogs_decomp_seasonal_m365_gap549",
        "Seasonality yearly của COGS đã lùi 549 ngày",
        "Tương tự Revenue nhưng áp dụng trên COGS với pmdarima.arima.decompose(type='additive', m=365)",
        "sales.csv[COGS, Date] + pmdarima.arima.decompose",
        "Date",
        "Bắt mùa vụ năm của COGS từ lịch sử hợp lệ",
    ),
    (
        "cogs_decomp_random_m365_gap549",
        "Residual yearly của COGS đã lùi 549 ngày",
        "Tương tự Revenue nhưng áp dụng trên COGS với pmdarima.arima.decompose(type='additive', m=365)",
        "sales.csv[COGS, Date] + pmdarima.arima.decompose",
        "Date",
        "Đo phần biến động COGS không giải thích bởi trend và yearly seasonality",
    ),
]


def _extract_original_formula(formula: str) -> str:
    if not isinstance(formula, str):
        return ""
    for marker in ["Công thức gốc:", "C?ng th?c g?c:"]:
        if marker in formula:
            return formula.split(marker, 1)[1].strip()
    if formula.startswith("Gap-safe:"):
        return formula.split(":", 1)[1].strip()
    return formula


def main() -> None:
    df = pd.read_csv(FEATURE_LIST_PATH)
    name_col, definition_col, formula_col, source_col, grain_col, meaning_col = df.columns.tolist()
    calendar_sources = {"sales.csv[Date]", "Mùa vụ rủi ro vận hành"}
    calendar_formula_overrides = {
        "quarter": "quarter(date)",
        "is_q3": "1 nếu quarter=3, ngược lại 0",
        "is_december": "1 nếu month=12, ngược lại 0",
    }

    for idx, row in df.iterrows():
        feature_name = str(row.get(name_col, ""))
        if feature_name in calendar_formula_overrides:
            df.at[idx, formula_col] = calendar_formula_overrides[feature_name]
            df.at[idx, meaning_col] = "Feature calendar dùng trực tiếp target date, không shift 549 ngày"
            continue
        source = str(row.get(source_col, ""))
        if source in calendar_sources:
            continue
        formula = _extract_original_formula(row.get(formula_col, ""))
        df.at[idx, formula_col] = (
            "Gap-safe: tính tại as_of_date = Date - 549 ngày; "
            f"rolling/lag thực hiện sau shift. Công thức gốc: {formula}"
        )
        meaning = row.get(meaning_col, "")
        if pd.isna(meaning) or "Feature phi-calendar" in str(meaning):
            df.at[idx, meaning_col] = (
                "Feature phi-calendar, dùng lịch sử đã lùi 549 ngày để tránh leakage "
                "khi forecast đến 2024-07-01"
            )

    by_name = {name: i for i, name in enumerate(df[name_col].astype(str))}
    for feature in CUSTOMER_MARKET_FEATURES + DECOMPOSE_FEATURES:
        row = dict(zip(df.columns, feature))
        if feature[0] in by_name:
            idx = by_name[feature[0]]
            for col, value in row.items():
                df.at[idx, col] = value
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(FEATURE_LIST_PATH, index=False, encoding="utf-8")
    print(f"Updated {FEATURE_LIST_PATH}: rows={len(df)}")


if __name__ == "__main__":
    main()
