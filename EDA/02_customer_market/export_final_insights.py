"""Xuất FINAL_INSIGHTS.md từ notebook customer market deep dive đã chạy."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "EDA" / "02_customer_market" / "02_customer_market_deep_dive.ipynb"
OUTPUT = ROOT / "EDA" / "02_customer_market" / "FINAL_INSIGHTS.md"


SECTION_PATTERN = re.compile(
    r"^\*\*Chú thích dưới mục này - (?P<section>.+?)\:\*\* (?P<headline>.+?)\n\n(?P<body>.+)$",
    re.DOTALL,
)

SECTION_ORDER = [
    "Framing doanh thu",
    "Acquisition & retention",
    "Loyalty erosion",
    "Demographics & market mix",
    "Promo & offer fit",
    "Service & traffic efficiency",
    "Chất lượng dữ liệu khách hàng",
]

FIGURE_MAP = {
    "Framing doanh thu": "../figures/02_customer_market_deep_revenue_bridge.png",
    "Acquisition & retention": "../figures/02_customer_market_deep_lifecycle.png",
    "Loyalty erosion": "../figures/02_customer_market_deep_loyalty.png",
    "Demographics & market mix": "../figures/02_customer_market_deep_mix_stability.png",
    "Promo & offer fit": "../figures/02_customer_market_deep_promo.png",
    "Service & traffic efficiency": "../figures/02_customer_market_deep_service_traffic.png",
    "Chất lượng dữ liệu khách hàng": "../figures/02_customer_market_deep_signup_quality.png",
}

SECTION_TITLES = {
    "Framing doanh thu": "1. Revenue bridge --> doanh thu giảm vì ít khách mua hơn và mua ít lần hơn",
    "Acquisition & retention": "2. Acquisition và retention --> customer engine bị gãy ở cả đầu vào lẫn đầu ra",
    "Loyalty erosion": "3. Loyalty erosion --> lớp khách tạo doanh thu cấu trúc đang co rút",
    "Demographics & market mix": "4. Demographics và market mix --> không có dấu hiệu đổi hẳn chân dung khách hàng",
    "Promo & offer fit": "5. Promo và offer --> được đẩy mạnh nhưng không cứu được tần suất mua",
    "Service & traffic efficiency": "6. Service và traffic efficiency --> service khá ổn nhưng conversion suy giảm mạnh",
    "Chất lượng dữ liệu khách hàng": "Ghi chú phương pháp --> không dùng signup_date làm cohort anchor",
}


def load_markdown_outputs(notebook_path: Path) -> list[str]:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown_outputs: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "text/markdown" not in data:
                continue
            value = data["text/markdown"]
            if isinstance(value, list):
                value = "".join(value)
            markdown_outputs.append(value.strip())
    return markdown_outputs


def split_outputs(markdown_outputs: list[str]) -> tuple[list[dict[str, object]], str]:
    section_notes: list[dict[str, object]] = []
    summary = ""
    for text in markdown_outputs:
        if text.startswith("## Tóm tắt customer market insights"):
            summary = text
            continue
        match = SECTION_PATTERN.match(text)
        if not match:
            continue
        section = match.group("section").strip()
        headline = match.group("headline").strip()
        body = match.group("body").strip()
        bullet_lines: list[str] = []
        implication = ""
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("- Hàm ý:"):
                implication = line.removeprefix("- Hàm ý:").strip()
            elif line.startswith("- "):
                bullet_lines.append(line.removeprefix("- ").strip())
        section_notes.append(
            {
                "section": section,
                "headline": headline,
                "bullets": bullet_lines,
                "implication": implication,
            }
        )
    return section_notes, summary


def extract_summary_parts(summary: str) -> dict[str, object]:
    lines = [line.rstrip() for line in summary.splitlines()]
    lead = ""
    thesis = ""
    score_lines: list[str] = []
    actions: list[str] = []

    mode = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("**Luận điểm chính"):
            lead = stripped
            continue
        if stripped == "### Bảng điểm các giả thuyết cạnh tranh":
            mode = "scores"
            continue
        if stripped == "### Chuỗi bằng chứng xuyên suốt notebook":
            mode = "support"
            continue
        if stripped == "### Hàm ý hành động cho phần customer market":
            mode = "actions"
            continue
        if mode == "scores" and stripped.startswith("- "):
            score_lines.append(stripped)
        elif mode == "actions" and stripped.startswith("- "):
            actions.append(stripped)
        elif not thesis and not stripped.startswith("## "):
            thesis = stripped

    return {
        "lead": lead,
        "thesis": thesis,
        "score_lines": score_lines,
        "actions": actions,
    }


def build_markdown(section_notes: list[dict[str, object]], summary_parts: dict[str, object]) -> str:
    lead = summary_parts["lead"]
    thesis = summary_parts["thesis"]
    score_lines = summary_parts["score_lines"]
    actions = summary_parts["actions"]

    if lead and not lead.startswith("**"):
        lead = f"**{lead}"

    if not actions:
        actions = [
            "- Ưu tiên phục hồi acquisition efficiency thay vì chỉ tăng traffic volume.",
            "- Tập trung retention journey cho cohort mới ngay sau đơn đầu.",
            "- Bảo vệ tệp khách mua nhiều lần vì đây là lớp doanh thu cấu trúc.",
            "- Dùng promo có chọn lọc, gắn với incremental conversion và lifetime value.",
        ]

    section_by_name = {str(note["section"]): note for note in section_notes}

    def render_aspect(section_name: str) -> str:
        note = section_by_name[section_name]
        headline = str(note["headline"]).strip().rstrip(".")
        bullets = list(note["bullets"])
        implication = str(note["implication"])
        figure = FIGURE_MAP.get(section_name)
        title = SECTION_TITLES.get(section_name, section_name)

        body_lines = [f"## {title}", ""]

        if section_name == "Framing doanh thu":
            body_lines += [
                f"{headline}. Đây là điểm mở khóa toàn bộ câu chuyện customer market: nếu AOV không giảm mà doanh thu vẫn rơi mạnh, thì vấn đề phải nằm ở số người mua và mức độ lặp lại hành vi mua.",
                "",
            ]
        elif section_name == "Acquisition & retention":
            body_lines += [
                f"{headline}. Đây là lớp bằng chứng quan trọng nhất vì nó cho thấy doanh nghiệp không chỉ hút được ít khách mới hơn, mà còn biến cohort mới thành khách quay lại ngày càng kém hơn.",
                "",
            ]
        elif section_name == "Loyalty erosion":
            body_lines += [
                f"{headline}. Khi lớp khách mua nhiều lần co lại, doanh nghiệp mất phần doanh thu ổn định nhất và buộc phải phụ thuộc nhiều hơn vào khách mua một lần.",
                "",
            ]
        elif section_name == "Demographics & market mix":
            body_lines += [
                f"{headline}. Vì mix theo tuổi, kênh và vùng gần như không xê dịch đáng kể, giả thuyết “khách hàng đổi hẳn tệp” không còn là lời giải thích chính.",
                "",
            ]
        elif section_name == "Promo & offer fit":
            body_lines += [
                f"{headline}. Điều này cho thấy doanh nghiệp đã dùng promo như một cách chống đỡ cầu yếu, nhưng promo không kéo được purchase frequency quay lại mức cũ.",
                "",
            ]
        elif section_name == "Service & traffic efficiency":
            body_lines += [
                f"{headline}. Nếu service tổng quan không xấu đi mạnh mà traffic vẫn tăng, thì điểm gãy nhiều khả năng nằm ở conversion, offer fit hoặc CRM journey.",
                "",
            ]
        elif section_name == "Chất lượng dữ liệu khách hàng":
            body_lines += [
                f"{headline}. Đây không phải business insight chính, nhưng rất quan trọng để tránh đọc sai cohort acquisition.",
                "",
            ]

        body_lines += [f"- {item}" for item in bullets]
        body_lines += ["", f"--> {implication}", ""]
        if figure:
            body_lines += [f"![{section_name}]({figure})", ""]
        return "\n".join(body_lines).strip()

    primary_sections = [name for name in SECTION_ORDER if name in section_by_name and name != "Chất lượng dữ liệu khách hàng"]
    method_sections = [name for name in SECTION_ORDER if name in section_by_name and name == "Chất lượng dữ liệu khách hàng"]

    executive = [
        "# FINAL INSIGHTS - Customer Market",
        "",
        "Tài liệu này được tổng hợp trực tiếp từ các insight evidence-driven trong `02_customer_market_deep_dive.ipynb` sau khi notebook đã chạy xong.",
        "",
        "## Executive Summary",
        "",
        lead,
        "",
        thesis,
        "",
        "Nói gọn theo góc nhìn business: doanh thu đi xuống sau 2016 không phải vì doanh nghiệp bán rẻ hơn, cũng không có bằng chứng mạnh cho thấy chân dung khách hàng đã đổi hẳn sang một tệp mới. Vấn đề nằm ở chỗ cỗ máy tăng trưởng khách hàng yếu đi: hút được ít khách mới chất lượng hơn, giữ khách kém hơn, và mất dần lớp khách mua nhiều lần vốn tạo ra doanh thu cấu trúc.",
        "",
        "## Business Story Theo Từng Aspect",
        "",
        *[piece for section in primary_sections for piece in (render_aspect(section), "")],
        "## Ghi chú phương pháp",
        "",
        *[piece for section in method_sections for piece in (render_aspect(section), "")],
        "",
        "## Kết luận chốt vấn đề",
        "",
        "Nếu phải diễn giải thành một câu cho business team: **đây là bài toán suy giảm chất lượng customer engine hơn là bài toán demographic shift hay service collapse**.",
        "",
        "Các bằng chứng khi ghép lại đang chỉ về một logic MECE khá rõ:",
        "",
        "- **Top of funnel** --> vẫn có traffic, nhưng hút được ít khách mua mới hơn.",
        "- **Retention layer** --> cohort mới quay lại kém hơn và retention sang năm sau xấu dần.",
        "- **Loyalty layer** --> nhóm khách mua nhiều lần, vốn tạo doanh thu cấu trúc, co rút rất mạnh.",
        "- **Segment mix** --> demographic mix gần như ổn định, nên không thể đổ lỗi chính cho việc “đổi tệp khách”.",
        "- **Commercial lever** --> promo được đẩy lên nhưng chỉ chống đỡ ngắn hạn, không sửa được behavior gốc.",
        "- **Ops / service** --> không xấu đi tương ứng với mức rơi doanh thu, nên không phải thủ phạm chính.",
        "",
        "Khi ghép tất cả lại, câu chuyện hợp lý nhất là: doanh nghiệp vẫn kéo được sự chú ý từ thị trường, nhưng **không còn chuyển hóa tốt traffic thành khách mua chất lượng và cũng không giữ được cohort mới trở thành loyal base**.",
        "",
        "## Bảng điểm giả thuyết cạnh tranh",
        "",
        *score_lines,
        "",
        "## Hàm ý hành động ưu tiên",
        "",
        *actions,
        "",
        "## Một câu chốt để dùng trong report/slide",
        "",
        "> Doanh thu giảm mạnh sau 2016 chủ yếu đến từ việc customer engine suy yếu: doanh nghiệp vừa hút ít khách mới hơn, vừa giữ khách kém hơn, trong khi mix demographic và service tổng quan không thay đổi đủ mạnh để giải thích cú giảm này.",
    ]

    return "\n".join(executive).strip() + "\n"


def main() -> None:
    markdown_outputs = load_markdown_outputs(NOTEBOOK)
    section_notes, summary = split_outputs(markdown_outputs)
    if not section_notes:
        raise RuntimeError("Không tìm thấy section insight markdown trong notebook.")
    if not summary:
        raise RuntimeError("Không tìm thấy phần summary markdown cuối notebook.")

    summary_parts = extract_summary_parts(summary)
    final_markdown = build_markdown(section_notes, summary_parts)
    OUTPUT.write_text(final_markdown, encoding="utf-8")
    print(f"Written: {OUTPUT}")


if __name__ == "__main__":
    main()
