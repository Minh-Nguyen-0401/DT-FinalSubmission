import pandas as pd
from pathlib import Path

md_path = Path(__file__).parent / "Domain5_Feature_Forecast.md"
out_path = Path(__file__).parent / "Domain5_Feature_Forecast.xlsx"

lines = md_path.read_text(encoding="utf-8").splitlines()

start = None
for i, line in enumerate(lines):
    if line.strip().startswith("| Nhóm | Tên đặc trưng |"):
        start = i
        break

if start is None:
    raise RuntimeError("Không tìm thấy bảng feature chính trong markdown.")

table_lines = []
for line in lines[start:]:
    if line.strip().startswith("|"):
        table_lines.append(line.strip())
    else:
        break

rows = []
for line in table_lines[2:]:
    cells = [c.strip() for c in line.strip("|").split("|")]
    if len(cells) >= 5:
        group, feature, formula, meaning, _priority = cells[:5]
        rows.append(
            {
                "Tên feature": feature,
                "Định nghĩa/Ý nghĩa": meaning,
                "Công thức tính": formula,
                "Source": group,
                "Grain": "",
            }
        )

df = pd.DataFrame(
    rows,
    columns=["Tên feature", "Định nghĩa/Ý nghĩa", "Công thức tính", "Source", "Grain"],
)

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Domain5_Features")

print(f"Đã tạo file: {out_path}")
print(f"Số feature: {len(df)}")
