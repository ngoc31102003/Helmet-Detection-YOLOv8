import os

# Mapping cho bộ cuối:
# index theo thứ tự trong names của bộ cuối:
# ['Helmet', 'Motorbike', 'Motorcyclist', 'Non_helmet', 'Plate', 'Plates']
# -> ta muốn: Motorbike -> Motorcyclist, Plates -> Plate
# nghĩa là:
# 0 -> 0  (Helmet)
# 1 -> 1  (Motorbike -> Motorcyclist)
# 2 -> 1  (Motorcyclist -> Motorcyclist)
# 3 -> 2  (Non_helmet)
# 4 -> 3  (Plate)
# 5 -> 3  (Plates)
remap = {
    "0": "0",
    "1": "3",
    "2": "1",
    "3": "2",
}

# Đường dẫn đến thư mục labels của bộ cuối
labels_dir = r"D:\HocTapUTC\DATN\Detect_mu_bao_hiem\goc_nhin_ngang_datasets_api490\train\labels"
output_dir = r"D:\HocTapUTC\DATN\Detect_mu_bao_hiem\goc_nhin_ngang_datasets_api490\train\labels_fixed"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(labels_dir):
    if not file.endswith(".txt"):
        continue
    src_path = os.path.join(labels_dir, file)
    dst_path = os.path.join(output_dir, file)
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        old_id = parts[0]
        if old_id not in remap:
            continue
        new_id = remap[old_id]
        new_lines.append(" ".join([new_id] + parts[1:]) + "\n")

    with open(dst_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

print("✅ Đã remap xong class cho toàn bộ label file!")
print("Các class chuẩn cuối cùng là: ['Helmet', 'Motorcyclist', 'Non_helmet', 'Plate']")
