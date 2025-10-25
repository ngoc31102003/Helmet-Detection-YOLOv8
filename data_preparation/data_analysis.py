import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

images_dir = r"D:\HocTapUTC\DATN\Detect_mu_bao_hiem\datasets_doi_mu_bao_hiem_new\train\balanced_train\images"
labels_dir = r"D:\HocTapUTC\DATN\Detect_mu_bao_hiem\datasets_doi_mu_bao_hiem_new\train\balanced_train\labels"

class_names = ['Helmet', 'Motorcyclist', 'Non_helmet', 'Plate']

image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) +
                     glob.glob(os.path.join(images_dir, "*.png")))

label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))

print(f"Tổng số ảnh: {len(image_files)}")
print(f"Tổng số file nhãn: {len(label_files)}")

class_counts = Counter()
box_widths = []
box_heights = []

for label_path in label_files:
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, bw, bh = map(float, parts)
            cls = int(cls)
            class_counts[cls] += 1
            box_widths.append(bw)
            box_heights.append(bh)

print("\nSố lượng đối tượng theo class:")
for i, name in enumerate(class_names):
    print(f" - {name:<15}: {class_counts[i]}")

plt.figure(figsize=(8, 5))
plt.bar(class_names, [class_counts[i] for i in range(len(class_names))], color='skyblue')
plt.title("Phân bố số lượng đối tượng theo class")
plt.xlabel("Class")
plt.ylabel("Số lượng")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(box_widths, bins=30, alpha=0.6, label='Width', color='orange')
plt.hist(box_heights, bins=30, alpha=0.6, label='Height', color='green')
plt.legend()
plt.title("📏 Phân bố kích thước bounding box (chuẩn hóa)")
plt.xlabel("Tỉ lệ so với kích thước ảnh")
plt.ylabel("Số lượng")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# === Hiển thị ngẫu nhiên một vài ảnh có bounding box ===
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
display_width, display_height = 1280, 720

sample_images = np.random.choice(image_files, size=min(5, len(image_files)), replace=False)
for img_path in sample_images:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_dir, base_name + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls, x, y, bw, bh = map(float, line.strip().split())
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                color = colors[int(cls) % len(colors)]
                label = class_names[int(cls)] if int(cls) < len(class_names) else f"cls_{int(cls)}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    img_resized = cv2.resize(img, (display_width, display_height))
    cv2.imshow("YOLO Dataset Visualization", img_resized)
    key = cv2.waitKey(1000)
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
