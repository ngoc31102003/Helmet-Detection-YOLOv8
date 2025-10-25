import cv2
import os
import glob

images_dir = r"D:\HocTapUTC\DATN\Detect_mu_bao_hiem\datasets_doi_mu_bao_hiem_new\train\images"
labels_dir = r"D:\HocTapUTC\DATN\Detect_mu_bao_hiem\datasets_doi_mu_bao_hiem_new\train\labels"

class_names = ['Helmet', 'Motorcyclist', 'Non_helmet', 'Plate']

image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) +
                     glob.glob(os.path.join(images_dir, "*.png")))

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

display_width = 1280
display_height = 720

for img_path in image_files:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_dir, base_name + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        continue

    orig_h, orig_w = img.shape[:2]

    # Đọc nhãn
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls, x, y, bw, bh = map(float, line.strip().split())

                x1 = int((x - bw / 2) * orig_w)
                y1 = int((y - bh / 2) * orig_h)
                x2 = int((x + bw / 2) * orig_w)
                y2 = int((y + bh / 2) * orig_h)

                color = colors[int(cls) % len(colors)]
                label = class_names[int(cls)] if int(cls) < len(class_names) else f"cls_{int(cls)}"

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    img_resized = cv2.resize(img, (display_width, display_height))

    cv2.imshow("YOLOv8 Dataset Visualization", img_resized)

    key = cv2.waitKey(1000)
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
