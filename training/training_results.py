import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


def display_images(post_training_path, image_files):
    """
    Hàm hiển thị các ảnh kết quả sau huấn luyện YOLO.
    """
    for image_file in image_files:
        image_path = os.path.join(post_training_path, image_file)

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10), dpi=120)
        plt.imshow(img)
        plt.title(image_file, fontsize=12)
        plt.axis('off')
        plt.show()

post_training_path = r"D:\HocTapUTC\DATN\Detect_mu_bao_hiem\runs_time\train_yolov8n"  # Thay đổi nếu cần

image_files = [
    'confusion_matrix_normalized.png',
    'F1_curve.png',
    'P_curve.png',
    'R_curve.png',
    'PR_curve.png',
    'results.png'
]

display_images(post_training_path, image_files)

results_csv = os.path.join(post_training_path, 'results.csv')

if os.path.exists(results_csv):
    Result_model = pd.read_csv(results_csv)
    print("10 dòng cuối cùng trong file kết quả huấn luyện:")
    print(Result_model.tail(10))
else:
    print("Không tìm thấy file 'results.csv' trong thư mục:", post_training_path)

model_path = os.path.join(post_training_path, 'weights', 'best.pt')

if os.path.exists(model_path):
    print("\nĐang tải mô hình và đánh giá trên tập validation...")
    Valid_model = YOLO(model_path)

    metrics = Valid_model.val(split='val')

    print("\nKết quả đánh giá mô hình:")
    print(f"Precision(B): {metrics.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall(B): {metrics.results_dict['metrics/recall(B)']:.4f}")
    print(f"mAP50(B): {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95(B): {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
else:
    print("Không tìm thấy mô hình 'best.pt' trong thư mục:", model_path)
