import cv2
import torch
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Thiết bị đang dùng: {device}")

model_path = r"C:\Users\admin\PycharmProjects\Hoc_DL_CV\Helmet-Detection-YOLOv8\runs\detect\viet_traffic_signs_v8n2\weights\best.pt"

model = YOLO(model_path)
model.to(device)
print("Đã load mô hình thành công!")

class_names = [
    "Co doi mu",
    "PTGT",
    "Khong doi mu",
    "Bien so xe"
]

colors = {
    0: (0, 255, 0),  # Xanh lá - Có mũ
    1: (255, 255, 0),  # Vàng - Người lái xe
    2: (0, 0, 255),  # Đỏ - Không mũ
    3: (255, 0, 0)  # Xanh dương - Biển số
}

conf_threshold = 0.35


def run_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Không mở được video!")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS video gốc: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (640, 640))
        results = model.predict(source=frame_resized, device=device, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Lọc theo confidence
                if conf < conf_threshold:
                    continue

                label = f"{class_names[cls_id]} {conf:.2f}"
                color = colors.get(cls_id, (255, 255, 255))

                x1 = int(x1 * frame.shape[1] / 640)
                x2 = int(x2 * frame.shape[1] / 640)
                y1 = int(y1 * frame.shape[0] / 640)
                y2 = int(y2 * frame.shape[0] / 640)

                # Vẽ bounding box và label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Phát hiện Mũ Bảo Hiểm & Biển Số (YOLOv8 CUDA)", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"D:\Tải xuống từ Chomre\test_1.mp4"
    run_video(video_path)
