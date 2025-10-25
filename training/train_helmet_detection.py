from ultralytics import YOLO
import torch

# Kiểm tra GPU
if torch.cuda.is_available():
    print("GPU đã sẵn sàng:", torch.cuda.get_device_name(0))
else:
    print("Chạy trên CPU, sẽ chậm hơn!")

model = YOLO("yolov8n.pt")

# Train
model.train(
    data= r"/root/yolov8/datasets_doi_mu_bao_hiem_new/dataset.yaml",  # đường dẫn đến file yaml
    imgsz=640,
    epochs=50,
    batch=160,
    name= "viet_mubaohiem_v8n",
    device=0,
    project="runs_time/train_yolov8n",
    workers = 8
)

metrics = model.val()
print("📊 Kết quả đánh giá:")
print(metrics)
