# File: utils/export_onnx.py
import torch
from ultralytics import YOLO
import os


def export_to_onnx():
    """
    Chuyen doi model YOLOv8 sang ONNX format
    """
    # Duong dan model
    model_path = r"C:\Users\admin\PycharmProjects\Hoc_DL_CV\Helmet-Detection-YOLOv8\runs\detect\viet_traffic_signs_v8n2\weights\best.pt"
    output_dir = r"C:\Users\admin\PycharmProjects\Hoc_DL_CV\Helmet-Detection-YOLOv8\exported_models"

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Dang tai model YOLOv8...")
    model = YOLO(model_path)

    # Export sang ONNX
    print("Dang export sang ONNX...")
    try:
        success = model.export(
            format='onnx',
            imgsz=640,
            dynamic=True,  # Ho tro dynamic batch size
            simplify=True,  # Simplify model
            opset=12,  # ONNX opset version
            workspace=4  # GPU memory limit
        )

        if success:
            onnx_path = model_path.replace('.pt', '.onnx')
            print(f"Export ONNX thanh cong: {onnx_path}")

            # Di chuyen file ONNX den thu muc exported_models
            if os.path.exists(onnx_path):
                new_onnx_path = os.path.join(output_dir, "helmet_detection.onnx")
                os.rename(onnx_path, new_onnx_path)
                print(f"Da di chuyen ONNX den: {new_onnx_path}")

                # Kiem tra model ONNX
                check_onnx_model(new_onnx_path)
        else:
            print("Export ONNX that bai")

    except Exception as e:
        print(f"Loi khi export ONNX: {e}")


def check_onnx_model(onnx_path):
    """
    Kiem tra model ONNX
    """
    try:
        import onnx
        import onnxruntime as ort

        # Load va kiem tra model ONNX
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("ONNX model hop le")

        # Kiem tra voi ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        print(f"Input name: {input_name}")
        print(f"Input shape: {input_shape}")
        print(f"Outputs: {len(ort_session.get_outputs())}")

        return True

    except Exception as e:
        print(f"Loi kiem tra ONNX model: {e}")
        return False


if __name__ == "__main__":
    export_to_onnx()