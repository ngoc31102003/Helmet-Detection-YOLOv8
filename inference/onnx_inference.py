# File: inference/onnx_inference_fixed.py
import cv2
import numpy as np
import onnxruntime as ort
import time
import os


class ONNXHelmetDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        """
        Khoi tao ONNX detector

        Args:
            model_path: Duong dan den file .onnx
            conf_threshold: Nguong confidence
        """
        self.conf_threshold = conf_threshold
        self.class_names = ['Helmet', 'Motorcyclist', 'Non_helmet', 'Plate']
        self.colors = [(0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 0)]

        # Khoi tao ONNX Runtime
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"Loi khi tai ONNX model: {e}")
            # Thu voi CPU only
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)

        # Lay thong tin input va output
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]

        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        print(f"ONNX Model loaded: {model_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Input name: {self.input_name}")
        print(f"Output names: {self.output_names}")

    def preprocess(self, image):
        """
        Tien xu anh dau vao
        """
        # Resize anh
        input_img = cv2.resize(image, (self.input_width, self.input_height))

        # Chuyen doi color space
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        # Normalize
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)  # HWC to CHW
        input_img = np.expand_dims(input_img, axis=0).astype(np.float32)

        return input_img

    def postprocess(self, outputs, original_shape):
        """
        Hau xu ket qua cho YOLOv8 ONNX format
        """
        # YOLOv8 ONNX co output shape: [1, 8, 8400]
        # Trong do: 8 = 4 (bbox) + 4 (classes)
        predictions = outputs[0]  # Shape: [1, 8, 8400]

        boxes = []
        scores = []
        class_ids = []

        orig_height, orig_width = original_shape[:2]

        # Remove batch dimension
        predictions = np.squeeze(predictions, 0)  # Shape: [8, 8400]

        # Tach boxes (4 first values) va class scores (4 last values)
        boxes_data = predictions[:4, :]  # [4, 8400] - x_center, y_center, width, height
        scores_data = predictions[4:, :]  # [4, 8400] - class scores

        # Chuyen doi boxes_data va scores_data
        boxes_data = boxes_data.transpose(1, 0)  # [8400, 4]
        scores_data = scores_data.transpose(1, 0)  # [8400, 4]

        # Tim class_id va confidence
        class_ids = np.argmax(scores_data, axis=1)  # [8400]
        confidences = np.max(scores_data, axis=1)  # [8400]

        # Loc theo confidence
        valid_detections = confidences > self.conf_threshold
        boxes_data = boxes_data[valid_detections]
        class_ids = class_ids[valid_detections]
        confidences = confidences[valid_detections]

        if len(boxes_data) == 0:
            return boxes, scores, class_ids

        # Chuyen doi toa do YOLO format ve pixel coordinates
        scale_x = orig_width / self.input_width
        scale_y = orig_height / self.input_height

        # Convert from center format to corner format
        x_center = boxes_data[:, 0] * scale_x
        y_center = boxes_data[:, 1] * scale_y
        width = boxes_data[:, 2] * scale_x
        height = boxes_data[:, 3] * scale_y

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Ensure coordinates are within image boundaries
        x1 = np.clip(x1, 0, orig_width)
        y1 = np.clip(y1, 0, orig_height)
        x2 = np.clip(x2, 0, orig_width)
        y2 = np.clip(y2, 0, orig_height)

        boxes = np.column_stack([x1, y1, x2, y2])
        scores = confidences

        return boxes, scores, class_ids

    def detect(self, image):
        """
        Phat hien doi tuong trong anh
        """
        original_shape = image.shape

        # Tien xu
        input_tensor = self.preprocess(image)

        # Inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        inference_time = time.time() - start_time

        # Hau xu
        boxes, scores, class_ids = self.postprocess(outputs, original_shape)

        return boxes, scores, class_ids, inference_time

    def draw_detections(self, image, boxes, scores, class_ids):
        """
        Ve bounding boxes len anh
        """
        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            x1, y1, x2, y2 = box.astype(int)

            # Ve bounding box
            color = self.colors[class_id % len(self.colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Ve label
            label = f"{self.class_names[class_id]}: {score:.2f}"

            # Tinh toan kich thuoc text
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Ve background cho label
            cv2.rectangle(image,
                          (x1, y1 - label_height - 10),
                          (x1 + label_width, y1),
                          color, -1)

            # Ve text
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return image


def test_onnx_inference():
    """
    Test ONNX inference
    """
    model_path = r"C:\Users\admin\PycharmProjects\Hoc_DL_CV\Helmet-Detection-YOLOv8\exported_models\helmet_detection.onnx"

    if not os.path.exists(model_path):
        print("Khong tim thay model ONNX. Hay chay export_onnx.py truoc")
        return

    print(f"Testing ONNX model: {model_path}")

    # Khoi tao detector
    detector = ONNXHelmetDetector(model_path, conf_threshold=0.5)

    # Test voi webcam
    cap = cv2.VideoCapture(0)  # Webcam

    if not cap.isOpened():
        print("Khong the mo webcam")
        # Thu voi video file
        video_path = r"D:\Tải xuống từ Chomre\test_1.mp4"  # Thay duong dan video cua ban
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Khong the mo video file. Thoat.")
        return

    frame_count = 0
    total_inference_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Khong doc duoc frame. Thoat.")
            break

        # Phat hien
        boxes, scores, class_ids, inference_time = detector.detect(frame)

        total_inference_time += inference_time
        frame_count += 1

        # Ve ket qua
        frame_with_detections = detector.draw_detections(frame.copy(), boxes, scores, class_ids)

        # Hien thi FPS
        fps = 1.0 / inference_time if inference_time > 0 else 0
        avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0

        cv2.putText(frame_with_detections, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_with_detections, f"Avg FPS: {avg_fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_with_detections, f"Detections: {len(boxes)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ONNX Helmet Detection", frame_with_detections)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space bar to pause
            cv2.waitKey(0)

    # Thong ke
    if frame_count > 0:
        avg_inference_time = total_inference_time / frame_count * 1000  # ms
        print(f"\nThong ke:")
        print(f"Tong so frame: {frame_count}")
        print(f"Thoi gian inference trung binh: {avg_inference_time:.1f} ms")
        print(f"FPS trung binh: {1000 / avg_inference_time:.1f}")

    cap.release()
    cv2.destroyAllWindows()


def test_onnx_with_image():
    """
    Test ONNX inference voi anh
    """
    model_path = r"C:\Users\admin\PycharmProjects\Hoc_DL_CV\Helmet-Detection-YOLOv8\exported_models\helmet_detection.onnx"

    if not os.path.exists(model_path):
        print("Khong tim thay model ONNX")
        return

    # Khoi tao detector
    detector = ONNXHelmetDetector(model_path, conf_threshold=0.5)

    # Test voi anh
    test_image_path = r"C:\Users\admin\PycharmProjects\Hoc_DL_CV\Helmet-Detection-YOLOv8\test_image.jpg"

    if not os.path.exists(test_image_path):
        # Tao anh test neu khong ton tai
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, test_image)
        print(f"Da tao anh test: {test_image_path}")

    # Doc anh
    image = cv2.imread(test_image_path)
    if image is None:
        print("Khong doc duoc anh")
        return

    print(f"Kich thuoc anh: {image.shape}")

    # Phat hien
    boxes, scores, class_ids, inference_time = detector.detect(image)

    print(f"Thoi gian inference: {inference_time * 1000:.1f} ms")
    print(f"So luong phat hien: {len(boxes)}")

    # Ve ket qua
    result_image = detector.draw_detections(image.copy(), boxes, scores, class_ids)

    # Hien thi ket qua
    cv2.putText(result_image, f"Inference: {inference_time * 1000:.1f}ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result_image, f"Detections: {len(boxes)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize de hien thi
    display_image = cv2.resize(result_image, (800, 600))
    cv2.imshow("ONNX Detection Result", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Luu ket qua
    output_path = r"C:\Users\admin\PycharmProjects\Hoc_DL_CV\Helmet-Detection-YOLOv8\onnx_detection_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"Da luu ket qua: {output_path}")


if __name__ == "__main__":
    print("Chon che do test:")
    print("1. Test voi webcam/video")
    print("2. Test voi anh")

    choice = input("Nhap lua chon (1 hoac 2): ").strip()

    if choice == "1":
        test_onnx_inference()
    elif choice == "2":
        test_onnx_with_image()
    else:
        print("Lua chon khong hop le. Thoat.")