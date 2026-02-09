from ultralytics import YOLO
from pathlib import Path
import time
import sys

import torch
import cv2

# ----------------------------
# Config
# ----------------------------
VIDEO_PATH = ""
MODEL_PATH = ""

# BEST for performance
MODEL_ONNX = ""
MODEL_ENGINE = ""

# Tracker
TRACKER_PATH = ""

# ----------------------------
# Check video
# ----------------------------
if Path(VIDEO_PATH).exists():
    print("Video exists")
else:
    print("Video does not exist")
    sys.exit(-1)

# ----------------------------
# Device selection
# ----------------------------
cuda_available = torch.cuda.is_available()

if cuda_available:
    device = 0
    half = True
    backend = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    half = False  # never on cpu
    backend = "onnx"

print(f"Backend selected: {backend}")

# ----------------------------
# Global optimizations
# ----------------------------
if device != "cpu":  # (CUDA only)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True


# ----------------------------
# Load model
# ----------------------------
def load_model():
    # ---------------- CUDA PATH ----------------
    if backend == "cuda":  # NVIDIA
        if Path(MODEL_ENGINE).exists():
            print("TensorRT engine exists")
            print("Using TensorRT engine")
            return YOLO(MODEL_ENGINE)
        else:
            print("Engine does not exist")
            print("Exporting TensorRT engine")
            model = YOLO(MODEL_PATH)
            path = model.export(format="engine", device=device)
            print("Model exported to path: ", path)
            return YOLO(path)
    # ---------------- CPU PATH ----------------
    else:  # ANY CPU
        if Path(MODEL_ONNX).exists():
            print("ONNX Runtime exists")
            print("Using ONNX Runtime")
            return YOLO(MODEL_ONNX)
        else:
            print("ONNX does not exist")
            print("Exporting ONNX")
            model = YOLO(MODEL_PATH)
            path = model.export(format="onnx", simplify=True)
            print("Model exported to path: ", path)
            return YOLO(path)


# Load model
model = load_model()

# Warmup (CUDA only)
if device != "cpu":
    dummy = torch.zeros(1, 3, 640, 640).to(device)
    model(dummy, device=device, half=half)

# ----------------------------
# Video
# ----------------------------

# Open the video
cap = cv2.VideoCapture(VIDEO_PATH)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# # Output video
# out = cv2.VideoWriter(
#     "annotated_match.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
# )

imgsz = 640  # (640 x 640)
display_scale = 0.5

frame_id = 0
frame_skip = 1
prev_time = time.time()

# ----------------------------
# Main loop
# ----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % frame_skip != 0:
        continue

    start = time.time()

    results = model(
        source=frame,
        device=device,
        imgsz=imgsz,
        half=half,
        # stream=True,
        verbose=False,
    )

    # results = model.track(
    #     source=frame,
    #     device=device,
    #     imgsz=imgsz,
    #     half=half,
    #     persist=True,
    #     tracker=TRACKER_PATH,
    #     # classes=[0, 1, 2, 3],
    #     verbose=False,
    # )

    annotated = results[0].plot(labels=True, boxes=True)

    # # Here is the bottleneck
    # # Draw detections manually
    # # (Much faster than plot)
    # for r in results:
    #     if r.boxes is None:
    #         continue

    #     boxes = r.boxes.xyxy.cpu().numpy()
    #     confs = r.boxes.conf.cpu().numpy()
    #     clss = r.boxes.cls.cpu().numpy()

    #     for box, conf, cls in zip(boxes, confs, clss):
    #         x1, y1, x2, y2 = map(int, box)
    #         cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(
    #             frame_resized,
    #             f"{model.names[int(cls)]} {conf:.2f}",
    #             (x1, y1 - 5),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             (0, 255, 0),
    #             1,
    #         )
    # # End of bottleneck

    # Compute FPS
    fps = 1 / (time.time() - start + 1e-8)
    print(f"FPS: {fps:.1f}")

    frame_resized = cv2.resize(
        src=annotated,
        dsize=None,
        fx=display_scale,
        fy=display_scale,
    )

    # Draw FPS
    cv2.putText(
        frame_resized,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Show output
    cv2.imshow("YOLO Detection", frame_resized)

    # Save output
    # out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
