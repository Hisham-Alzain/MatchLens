from utils import read_video, save_video
from trackers import Assigner
from trackers import Tracker
from ultralytics import YOLO
from pathlib import Path
import torch
import time
import cv2
import sys


def main():
    # ----------------------------
    # Config
    # ----------------------------
    VIDEO_PATH = "videos/test.mp4"
    MODEL_PATH = "models/yolo26/weights/best.pt"

    # BEST for performance
    MODEL_ONNX = "models/yolo26/weights/best.onnx"
    MODEL_ENGINE = "models/yolo26/weights/best.engine"

    TRACKERS = [
        "botsort",
        "boosttrack",
        "strongsort",
        "deepocsort",
        "bytetrack",
        "hybridsort",
        "ocsort",
    ]
    TRACKER_NAME = TRACKERS[4]

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
                path = model.export(
                    format="engine", device=device, half=half, dynamic=True
                )
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
                path = model.export(
                    format="onnx", simplify=True, half=half, dynamic=True
                )
                print("Model exported to path: ", path)
                return YOLO(path)

    # Load model
    model = load_model()

    imgsz = 640  # (640 x 640)
    tracker = Tracker(model, TRACKER_NAME, imgsz, device, half)

    # Init Assigner
    assigner = Assigner()
    #

    # Warmup (CUDA only)
    if device != "cpu":
        dummy = torch.zeros(1, 3, 640, 640).to(device)
        model(dummy, device=device, half=half)

    # ----------------------------
    # Video
    # ----------------------------

    # Open the video
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Try to limit FPS
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # # Output video
    # out = cv2.VideoWriter(
    #     "videos/out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    # )

    frame_id = 0
    frame_skip = 1
    display_scale = 0.5

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

        tracks = tracker.track_frame(frame)

        if frame_id < 60:
            assigner.collect_team_colors(frame, tracks["players"])
        elif frame_id == 60:
            assigner.finalize_team_colors()
        else:
            for track_id, player in tracks["players"].items():
                team = assigner.get_player_team_stable(frame, player["bbox"], track_id)

                # Assign each player to it's team
                tracks["players"][track_id]["team"] = team
                tracks["players"][track_id]["team_color"] = assigner.hsv_to_bgr(
                    assigner.team_colors[team]
                )

        annotated = tracker.draw_annotations(frame, tracks)

        # boxes, scores, classes, names = tracker.detect_frame(frame)
        # annotated = tracker.draw_detections_custom(frame, boxes, scores, classes, names)

        # Compute FPS
        fps = 1 / (time.time() - start + 1e-8)
        print(f"FPS: {fps:.1f}")

        frame_resized = cv2.resize(src=annotated, dsize=(854, 480))

        # frame_resized = cv2.resize(
        #     src=annotated,
        #     dsize=None,
        #     fx=display_scale,
        #     fy=display_scale,
        # )

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
        # out.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
