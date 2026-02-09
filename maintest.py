from utils import read_video, save_video
from trackers import Tracker
from homography_test import SoccerHomographyPipeline, HomographyResult
from homography.pitch_visualizer import LivePitchVisualizer, TacticalDisplay, TrackedObject
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import threading
import torch
import time
import cv2
import sys


# ==============================================================================
# Background homography worker
# ==============================================================================


class HomographyWorker:
    """
    Runs the homography pipeline in a background thread so the main
    video loop never blocks.

    Usage:
        worker = HomographyWorker(pipeline)
        worker.start()

        # In the main loop — non-blocking:
        if frame_id % 10 == 0:
            worker.submit(frame)       # drops the frame in, returns instantly

        H_result = worker.get_result() # returns latest valid result (or None)

        # On exit:
        worker.stop()
    """

    def __init__(self, pipeline: SoccerHomographyPipeline):
        self._pipeline = pipeline

        # Latest valid result — read from main thread, written by worker
        self._result: HomographyResult | None = None
        self._result_lock = threading.Lock()

        # Pending frame — written by main thread, consumed by worker
        self._pending_frame: np.ndarray | None = None
        self._pending_lock = threading.Lock()
        self._event = threading.Event()  # signals "new frame available"

        self._running = False
        self._thread: threading.Thread | None = None

        # Stats (informational, no lock needed for a single float)
        self.last_compute_ms: float = 0.0

    # ── control ───────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ── main-thread API (non-blocking) ────────────────────────────────

    def submit(self, frame: np.ndarray):
        """
        Hand a frame to the worker.  Returns immediately.

        If the worker is still busy with a previous frame the pending
        frame is *replaced* (latest-only — we never queue stale frames).
        """
        with self._pending_lock:
            self._pending_frame = frame.copy()
        self._event.set()

    def get_result(self) -> HomographyResult | None:
        """Return the latest valid result (or None).  Non-blocking."""
        with self._result_lock:
            return self._result

    # ── background loop ───────────────────────────────────────────────

    def _loop(self):
        while self._running:
            self._event.wait()
            self._event.clear()

            if not self._running:
                break

            with self._pending_lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                continue

            t0 = time.perf_counter()
            try:
                h_result = self._pipeline.process(frame)
                self.last_compute_ms = (time.perf_counter() - t0) * 1000.0

                if h_result.is_valid:
                    with self._result_lock:
                        self._result = h_result
                    print(
                        f"[homography bg] UPDATED — "
                        f"{h_result.inlier_count} inliers, "
                        f"err={h_result.reprojection_error:.3f}m  "
                        f"({self.last_compute_ms:.0f} ms)"
                    )
                else:
                    print(
                        f"[homography bg] FAILED — keeping previous  "
                        f"({self.last_compute_ms:.0f} ms)"
                    )
            except Exception as e:
                self.last_compute_ms = (time.perf_counter() - t0) * 1000.0
                print(f"[homography bg] exception: {e}")


# ==============================================================================
# Tracker output adapter
# ==============================================================================


def parse_tracks(tracks, tracker=None) -> list:
    objects = []

    # Pattern A — ultralytics Results
    try:
        for result in tracks if isinstance(tracks, list) else [tracks]:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            ids = (
                boxes.id.cpu().numpy().astype(int)
                if boxes.id is not None
                else np.arange(len(xyxy))
            )
            names = result.names
            for i in range(len(xyxy)):
                objects.append(
                    TrackedObject(
                        track_id=int(ids[i]),
                        class_id=int(cls[i]),
                        class_name=names.get(int(cls[i]), "player"),
                        bbox=tuple(xyxy[i].tolist()),
                    )
                )
        return objects
    except (AttributeError, TypeError):
        pass

    # Pattern B — nested dict
    try:
        if isinstance(tracks, dict):
            for class_name, id_dict in tracks.items():
                for track_id, info in id_dict.items():
                    bbox = info.get("bbox", info.get("box", None))
                    if bbox is None:
                        continue
                    objects.append(
                        TrackedObject(
                            track_id=int(track_id),
                            class_id=0,
                            class_name=str(class_name),
                            bbox=tuple(bbox),
                        )
                    )
            return objects
    except (AttributeError, TypeError):
        pass

    # Pattern C — list of dicts/tuples
    if isinstance(tracks, list):
        for t in tracks:
            if isinstance(t, dict):
                objects.append(
                    TrackedObject(
                        track_id=int(t.get("track_id", t.get("id", 0))),
                        class_id=int(t.get("class_id", t.get("cls", 0))),
                        class_name=str(t.get("class_name", t.get("name", "player"))),
                        bbox=tuple(t.get("bbox", t.get("box", (0, 0, 0, 0)))),
                    )
                )
            elif isinstance(t, (list, tuple)) and len(t) >= 6:
                objects.append(
                    TrackedObject(
                        track_id=int(t[4]),
                        class_id=int(t[5]),
                        class_name=str(t[6]) if len(t) > 6 else "player",
                        bbox=(float(t[0]), float(t[1]), float(t[2]), float(t[3])),
                    )
                )

    return objects


# ==============================================================================
# Main
# ==============================================================================


def main():
    # ----------------------------
    # Config
    # ----------------------------
    VIDEO_PATH = "D:/ITE/year5/graduation project/5th Dataset/Full Match Tactical Cam UCL UEFA Champions League  24-25  2nd Leg - Liverpool vs PSG (11 Mar 2025).mp4"
    MODEL_PATH = "models/yolo26/weights/best.pt"

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

    HOMOGRAPHY_INTERVAL = 10

    # ----------------------------
    # Check video
    # ----------------------------
    if not Path(VIDEO_PATH).exists():
        print("Video does not exist")
        sys.exit(-1)
    print("Video exists")

    # ----------------------------
    # Device
    # ----------------------------
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = 0
        half = True
        backend = "cuda"
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        half = False
        backend = "onnx"
    print(f"Backend: {backend}")

    if device != "cpu":
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True

    # ----------------------------
    # Model
    # ----------------------------
    def load_model():
        if backend == "cuda":
            if Path(MODEL_ENGINE).exists():
                return YOLO(MODEL_ENGINE)
            model = YOLO(MODEL_PATH)
            path = model.export(format="engine", device=device, half=half, dynamic=True)
            return YOLO(path)
        else:
            if Path(MODEL_ONNX).exists():
                return YOLO(MODEL_ONNX)
            model = YOLO(MODEL_PATH)
            path = model.export(format="onnx", simplify=True, half=half, dynamic=True)
            return YOLO(path)

    model = load_model()
    imgsz = 640
    tracker = Tracker(model, TRACKER_NAME, imgsz, device, half)

    if device != "cpu":
        dummy = torch.zeros(1, 3, 640, 640).to(device)
        model(dummy, device=device, half=half)

    # ----------------------------
    # Homography — background worker
    # ----------------------------
    homography_config = {
        "hue_range": (35, 85),
        "sat_range": (40, 255),
        "val_range": (40, 255),
        "adaptive_k": 1.1,
        "block_size": 101,
        "hough_threshold": 30,
        "min_length": 30,
        "angle_h_thresh": 30,
        "angle_v_thresh": 30,
        "min_contour_length": 60,
        "min_aspect_ratio": 0.15,
        "min_area": 50,
        "close_iters": 2,
        "adaptive_k_edge": 0.8,
        "edge_band": 0.25,
        "min_confidence_for_selection": 0.2,
        "min_touchline_length_ratio": 0.25,
        "refiner_frame_margin": 50.0,
        "refiner_snap_tolerance": 0.3,
    }

    h_worker = HomographyWorker(SoccerHomographyPipeline(homography_config))
    h_worker.start()

    # ----------------------------
    # Pitch visualizer
    # ----------------------------
    pitch_viz = LivePitchVisualizer(
        canvas_width=1050,
        canvas_height=680,
        margin=50,
        trail_length=30,
    )
    tactical_display = TacticalDisplay(
        video_width=854,
        video_height=480,
        pitch_width=525,
        pitch_height=480,
    )

    # ----------------------------
    # Video
    # ----------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FPS, 30)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_id = 0
    frame_skip = 1

    # ----------------------------
    # Main loop  (never blocks on homography)
    # ----------------------------
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % frame_skip != 0:
                continue

            start = time.time()

            # 1) Track (runs every frame)
            tracks = tracker.track_frame(frame)
            annotated = tracker.draw_annotations(frame, tracks)

            # 2) Submit frame for background homography (instant return)
            if frame_id % HOMOGRAPHY_INTERVAL == 0:
                h_worker.submit(frame)

            # 3) Read latest homography (instant return)
            h_result = h_worker.get_result()
            H = (
                h_result.homography
                if h_result is not None and h_result.is_valid
                else None
            )

            # 4) Parse tracker output
            tracked_objects = parse_tracks(tracks, tracker)

            # 5) Render 2D pitch
            pitch_img = pitch_viz.update(
                tracks=tracked_objects,
                H=H,
                show_trails=True,
                show_ids=True,
            )

            # 6) Status overlay on video
            if H is not None:
                cv2.putText(
                    annotated,
                    f"H: {h_result.inlier_count}pts "
                    f"err={h_result.reprojection_error:.2f}m "
                    f"({h_worker.last_compute_ms:.0f}ms bg)",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    annotated,
                    "H: computing...",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            # 7) FPS
            fps = 1.0 / (time.time() - start + 1e-8)

            # 8) Side-by-side display
            combined = tactical_display.combine(
                annotated,
                pitch_img,
                info_text=f"Frame {frame_id}  |  FPS: {fps:.1f}",
            )
            cv2.imshow("TactixAI", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        h_worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
