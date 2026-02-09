"""
Homography Diagnostic Tool - Full Pipeline Version

This script runs the full detection + tracking pipeline while showing
detailed homography diagnostic information to help debug issues.

Usage:
    python diagnose_homography.py

Press keys during playback:
    d - Toggle detailed debug output
    c - Force recalibration
    p - Pause/resume
    s - Step one frame (when paused)
    q - Quit
"""

from utils import read_video, save_video
from trackers import Tracker
from homography.homography import AutoHomographyCalibrator, PitchTemplate
from ultralytics import YOLO
from pathlib import Path
import torch
import time
import cv2
import sys
import numpy as np
from homography.pitch_visualizer import TacticalDisplay, PitchVisualizer


def print_diagnostic_report(detection, calibrator, frame_shape):
    """Print detailed diagnostic information about the homography."""
    h, w = frame_shape[:2]

    print(f"\n{'='*70}")
    print("HOMOGRAPHY DIAGNOSTIC REPORT")
    print(f"{'='*70}")

    # ============== GRASS SEGMENTATION ==============
    print(f"\n[1] GRASS SEGMENTATION")
    mask_field = detection["mask_field"]
    grass_pixels = np.sum(mask_field > 0)
    grass_percent = 100 * grass_pixels / (w * h)
    status = "✓" if grass_percent > 30 else "⚠️"
    print(f"    {status} Grass coverage: {grass_percent:.1f}% of frame")

    # ============== LINE DETECTION ==============
    print(f"\n[2] LINE DETECTION")
    lines = detection["lines"]
    vertical = detection["vertical_lines"]
    horizontal = detection["horizontal_lines"]
    print(f"    Total lines: {len(lines)}")
    print(f"    Vertical: {len(vertical)}, Horizontal: {len(horizontal)}")

    if len(vertical) > 0:
        best_vert = max(vertical, key=lambda l: l.length)
        print(
            f"    Best vertical: length={best_vert.length:.0f}px at x={best_vert.midpoint[0]:.0f}"
        )
    else:
        print(f"    ⚠️ No vertical lines detected!")

    if len(horizontal) > 0:
        for line in sorted(horizontal, key=lambda l: l.length, reverse=True)[:2]:
            pos = "TOP" if line.midpoint[1] < h * 0.5 else "BOTTOM"
            print(
                f"    Horizontal ({pos}): length={line.length:.0f}px at y={line.midpoint[1]:.0f}"
            )
    else:
        print(f"    ⚠️ No horizontal lines detected (using field mask boundary instead)")

    # ============== CIRCLE DETECTION ==============
    print(f"\n[3] CIRCLE/ELLIPSE DETECTION")
    circle = detection["circle"]
    ellipse = detection["ellipse"]

    if circle:
        cx, cy, r = circle
        rel_x = 100 * cx / w
        rel_y = 100 * cy / h
        status = "✓" if 30 < rel_x < 70 else "⚠️"
        print(f"    {status} Circle: center=({cx:.0f}, {cy:.0f}), radius={r:.0f}px")
        print(f"       Relative position: ({rel_x:.0f}%, {rel_y:.0f}%)")
    else:
        print(f"    ❌ No circle detected!")

    if ellipse:
        (ecx, ecy), (ma, MA), angle = ellipse
        print(f"    Ellipse: axes=({ma:.0f}, {MA:.0f}), angle={angle:.0f}°")

    # ============== KEYPOINTS ==============
    print(f"\n[4] KEYPOINT EXTRACTION")
    keypoints = detection["keypoints"]

    critical = ["center_spot", "halfway_top", "halfway_bottom"]
    found = [k for k in critical if k in keypoints]
    missing = [k for k in critical if k not in keypoints]

    print(f"    Found: {found}")
    if missing:
        print(f"    ❌ MISSING: {missing}")
    else:
        print(f"    ✓ All critical keypoints found!")

    for name in critical:
        if name in keypoints:
            kp = keypoints[name]
            if isinstance(kp, (tuple, list)) and len(kp) >= 2:
                print(f"    - {name}: ({kp[0]:.0f}, {kp[1]:.0f})px")

    if "center_circle_radius" in keypoints:
        print(f"    - center_circle_radius: {keypoints['center_circle_radius']:.0f}px")

    # Check touchline order
    if "halfway_top" in keypoints and "halfway_bottom" in keypoints:
        top_y = keypoints["halfway_top"][1]
        bottom_y = keypoints["halfway_bottom"][1]
        if top_y < bottom_y:
            print(
                f"    ✓ Touchline order correct: top_y={top_y:.0f} < bottom_y={bottom_y:.0f}"
            )
        else:
            print(
                f"    ❌ Touchline order INVERTED: top_y={top_y:.0f} > bottom_y={bottom_y:.0f}"
            )

    # ============== HOMOGRAPHY STATUS ==============
    print(f"\n[5] HOMOGRAPHY CALIBRATION")
    if calibrator.estimator.is_calibrated:
        print(f"    ✓ CALIBRATED")
        print(f"    Error: {calibrator.estimator.calibration_error:.2f}px")
        print(f"    Inliers: {calibrator.estimator.num_inliers}")

        if calibrator.estimator.calibration_error < 0.1:
            print(f"    ⚠️ Warning: Error near 0 suggests overfitting!")

        # Test projection
        print(f"\n[6] PROJECTION TEST")

        # Project pitch center
        pitch_center = np.array([[52.5, 34.0]])
        img_center = calibrator.estimator.pitch_to_image(pitch_center)[0]
        print(
            f"    Pitch center (52.5, 34) → Image ({img_center[0]:.0f}, {img_center[1]:.0f})"
        )

        if circle:
            dist = np.sqrt(
                (img_center[0] - circle[0]) ** 2 + (img_center[1] - circle[1]) ** 2
            )
            status = "✓" if dist < 30 else "⚠️"
            print(f"    {status} Distance to detected circle center: {dist:.0f}px")

        # Test corners
        corners = [
            (w / 2, 0, "Top-center"),
            (w / 2, h, "Bottom-center"),
        ]

        print(f"\n    Image → Pitch projection:")
        for x, y, name in corners:
            pitch_pt = calibrator.estimator.image_to_pitch(np.array([[x, y]]))[0]
            in_bounds = 0 <= pitch_pt[0] <= 105 and 0 <= pitch_pt[1] <= 68
            status = "✓" if in_bounds else "❌"
            print(
                f"    {status} {name:15s} ({x:4.0f}, {y:4.0f}) → ({pitch_pt[0]:6.1f}, {pitch_pt[1]:6.1f})m"
            )

        # Check orientation
        top_pitch = calibrator.estimator.image_to_pitch(np.array([[w / 2, 0]]))[0]
        bottom_pitch = calibrator.estimator.image_to_pitch(np.array([[w / 2, h]]))[0]

        print(f"\n[7] ORIENTATION CHECK")
        print(f"    Top of image → pitch Y = {top_pitch[1]:.1f}m")
        print(f"    Bottom of image → pitch Y = {bottom_pitch[1]:.1f}m")

        if top_pitch[1] > bottom_pitch[1]:
            print(f"    ✓ CORRECT: Top of image = far touchline (larger Y)")
        else:
            print(f"    ❌ INVERTED: Top of image = near touchline (smaller Y)")
            print(f"       This will cause players to appear in wrong positions!")
    else:
        print(f"    ❌ NOT CALIBRATED")
        print(f"    Reason: Not enough keypoints or inliers")

    print(f"\n{'='*70}\n")


def create_diagnostic_visualization(frame, detection, calibrator):
    """Create a detailed diagnostic visualization overlay."""
    h, w = frame.shape[:2]
    vis = frame.copy()

    # Draw field mask boundary (blue)
    mask = detection["mask_field"]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 100, 0), 2)

    # Draw all detected lines
    for line in detection["lines"]:
        if line.is_vertical():
            color = (0, 255, 0)  # Green = vertical
            thickness = 3
        elif line.is_horizontal():
            color = (0, 0, 255)  # Red = horizontal
            thickness = 3
        else:
            color = (128, 128, 128)  # Gray = other
            thickness = 1
        cv2.line(
            vis,
            (int(line.x1), int(line.y1)),
            (int(line.x2), int(line.y2)),
            color,
            thickness,
        )

    # Draw circle (yellow)
    circle = detection.get("circle")
    if circle:
        cx, cy, r = circle
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 255), 3)
        cv2.circle(vis, (int(cx), int(cy)), 8, (0, 255, 255), -1)
        cv2.putText(
            vis,
            "CENTER",
            (int(cx) + 15, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    # Draw ellipse (cyan dashed effect)
    ellipse = detection.get("ellipse")
    if ellipse:
        cv2.ellipse(vis, ellipse, (255, 255, 0), 2)

    # Draw keypoints (magenta)
    keypoints = detection["keypoints"]
    for name, value in keypoints.items():
        if name in ("center_circle_radius", "ellipse"):
            continue
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            x, y = int(value[0]), int(value[1])
            cv2.circle(vis, (x, y), 12, (255, 0, 255), -1)
            cv2.putText(
                vis,
                name,
                (x + 15, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )

    # Draw projected pitch template (cyan) if calibrated
    if calibrator.estimator.is_calibrated:
        pitch = calibrator.pitch
        for start, end in pitch.get_all_lines():
            try:
                img_pts = calibrator.estimator.pitch_to_image(np.array([start, end]))
                pt1 = tuple(img_pts[0].astype(int))
                pt2 = tuple(img_pts[1].astype(int))
                cv2.line(vis, pt1, pt2, (0, 200, 200), 2, cv2.LINE_AA)
            except:
                pass

        # Draw center circle projection
        try:
            circle_pts = pitch.get_center_circle_points(32)
            img_circle_pts = calibrator.estimator.pitch_to_image(circle_pts).astype(int)
            for i in range(len(img_circle_pts)):
                pt1 = tuple(img_circle_pts[i])
                pt2 = tuple(img_circle_pts[(i + 1) % len(img_circle_pts)])
                cv2.line(vis, pt1, pt2, (0, 200, 200), 2, cv2.LINE_AA)
        except:
            pass

    # Add legend
    legend_x = 10
    legend_y = h - 150
    cv2.rectangle(vis, (legend_x, legend_y), (legend_x + 220, h - 10), (0, 0, 0), -1)
    cv2.rectangle(
        vis, (legend_x, legend_y), (legend_x + 220, h - 10), (255, 255, 255), 1
    )

    legend_y += 20
    cv2.putText(
        vis,
        "LEGEND:",
        (legend_x + 5, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    legend_y += 20
    cv2.line(
        vis, (legend_x + 5, legend_y - 5), (legend_x + 25, legend_y - 5), (0, 255, 0), 2
    )
    cv2.putText(
        vis,
        "Vertical lines",
        (legend_x + 30, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1,
    )
    legend_y += 18
    cv2.line(
        vis, (legend_x + 5, legend_y - 5), (legend_x + 25, legend_y - 5), (0, 0, 255), 2
    )
    cv2.putText(
        vis,
        "Horizontal lines",
        (legend_x + 30, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 255),
        1,
    )
    legend_y += 18
    cv2.circle(vis, (legend_x + 15, legend_y - 5), 6, (0, 255, 255), -1)
    cv2.putText(
        vis,
        "Center circle",
        (legend_x + 30, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
    )
    legend_y += 18
    cv2.circle(vis, (legend_x + 15, legend_y - 5), 6, (255, 0, 255), -1)
    cv2.putText(
        vis,
        "Keypoints",
        (legend_x + 30, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 0, 255),
        1,
    )
    legend_y += 18
    cv2.line(
        vis,
        (legend_x + 5, legend_y - 5),
        (legend_x + 25, legend_y - 5),
        (0, 200, 200),
        2,
    )
    cv2.putText(
        vis,
        "Projected pitch",
        (legend_x + 30, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 200, 200),
        1,
    )

    return vis


def main():
    # ----------------------------
    # Config
    # ----------------------------
    VIDEO_PATH = "D:/ITE/year5/graduation project/5th Dataset/Full Match Tactical Cam UCL UEFA Champions League  24-25  2nd Leg - Liverpool vs PSG (11 Mar 2025).mp4"
    MODEL_PATH = "models/finetuned/weights/best.pt"
    MODEL_ONNX = "models/finetuned/weights/best.onnx"
    MODEL_ENGINE = "models/finetuned/weights/best.engine"

    TRACKER_NAME = "bytetrack"

    # ----------------------------
    # Check video
    # ----------------------------
    if not Path(VIDEO_PATH).exists():
        print("Video does not exist")
        sys.exit(-1)
    print("Video exists")

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
        half = False
        backend = "onnx"

    print(f"Backend selected: {backend}")

    # ----------------------------
    # Global optimizations
    # ----------------------------
    if device != "cpu":
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True

    # ----------------------------
    # Load model
    # ----------------------------
    def load_model():
        if backend == "cuda":
            if Path(MODEL_ENGINE).exists():
                print("Using TensorRT engine")
                return YOLO(MODEL_ENGINE)
            else:
                print("Exporting TensorRT engine")
                model = YOLO(MODEL_PATH)
                path = model.export(format="engine", device=device, dynamic=True)
                return YOLO(path)
        else:
            if Path(MODEL_ONNX).exists():
                print("Using ONNX Runtime")
                return YOLO(MODEL_ONNX)
            else:
                print("Exporting ONNX")
                model = YOLO(MODEL_PATH)
                path = model.export(format="onnx", simplify=True, dynamic=True)
                return YOLO(path)

    model = load_model()
    imgsz = 640
    tracker = Tracker(model, TRACKER_NAME, imgsz, device, half)

    # Warmup
    if device != "cpu":
        dummy = torch.zeros(1, 3, 640, 640).to(device)
        model(dummy, device=device, half=half)

    # ----------------------------
    # Initialize Homography Calibrator
    # ----------------------------
    homography_calibrator = AutoHomographyCalibrator(
        calibration_interval=30,
    )

    # ----------------------------
    # Initialize Tactical Display
    # ----------------------------
    tactical_display = TacticalDisplay(
        video_width=854,
        video_height=480,
        pitch_width=425,
        pitch_height=340,
    )

    # ----------------------------
    # Video setup
    # ----------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    frame_id = 0
    paused = False
    step_frame = False
    show_detailed_debug = True
    show_diagnostic_overlay = True
    show_tactical_view = True
    show_trails = True
    last_diagnostic_frame = 0

    print("\n" + "=" * 50)
    print("CONTROLS")
    print("=" * 50)
    print("d - Toggle detailed debug output (console)")
    print("o - Toggle diagnostic overlay (visual)")
    print("t - Toggle tactical 2D view")
    print("r - Toggle trails")
    print("c - Force recalibration")
    print("p - Pause/resume")
    print("s - Step one frame (when paused)")
    print("q - Quit")
    print("=" * 50 + "\n")

    # ----------------------------
    # Main loop
    # ----------------------------
    while cap.isOpened():
        if not paused or step_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            step_frame = False

        start = time.time()

        # 1. Run tracking
        tracks = tracker.track_frame(frame)

        # 2. Update homography calibration
        homography_result = homography_calibrator.process_frame(frame)
        detection = (
            homography_result.get("detection") or homography_calibrator.last_detection
        )

        # 3. Print diagnostic report on first frame or when requested
        if (
            detection
            and show_detailed_debug
            and (frame_id == 1 or frame_id - last_diagnostic_frame >= 60)
        ):
            print_diagnostic_report(detection, homography_calibrator, frame.shape)
            last_diagnostic_frame = frame_id

        # 4. Project player positions
        pitch_positions = {}
        ball_pitch_pos = None

        if homography_calibrator.estimator.is_calibrated:
            for player_id, player_data in tracks["players"].items():
                bbox = player_data["bbox"]
                pitch_pos = homography_calibrator.project_bbox_to_pitch(tuple(bbox))
                if pitch_pos:
                    x, y = pitch_pos
                    if 0 <= x <= 105 and 0 <= y <= 68:
                        pitch_positions[f"player_{player_id}"] = pitch_pos

            for gk_id, gk_data in tracks["goalkeepers"].items():
                bbox = gk_data["bbox"]
                pitch_pos = homography_calibrator.project_bbox_to_pitch(tuple(bbox))
                if pitch_pos:
                    x, y = pitch_pos
                    if 0 <= x <= 105 and 0 <= y <= 68:
                        pitch_positions[f"goalkeeper_{gk_id}"] = pitch_pos

            for ref_id, ref_data in tracks["referees"].items():
                bbox = ref_data["bbox"]
                pitch_pos = homography_calibrator.project_bbox_to_pitch(tuple(bbox))
                if pitch_pos:
                    x, y = pitch_pos
                    if 0 <= x <= 105 and 0 <= y <= 68:
                        pitch_positions[f"referee_{ref_id}"] = pitch_pos

            for ball_id, ball_data in tracks["ball"].items():
                bbox = ball_data["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                pitch_pos = homography_calibrator.project_to_pitch((cx, cy))
                if pitch_pos:
                    x, y = pitch_pos
                    if 0 <= x <= 105 and 0 <= y <= 68:
                        ball_pitch_pos = pitch_pos

        # 5. Draw annotations
        annotated = tracker.draw_annotations(frame, tracks)

        # 6. Draw diagnostic overlay
        if show_diagnostic_overlay and detection:
            annotated = create_diagnostic_visualization(
                annotated, detection, homography_calibrator
            )
        elif show_diagnostic_overlay:
            annotated = homography_calibrator.draw_debug(annotated)

        # 7. Draw ball position
        if ball_pitch_pos:
            cv2.putText(
                annotated,
                f"Ball: ({ball_pitch_pos[0]:.1f}m, {ball_pitch_pos[1]:.1f}m)",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        # Compute FPS
        loop_fps = 1 / (time.time() - start + 1e-8)

        # 8. Create final display
        if show_tactical_view and homography_calibrator.estimator.is_calibrated:
            display_frame = tactical_display.render(
                video_frame=annotated,
                player_positions=pitch_positions,
                ball_position=ball_pitch_pos,
                team_assignments=None,
                show_trails=show_trails,
            )
        else:
            display_frame = cv2.resize(annotated, (854, 480))

        # Draw status bar
        cv2.rectangle(display_frame, (0, 0), (400, 100), (0, 0, 0), -1)

        cv2.putText(
            display_frame,
            f"FPS: {loop_fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display_frame,
            f"Frame: {frame_id}/{total_frames}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # Calibration status
        if homography_calibrator.estimator.is_calibrated:
            err = homography_calibrator.estimator.calibration_error
            inliers = homography_calibrator.estimator.num_inliers
            status_text = f"CALIBRATED | err={err:.1f}px | inliers={inliers}"
            status_color = (0, 255, 0) if err < 10 else (0, 255, 255)
        else:
            status_text = "NOT CALIBRATED"
            status_color = (0, 0, 255)

        cv2.putText(
            display_frame,
            status_text,
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
        )

        # Keypoint status
        if detection:
            kps = detection["keypoints"]
            kp_names = [
                k for k in kps.keys() if k not in ("ellipse", "center_circle_radius")
            ]
            cv2.putText(
                display_frame,
                f"Keypoints: {kp_names}",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )

        # Paused indicator
        if paused:
            cv2.putText(
                display_frame,
                "PAUSED",
                (display_frame.shape[1] - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # Controls
        help_text = (
            "[d]ebug [o]verlay [t]actical [r]trails [c]alibrate [p]ause [s]tep [q]uit"
        )
        cv2.putText(
            display_frame,
            help_text,
            (10, display_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
        )

        # Show frame
        cv2.imshow("Homography Diagnostic", display_frame)

        # Handle keyboard input
        key = cv2.waitKey(1 if not paused else 50) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == ord("s"):
            step_frame = True
        elif key == ord("d"):
            show_detailed_debug = not show_detailed_debug
            print(f"Detailed debug: {show_detailed_debug}")
            if show_detailed_debug and detection:
                print_diagnostic_report(detection, homography_calibrator, frame.shape)
                last_diagnostic_frame = frame_id
        elif key == ord("o"):
            show_diagnostic_overlay = not show_diagnostic_overlay
            print(f"Diagnostic overlay: {show_diagnostic_overlay}")
        elif key == ord("t"):
            show_tactical_view = not show_tactical_view
            print(f"Tactical view: {show_tactical_view}")
        elif key == ord("r"):
            show_trails = not show_trails
            print(f"Trails: {show_trails}")
        elif key == ord("c"):
            print("\n--- FORCING RECALIBRATION ---")
            homography_result = homography_calibrator.process_frame(
                frame, force_calibrate=True
            )
            detection = homography_result.get("detection")
            if detection:
                print_diagnostic_report(detection, homography_calibrator, frame.shape)
                last_diagnostic_frame = frame_id

    cap.release()
    cv2.destroyAllWindows()

    # ----------------------------
    # Final summary
    # ----------------------------
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"Frames processed: {frame_id}")
    print(f"Calibration status: {homography_calibrator.estimator.is_calibrated}")
    if homography_calibrator.estimator.is_calibrated:
        print(f"Final error: {homography_calibrator.estimator.calibration_error:.2f}px")
        print(f"Final inliers: {homography_calibrator.estimator.num_inliers}")


if __name__ == "__main__":
    main()
