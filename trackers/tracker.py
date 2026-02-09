from trackers import BallKalman
from utils import (
    get_bbox_center,
    get_bbox_width,
    xyxy_to_xywh,
    xywh_to_xyxy,
    save_results_txt,
    save_results_json,
)
from boxmot import (
    BotSort,
    BoostTrack,
    StrongSort,
    DeepOcSort,
    ByteTrack,
    HybridSort,
    OcSort,
)
import numpy as np
import pickle
import cv2


class Tracker:
    def __init__(self, model, tracker="bytetrack", imgsz=640, device="cpu", half=True):
        self.model = model
        self.imgsz = imgsz

        # Device Setup (GPU or CPU)
        self.device = device
        self.half = half

        reid_weights = "trackers/reid_weights/osnet_x0_25_msmt17.pt"

        # Initialize the chosen tracker
        if tracker == "botsort":  # Tested: Too slow, too low fps
            self.tracker = BotSort(
                reid_weights=reid_weights, device=self.device, half=half
            )
        elif tracker == "boosttrack":  # Tested: Good, not bad
            self.tracker = BoostTrack(device=self.device, half=half)
        elif tracker == "strongsort":  # Tested: Too slow, too low fps
            self.tracker = StrongSort(
                reid_weights=reid_weights, device=self.device, half=half
            )
        elif tracker == "deepocsort":  # Tested: Good, slow
            self.tracker = DeepOcSort(
                reid_weights=reid_weights, device=self.device, half=half
            )
        elif tracker == "bytetrack":  # Tested: Good, Fast
            self.tracker = ByteTrack(
                device=self.device,
                half=half,
                track_thresh=0.5,
                track_buffer=90,
                match_thresh=0.8,
            )
        elif tracker == "hybridsort":  # Tested: Best acc, Slow
            self.tracker = HybridSort(
                reid_weights=reid_weights, device=self.device, half=half
            )
        elif tracker == "ocsort":  # Tested: Good, Fastest
            self.tracker = OcSort(device=self.device, half=half)
        else:
            # Fallback
            self.tracker = ByteTrack(device=self.device, half=half)

        # Ball Tracker
        self.ball_kalman = BallKalman()

        # id limiting
        self.MAX_HUMANS = 30

        self.CLASS_RULES = {
            "goalkeeper": {
                "max_missing": 300,
                "max_dist": 100,
            },
            "player": {
                "max_missing": 10,
                "max_dist": 150,
            },
            "referee": {
                "max_missing": 150,
                "max_dist": 150,
            },
        }

        self.humans = {}  # sid -> memory
        self.trackerid_to_sid = {}
        self.free_sids = list(range(1, self.MAX_HUMANS + 1))

        self.frame_idx = 0

    def detect_frame(self, frame):
        results = self.model(
            source=frame,
            device=self.device,
            imgsz=self.imgsz,
            half=self.half,
            verbose=False,
            conf=0.01,  # keep >= confidence
        )

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()  # shape (N, 4)
        scores = r.boxes.conf.cpu().numpy()  # shape (N,)
        classes = r.boxes.cls.cpu().numpy().astype(int)  # shape (N,)

        # class_names = np.array([self.model.names[c] for c in classes])

        # conf_thresholds = {
        #     "ball": 0.05,
        #     "goalkeeper": 0.25,
        #     "player": 0.60,
        #     "referee": 0.20,
        #     "default": 0.50,
        # }

        # thresholds = np.array(
        #     [
        #         conf_thresholds.get(name, conf_thresholds["default"])
        #         for name in class_names
        #     ]
        # )

        class_thresholds = np.array(
            [0.20, 0.30, 0.80, 0.45], dtype=float
        )  # ball, goalkeeper, player, referee
        default_threshold = 0.25

        thresholds = np.array(
            [
                class_thresholds[c] if c < len(class_thresholds) else default_threshold
                for c in classes
            ]
        )

        # Threshold using a boolean mask
        mask = scores >= thresholds

        # Filter arrays
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_classes = classes[mask]

        # Get class names
        names = self.model.names

        # # Save to TXT
        # save_results_txt(
        #     filtered_boxes, filtered_scores, filtered_classes, names, append=True
        # )

        # # Save to JSON
        # save_results_json(
        #     filtered_boxes, filtered_scores, filtered_classes, names, append=True
        # )

        # return results[0]

        return filtered_boxes, filtered_scores, filtered_classes, names

    def track_frame(self, frame, stub_path=None):
        self.frame_idx += 1

        # Expire humans
        expired_sids = set()
        for sid, mem in self.humans.items():
            cls = mem["cls"]  # current class
            max_missing = self.CLASS_RULES.get(cls, {}).get("max_missing", 150)

            if mem["active"] and self.frame_idx - mem["last_seen"] > max_missing:
                mem["active"] = False
                expired_sids.add(sid)
                self.free_sids.append(sid)

        # Clean tracker
        # → stable ID mapping
        self.trackerid_to_sid = {
            tid: sid
            for tid, sid in self.trackerid_to_sid.items()
            if sid not in expired_sids
        }

        # 1. Run your existing filtered detection logic
        boxes, scores, classes, cls_names = self.detect_frame(frame)

        tracks = {"ball": {}, "goalkeepers": {}, "players": {}, "referees": {}}

        if len(boxes) == 0:
            return tracks

        cls_names_inv = {v: k for k, v in cls_names.items()}

        # 2. Format detections for the tracker
        # -> Single numpy array [x1, y1, x2, y2, conf, class]
        detections = np.column_stack((boxes, scores, classes))

        # 3. Update tracker
        # Returns: [x1, y1, x2, y2, track_id, conf, class, ind]
        if len(detections) > 0:
            detection_with_tracks = self.tracker.update(detections, frame)
        else:
            detection_with_tracks = np.empty((0, 7), frame)

        # Goalkeepers & Players & Referees
        for det in detection_with_tracks:
            # Correct Unpacking for Trackers
            x1, y1, x2, y2, track_id, conf, cls_id = det[0:7]
            bbox = [x1, y1, x2, y2]
            track_id = int(track_id)

            cls_name = cls_names[cls_id]

            if cls_name == "ball":
                continue

            # stable_id = self.assign_stable_id(track_id, bbox, cls_name)
            # sure_cls = self.update_class(stable_id, cls_name)
            # cls_id = cls_names_inv[sure_cls]

            if cls_id == cls_names_inv["player"]:
                tracks["players"][track_id] = {"bbox": bbox, "conf": conf}
            elif cls_id == cls_names_inv["referee"]:
                tracks["referees"][track_id] = {"bbox": bbox, "conf": conf}
            elif cls_id == cls_names_inv["goalkeeper"]:
                tracks["goalkeepers"][track_id] = {"bbox": bbox, "conf": conf}
            else:
                pass

        # Ball (no tracker id)
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det

            if cls_id == cls_names_inv["ball"]:
                tracks["ball"][1] = {"bbox": [x1, y1, x2, y2], "conf": conf}

        # # ---------------- BALL TRACKING ----------------
        # self.last_ball_w = getattr(self, "last_ball_w", 10)
        # self.last_ball_h = getattr(self, "last_ball_h", 10)

        # ball_dets = []
        # for det in detections:
        #     x1, y1, x2, y2, conf, cls_id = det
        #     if cls_id == cls_names_inv["ball"]:
        #         cx, cy = get_bbox_center((x1, y1, x2, y2))
        #         ball_dets.append((cx, cy, [x1, y1, x2, y2], conf))

        # # highest confidence
        # if len(ball_dets) > 0:
        #     cx, cy, bbox, conf = max(ball_dets, key=lambda x: x[3])
        #     self.ball_kalman.update(cx, cy, conf)
        #     self.last_ball_w = bbox[2] - bbox[0]
        #     self.last_ball_h = bbox[3] - bbox[1]
        #     tracks["ball"][1] = {"bbox": bbox, "conf": conf}
        # else:
        #     # No detection → predict
        #     px, py = self.ball_kalman.predict()

        #     # Reconstruct bbox using last known size
        #     half_w = self.last_ball_w / 2
        #     half_h = self.last_ball_h / 2
        #     tracks["ball"][1] = {
        #         "bbox": [px - half_w, py - half_h, px + half_w, py + half_h],
        #         "conf": None,  # This is a prediction
        #     }

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_bbox_center(bbox)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ]
        )
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, frame, tracks):
        frame = frame.copy()

        ball_dict = tracks["ball"]
        goalkeeper_dict = tracks["goalkeepers"]
        player_dict = tracks["players"]
        referee_dict = tracks["referees"]

        # Draw Ball
        for track_id, ball in ball_dict.items():
            bbox = ball["bbox"]
            conf = ball.get("conf", None)

            frame = self.draw_traingle(frame, bbox, (0, 128, 255))

            if conf is not None:
                x1, _, _, y2 = map(int, bbox)
                self.draw_conf_text(frame, f"{conf:.2f}", x1, y2, (0, 128, 255))

        # Draw Goalkeepers
        for track_id, goalkeeper in goalkeeper_dict.items():
            bbox = goalkeeper["bbox"]
            conf = goalkeeper.get("conf", None)

            frame = self.draw_ellipse(frame, bbox, (255, 0, 255), track_id)

            if conf is not None:
                x1, _, _, y2 = map(int, bbox)
                self.draw_conf_text(frame, f"{conf:.2f}", x1, y2, (255, 0, 255))

        # Draw Players
        for track_id, player in player_dict.items():
            bbox = player["bbox"]
            conf = player.get("conf", None)
            team = player.get("team", None)
            team_color = player.get("team_color", (255, 255, 255))

            frame = self.draw_ellipse(frame, bbox, team_color, track_id)

            if conf is not None:
                x1, _, _, y2 = map(int, bbox)
                self.draw_conf_text(frame, f"{conf:.2f}", x1, y2, team_color)

            if team is not None:
                x1, _, _, y2 = map(int, bbox)
                self.draw_team_text(frame, f"t: {team}", x1, y2, team_color)

        # Draw Referees
        for track_id, referee in referee_dict.items():
            bbox = referee["bbox"]
            conf = referee.get("conf", None)

            frame = self.draw_ellipse(frame, bbox, (0, 255, 255), track_id)

            if conf is not None:
                x1, _, _, y2 = map(int, bbox)
                self.draw_conf_text(frame, f"{conf:.2f}", x1, y2, (0, 255, 255))

        return frame

    def draw_conf_text(self, frame, text, x, y, color):
        cv2.putText(
            frame,
            text,
            (x, y + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,  # small font
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_team_text(self, frame, text, x, y, color):
        cv2.putText(
            frame,
            text,
            (x, y + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,  # small font
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_detections_custom(self, frame, boxes, scores, classes, names):
        frame = frame.copy()

        for box, conf, cls_id in zip(boxes, scores, classes):
            class_name = names[cls_id]

            if class_name == "ball":
                color = (0, 128, 255)
                frame = self.draw_traingle(frame, box, color)
            elif class_name == "goalkeeper":
                color = (255, 0, 255)
                frame = self.draw_ellipse(frame, box, color)
            elif class_name == "player":
                color = (255, 0, 0)
                frame = self.draw_ellipse(frame, box, color)
            elif class_name == "referee":
                color = (0, 255, 255)
                frame = self.draw_ellipse(frame, box, color)
            else:
                color = (255, 255, 255)
                continue

            # Draw confidence under shape
            x1, _, _, y2 = map(int, box)
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return frame

    def assign_stable_id(self, tracker_id, bbox, cls_name):
        center = np.array(get_bbox_center(bbox))

        # 1️⃣ Tracker ID reuse protection
        if tracker_id in self.trackerid_to_sid:
            sid = self.trackerid_to_sid[tracker_id]
            mem = self.humans.get(sid)

            if mem:
                dist = np.linalg.norm(center - mem["center"])
                max_dist = self.CLASS_RULES.get(mem["cls"], {}).get("max_dist", 150)
                max_missing = self.CLASS_RULES.get(mem["cls"], {}).get(
                    "max_missing", 150
                )

                if (
                    self.frame_idx - mem["last_seen"] < max_missing * 0.5
                    and dist < max_dist * 0.5
                ):
                    mem.update(
                        {
                            "center": center,
                            "last_seen": self.frame_idx,
                            "cls": cls_name,
                            "active": True,
                        }
                    )
                    return sid
                else:
                    del self.trackerid_to_sid[tracker_id]

        # 2️⃣ Match inactive humans
        best_sid = None
        best_score = float("inf")

        for sid, mem in self.humans.items():
            if mem["active"]:
                continue

            cls = mem["cls"]  # current class
            max_dist = self.CLASS_RULES.get(cls, {}).get("max_dist", 150)
            dist = np.linalg.norm(center - mem["center"])
            if dist < max_dist:
                if dist < best_score:
                    best_score = dist
                    best_sid = sid

        if best_sid is not None:
            self.trackerid_to_sid[tracker_id] = best_sid
            self.humans[best_sid].update(
                {
                    "center": center,
                    "last_seen": self.frame_idx,
                    "cls": cls_name,
                    "active": True,
                }
            )
            return best_sid

        # 3️⃣ Allocate new ID
        if self.free_sids:
            sid = self.free_sids.pop(0)
            self.trackerid_to_sid[tracker_id] = sid
            self.humans[sid] = {
                "center": center,
                "last_seen": self.frame_idx,
                "cls": cls_name,
                "active": True,
                "class_votes": {
                    "player": 0,
                    "goalkeeper": 0,
                    "referee": 0,
                },
            }

            return sid

        # 4️⃣ Absolute fallback (should almost never happen)
        sid = min(
            self.humans, key=lambda s: np.linalg.norm(center - self.humans[s]["center"])
        )
        self.trackerid_to_sid[tracker_id] = sid
        self.humans[sid]["center"] = center
        self.humans[sid]["last_seen"] = self.frame_idx
        return sid

    def update_class(self, sid, yolo_cls):
        votes = self.humans[sid]["class_votes"]

        if yolo_cls not in votes:
            return self.humans[sid]["cls"]

        # decay
        for k in votes:
            votes[k] *= 0.98

        votes[yolo_cls] += 1.0

        final_cls = max(votes, key=votes.get)
        self.humans[sid]["cls"] = final_cls
        return final_cls
