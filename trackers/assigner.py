from collections import defaultdict, deque
import numpy as np
import cv2


class Assigner:
    def __init__(self):
        self.is_initialized = False  # Only once

        # Final learned team colors (HSV, only H+S)
        self.team_colors = {}

        # Temporary storage during warmup
        self.collected_colors = []

        # Track → recent team history
        self.player_team_history = defaultdict(lambda: deque(maxlen=10))

    # --------------------------------------------------
    # Jersey color extraction (K-MEANS ON PIXELS, HSV)
    # Only H + S, ignore V completely
    # --------------------------------------------------
    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Top half only (shirt region)
        crop = crop[: crop.shape[0] // 2]

        # Resize aggressively for speed
        crop = cv2.resize(crop, (16, 16))

        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Flatten pixels
        pixels = hsv.reshape((-1, 3)).astype(np.float32)

        # KMeans clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels,
            2,  # clusters
            None,  # Optional starting labels
            criteria,  # (type, max_iter, epsilon)
            1,  # attempts (fast)
            cv2.KMEANS_PP_CENTERS,
        )

        labels = labels.flatten()

        # Map cluster to jersey using corners
        h, w = hsv.shape[:2]
        clustered = labels.reshape(h, w)

        # Grass cluster
        corner_labels = [
            clustered[0, 0],
            clustered[0, -1],
            clustered[-1, 0],
            clustered[-1, -1],
        ]

        background = max(set(corner_labels), key=corner_labels.count)
        jersey_cluster = 1 - background

        # Return only H + S
        return centers[jersey_cluster][:2]

    # --------------------------------------------------
    # Collect colors during warmup
    # --------------------------------------------------
    def collect_team_colors(self, frame, players):
        for player in players.values():
            color = self.get_player_color(frame, player["bbox"])

            if color is not None:
                self.collected_colors.append(color)

    # --------------------------------------------------
    # Finalize team colors once (H + S only)
    # --------------------------------------------------
    def finalize_team_colors(self):
        if len(self.collected_colors) < 10:
            print("⚠️ Not enough samples to initialize teams")
            return

        data = np.array(self.collected_colors, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
        _, labels, centers = cv2.kmeans(
            data,
            2,  # clusters
            None,  # Optional starting labels
            criteria,  # (type, max_iter, epsilon)
            3,  # attempts (slower)
            cv2.KMEANS_PP_CENTERS,
        )

        # Sort by Hue to lock team IDs
        order = np.argsort(centers[:, 0])
        self.team_colors[1] = centers[order[0]]  # (H,S)
        self.team_colors[2] = centers[order[1]]  # (H,S)

        self.is_initialized = True
        self.collected_colors.clear()
        print("✅ Team colors locked (HSV H+S)")

    # --------------------------------------------------
    # Compute circular hue distance
    # --------------------------------------------------
    def color_cosine_similarity(self, c1, c2):
        v1 = np.array(c1)
        v2 = np.array(c2)
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return np.dot(v1, v2)

    # --------------------------------------------------
    # Stable per-player assignment
    # --------------------------------------------------
    def get_player_team_stable(self, frame, bbox, track_id):
        if not self.is_initialized:
            return None

        color = self.get_player_color(frame, bbox)
        if color is None:
            return None

        sim1 = self.color_cosine_similarity(color, self.team_colors[1])
        sim2 = self.color_cosine_similarity(color, self.team_colors[2])
        team = 1 if sim1 > sim2 else 2

        # Temporal smoothing
        history = self.player_team_history[track_id]
        history.append(team)

        stable_team = max(set(history), key=history.count)
        return stable_team

    # --------------------------------------------------
    # Visualization helpers (assign fixed V for display)
    # --------------------------------------------------
    def hsv_to_bgr(self, hs, v=200):
        hsv = np.uint8([[[hs[0], hs[1], v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(x) for x in bgr)
