"""
================================================================================
SOCCER FIELD HOMOGRAPHY PIPELINE — GEOMETRY-CONSTRAINED (v2)
================================================================================

A production-grade pipeline for computing homography from tactical camera views.

FEATURES:
    - Detects lines: touchlines, center line, goal lines, penalty areas
    - Detects ellipses: center circle, penalty arcs
    - Uses spatial relationships to classify features
    - Computes homography using FIFA standard dimensions
    - Works with partial field views (tactical cameras)

GEOMETRIC CONSTRAINTS (prior knowledge):
    1. Touchlines span the full visible frame width (clipped by goal lines)
    2. The center line connects the far and near touchlines
    3. A penalty arc is only valid if its penalty-area front line is present
    4. The center circle has the center line as a diameter
    5. The frame is always a partial view of the pitch

v2 IMPROVEMENTS:
    - Perspective-aware center circle rendering (depth-dependent thickness)
    - Algebraic ellipse–line intersection for center circle keypoints
    - Hard field boundary constraints (reject lines outside touchlines/goal lines)
    - Explicit penalty-area 3-side labeling (Front / Far-Side / Near-Side)

COORDINATE SYSTEM:
    World: Origin at center spot, X along length, Y along width (meters)
    Image: Standard pixel coordinates (0,0) at top-left
================================================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum, auto
from collections import defaultdict
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings("ignore")


# ==============================================================================
# SECTION 1: FIELD DIMENSIONS (FIFA STANDARD)
# ==============================================================================


@dataclass(frozen=True)
class FIFADimensions:
    """FIFA International Standard Field Dimensions (meters). Origin at CENTER SPOT."""

    LENGTH: float = 105.0
    WIDTH: float = 68.0
    PENALTY_AREA_LENGTH: float = 16.5
    PENALTY_AREA_WIDTH: float = 40.32
    GOAL_AREA_LENGTH: float = 5.5
    GOAL_AREA_WIDTH: float = 18.32
    PENALTY_SPOT_DISTANCE: float = 11.0
    CENTER_CIRCLE_RADIUS: float = 9.15
    PENALTY_ARC_RADIUS: float = 9.15
    CORNER_ARC_RADIUS: float = 1.0
    GOAL_WIDTH: float = 7.32
    GOAL_HEIGHT: float = 2.44

    @property
    def half_length(self) -> float:
        return self.LENGTH / 2

    @property
    def half_width(self) -> float:
        return self.WIDTH / 2

    @property
    def penalty_area_half_width(self) -> float:
        return self.PENALTY_AREA_WIDTH / 2

    @property
    def goal_area_half_width(self) -> float:
        return self.GOAL_AREA_WIDTH / 2


FIFA = FIFADimensions()


# ==============================================================================
# SECTION 2: KEYPOINT DEFINITIONS
# ==============================================================================


class KeypointID(Enum):
    """All detectable field keypoints with semantic meaning."""

    CORNER_FAR_LEFT = auto()
    CORNER_FAR_RIGHT = auto()
    CORNER_NEAR_LEFT = auto()
    CORNER_NEAR_RIGHT = auto()

    CENTER_FAR = auto()
    CENTER_NEAR = auto()
    CENTER_SPOT = auto()

    # Center circle ∩ center line (diameter endpoints)
    CENTER_CIRCLE_FAR = auto()
    CENTER_CIRCLE_NEAR = auto()

    L_PENALTY_FAR_OUTER = auto()
    L_PENALTY_NEAR_OUTER = auto()
    L_PENALTY_FAR_INNER = auto()
    L_PENALTY_NEAR_INNER = auto()

    L_GOAL_AREA_FAR_OUTER = auto()
    L_GOAL_AREA_NEAR_OUTER = auto()
    L_GOAL_AREA_FAR_INNER = auto()
    L_GOAL_AREA_NEAR_INNER = auto()

    L_PENALTY_SPOT = auto()
    L_PENALTY_ARC_FAR = auto()
    L_PENALTY_ARC_NEAR = auto()

    R_PENALTY_FAR_OUTER = auto()
    R_PENALTY_NEAR_OUTER = auto()
    R_PENALTY_FAR_INNER = auto()
    R_PENALTY_NEAR_INNER = auto()

    R_GOAL_AREA_FAR_OUTER = auto()
    R_GOAL_AREA_NEAR_OUTER = auto()
    R_GOAL_AREA_FAR_INNER = auto()
    R_GOAL_AREA_NEAR_INNER = auto()

    R_PENALTY_SPOT = auto()
    R_PENALTY_ARC_FAR = auto()
    R_PENALTY_ARC_NEAR = auto()


def get_world_coordinates(kp_id: KeypointID) -> Tuple[float, float]:
    """
    Get world coordinates (meters) for a keypoint.
    Origin: Center spot (0, 0). X: left goal to right goal. Y: far to near touchline.
    """
    HL = FIFA.half_length
    HW = FIFA.half_width
    PA_L = FIFA.PENALTY_AREA_LENGTH
    PA_HW = FIFA.penalty_area_half_width
    GA_L = FIFA.GOAL_AREA_LENGTH
    GA_HW = FIFA.goal_area_half_width
    PS_D = FIFA.PENALTY_SPOT_DISTANCE
    ARC_R = FIFA.PENALTY_ARC_RADIUS
    CCR = FIFA.CENTER_CIRCLE_RADIUS

    arc_dx = PA_L - PS_D
    arc_dy = np.sqrt(ARC_R**2 - arc_dx**2) if ARC_R > arc_dx else 0

    coords = {
        KeypointID.CORNER_FAR_LEFT: (-HL, -HW),
        KeypointID.CORNER_FAR_RIGHT: (HL, -HW),
        KeypointID.CORNER_NEAR_LEFT: (-HL, HW),
        KeypointID.CORNER_NEAR_RIGHT: (HL, HW),
        KeypointID.CENTER_FAR: (0, -HW),
        KeypointID.CENTER_NEAR: (0, HW),
        KeypointID.CENTER_SPOT: (0, 0),
        KeypointID.CENTER_CIRCLE_FAR: (0, -CCR),
        KeypointID.CENTER_CIRCLE_NEAR: (0, CCR),
        KeypointID.L_PENALTY_FAR_OUTER: (-HL, -PA_HW),
        KeypointID.L_PENALTY_NEAR_OUTER: (-HL, PA_HW),
        KeypointID.L_PENALTY_FAR_INNER: (-HL + PA_L, -PA_HW),
        KeypointID.L_PENALTY_NEAR_INNER: (-HL + PA_L, PA_HW),
        KeypointID.L_GOAL_AREA_FAR_OUTER: (-HL, -GA_HW),
        KeypointID.L_GOAL_AREA_NEAR_OUTER: (-HL, GA_HW),
        KeypointID.L_GOAL_AREA_FAR_INNER: (-HL + GA_L, -GA_HW),
        KeypointID.L_GOAL_AREA_NEAR_INNER: (-HL + GA_L, GA_HW),
        KeypointID.L_PENALTY_SPOT: (-HL + PS_D, 0),
        KeypointID.L_PENALTY_ARC_FAR: (-HL + PA_L, -arc_dy),
        KeypointID.L_PENALTY_ARC_NEAR: (-HL + PA_L, arc_dy),
        KeypointID.R_PENALTY_FAR_OUTER: (HL, -PA_HW),
        KeypointID.R_PENALTY_NEAR_OUTER: (HL, PA_HW),
        KeypointID.R_PENALTY_FAR_INNER: (HL - PA_L, -PA_HW),
        KeypointID.R_PENALTY_NEAR_INNER: (HL - PA_L, PA_HW),
        KeypointID.R_GOAL_AREA_FAR_OUTER: (HL, -GA_HW),
        KeypointID.R_GOAL_AREA_NEAR_OUTER: (HL, GA_HW),
        KeypointID.R_GOAL_AREA_FAR_INNER: (HL - GA_L, -GA_HW),
        KeypointID.R_GOAL_AREA_NEAR_INNER: (HL - GA_L, GA_HW),
        KeypointID.R_PENALTY_SPOT: (HL - PS_D, 0),
        KeypointID.R_PENALTY_ARC_FAR: (HL - PA_L, -arc_dy),
        KeypointID.R_PENALTY_ARC_NEAR: (HL - PA_L, arc_dy),
    }

    return coords.get(kp_id, (0, 0))


# ==============================================================================
# SECTION 3: DATA STRUCTURES
# ==============================================================================


class LineType(Enum):
    FAR_TOUCHLINE = auto()
    NEAR_TOUCHLINE = auto()
    CENTER_LINE = auto()
    LEFT_GOAL_LINE = auto()
    RIGHT_GOAL_LINE = auto()

    # Penalty area: explicit 3-side labelling
    #   "FRONT" = the line parallel to (and away from) the goal line
    #   "FAR"   = the side closer to the far touchline (top of image)
    #   "NEAR"  = the side closer to the near touchline (bottom of image)
    L_PENALTY_FAR = auto()  # Left penalty box — far side
    L_PENALTY_NEAR = auto()  # Left penalty box — near side
    L_PENALTY_FRONT = auto()  # Left penalty box — front (parallel to goal line)

    R_PENALTY_FAR = auto()  # Right penalty box — far side
    R_PENALTY_NEAR = auto()  # Right penalty box — near side
    R_PENALTY_FRONT = auto()  # Right penalty box — front (parallel to goal line)

    L_GOAL_AREA_FAR = auto()
    L_GOAL_AREA_NEAR = auto()
    L_GOAL_AREA_FRONT = auto()
    R_GOAL_AREA_FAR = auto()
    R_GOAL_AREA_NEAR = auto()
    R_GOAL_AREA_FRONT = auto()

    UNKNOWN = auto()


# ---------- Friendly aliases for readability ----------
# These are NOT new enum members — they are just module-level references so
# that downstream code can write e.g. ``LineType.L_PENALTY_BOX_TOP`` and
# have it resolve to the correct member.
LineType.L_PENALTY_BOX_TOP = LineType.L_PENALTY_FRONT
LineType.L_PENALTY_BOX_LEFT = LineType.L_PENALTY_FAR
LineType.L_PENALTY_BOX_RIGHT = LineType.L_PENALTY_NEAR
LineType.R_PENALTY_BOX_TOP = LineType.R_PENALTY_FRONT
LineType.R_PENALTY_BOX_LEFT = LineType.R_PENALTY_FAR
LineType.R_PENALTY_BOX_RIGHT = LineType.R_PENALTY_NEAR


# Descriptive mapping used by visualization and logging
PENALTY_SIDE_DESCRIPTIONS: Dict[LineType, str] = {
    LineType.L_PENALTY_FRONT: "L Penalty Box — Front (top, parallel to goal line)",
    LineType.L_PENALTY_FAR: "L Penalty Box — Far Side (connects goal line → front, far/top)",
    LineType.L_PENALTY_NEAR: "L Penalty Box — Near Side (connects goal line → front, near/bot)",
    LineType.R_PENALTY_FRONT: "R Penalty Box — Front (top, parallel to goal line)",
    LineType.R_PENALTY_FAR: "R Penalty Box — Far Side (connects goal line → front, far/top)",
    LineType.R_PENALTY_NEAR: "R Penalty Box — Near Side (connects goal line → front, near/bot)",
}


class EllipseType(Enum):
    CENTER_CIRCLE = auto()
    LEFT_PENALTY_ARC = auto()
    RIGHT_PENALTY_ARC = auto()
    CORNER_ARC = auto()
    UNKNOWN = auto()


@dataclass
class DetectedLine:
    x1: int
    y1: int
    x2: int
    y2: int
    line_type: LineType = LineType.UNKNOWN
    confidence: float = 0.0

    @property
    def length(self) -> float:
        return np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    @property
    def angle(self) -> float:
        return np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1)) % 180

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def endpoints(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_homogeneous(self) -> np.ndarray:
        p1 = np.array([self.x1, self.y1, 1.0])
        p2 = np.array([self.x2, self.y2, 1.0])
        line = np.cross(p1, p2)
        norm = np.sqrt(line[0] ** 2 + line[1] ** 2)
        return line / norm if norm > 1e-10 else line


@dataclass
class DetectedEllipse:
    center_x: float
    center_y: float
    axis_major: float
    axis_minor: float
    angle: float
    ellipse_type: EllipseType = EllipseType.UNKNOWN
    confidence: float = 0.0
    contour: Optional[np.ndarray] = None

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)

    @property
    def aspect_ratio(self) -> float:
        return self.axis_minor / self.axis_major if self.axis_major > 0 else 0


@dataclass
class DetectedKeypoint:
    image_x: float
    image_y: float
    keypoint_id: KeypointID
    confidence: float = 1.0
    source: str = ""

    @property
    def image_point(self) -> Tuple[float, float]:
        return (self.image_x, self.image_y)

    @property
    def world_point(self) -> Tuple[float, float]:
        return get_world_coordinates(self.keypoint_id)


@dataclass
class HomographyResult:
    lines: List[DetectedLine] = field(default_factory=list)
    ellipses: List[DetectedEllipse] = field(default_factory=list)
    keypoints: List[DetectedKeypoint] = field(default_factory=list)
    grass_mask: Optional[np.ndarray] = None
    white_mask: Optional[np.ndarray] = None
    skeleton: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None
    homography_inv: Optional[np.ndarray] = None
    inlier_count: int = 0
    reprojection_error: float = float("inf")
    median_reprojection_error: float = float("inf")

    @property
    def is_valid(self) -> bool:
        return self.homography is not None and self.inlier_count >= 4


# ==============================================================================
# SECTION 4: IMAGE PREPROCESSING
# ==============================================================================


class Preprocessor:
    def __init__(
        self,
        hue_range: Tuple[int, int] = (35, 85),
        sat_range: Tuple[int, int] = (40, 255),
        val_range: Tuple[int, int] = (40, 255),
    ):
        self.hue_range = hue_range
        self.sat_range = sat_range
        self.val_range = val_range

    def normalize_illumination(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def segment_grass(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([self.hue_range[0], self.sat_range[0], self.val_range[0]])
        upper = np.array([self.hue_range[1], self.sat_range[1], self.val_range[1]])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def get_grass_bounds(self, grass_mask: np.ndarray) -> Dict[str, int]:
        rows = np.any(grass_mask > 0, axis=1)
        cols = np.any(grass_mask > 0, axis=0)
        h, w = grass_mask.shape
        return {
            "top": int(np.argmax(rows)) if np.any(rows) else 0,
            "bottom": int(h - np.argmax(rows[::-1])) if np.any(rows) else h,
            "left": int(np.argmax(cols)) if np.any(cols) else 0,
            "right": int(w - np.argmax(cols[::-1])) if np.any(cols) else w,
        }


# ==============================================================================
# SECTION 5: WHITE LINE EXTRACTION
# ==============================================================================


class WhiteLineExtractor:
    def __init__(
        self,
        sat_thresh: int = 70,
        adaptive_k: float = 1.2,
        adaptive_k_edge: float = 0.8,
        edge_band: float = 0.25,
        block_size: int = 101,
        min_val: int = 100,
        h_kernel: int = 20,
        v_kernel: int = 20,
        min_area: int = 50,
        close_iters: int = 2,
    ):
        self.sat_thresh = sat_thresh
        self.adaptive_k = adaptive_k
        self.adaptive_k_edge = adaptive_k_edge
        self.edge_band = edge_band
        self.block_size = block_size
        self.min_val = min_val
        self.h_kernel = h_kernel
        self.v_kernel = v_kernel
        self.min_area = min_area
        self.close_iters = close_iters

    def extract(
        self, img: np.ndarray, grass_mask: np.ndarray, grass_bounds: Dict[str, int]
    ) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)

        low_sat = s < self.sat_thresh
        v_float = v.astype(np.float32)
        m = (grass_mask > 0).astype(np.float32)

        k = (self.block_size, self.block_size)
        m_sum = cv2.boxFilter(m, ddepth=-1, ksize=k, normalize=False)

        eps = 1e-6
        inv_m_sum = 1.0 / np.maximum(m_sum, eps)

        v_m_sum = cv2.boxFilter(v_float * m, ddepth=-1, ksize=k, normalize=False)
        v2_m_sum = cv2.boxFilter(
            (v_float * v_float) * m, ddepth=-1, ksize=k, normalize=False
        )

        local_mean = v_m_sum * inv_m_sum
        local_sq_mean = v2_m_sum * inv_m_sum
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))

        h, w = v.shape
        g_top = grass_bounds["top"]
        g_bottom = grass_bounds["bottom"]
        g_height = max(g_bottom - g_top, 1)

        y = np.arange(h, dtype=np.float32)
        y_rel = np.clip((y - g_top) / g_height, 0.0, 1.0)

        k_row = np.full((h,), self.adaptive_k, dtype=np.float32)
        edge = (y_rel < self.edge_band) | (y_rel > (1.0 - self.edge_band))
        k_row[edge] = self.adaptive_k_edge
        k_map = k_row[:, None]

        high_val = (v_float > (local_mean + k_map * local_std)) & (v > self.min_val)

        white_mask = (low_sat & high_val).astype(np.uint8) * 255
        white_mask = cv2.bitwise_and(white_mask, grass_mask)
        return white_mask

    def bridge_multiscale(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        out = mask.copy()
        for scale, ksz in [(1.0, 7), (0.5, 5)]:
            if scale != 1.0:
                m = cv2.resize(
                    mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )
            else:
                m = mask.copy()
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
            if scale != 1.0:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            out = cv2.bitwise_or(out, m)
        return out

    def clean(self, white_mask: np.ndarray) -> np.ndarray:
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (self.h_kernel, 1))
        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.v_kernel))

        closed_h = white_mask.copy()
        closed_v = white_mask.copy()
        for _ in range(self.close_iters):
            closed_h = cv2.morphologyEx(closed_h, cv2.MORPH_CLOSE, k_h)
            closed_v = cv2.morphologyEx(closed_v, cv2.MORPH_CLOSE, k_v)

        combined = cv2.bitwise_or(closed_h, closed_v)

        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_small, iterations=1)
        combined = self.bridge_multiscale(combined)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
        cleaned = np.zeros_like(combined)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                cleaned[labels == i] = 255

        return cleaned

    def skeletonize(
        self, mask: np.ndarray, scales: List[float] = [1.0, 0.5]
    ) -> np.ndarray:
        h, w = mask.shape
        result = np.zeros((h, w), dtype=np.uint8)
        for scale in scales:
            if scale == 1.0:
                scaled = mask
            else:
                scaled = cv2.resize(
                    mask,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            skel = (skeletonize(scaled > 0) * 255).astype(np.uint8)
            if scale != 1.0:
                skel = cv2.resize(skel, (w, h), interpolation=cv2.INTER_NEAREST)
            result = cv2.bitwise_or(result, skel)
        return result


# ==============================================================================
# SECTION 6: LINE DETECTION & CLASSIFICATION
# ==============================================================================


class LineDetector:
    """Base line detector with Hough transform, merging, and classification."""

    def __init__(
        self,
        hough_threshold: int = 30,
        min_length: int = 30,
        max_gap: int = 20,
        angle_h_thresh: float = 25,
        angle_v_thresh: float = 25,
        merge_angle_thresh: float = 10,
        merge_dist_thresh: float = 30,
        merge_gap_thresh: float = 150,
    ):
        self.hough_threshold = hough_threshold
        self.min_length = min_length
        self.max_gap = max_gap
        self.angle_h_thresh = angle_h_thresh
        self.angle_v_thresh = angle_v_thresh
        self.merge_angle_thresh = merge_angle_thresh
        self.merge_dist_thresh = merge_dist_thresh
        self.merge_gap_thresh = merge_gap_thresh

    def detect(self, skeleton: np.ndarray) -> List[DetectedLine]:
        edges = cv2.Canny(skeleton, 50, 150)
        hough_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_length,
            maxLineGap=self.max_gap,
        )
        if hough_lines is None:
            return []
        return [DetectedLine(x1, y1, x2, y2) for x1, y1, x2, y2 in hough_lines[:, 0]]

    def merge_collinear(self, lines: List[DetectedLine]) -> List[DetectedLine]:
        if len(lines) < 2:
            return lines

        def perpendicular_dist(px, py, x1, y1, x2, y2):
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length < 1e-6:
                return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
            return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length

        def are_collinear(l1: DetectedLine, l2: DetectedLine) -> bool:
            angle_diff = abs(l1.angle - l2.angle)
            angle_diff = min(angle_diff, 180 - angle_diff)
            if angle_diff > self.merge_angle_thresh:
                return False
            m1, m2 = l1.midpoint, l2.midpoint
            d1 = perpendicular_dist(m1[0], m1[1], l2.x1, l2.y1, l2.x2, l2.y2)
            d2 = perpendicular_dist(m2[0], m2[1], l1.x1, l1.y1, l1.x2, l1.y2)
            return d1 < self.merge_dist_thresh and d2 < self.merge_dist_thresh

        def gap_between(l1: DetectedLine, l2: DetectedLine) -> float:
            pts1 = [(l1.x1, l1.y1), (l1.x2, l1.y2)]
            pts2 = [(l2.x1, l2.y1), (l2.x2, l2.y2)]
            return min(
                np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                for p1 in pts1
                for p2 in pts2
            )

        n = len(lines)
        used = [False] * n
        groups = []
        for i in range(n):
            if used[i]:
                continue
            group = [i]
            used[i] = True
            for j in range(i + 1, n):
                if used[j]:
                    continue
                for k in group:
                    if are_collinear(lines[k], lines[j]):
                        if gap_between(lines[k], lines[j]) < self.merge_gap_thresh:
                            group.append(j)
                            used[j] = True
                            break
            groups.append(group)

        merged = []
        for group in groups:
            if len(group) == 1:
                merged.append(lines[group[0]])
            else:
                all_pts = []
                for idx in group:
                    l = lines[idx]
                    all_pts.extend([(l.x1, l.y1), (l.x2, l.y2)])
                pts = np.array(all_pts)
                dists = cdist(pts, pts)
                i, j = np.unravel_index(np.argmax(dists), dists.shape)
                merged.append(
                    DetectedLine(
                        int(pts[i, 0]), int(pts[i, 1]), int(pts[j, 0]), int(pts[j, 1])
                    )
                )
        return merged

    def classify(
        self,
        lines: List[DetectedLine],
        grass_bounds: Dict[str, int],
        ellipses: List[DetectedEllipse],
    ) -> List[DetectedLine]:
        g_top = grass_bounds["top"]
        g_bottom = grass_bounds["bottom"]
        g_left = grass_bounds["left"]
        g_right = grass_bounds["right"]
        g_height = max(g_bottom - g_top, 1)
        g_width = max(g_right - g_left, 1)

        center_circle = None
        left_penalty_arc = None
        right_penalty_arc = None
        for e in ellipses:
            if e.ellipse_type == EllipseType.CENTER_CIRCLE:
                center_circle = e
            elif e.ellipse_type == EllipseType.LEFT_PENALTY_ARC:
                left_penalty_arc = e
            elif e.ellipse_type == EllipseType.RIGHT_PENALTY_ARC:
                right_penalty_arc = e

        classified = []
        for line in lines:
            angle = line.angle
            mid_x, mid_y = line.midpoint
            x_rel = (mid_x - g_left) / g_width
            y_rel = (mid_y - g_top) / g_height

            is_horizontal = (
                angle < self.angle_h_thresh or angle > 180 - self.angle_h_thresh
            )
            is_vertical = abs(angle - 90) < self.angle_v_thresh

            line_type = LineType.UNKNOWN
            confidence = 0.5

            if is_horizontal:
                if center_circle is not None:
                    cc_y = center_circle.center_y
                    if abs(mid_y - cc_y) < center_circle.axis_major * 0.5:
                        if 0.3 < x_rel < 0.7:
                            line_type = LineType.CENTER_LINE
                            confidence = 0.9

                if line_type == LineType.UNKNOWN:
                    if y_rel < 0.2:
                        line_type = LineType.FAR_TOUCHLINE
                        confidence = 0.8
                    elif y_rel > 0.8:
                        line_type = LineType.NEAR_TOUCHLINE
                        confidence = 0.8
                    elif 0.4 < y_rel < 0.6:
                        line_type = LineType.CENTER_LINE
                        confidence = 0.6
                    else:
                        if x_rel < 0.35:
                            line_type = (
                                LineType.L_PENALTY_FAR
                                if y_rel < 0.5
                                else LineType.L_PENALTY_NEAR
                            )
                            confidence = 0.7
                        elif x_rel > 0.65:
                            line_type = (
                                LineType.R_PENALTY_FAR
                                if y_rel < 0.5
                                else LineType.R_PENALTY_NEAR
                            )
                            confidence = 0.7

            elif is_vertical:
                near_left_arc = (
                    left_penalty_arc is not None
                    and abs(mid_x - left_penalty_arc.center_x)
                    < left_penalty_arc.axis_major * 1.5
                )
                near_right_arc = (
                    right_penalty_arc is not None
                    and abs(mid_x - right_penalty_arc.center_x)
                    < right_penalty_arc.axis_major * 1.5
                )

                if x_rel < 0.15:
                    line_type = LineType.LEFT_GOAL_LINE
                    confidence = 0.85
                elif x_rel > 0.85:
                    line_type = LineType.RIGHT_GOAL_LINE
                    confidence = 0.85
                elif 0.35 < x_rel < 0.65:
                    height_ratio = line.length / g_height
                    near_center_circle = (
                        center_circle is not None
                        and abs(mid_x - center_circle.center_x)
                        < center_circle.axis_major * 1.5
                    )
                    if height_ratio > 0.4 or near_center_circle:
                        line_type = LineType.CENTER_LINE
                        confidence = 0.85
                    else:
                        line_type = LineType.CENTER_LINE
                        confidence = 0.6
                elif near_left_arc or (0.15 < x_rel < 0.35):
                    line_type = LineType.L_PENALTY_FRONT
                    confidence = 0.75
                elif near_right_arc or (0.65 < x_rel < 0.85):
                    line_type = LineType.R_PENALTY_FRONT
                    confidence = 0.75

            else:
                if y_rel < 0.25:
                    line_type = LineType.FAR_TOUCHLINE
                    confidence = 0.5
                elif y_rel > 0.75:
                    line_type = LineType.NEAR_TOUCHLINE
                    confidence = 0.5
                elif x_rel < 0.3:
                    line_type = (
                        LineType.L_PENALTY_FAR
                        if y_rel < 0.5
                        else LineType.L_PENALTY_NEAR
                    )
                    confidence = 0.5
                elif x_rel > 0.7:
                    line_type = (
                        LineType.R_PENALTY_FAR
                        if y_rel < 0.5
                        else LineType.R_PENALTY_NEAR
                    )
                    confidence = 0.5

            classified.append(
                DetectedLine(
                    line.x1,
                    line.y1,
                    line.x2,
                    line.y2,
                    line_type=line_type,
                    confidence=confidence,
                )
            )
        return classified


class EnhancedLineDetector(LineDetector):
    """Keeps only the best candidate for each line type."""

    def __init__(
        self,
        hough_threshold: int = 30,
        min_length: int = 30,
        max_gap: int = 20,
        angle_h_thresh: float = 25,
        angle_v_thresh: float = 25,
        merge_angle_thresh: float = 10,
        merge_dist_thresh: float = 30,
        merge_gap_thresh: float = 150,
        min_confidence_for_selection: float = 0.3,
        min_touchline_length_ratio: float = 0.3,
    ):
        super().__init__(
            hough_threshold,
            min_length,
            max_gap,
            angle_h_thresh,
            angle_v_thresh,
            merge_angle_thresh,
            merge_dist_thresh,
            merge_gap_thresh,
        )
        self.min_confidence_for_selection = min_confidence_for_selection
        self.min_touchline_length_ratio = min_touchline_length_ratio

    def select_best_by_type(
        self, lines: List[DetectedLine], img_width: int
    ) -> List[DetectedLine]:
        lines_by_type = defaultdict(list)
        for line in lines:
            if line.line_type != LineType.UNKNOWN:
                lines_by_type[line.line_type].append(line)

        selected_lines = []

        w_touchline = {"len": 0.60, "conf": 0.20, "ang": 0.20, "target": 0}
        w_goal_line = {"len": 0.50, "conf": 0.30, "ang": 0.20, "target": 90}
        w_center_line = {"len": 0.65, "conf": 0.15, "ang": 0.20, "target": 90}
        w_penalty = {"len": 0.40, "conf": 0.40, "ang": 0.20, "target": 90}

        def calculate_score(line: DetectedLine, weights: dict) -> float:
            norm_len = min(line.length / img_width, 1.0)
            angle = line.angle
            target = weights["target"]
            if target == 0:
                diff = min(angle, 180 - angle)
            else:
                diff = abs(angle - 90)
            ang_score = max(0, 1.0 - (diff / 45.0))
            score = (
                (weights["len"] * norm_len)
                + (weights["conf"] * line.confidence)
                + (weights["ang"] * ang_score)
            )
            return score

        for line_type, type_lines in lines_by_type.items():
            valid_lines = [
                l
                for l in type_lines
                if l.confidence >= self.min_confidence_for_selection
            ]
            if not valid_lines:
                continue

            if line_type in (LineType.FAR_TOUCHLINE, LineType.NEAR_TOUCHLINE):
                params = w_touchline
            elif line_type in (LineType.LEFT_GOAL_LINE, LineType.RIGHT_GOAL_LINE):
                params = w_goal_line
            elif line_type == LineType.CENTER_LINE:
                params = w_center_line
            elif line_type in (LineType.L_PENALTY_FRONT, LineType.R_PENALTY_FRONT):
                params = w_penalty
            elif line_type in (
                LineType.L_PENALTY_FAR,
                LineType.L_PENALTY_NEAR,
                LineType.R_PENALTY_FAR,
                LineType.R_PENALTY_NEAR,
            ):
                params = {"len": 0.4, "conf": 0.4, "ang": 0.2, "target": 0}
            else:
                params = {"len": 0.5, "conf": 0.5, "ang": 0.0, "target": 0}

            best_line = max(valid_lines, key=lambda l: calculate_score(l, params))

            if line_type in (LineType.FAR_TOUCHLINE, LineType.NEAR_TOUCHLINE):
                if best_line.length / img_width < self.min_touchline_length_ratio:
                    continue

            selected_lines.append(best_line)

        return selected_lines

    def filter_overlapping_lines(
        self, lines: List[DetectedLine], angle_threshold: float = 15
    ) -> List[DetectedLine]:
        if len(lines) <= 1:
            return lines
        lines_sorted = sorted(lines, key=lambda l: l.confidence, reverse=True)
        filtered = []
        for current_line in lines_sorted:
            is_redundant = False
            for kept_line in filtered:
                if current_line.line_type == kept_line.line_type:
                    angle_diff = abs(current_line.angle - kept_line.angle)
                    angle_diff = min(angle_diff, 180 - angle_diff)
                    if angle_diff < angle_threshold:
                        mid1 = current_line.midpoint
                        mid2 = kept_line.midpoint
                        dist = np.sqrt(
                            (mid1[0] - mid2[0]) ** 2 + (mid1[1] - mid2[1]) ** 2
                        )
                        if dist < max(current_line.length, kept_line.length) * 0.5:
                            is_redundant = True
                            break
            if not is_redundant:
                filtered.append(current_line)
        return filtered

    def enforce_field_geometry_constraints(
        self, lines: List[DetectedLine], img_width: int, img_height: int
    ) -> List[DetectedLine]:
        lines_by_type = defaultdict(list)
        for line in lines:
            lines_by_type[line.line_type].append(line)

        final_lines = []
        for line_type, type_lines in lines_by_type.items():
            if not type_lines:
                continue
            if line_type == LineType.CENTER_LINE:
                best = max(
                    type_lines,
                    key=lambda l: (
                        l.confidence,
                        -abs(l.angle) if l.angle < 90 else abs(l.angle - 180),
                        l.length,
                    ),
                )
                final_lines.append(best)
            elif line_type in (LineType.FAR_TOUCHLINE, LineType.NEAR_TOUCHLINE):
                valid = [l for l in type_lines if l.length / img_width > 0.25]
                if valid:
                    best = max(valid, key=lambda l: (l.confidence, l.length))
                    final_lines.append(best)
            elif line_type in (LineType.LEFT_GOAL_LINE, LineType.RIGHT_GOAL_LINE):
                valid = [l for l in type_lines if abs(l.angle - 90) < 30]
                if valid:
                    best = max(
                        valid,
                        key=lambda l: (l.confidence, l.length, abs(l.angle - 90)),
                    )
                    final_lines.append(best)
            else:
                best = max(type_lines, key=lambda l: l.confidence)
                final_lines.append(best)

        # Ensure far touchline is above near touchline
        far = [l for l in final_lines if l.line_type == LineType.FAR_TOUCHLINE]
        near = [l for l in final_lines if l.line_type == LineType.NEAR_TOUCHLINE]
        if far and near and far[0].midpoint[1] > near[0].midpoint[1]:
            for line in final_lines:
                if line.line_type == LineType.FAR_TOUCHLINE:
                    line.line_type = LineType.NEAR_TOUCHLINE
                elif line.line_type == LineType.NEAR_TOUCHLINE:
                    line.line_type = LineType.FAR_TOUCHLINE

        return final_lines

    def classify(
        self,
        lines: List[DetectedLine],
        grass_bounds: Dict[str, int],
        ellipses: List[DetectedEllipse],
        img_shape: Tuple[int, int] = None,
    ) -> List[DetectedLine]:
        classified = super().classify(lines, grass_bounds, ellipses)
        if img_shape is None:
            img_height = grass_bounds["bottom"] - grass_bounds["top"]
            img_width = grass_bounds["right"] - grass_bounds["left"]
        else:
            img_height, img_width = img_shape[:2]

        best_lines = self.select_best_by_type(classified, img_width)
        filtered_lines = self.filter_overlapping_lines(best_lines)
        final_lines = self.enforce_field_geometry_constraints(
            filtered_lines, img_width, img_height
        )
        return final_lines


# ==============================================================================
# SECTION 7: ELLIPSE DETECTION & CLASSIFICATION
# ==============================================================================


class EllipseDetector:
    """
    Line-guided ellipse detector.

    Uses detected lines (center line, touchlines) to:
        1. Predict WHERE the center circle should be (on the center line)
        2. Predict HOW BIG it should be (from touchline separation → scale)
        3. Search ONLY in the predicted region
        4. Enforce symmetry around the center line
    """

    def __init__(
        self,
        min_contour_length: int = 80,
        max_contour_length: int = 3000,
        min_aspect_ratio: float = 0.2,
        min_points: int = 5,
        search_margin_factor: float = 2.5,
        line_suppress_thickness: int = 14,
        ransac_inlier_thresh: float = 0.15,
        ransac_iterations: int = 3,
    ):
        self.min_contour_length = min_contour_length
        self.max_contour_length = max_contour_length
        self.min_aspect_ratio = min_aspect_ratio
        self.min_points = min_points
        self.search_margin_factor = search_margin_factor
        self.line_suppress_thickness = line_suppress_thickness
        self.ransac_inlier_thresh = ransac_inlier_thresh
        self.ransac_iterations = ransac_iterations

    def detect(
        self,
        white_mask: np.ndarray,
        lines: List[DetectedLine],
        grass_mask: np.ndarray,
        grass_bounds: Dict[str, int],
        img_shape: Tuple[int, int],
    ) -> List[DetectedEllipse]:
        center_line = None
        touchlines = []
        for l in lines:
            if l.line_type == LineType.CENTER_LINE:
                center_line = l
            elif l.line_type in (LineType.FAR_TOUCHLINE, LineType.NEAR_TOUCHLINE):
                touchlines.append(l)

        if center_line is None:
            return []

        expected_radius_px = self._estimate_circle_radius(
            center_line, touchlines, grass_bounds
        )

        suppressed = self._suppress_lines(white_mask, lines)
        suppressed = cv2.bitwise_and(suppressed, grass_mask)

        search_r = expected_radius_px * self.search_margin_factor
        cl_mx, cl_my = center_line.midpoint
        h, w = img_shape[:2]

        cl_angle = center_line.angle
        cl_is_vertical = abs(cl_angle - 90) < 35

        if cl_is_vertical:
            x1 = max(0, int(cl_mx - search_r))
            x2 = min(w, int(cl_mx + search_r))
            y1 = max(grass_bounds["top"], int(cl_my - search_r * 1.3))
            y2 = min(grass_bounds["bottom"], int(cl_my + search_r * 1.3))
        else:
            x1 = max(grass_bounds["left"], int(cl_mx - search_r * 1.3))
            x2 = min(grass_bounds["right"], int(cl_mx + search_r * 1.3))
            y1 = max(0, int(cl_my - search_r))
            y2 = min(h, int(cl_my + search_r))

        roi = suppressed[y1:y2, x1:x2]
        if roi.size == 0 or np.sum(roi > 0) < 20:
            return []

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        candidate_pts = []
        for c in contours:
            if len(c) < 8:
                continue
            arc = cv2.arcLength(c, True)
            if arc < 25:
                continue
            c_global = c.copy()
            c_global[:, :, 0] += x1
            c_global[:, :, 1] += y1
            candidate_pts.append(c_global.reshape(-1, 2))

        if not candidate_pts:
            return []

        all_pts = np.vstack(candidate_pts).astype(np.float32)
        if len(all_pts) < self.min_points:
            return []

        ellipse = self._fit_ellipse_symmetric(all_pts, center_line, expected_radius_px)

        if ellipse is None:
            return []

        return [ellipse]

    def _estimate_circle_radius(
        self,
        center_line: DetectedLine,
        touchlines: List[DetectedLine],
        grass_bounds: Dict[str, int],
    ) -> float:
        if len(touchlines) >= 2:
            ys = [l.midpoint[1] for l in touchlines]
            touchline_sep = abs(ys[0] - ys[1])
        else:
            touchline_sep = grass_bounds["bottom"] - grass_bounds["top"]

        radius_px = touchline_sep * (FIFA.CENTER_CIRCLE_RADIUS / FIFA.WIDTH)
        return float(np.clip(radius_px, 20, 200))

    def _suppress_lines(
        self, white_mask: np.ndarray, lines: List[DetectedLine]
    ) -> np.ndarray:
        suppressed = white_mask.copy()
        for l in lines:
            cv2.line(
                suppressed,
                (l.x1, l.y1),
                (l.x2, l.y2),
                0,
                self.line_suppress_thickness,
            )
        return suppressed

    def _fit_ellipse_symmetric(
        self,
        pts: np.ndarray,
        center_line: DetectedLine,
        expected_radius: float,
    ) -> Optional[DetectedEllipse]:
        if len(pts) < self.min_points:
            return None

        homo = center_line.to_homogeneous()
        a, b, c = homo

        mirrored = self._mirror_points(pts, a, b, c)
        combined = np.vstack([pts, mirrored]).astype(np.float32)

        if len(combined) < self.min_points:
            return None

        try:
            ell = cv2.fitEllipse(combined)
        except cv2.error:
            return None

        for _ in range(self.ransac_iterations):
            dists = self._point_ellipse_dist(combined, ell)
            inlier_mask = dists < self.ransac_inlier_thresh
            inliers = combined[inlier_mask]
            if len(inliers) < self.min_points:
                break
            try:
                ell = cv2.fitEllipse(inliers)
            except cv2.error:
                break
            combined = inliers

        (ex, ey), (ax1, ax2), angle = ell
        major = max(ax1, ax2) / 2.0
        minor = min(ax1, ax2) / 2.0

        dist_to_line = abs(a * ex + b * ey + c)
        if dist_to_line > major * 0.3:
            return None

        if major < expected_radius * 0.3 or major > expected_radius * 3.0:
            return None

        if minor < 5 or major < 10:
            return None

        ar = minor / major
        if ar < self.min_aspect_ratio:
            return None

        denom = a * a + b * b
        if denom > 1e-10:
            factor = (a * ex + b * ey + c) / denom
            ex = ex - a * factor
            ey = ey - b * factor

        # Build contour from ellipse parameters (dense sampling for
        # accurate intersection computation later)
        t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        cos_a_val = np.cos(np.radians(angle))
        sin_a_val = np.sin(np.radians(angle))
        xs = ex + (ax1 / 2) * np.cos(t) * cos_a_val - (ax2 / 2) * np.sin(t) * sin_a_val
        ys = ey + (ax1 / 2) * np.cos(t) * sin_a_val + (ax2 / 2) * np.sin(t) * cos_a_val
        contour = np.column_stack([xs, ys]).astype(np.float32)

        confidence = min(ar * 1.2, 1.0)

        return DetectedEllipse(
            center_x=float(ex),
            center_y=float(ey),
            axis_major=major,
            axis_minor=minor,
            angle=angle,
            ellipse_type=EllipseType.CENTER_CIRCLE,
            confidence=confidence,
            contour=contour,
        )

    @staticmethod
    def _mirror_points(pts: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        denom = a * a + b * b
        if denom < 1e-10:
            return pts.copy()
        factor = 2.0 * (a * pts[:, 0] + b * pts[:, 1] + c) / denom
        mirrored = pts.copy()
        mirrored[:, 0] = pts[:, 0] - a * factor
        mirrored[:, 1] = pts[:, 1] - b * factor
        return mirrored

    @staticmethod
    def _point_ellipse_dist(pts: np.ndarray, ellipse_params) -> np.ndarray:
        (cx, cy), (w, h), ang = ellipse_params
        cos_a = np.cos(np.radians(ang))
        sin_a = np.sin(np.radians(ang))
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a
        norm_d = np.sqrt((rx / (w / 2 + 0.001)) ** 2 + (ry / (h / 2 + 0.001)) ** 2)
        return np.abs(norm_d - 1.0)

    def classify(
        self,
        ellipses: List[DetectedEllipse],
        grass_bounds: Dict[str, int],
        lines: List[DetectedLine],
    ) -> List[DetectedEllipse]:
        g_left = grass_bounds["left"]
        g_right = grass_bounds["right"]
        g_width = max(g_right - g_left, 1)

        classified = []
        for ellipse in ellipses:
            x_rel = (ellipse.center_x - g_left) / g_width

            if ellipse.ellipse_type == EllipseType.CENTER_CIRCLE:
                if 0.2 < x_rel < 0.8:
                    classified.append(ellipse)
            elif ellipse.ellipse_type == EllipseType.UNKNOWN:
                if x_rel < 0.35:
                    ellipse.ellipse_type = EllipseType.LEFT_PENALTY_ARC
                elif x_rel > 0.65:
                    ellipse.ellipse_type = EllipseType.RIGHT_PENALTY_ARC
                classified.append(ellipse)
            else:
                classified.append(ellipse)

        return classified


# ==============================================================================
# SECTION 8: GEOMETRIC REFINER (PRIOR KNOWLEDGE CONSTRAINTS)
# ==============================================================================


class GeometricRefiner:
    """
    Applies field geometry constraints to refine detected lines and ellipses.

    Constraints enforced:
        1. Touchlines span the full frame width (clipped at goal lines if visible)
        2. The center line must connect far and near touchlines
        3. Penalty arcs require their penalty-area front line to be present
        4. The center circle has the center line as a diameter
        5. The frame is always a partial view (no whole-pitch assumptions)
        6. **NEW** Hard field boundary: no horizontal line outside touchlines,
           no vertical line outside goal lines
        7. **NEW** Penalty area 3-side validation
    """

    def __init__(
        self,
        frame_margin: float = 50.0,
        snap_tolerance: float = 0.3,
    ):
        self.frame_margin = frame_margin
        self.snap_tolerance = snap_tolerance

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def refine(
        self,
        lines: List[DetectedLine],
        ellipses: List[DetectedEllipse],
        img_shape: Tuple[int, ...],
        grass_bounds: Dict[str, int],
        grass_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[DetectedLine], List[DetectedEllipse]]:
        h, w = img_shape[:2]

        # Constraint 1: extend touchlines to frame / goal-line boundaries
        lines = self._extend_touchlines(lines, h, w)

        # Constraint 2: extend center line to meet the two touchlines
        lines = self._extend_center_line(lines, h, w)

        # Constraint 3: reject penalty arcs that lack a penalty front line
        ellipses = self._gate_penalty_arcs(ellipses, lines)

        # Constraint 4: snap center circle onto center line
        ellipses = self._snap_center_circle_to_center_line(ellipses, lines)

        # Constraint 5: GRASS BOUNDARY — nothing exists outside the grass
        if grass_mask is not None:
            lines = self._clip_lines_to_grass(lines, grass_mask)
            ellipses = self._filter_ellipses_by_grass(ellipses, grass_mask)

        # ── NEW Constraint 6: Hard field boundary ────────────────────
        lines = self._reject_lines_outside_field_boundaries(lines, h, w)

        # ── NEW Constraint 7: Penalty area 3-side validation ─────────
        lines = self._validate_penalty_area_sides(lines)

        return lines, ellipses

    # ------------------------------------------------------------------
    # NEW — Constraint 6: Hard field boundary rejection
    # ------------------------------------------------------------------

    @staticmethod
    def _reject_lines_outside_field_boundaries(
        lines: List[DetectedLine],
        img_h: int,
        img_w: int,
        y_margin: float = 15.0,
        x_margin: float = 15.0,
    ) -> List[DetectedLine]:
        """
        Reject false-positive lines that lie outside the established
        pitch boundaries formed by touchlines and goal lines.

        Rules:
            Y-axis (horizontal lines):
              • Reject if BOTH endpoints are above the far touchline.
              • Reject if BOTH endpoints are below the near touchline.
            X-axis (vertical lines):
              • Reject if BOTH endpoints are left of the left goal line.
              • Reject if BOTH endpoints are right of the right goal line.

        Boundary lines themselves are never rejected.  A generous pixel
        margin is used so that small extrapolation errors don't kill
        valid lines.
        """
        # Collect boundary references
        far_tl = None
        near_tl = None
        left_gl = None
        right_gl = None
        for l in lines:
            if l.line_type == LineType.FAR_TOUCHLINE:
                far_tl = l
            elif l.line_type == LineType.NEAR_TOUCHLINE:
                near_tl = l
            elif l.line_type == LineType.LEFT_GOAL_LINE:
                left_gl = l
            elif l.line_type == LineType.RIGHT_GOAL_LINE:
                right_gl = l

        # Compute boundary y/x values (use midpoint as representative)
        far_y = far_tl.midpoint[1] if far_tl else None
        near_y = near_tl.midpoint[1] if near_tl else None
        left_x = left_gl.midpoint[0] if left_gl else None
        right_x = right_gl.midpoint[0] if right_gl else None

        # A line is "horizontal-ish" if its angle is within 35° of 0/180
        def is_horizontal(line: DetectedLine) -> bool:
            a = line.angle
            return a < 35 or a > 145

        def is_vertical(line: DetectedLine) -> bool:
            return abs(line.angle - 90) < 35

        boundary_types = {
            LineType.FAR_TOUCHLINE,
            LineType.NEAR_TOUCHLINE,
            LineType.LEFT_GOAL_LINE,
            LineType.RIGHT_GOAL_LINE,
        }

        kept = []
        for line in lines:
            # Never reject boundary lines themselves
            if line.line_type in boundary_types:
                kept.append(line)
                continue

            rejected = False

            # ── Y-axis rejection (horizontal lines) ──
            if is_horizontal(line):
                both_y = [line.y1, line.y2]
                if far_y is not None:
                    if all(y < far_y - y_margin for y in both_y):
                        rejected = True
                if near_y is not None:
                    if all(y > near_y + y_margin for y in both_y):
                        rejected = True

            # ── X-axis rejection (vertical lines) ──
            if is_vertical(line) and not rejected:
                both_x = [line.x1, line.x2]
                if left_x is not None:
                    if all(x < left_x - x_margin for x in both_x):
                        rejected = True
                if right_x is not None:
                    if all(x > right_x + x_margin for x in both_x):
                        rejected = True

            if not rejected:
                kept.append(line)

        n_removed = len(lines) - len(kept)
        if n_removed > 0:
            print(
                f"      [boundary filter] Rejected {n_removed} line(s) "
                f"outside field boundaries"
            )

        return kept

    # ------------------------------------------------------------------
    # NEW — Constraint 7: Penalty area 3-side validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_penalty_area_sides(
        lines: List[DetectedLine],
    ) -> List[DetectedLine]:
        """
        Validate that penalty-area side lines (FAR / NEAR) are spatially
        consistent with their FRONT line and the corresponding goal line.

        For each side (L/R penalty):
            • The FRONT line must be roughly perpendicular to the two
              SIDES and lie between the SIDES in the far/near direction.
            • A SIDE whose midpoint is farther from the goal line than
              the FRONT is demoted to UNKNOWN (likely a false positive).

        This also prints descriptive labels for the three sides of each
        detected penalty area, clarifying their role.
        """
        lines_by_type = {}
        for l in lines:
            lines_by_type.setdefault(l.line_type, []).append(l)

        # Validate each penalty area (left and right)
        for prefix, goal_type, front_type, far_type, near_type in [
            (
                "L",
                LineType.LEFT_GOAL_LINE,
                LineType.L_PENALTY_FRONT,
                LineType.L_PENALTY_FAR,
                LineType.L_PENALTY_NEAR,
            ),
            (
                "R",
                LineType.RIGHT_GOAL_LINE,
                LineType.R_PENALTY_FRONT,
                LineType.R_PENALTY_FAR,
                LineType.R_PENALTY_NEAR,
            ),
        ]:
            front = lines_by_type.get(front_type, [])
            far_sides = lines_by_type.get(far_type, [])
            near_sides = lines_by_type.get(near_type, [])

            if front:
                desc = PENALTY_SIDE_DESCRIPTIONS.get(front_type, front_type.name)
                print(f"      [penalty-area] {desc}")
            if far_sides:
                desc = PENALTY_SIDE_DESCRIPTIONS.get(far_type, far_type.name)
                print(f"      [penalty-area] {desc}")
            if near_sides:
                desc = PENALTY_SIDE_DESCRIPTIONS.get(near_type, near_type.name)
                print(f"      [penalty-area] {desc}")

            # If we have the front line, validate sides are on the
            # correct side of it (between front and goal line).
            goal_lines = lines_by_type.get(goal_type, [])
            if not front or not goal_lines:
                continue

            front_x = front[0].midpoint[0]
            goal_x = goal_lines[0].midpoint[0]

            # For left penalty: goal_x < front_x.
            # For right penalty: goal_x > front_x.
            # A side line's midpoint x must be between goal_x and front_x.
            lo_x = min(goal_x, front_x)
            hi_x = max(goal_x, front_x)
            margin = abs(hi_x - lo_x) * 0.3  # 30% tolerance

            for side_type in [far_type, near_type]:
                for side_line in lines_by_type.get(side_type, []):
                    sx = side_line.midpoint[0]
                    if sx < lo_x - margin or sx > hi_x + margin:
                        print(
                            f"      [penalty-area] Demoting {side_type.name} "
                            f"(midpoint x={sx:.0f} outside [{lo_x:.0f}, {hi_x:.0f}])"
                        )
                        side_line.line_type = LineType.UNKNOWN
                        side_line.confidence *= 0.3

        return lines

    # ------------------------------------------------------------------
    # Keypoint filters
    # ------------------------------------------------------------------

    def filter_keypoints_by_grass(
        self,
        keypoints: List[DetectedKeypoint],
        grass_mask: np.ndarray,
        margin_px: int = 15,
    ) -> List[DetectedKeypoint]:
        h, w = grass_mask.shape[:2]

        if margin_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1)
            )
            expanded = cv2.dilate(grass_mask, kernel)
        else:
            expanded = grass_mask

        kept = []
        for kp in keypoints:
            ix = int(round(kp.image_x))
            iy = int(round(kp.image_y))
            ix = max(0, min(w - 1, ix))
            iy = max(0, min(h - 1, iy))
            if expanded[iy, ix] > 0:
                kept.append(kp)
        return kept

    # ------------------------------------------------------------------
    # Center circle ∩ center line keypoints  (IMPROVED — algebraic solver)
    # ------------------------------------------------------------------

    def generate_circle_line_keypoints(
        self,
        lines: List[DetectedLine],
        ellipses: List[DetectedEllipse],
    ) -> List[DetectedKeypoint]:
        """
        Produce two keypoints where the center circle crosses the center
        line (diameter endpoints).

        v2 improvement: uses an **algebraic** ellipse–line intersection
        (quadratic in the line parameter) instead of contour sampling, so
        that the result is accurate even for tilted or near-vertical center
        lines and for ellipses with any orientation.

        World coords: (0, −9.15) and (0, +9.15).
        """
        center_line = self._find_line(lines, LineType.CENTER_LINE)
        center_circle = self._find_ellipse(ellipses, EllipseType.CENTER_CIRCLE)
        if center_line is None or center_circle is None:
            return []

        pts = self._algebraic_ellipse_line_intersection(center_circle, center_line)

        if len(pts) < 2:
            # Fallback to contour-based method
            pts = self._contour_ellipse_line_intersection(center_circle, center_line)

        if len(pts) < 2:
            return []

        # Sort: far = smaller y (top of image), near = larger y (bottom)
        pts.sort(key=lambda p: p[1])

        conf = center_circle.confidence * 0.85
        return [
            DetectedKeypoint(
                image_x=pts[0][0],
                image_y=pts[0][1],
                keypoint_id=KeypointID.CENTER_CIRCLE_FAR,
                confidence=conf,
                source="center_circle ∩ center_line [algebraic] (far)",
            ),
            DetectedKeypoint(
                image_x=pts[1][0],
                image_y=pts[1][1],
                keypoint_id=KeypointID.CENTER_CIRCLE_NEAR,
                confidence=conf,
                source="center_circle ∩ center_line [algebraic] (near)",
            ),
        ]

    # ------------------------------------------------------------------
    # Algebraic ellipse–line intersection (NEW)
    # ------------------------------------------------------------------

    @staticmethod
    def _algebraic_ellipse_line_intersection(
        ellipse: DetectedEllipse,
        line: DetectedLine,
    ) -> List[Tuple[float, float]]:
        """
        Analytically intersect a conic (ellipse) with a line.

        The ellipse is parameterised by center (cx, cy), semi-axes a, b
        and rotation angle θ.  The line is given by two endpoints which
        we parameterise as  P(t) = P1 + t·(P2 − P1),  t ∈ ℝ.

        We substitute P(t) into the standard-form ellipse equation
        (after rotating into the ellipse frame) and solve the resulting
        quadratic in t.  This handles arbitrary line orientations
        (vertical, horizontal, tilted) without special-casing.
        """
        cx, cy = ellipse.center_x, ellipse.center_y
        # Semi-axes — DetectedEllipse stores these as *half* the cv2
        # fitEllipse output which uses full axis lengths internally,
        # but our dataclass already stores the half-values in axis_major
        # / axis_minor.
        a = ellipse.axis_major  # semi-major
        b = ellipse.axis_minor  # semi-minor
        theta = np.radians(ellipse.angle)

        if a < 1e-6 or b < 1e-6:
            return []

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Line endpoints
        x1, y1 = float(line.x1), float(line.y1)
        x2, y2 = float(line.x2), float(line.y2)
        dx = x2 - x1
        dy = y2 - y1

        # Translate so ellipse center is at origin
        ox = x1 - cx
        oy = y1 - cy

        # Rotate into ellipse-aligned frame
        ox_r = cos_t * ox + sin_t * oy
        oy_r = -sin_t * ox + cos_t * oy
        dx_r = cos_t * dx + sin_t * dy
        dy_r = -sin_t * dx + cos_t * dy

        # Ellipse equation in aligned frame: (X/a)² + (Y/b)² = 1
        # Substitute X = ox_r + t·dx_r, Y = oy_r + t·dy_r :
        #   (ox_r + t·dx_r)²/a² + (oy_r + t·dy_r)²/b² = 1
        # Expand to  A·t² + B·t + C = 0
        A = (dx_r / a) ** 2 + (dy_r / b) ** 2
        B = 2.0 * ((ox_r * dx_r) / (a * a) + (oy_r * dy_r) / (b * b))
        C = (ox_r / a) ** 2 + (oy_r / b) ** 2 - 1.0

        disc = B * B - 4.0 * A * C
        if disc < -1e-8:
            return []

        disc = max(disc, 0.0)
        sqrt_disc = np.sqrt(disc)

        if abs(A) < 1e-12:
            # Degenerate — line is tangent or misses
            if abs(B) < 1e-12:
                return []
            ts = [-C / B]
        else:
            ts = [(-B + sqrt_disc) / (2.0 * A), (-B - sqrt_disc) / (2.0 * A)]

        results = []
        for t in ts:
            px = x1 + t * dx
            py = y1 + t * dy
            results.append((float(px), float(py)))

        return results

    @staticmethod
    def _contour_ellipse_line_intersection(
        ellipse: DetectedEllipse, line: DetectedLine
    ) -> List[Tuple[float, float]]:
        """Fallback: contour-based zero-crossing method."""
        homo = line.to_homogeneous()
        a, b, c = homo

        if ellipse.contour is not None and len(ellipse.contour) >= 10:
            pts = ellipse.contour.reshape(-1, 2)
        else:
            cx, cy = ellipse.center_x, ellipse.center_y
            ma, mi = ellipse.axis_major, ellipse.axis_minor
            ang = np.radians(ellipse.angle)
            t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
            cos_a, sin_a = np.cos(ang), np.sin(ang)
            xs = cx + ma * np.cos(t) * cos_a - mi * np.sin(t) * sin_a
            ys = cy + ma * np.cos(t) * sin_a + mi * np.sin(t) * cos_a
            pts = np.column_stack([xs, ys])

        dists = a * pts[:, 0] + b * pts[:, 1] + c

        intersections = []
        n = len(dists)
        for i in range(n):
            j = (i + 1) % n
            if dists[i] * dists[j] < 0:
                t_val = abs(dists[i]) / (abs(dists[i]) + abs(dists[j]) + 1e-12)
                x = pts[i, 0] + t_val * (pts[j, 0] - pts[i, 0])
                y = pts[i, 1] + t_val * (pts[j, 1] - pts[i, 1])
                intersections.append((float(x), float(y)))

        return intersections

    # ------------------------------------------------------------------
    # Helpers: find a line / ellipse by type
    # ------------------------------------------------------------------

    @staticmethod
    def _find_line(lines: List[DetectedLine], lt: LineType) -> Optional[DetectedLine]:
        for l in lines:
            if l.line_type == lt:
                return l
        return None

    @staticmethod
    def _find_ellipse(
        ellipses: List[DetectedEllipse], et: EllipseType
    ) -> Optional[DetectedEllipse]:
        for e in ellipses:
            if e.ellipse_type == et:
                return e
        return None

    # ------------------------------------------------------------------
    # Homogeneous geometry utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _line_line_intersection(
        l1: DetectedLine, l2: DetectedLine
    ) -> Optional[Tuple[float, float]]:
        h1 = l1.to_homogeneous()
        h2 = l2.to_homogeneous()
        pt = np.cross(h1, h2)
        if abs(pt[2]) < 1e-10:
            return None
        return (pt[0] / pt[2], pt[1] / pt[2])

    @staticmethod
    def _extend_segment_to_boundaries(
        line: DetectedLine, h: int, w: int, margin: float = 50.0
    ) -> DetectedLine:
        homo = line.to_homogeneous()
        boundaries = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, -float(w)]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, -float(h)]),
        ]

        pts = []
        for boundary in boundaries:
            pt = np.cross(homo, boundary)
            if abs(pt[2]) > 1e-10:
                x, y = pt[0] / pt[2], pt[1] / pt[2]
                if -margin <= x <= w + margin and -margin <= y <= h + margin:
                    pts.append((x, y))

        if len(pts) < 2:
            return line

        pts_arr = np.array(pts)
        dists = cdist(pts_arr, pts_arr)
        i, j = np.unravel_index(np.argmax(dists), dists.shape)

        return DetectedLine(
            int(round(pts_arr[i, 0])),
            int(round(pts_arr[i, 1])),
            int(round(pts_arr[j, 0])),
            int(round(pts_arr[j, 1])),
            line_type=line.line_type,
            confidence=line.confidence,
        )

    @staticmethod
    def _extend_segment_to_line(
        segment: DetectedLine,
        boundary_line: DetectedLine,
        which_end: str,
        h: int,
        w: int,
        margin: float = 50.0,
    ) -> DetectedLine:
        h_seg = segment.to_homogeneous()
        h_bnd = boundary_line.to_homogeneous()
        pt = np.cross(h_seg, h_bnd)

        if abs(pt[2]) < 1e-10:
            return segment

        ix, iy = pt[0] / pt[2], pt[1] / pt[2]

        if not (-margin <= ix <= w + margin and -margin <= iy <= h + margin):
            return segment

        x1, y1, x2, y2 = segment.x1, segment.y1, segment.x2, segment.y2
        ix_int, iy_int = int(round(ix)), int(round(iy))

        if which_end == "left":
            if x1 <= x2:
                x1, y1 = ix_int, iy_int
            else:
                x2, y2 = ix_int, iy_int
        elif which_end == "right":
            if x1 >= x2:
                x1, y1 = ix_int, iy_int
            else:
                x2, y2 = ix_int, iy_int
        elif which_end == "top":
            if y1 <= y2:
                x1, y1 = ix_int, iy_int
            else:
                x2, y2 = ix_int, iy_int
        elif which_end == "bottom":
            if y1 >= y2:
                x1, y1 = ix_int, iy_int
            else:
                x2, y2 = ix_int, iy_int

        return DetectedLine(
            x1,
            y1,
            x2,
            y2,
            line_type=segment.line_type,
            confidence=segment.confidence,
        )

    # ------------------------------------------------------------------
    # Constraint 1: Touchlines span the full visible width
    # ------------------------------------------------------------------

    def _extend_touchlines(
        self, lines: List[DetectedLine], h: int, w: int
    ) -> List[DetectedLine]:
        left_goal = self._find_line(lines, LineType.LEFT_GOAL_LINE)
        right_goal = self._find_line(lines, LineType.RIGHT_GOAL_LINE)

        result = []
        for line in lines:
            if line.line_type not in (LineType.FAR_TOUCHLINE, LineType.NEAR_TOUCHLINE):
                result.append(line)
                continue

            extended = self._extend_segment_to_boundaries(line, h, w, self.frame_margin)

            if left_goal is not None:
                extended = self._extend_segment_to_line(
                    extended, left_goal, "left", h, w, self.frame_margin
                )

            if right_goal is not None:
                extended = self._extend_segment_to_line(
                    extended, right_goal, "right", h, w, self.frame_margin
                )

            result.append(extended)

        return result

    # ------------------------------------------------------------------
    # Constraint 2: Center line connects the two touchlines
    # ------------------------------------------------------------------

    def _extend_center_line(
        self, lines: List[DetectedLine], h: int, w: int
    ) -> List[DetectedLine]:
        center = self._find_line(lines, LineType.CENTER_LINE)
        if center is None:
            return lines

        far_tl = self._find_line(lines, LineType.FAR_TOUCHLINE)
        near_tl = self._find_line(lines, LineType.NEAR_TOUCHLINE)

        if far_tl is None and near_tl is None:
            extended = self._extend_segment_to_boundaries(
                center, h, w, self.frame_margin
            )
        elif far_tl is not None and near_tl is not None:
            pt_far = self._line_line_intersection(center, far_tl)
            pt_near = self._line_line_intersection(center, near_tl)
            if pt_far is not None and pt_near is not None:
                extended = DetectedLine(
                    int(round(pt_far[0])),
                    int(round(pt_far[1])),
                    int(round(pt_near[0])),
                    int(round(pt_near[1])),
                    line_type=LineType.CENTER_LINE,
                    confidence=center.confidence,
                )
            else:
                extended = self._extend_segment_to_boundaries(
                    center, h, w, self.frame_margin
                )
        else:
            available_tl = far_tl if far_tl is not None else near_tl
            which = "top" if far_tl is not None else "bottom"

            extended = self._extend_segment_to_boundaries(
                center, h, w, self.frame_margin
            )
            extended = self._extend_segment_to_line(
                extended, available_tl, which, h, w, self.frame_margin
            )

        result = []
        for line in lines:
            if line.line_type == LineType.CENTER_LINE:
                result.append(extended)
            else:
                result.append(line)
        return result

    # ------------------------------------------------------------------
    # Constraint 3: Penalty arcs need their penalty front line
    # ------------------------------------------------------------------

    def _gate_penalty_arcs(
        self,
        ellipses: List[DetectedEllipse],
        lines: List[DetectedLine],
    ) -> List[DetectedEllipse]:
        has_l_front = self._find_line(lines, LineType.L_PENALTY_FRONT) is not None
        has_r_front = self._find_line(lines, LineType.R_PENALTY_FRONT) is not None

        result = []
        for e in ellipses:
            if e.ellipse_type == EllipseType.LEFT_PENALTY_ARC and not has_l_front:
                continue
            if e.ellipse_type == EllipseType.RIGHT_PENALTY_ARC and not has_r_front:
                continue
            result.append(e)
        return result

    # ------------------------------------------------------------------
    # Constraint 4: Center circle has center line as a diameter
    # ------------------------------------------------------------------

    def _snap_center_circle_to_center_line(
        self,
        ellipses: List[DetectedEllipse],
        lines: List[DetectedLine],
    ) -> List[DetectedEllipse]:
        center_line = self._find_line(lines, LineType.CENTER_LINE)
        if center_line is None:
            return ellipses

        homo = center_line.to_homogeneous()
        a, b, c = homo

        result = []
        for e in ellipses:
            if e.ellipse_type != EllipseType.CENTER_CIRCLE:
                result.append(e)
                continue

            dist = abs(a * e.center_x + b * e.center_y + c)
            tolerance = self.snap_tolerance * e.axis_major

            if dist > tolerance:
                demoted = DetectedEllipse(
                    center_x=e.center_x,
                    center_y=e.center_y,
                    axis_major=e.axis_major,
                    axis_minor=e.axis_minor,
                    angle=e.angle,
                    ellipse_type=EllipseType.UNKNOWN,
                    confidence=e.confidence * 0.3,
                    contour=e.contour,
                )
                result.append(demoted)
                continue

            denom = a * a + b * b
            if denom < 1e-10:
                result.append(e)
                continue

            factor = (a * e.center_x + b * e.center_y + c) / denom
            new_cx = e.center_x - a * factor
            new_cy = e.center_y - b * factor

            snapped = DetectedEllipse(
                center_x=new_cx,
                center_y=new_cy,
                axis_major=e.axis_major,
                axis_minor=e.axis_minor,
                angle=e.angle,
                ellipse_type=EllipseType.CENTER_CIRCLE,
                confidence=e.confidence,
                contour=e.contour,
            )
            result.append(snapped)

        return result

    # ------------------------------------------------------------------
    # Constraint 5: Grass boundary enforcement
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_lines_to_grass(
        lines: List[DetectedLine],
        grass_mask: np.ndarray,
    ) -> List[DetectedLine]:
        h, w = grass_mask.shape[:2]
        result = []

        for line in lines:
            x1, y1 = line.x1, line.y1
            x2, y2 = line.x2, line.y2

            p1_inside = GeometricRefiner._point_in_grass(x1, y1, grass_mask, h, w)
            p2_inside = GeometricRefiner._point_in_grass(x2, y2, grass_mask, h, w)

            if not p1_inside:
                clipped = GeometricRefiner._walk_to_grass(
                    x1, y1, x2, y2, grass_mask, h, w
                )
                if clipped is not None:
                    x1, y1 = clipped
                else:
                    continue

            if not p2_inside:
                clipped = GeometricRefiner._walk_to_grass(
                    x2, y2, x1, y1, grass_mask, h, w
                )
                if clipped is not None:
                    x2, y2 = clipped
                else:
                    continue

            result.append(
                DetectedLine(
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                    line_type=line.line_type,
                    confidence=line.confidence,
                )
            )

        return result

    @staticmethod
    def _point_in_grass(
        x: float, y: float, grass_mask: np.ndarray, h: int, w: int, margin: int = 3
    ) -> bool:
        ix = int(round(x))
        iy = int(round(y))
        for dy in range(-margin, margin + 1):
            for dx in range(-margin, margin + 1):
                ny, nx = iy + dy, ix + dx
                if 0 <= ny < h and 0 <= nx < w and grass_mask[ny, nx] > 0:
                    return True
        return False

    @staticmethod
    def _walk_to_grass(
        x_out: float,
        y_out: float,
        x_in: float,
        y_in: float,
        grass_mask: np.ndarray,
        h: int,
        w: int,
        steps: int = 500,
    ) -> Optional[Tuple[int, int]]:
        for i in range(1, steps + 1):
            t = i / steps
            x = x_out + t * (x_in - x_out)
            y = y_out + t * (y_in - y_out)
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < h and 0 <= ix < w and grass_mask[iy, ix] > 0:
                return (ix, iy)
        return None

    @staticmethod
    def _filter_ellipses_by_grass(
        ellipses: List[DetectedEllipse],
        grass_mask: np.ndarray,
    ) -> List[DetectedEllipse]:
        h, w = grass_mask.shape[:2]
        result = []
        for e in ellipses:
            ix = int(round(e.center_x))
            iy = int(round(e.center_y))
            ix = max(0, min(w - 1, ix))
            iy = max(0, min(h - 1, iy))
            if grass_mask[iy, ix] > 0:
                result.append(e)
        return result


# ==============================================================================
# SECTION 9: KEYPOINT GENERATION
# ==============================================================================


class KeypointGenerator:
    def __init__(self, max_intersection_dist: float = 2000):
        self.max_intersection_dist = max_intersection_dist

    def _intersect_lines(
        self, l1: DetectedLine, l2: DetectedLine
    ) -> Optional[Tuple[float, float]]:
        h1 = l1.to_homogeneous()
        h2 = l2.to_homogeneous()
        pt = np.cross(h1, h2)
        if abs(pt[2]) < 1e-10:
            return None
        x, y = pt[0] / pt[2], pt[1] / pt[2]
        if abs(x) > self.max_intersection_dist or abs(y) > self.max_intersection_dist:
            return None
        return (x, y)

    def _get_keypoint_id_for_intersection(
        self, type1: LineType, type2: LineType
    ) -> Optional[KeypointID]:
        mapping = {
            frozenset(
                [LineType.FAR_TOUCHLINE, LineType.LEFT_GOAL_LINE]
            ): KeypointID.CORNER_FAR_LEFT,
            frozenset(
                [LineType.FAR_TOUCHLINE, LineType.RIGHT_GOAL_LINE]
            ): KeypointID.CORNER_FAR_RIGHT,
            frozenset(
                [LineType.NEAR_TOUCHLINE, LineType.LEFT_GOAL_LINE]
            ): KeypointID.CORNER_NEAR_LEFT,
            frozenset(
                [LineType.NEAR_TOUCHLINE, LineType.RIGHT_GOAL_LINE]
            ): KeypointID.CORNER_NEAR_RIGHT,
            frozenset(
                [LineType.CENTER_LINE, LineType.FAR_TOUCHLINE]
            ): KeypointID.CENTER_FAR,
            frozenset(
                [LineType.CENTER_LINE, LineType.NEAR_TOUCHLINE]
            ): KeypointID.CENTER_NEAR,
            frozenset(
                [LineType.L_PENALTY_FAR, LineType.LEFT_GOAL_LINE]
            ): KeypointID.L_PENALTY_FAR_OUTER,
            frozenset(
                [LineType.L_PENALTY_NEAR, LineType.LEFT_GOAL_LINE]
            ): KeypointID.L_PENALTY_NEAR_OUTER,
            frozenset(
                [LineType.L_PENALTY_FAR, LineType.L_PENALTY_FRONT]
            ): KeypointID.L_PENALTY_FAR_INNER,
            frozenset(
                [LineType.L_PENALTY_NEAR, LineType.L_PENALTY_FRONT]
            ): KeypointID.L_PENALTY_NEAR_INNER,
            frozenset(
                [LineType.R_PENALTY_FAR, LineType.RIGHT_GOAL_LINE]
            ): KeypointID.R_PENALTY_FAR_OUTER,
            frozenset(
                [LineType.R_PENALTY_NEAR, LineType.RIGHT_GOAL_LINE]
            ): KeypointID.R_PENALTY_NEAR_OUTER,
            frozenset(
                [LineType.R_PENALTY_FAR, LineType.R_PENALTY_FRONT]
            ): KeypointID.R_PENALTY_FAR_INNER,
            frozenset(
                [LineType.R_PENALTY_NEAR, LineType.R_PENALTY_FRONT]
            ): KeypointID.R_PENALTY_NEAR_INNER,
        }
        return mapping.get(frozenset([type1, type2]))

    def generate(
        self,
        lines: List[DetectedLine],
        ellipses: List[DetectedEllipse],
        img_shape: Tuple[int, int],
        extra_keypoints: Optional[List[DetectedKeypoint]] = None,
    ) -> List[DetectedKeypoint]:
        keypoints = []
        h, w = img_shape[:2]

        for i, l1 in enumerate(lines):
            if l1.line_type == LineType.UNKNOWN:
                continue
            for l2 in lines[i + 1 :]:
                if l2.line_type == LineType.UNKNOWN:
                    continue
                kp_id = self._get_keypoint_id_for_intersection(
                    l1.line_type, l2.line_type
                )
                if kp_id is None:
                    continue
                intersection = self._intersect_lines(l1, l2)
                if intersection is None:
                    continue
                x, y = intersection
                margin = 0.3 * max(h, w)
                if not (-margin < x < w + margin and -margin < y < h + margin):
                    continue
                keypoints.append(
                    DetectedKeypoint(
                        image_x=x,
                        image_y=y,
                        keypoint_id=kp_id,
                        confidence=min(l1.confidence, l2.confidence),
                        source=f"{l1.line_type.name} ∩ {l2.line_type.name}",
                    )
                )

        for ellipse in ellipses:
            if ellipse.ellipse_type == EllipseType.CENTER_CIRCLE:
                keypoints.append(
                    DetectedKeypoint(
                        image_x=ellipse.center_x,
                        image_y=ellipse.center_y,
                        keypoint_id=KeypointID.CENTER_SPOT,
                        confidence=ellipse.confidence,
                        source="center_circle",
                    )
                )
            elif ellipse.ellipse_type == EllipseType.LEFT_PENALTY_ARC:
                keypoints.append(
                    DetectedKeypoint(
                        image_x=ellipse.center_x,
                        image_y=ellipse.center_y,
                        keypoint_id=KeypointID.L_PENALTY_SPOT,
                        confidence=ellipse.confidence * 0.7,
                        source="left_penalty_arc_center",
                    )
                )
            elif ellipse.ellipse_type == EllipseType.RIGHT_PENALTY_ARC:
                keypoints.append(
                    DetectedKeypoint(
                        image_x=ellipse.center_x,
                        image_y=ellipse.center_y,
                        keypoint_id=KeypointID.R_PENALTY_SPOT,
                        confidence=ellipse.confidence * 0.7,
                        source="right_penalty_arc_center",
                    )
                )

        if extra_keypoints:
            keypoints.extend(extra_keypoints)

        seen: Dict[KeypointID, DetectedKeypoint] = {}
        for kp in keypoints:
            if (
                kp.keypoint_id not in seen
                or kp.confidence > seen[kp.keypoint_id].confidence
            ):
                seen[kp.keypoint_id] = kp

        return list(seen.values())


# ==============================================================================
# SECTION 10: HOMOGRAPHY ESTIMATION
# ==============================================================================


class HomographyEstimator:
    def __init__(self, ransac_threshold: float = 5.0, min_keypoints: int = 4):
        self.ransac_threshold = ransac_threshold
        self.min_keypoints = min_keypoints

    def compute(
        self, keypoints: List[DetectedKeypoint]
    ) -> Tuple[Optional[np.ndarray], int, float, float]:
        if len(keypoints) < self.min_keypoints:
            return None, 0, float("inf"), float("inf")

        src = np.array([kp.image_point for kp in keypoints], dtype=np.float32)
        dst = np.array([kp.world_point for kp in keypoints], dtype=np.float32)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_threshold)
        if H is None or mask is None:
            return None, 0, float("inf"), float("inf")

        inlier_idx = mask.ravel() == 1
        inliers = int(np.sum(inlier_idx))
        if inliers < self.min_keypoints:
            return None, inliers, float("inf"), float("inf")

        inlier_src = src[inlier_idx]
        inlier_dst = dst[inlier_idx]
        H_refined, _ = cv2.findHomography(inlier_src, inlier_dst, 0)
        if H_refined is not None:
            H = H_refined

        mean_err, med_err = self._reprojection_error_inliers(H, inlier_src, inlier_dst)
        return H, inliers, mean_err, med_err

    def _reprojection_error_inliers(
        self, H: np.ndarray, src: np.ndarray, dst: np.ndarray
    ) -> Tuple[float, float]:
        src_h = np.hstack([src, np.ones((len(src), 1), dtype=np.float32)])
        proj = (H @ src_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        per_point = np.sqrt(np.sum((proj - dst) ** 2, axis=1))
        return float(np.mean(per_point)), float(np.median(per_point))

    def image_to_world(self, H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        pt = np.array([[x, y, 1.0]])
        proj = (H @ pt.T).T
        return (proj[0, 0] / proj[0, 2], proj[0, 1] / proj[0, 2])

    def world_to_image(self, H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        H_inv = np.linalg.inv(H)
        return self.image_to_world(H_inv, x, y)


# ==============================================================================
# SECTION 11: PERSPECTIVE-AWARE DRAWING UTILITIES
# ==============================================================================


class PerspectiveDrawing:
    """
    Static helpers for rendering soccer-field features with realistic
    perspective distortion in the visualization overlay.

    Key idea — in a tactical / broadcast camera view the **near** half of
    the pitch (bottom of the image) is closer to the camera, so:
        • Lines appear *thicker* at the bottom, *thinner* at the top.
        • The center circle's near arc (bottom half) looks 'heavier'
          than the far arc (top half).

    The depth gradient is approximated by mapping each point's y-
    coordinate within the grass region to a thickness multiplier via
    linear interpolation:
        thickness(y) = thick_near  +  (y − y_near) / (y_far − y_near)
                                        × (thick_far − thick_near)

    Because y_far < y_near in image coordinates (top of image), the
    formula naturally yields larger thickness at the bottom.
    """

    @staticmethod
    def depth_thickness(
        y: float,
        y_far: float,
        y_near: float,
        thick_far: float = 1.0,
        thick_near: float = 3.0,
    ) -> float:
        """Return the pixel thickness appropriate for image-y position *y*."""
        span = y_near - y_far
        if abs(span) < 1e-6:
            return (thick_far + thick_near) / 2.0
        t = np.clip((y - y_far) / span, 0.0, 1.0)
        return thick_far + t * (thick_near - thick_far)

    @staticmethod
    def draw_perspective_ellipse(
        img: np.ndarray,
        ellipse: DetectedEllipse,
        color: Tuple[int, int, int],
        y_far: float,
        y_near: float,
        thick_far: float = 1.0,
        thick_near: float = 3.5,
        n_segments: int = 128,
    ) -> np.ndarray:
        """
        Draw an ellipse with depth-dependent line thickness.

        The ellipse is tessellated into *n_segments* short arcs.  Each
        arc's thickness is determined by the average y-coordinate of its
        two endpoints, producing a smooth taper from thick (near /
        bottom) to thin (far / top).
        """
        cx, cy = ellipse.center_x, ellipse.center_y
        a = ellipse.axis_major
        b = ellipse.axis_minor
        theta = np.radians(ellipse.angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        t = np.linspace(0, 2 * np.pi, n_segments + 1)
        xs = cx + a * np.cos(t) * cos_t - b * np.sin(t) * sin_t
        ys = cy + a * np.cos(t) * sin_t + b * np.sin(t) * cos_t

        for i in range(n_segments):
            p1 = (int(round(xs[i])), int(round(ys[i])))
            p2 = (int(round(xs[i + 1])), int(round(ys[i + 1])))
            avg_y = (ys[i] + ys[i + 1]) / 2.0
            thick = PerspectiveDrawing.depth_thickness(
                avg_y, y_far, y_near, thick_far, thick_near
            )
            cv2.line(img, p1, p2, color, max(1, int(round(thick))), cv2.LINE_AA)

        return img

    @staticmethod
    def draw_perspective_line(
        img: np.ndarray,
        line: DetectedLine,
        color: Tuple[int, int, int],
        y_far: float,
        y_near: float,
        thick_far: float = 1.0,
        thick_near: float = 3.5,
        n_segments: int = 32,
    ) -> np.ndarray:
        """
        Draw a straight line with depth-dependent thickness.

        Tessellated into *n_segments* short segments, each rendered at
        the thickness dictated by its average y-coordinate.
        """
        xs = np.linspace(line.x1, line.x2, n_segments + 1)
        ys = np.linspace(line.y1, line.y2, n_segments + 1)

        for i in range(n_segments):
            p1 = (int(round(xs[i])), int(round(ys[i])))
            p2 = (int(round(xs[i + 1])), int(round(ys[i + 1])))
            avg_y = (ys[i] + ys[i + 1]) / 2.0
            thick = PerspectiveDrawing.depth_thickness(
                avg_y, y_far, y_near, thick_far, thick_near
            )
            cv2.line(img, p1, p2, color, max(1, int(round(thick))), cv2.LINE_AA)

        return img


# ==============================================================================
# SECTION 12: MAIN PIPELINE
# ==============================================================================


class SoccerHomographyPipeline:
    """
    Complete pipeline for soccer field homography estimation.

    Usage:
        pipeline = SoccerHomographyPipeline()
        result = pipeline.process(image)

        if result.is_valid:
            world_x, world_y = pipeline.image_to_world(result.homography, px, py)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        self.preprocessor = Preprocessor(
            hue_range=config.get("hue_range", (35, 85)),
            sat_range=config.get("sat_range", (40, 255)),
            val_range=config.get("val_range", (40, 255)),
        )

        self.white_extractor = WhiteLineExtractor(
            adaptive_k=config.get("adaptive_k", 1.1),
            adaptive_k_edge=config.get("adaptive_k_edge", 0.8),
            edge_band=config.get("edge_band", 0.25),
            block_size=config.get("block_size", 101),
            h_kernel=config.get("h_kernel", 30),
            v_kernel=config.get("v_kernel", 30),
            min_area=config.get("min_area", 50),
            close_iters=config.get("close_iters", 2),
        )

        self.line_detector = EnhancedLineDetector(
            hough_threshold=config.get("hough_threshold", 30),
            min_length=config.get("min_length", 30),
            max_gap=config.get("max_gap", 20),
            angle_h_thresh=config.get("angle_h_thresh", 25),
            angle_v_thresh=config.get("angle_v_thresh", 25),
            min_confidence_for_selection=config.get(
                "min_confidence_for_selection", 0.3
            ),
            min_touchline_length_ratio=config.get("min_touchline_length_ratio", 0.3),
        )

        self.ellipse_detector = EllipseDetector(
            min_contour_length=config.get("min_contour_length", 80),
            min_aspect_ratio=config.get("min_aspect_ratio", 0.2),
            search_margin_factor=config.get("ellipse_search_margin", 2.5),
            line_suppress_thickness=config.get("line_suppress_thickness", 14),
        )

        self.geometric_refiner = GeometricRefiner(
            frame_margin=config.get("refiner_frame_margin", 50.0),
            snap_tolerance=config.get("refiner_snap_tolerance", 0.3),
        )

        self.keypoint_generator = KeypointGenerator()

        self.homography_estimator = HomographyEstimator(
            ransac_threshold=config.get("ransac_threshold", 5.0),
        )

    def process(self, image: np.ndarray) -> HomographyResult:
        result = HomographyResult()
        h, w = image.shape[:2]

        print("=" * 70)
        print("SOCCER FIELD HOMOGRAPHY PIPELINE v2 (geometry-constrained)")
        print(f"Image size: {w} x {h}")
        print("=" * 70)

        # Stage 1: Preprocessing
        print("[1/8] Preprocessing...")
        normalized = self.preprocessor.normalize_illumination(image)
        grass_mask = self.preprocessor.segment_grass(normalized)
        grass_bounds = self.preprocessor.get_grass_bounds(grass_mask)

        coverage = np.sum(grass_mask > 0) / grass_mask.size * 100
        print(f"      Grass coverage: {coverage:.1f}%")
        result.grass_mask = grass_mask

        # Stage 2: White extraction
        print("[2/8] Extracting white markings...")
        white_raw = self.white_extractor.extract(normalized, grass_mask, grass_bounds)
        white_clean = self.white_extractor.clean(white_raw)
        white_clean = cv2.bitwise_and(white_clean, grass_mask)
        skeleton = self.white_extractor.skeletonize(white_clean)
        skeleton = cv2.bitwise_and(skeleton, grass_mask)

        print(f"      Skeleton pixels: {np.sum(skeleton > 0)}")
        result.white_mask = white_clean
        result.skeleton = skeleton

        # Stage 3: Line detection
        print("[3/8] Detecting lines...")
        raw_lines = self.line_detector.detect(skeleton)
        merged_lines = self.line_detector.merge_collinear(raw_lines)
        print(f"      Raw: {len(raw_lines)}, Merged: {len(merged_lines)}")

        # Stage 4: Line classification (BEFORE ellipse detection)
        print("[4/8] Classifying lines...")
        classified_lines = self.line_detector.classify(
            merged_lines,
            grass_bounds,
            [],
            img_shape=image.shape,
        )
        type_counts: Dict[LineType, int] = defaultdict(int)
        for line in classified_lines:
            type_counts[line.line_type] += 1
        for lt, count in sorted(type_counts.items(), key=lambda x: x[0].name):
            print(f"        - {lt.name}: {count}")

        # Stage 5: Line-guided ellipse detection
        print("[5/8] Detecting ellipses (line-guided)...")
        classified_ellipses = self.ellipse_detector.detect(
            white_clean,
            classified_lines,
            grass_mask,
            grass_bounds,
            image.shape,
        )
        print(f"      Found: {len(classified_ellipses)}")
        for e in classified_ellipses:
            print(
                f"        - {e.ellipse_type.name}: "
                f"center=({e.center_x:.0f}, {e.center_y:.0f}), "
                f"axes=({e.axis_major:.0f}, {e.axis_minor:.0f}), "
                f"conf={e.confidence:.2f}"
            )

        # Stage 6: Re-classify lines with ellipse context
        print("[6/8] Refining line classification with ellipse context...")
        classified_lines = self.line_detector.classify(
            merged_lines,
            grass_bounds,
            classified_ellipses,
            img_shape=image.shape,
        )

        # Stage 7: Geometric refinement (includes NEW constraints 6 & 7)
        print("[7/8] Applying geometric constraints...")
        refined_lines, refined_ellipses = self.geometric_refiner.refine(
            classified_lines,
            classified_ellipses,
            image.shape,
            grass_bounds,
            grass_mask=grass_mask,
        )

        n_lines_before = len(classified_lines)
        n_lines_after = len(refined_lines)
        n_ell_before = len(classified_ellipses)
        n_ell_after = len(refined_ellipses)
        print(f"      Lines:    {n_lines_before} → {n_lines_after}")
        print(f"      Ellipses: {n_ell_before} → {n_ell_after}")

        for line in refined_lines:
            if line.line_type in (
                LineType.FAR_TOUCHLINE,
                LineType.NEAR_TOUCHLINE,
                LineType.CENTER_LINE,
            ):
                print(
                    f"        {line.line_type.name}: "
                    f"({line.x1},{line.y1})→({line.x2},{line.y2}) len={line.length:.0f}"
                )

        result.lines = refined_lines
        result.ellipses = refined_ellipses

        # Generate bonus keypoints from circle ∩ center line (algebraic)
        circle_line_kps = self.geometric_refiner.generate_circle_line_keypoints(
            refined_lines, refined_ellipses
        )
        if circle_line_kps:
            print(f"      Circle∩Line keypoints (algebraic): {len(circle_line_kps)}")
            for kp in circle_line_kps:
                print(
                    f"        ► {kp.keypoint_id.name}: "
                    f"img=({kp.image_x:.1f}, {kp.image_y:.1f})"
                )

        # Stage 8: Keypoint generation
        print("[8/8] Generating keypoints...")
        keypoints = self.keypoint_generator.generate(
            refined_lines,
            refined_ellipses,
            image.shape,
            extra_keypoints=circle_line_kps,
        )

        print(f"      Generated: {len(keypoints)} keypoints")

        keypoints = self.geometric_refiner.filter_keypoints_by_grass(
            keypoints, grass_mask, margin_px=15
        )
        print(f"      After grass filter: {len(keypoints)} keypoints")

        for kp in keypoints:
            wx, wy = kp.world_point
            print(
                f"        - {kp.keypoint_id.name}: "
                f"img=({kp.image_x:.0f}, {kp.image_y:.0f}) "
                f"→ world=({wx:.1f}, {wy:.1f})  [{kp.source}]"
            )
        result.keypoints = keypoints

        # Stage 9: Homography
        print("\n[HOMOGRAPHY]")
        H, inliers, mean_err, med_err = self.homography_estimator.compute(keypoints)

        result.homography = H
        result.inlier_count = inliers
        result.reprojection_error = mean_err
        result.median_reprojection_error = med_err

        if H is not None:
            result.homography_inv = np.linalg.inv(H)
            print(
                f"      SUCCESS: {inliers} inliers, error = {mean_err:.3f}m "
                f"(median {med_err:.3f}m)"
            )
        else:
            print(f"      FAILED: Need at least 4 keypoints (got {len(keypoints)})")

        print("=" * 70)
        return result

    # ------------------------------------------------------------------
    # Visualization  (v2: perspective-aware rendering)
    # ------------------------------------------------------------------

    def visualize(
        self,
        image: np.ndarray,
        result: HomographyResult,
        save_path: Optional[str] = None,
    ) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Soccer Field Homography Pipeline v2", fontsize=14, fontweight="bold"
        )

        # --- Determine far / near y for perspective thickness ---
        y_far, y_near = self._get_far_near_y(result)

        # 1. Original
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("1. Original Image")
        axes[0, 0].axis("off")

        # 2. Grass mask
        axes[0, 1].imshow(result.grass_mask, cmap="gray")
        coverage = np.sum(result.grass_mask > 0) / result.grass_mask.size * 100
        axes[0, 1].set_title(f"2. Grass Mask ({coverage:.1f}%)")
        axes[0, 1].axis("off")

        # --- 3. Detected lines (perspective thickness + legend) ---
        line_img = image.copy()

        colors_bgr = {
            LineType.FAR_TOUCHLINE: (0, 255, 255),
            LineType.NEAR_TOUCHLINE: (0, 255, 0),
            LineType.CENTER_LINE: (255, 0, 255),
            LineType.LEFT_GOAL_LINE: (255, 0, 0),
            LineType.RIGHT_GOAL_LINE: (0, 0, 255),
            LineType.L_PENALTY_FRONT: (255, 128, 0),
            LineType.R_PENALTY_FRONT: (0, 128, 255),
            LineType.L_PENALTY_FAR: (255, 200, 0),
            LineType.L_PENALTY_NEAR: (255, 200, 0),
            LineType.R_PENALTY_FAR: (0, 200, 255),
            LineType.R_PENALTY_NEAR: (0, 200, 255),
        }

        found_types = set()
        for line in result.lines:
            color = colors_bgr.get(line.line_type, (128, 128, 128))
            PerspectiveDrawing.draw_perspective_line(
                line_img,
                line,
                color,
                y_far,
                y_near,
                thick_far=1.5,
                thick_near=4.0,
                n_segments=48,
            )
            found_types.add(line.line_type)

        axes[0, 2].imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f"3. Refined Lines ({len(result.lines)})")
        axes[0, 2].axis("off")

        line_patches = []
        for l_type in found_types:
            bgr = colors_bgr.get(l_type, (128, 128, 128))
            rgb = (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)
            # Use descriptive name for penalty sides if available
            label = PENALTY_SIDE_DESCRIPTIONS.get(l_type, l_type.name)
            # Shorten for legend
            label = l_type.name
            line_patches.append(mpatches.Patch(color=rgb, label=label))

        if line_patches:
            axes[0, 2].legend(
                handles=line_patches,
                loc="upper right",
                fontsize="x-small",
                framealpha=0.7,
            )

        # --- 4. Ellipses (perspective thickness + legend) ---
        ellipse_img = image.copy()
        found_ellipses = set()

        color_center = (0, 255, 255)
        color_penalty = (0, 128, 255)

        for e in result.ellipses:
            is_center = e.ellipse_type == EllipseType.CENTER_CIRCLE
            color = color_center if is_center else color_penalty
            found_ellipses.add("Center Circle" if is_center else "Penalty Arc")

            # Perspective-aware ellipse drawing
            PerspectiveDrawing.draw_perspective_ellipse(
                ellipse_img,
                e,
                color,
                y_far,
                y_near,
                thick_far=1.0,
                thick_near=3.5,
                n_segments=128,
            )

            center = (int(e.center_x), int(e.center_y))
            cv2.circle(ellipse_img, center, 5, color, -1)

        axes[1, 0].imshow(cv2.cvtColor(ellipse_img, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"4. Ellipses — perspective ({len(result.ellipses)})")
        axes[1, 0].axis("off")

        ell_patches = []
        if "Center Circle" in found_ellipses:
            ell_patches.append(mpatches.Patch(color=(1, 1, 0), label="Center Circle"))
        if "Penalty Arc" in found_ellipses:
            ell_patches.append(mpatches.Patch(color=(1, 0.5, 0), label="Penalty Arc"))
        if ell_patches:
            axes[1, 0].legend(
                handles=ell_patches, loc="upper right", fontsize="x-small"
            )

        # --- 5. Keypoints ---
        kp_img = image.copy()
        kp_color = (0, 255, 0)

        for kp in result.keypoints:
            x, y = int(kp.image_x), int(kp.image_y)
            cv2.circle(kp_img, (x, y), 6, kp_color, -1)
            cv2.circle(kp_img, (x, y), 8, (255, 255, 255), 2)

            label = kp.keypoint_id.name.replace("CORNER_", "").replace(
                "PENALTY_", "PEN_"
            )

            cv2.putText(
                kp_img,
                label,
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                kp_img,
                label,
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        axes[1, 1].imshow(cv2.cvtColor(kp_img, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f"5. Keypoints ({len(result.keypoints)})")
        axes[1, 1].axis("off")

        # 6. Homography result
        if result.is_valid:
            proj_img = image.copy()

            corners = [
                (-FIFA.half_length, -FIFA.half_width),
                (FIFA.half_length, -FIFA.half_width),
                (FIFA.half_length, FIFA.half_width),
                (-FIFA.half_length, FIFA.half_width),
            ]
            for i in range(4):
                w1 = corners[i]
                w2 = corners[(i + 1) % 4]
                try:
                    p1 = self.homography_estimator.world_to_image(
                        result.homography, *w1
                    )
                    p2 = self.homography_estimator.world_to_image(
                        result.homography, *w2
                    )
                    cv2.line(
                        proj_img,
                        (int(p1[0]), int(p1[1])),
                        (int(p2[0]), int(p2[1])),
                        (0, 255, 255),
                        2,
                    )
                except Exception:
                    pass

            try:
                c1 = self.homography_estimator.world_to_image(
                    result.homography, 0, -FIFA.half_width
                )
                c2 = self.homography_estimator.world_to_image(
                    result.homography, 0, FIFA.half_width
                )
                cv2.line(
                    proj_img,
                    (int(c1[0]), int(c1[1])),
                    (int(c2[0]), int(c2[1])),
                    (255, 0, 255),
                    2,
                )
            except Exception:
                pass

            axes[1, 2].imshow(cv2.cvtColor(proj_img, cv2.COLOR_BGR2RGB))
            axes[1, 2].set_title(
                f"6. Homography (err={result.reprojection_error:.2f}m)"
            )
        else:
            axes[1, 2].text(
                0.5,
                0.5,
                "HOMOGRAPHY FAILED\n\nNeed >= 4 keypoints",
                ha="center",
                va="center",
                fontsize=14,
                transform=axes[1, 2].transAxes,
            )
            axes[1, 2].set_title("6. Homography (FAILED)")
        axes[1, 2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    # Helper: extract far / near y from result for perspective thickness
    # ------------------------------------------------------------------

    @staticmethod
    def _get_far_near_y(result: HomographyResult) -> Tuple[float, float]:
        """
        Return (y_far, y_near) from touchlines or grass bounds.
        y_far = top of pitch (small y), y_near = bottom (large y).
        """
        y_far = None
        y_near = None
        for l in result.lines:
            if l.line_type == LineType.FAR_TOUCHLINE:
                y_far = l.midpoint[1]
            elif l.line_type == LineType.NEAR_TOUCHLINE:
                y_near = l.midpoint[1]

        if result.grass_mask is not None:
            rows = np.any(result.grass_mask > 0, axis=1)
            if np.any(rows):
                if y_far is None:
                    y_far = float(np.argmax(rows))
                if y_near is None:
                    y_near = float(result.grass_mask.shape[0] - np.argmax(rows[::-1]))

        if y_far is None:
            y_far = 0.0
        if y_near is None:
            y_near = float(
                result.grass_mask.shape[0] if result.grass_mask is not None else 720
            )

        return y_far, y_near

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------

    def image_to_world(self, H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        return self.homography_estimator.image_to_world(H, x, y)

    def world_to_image(self, H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        return self.homography_estimator.world_to_image(H, x, y)

    def draw_field_template(self, scale: float = 8.0, margin: int = 30) -> np.ndarray:
        w = int(FIFA.LENGTH * scale) + 2 * margin
        h = int(FIFA.WIDTH * scale) + 2 * margin
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (34, 139, 34)

        def w2i(wx: float, wy: float) -> Tuple[int, int]:
            px = int((wx + FIFA.half_length) * scale + margin)
            py = int((wy + FIFA.half_width) * scale + margin)
            return (px, py)

        white = (255, 255, 255)
        thick = 2

        cv2.rectangle(
            img,
            w2i(-FIFA.half_length, -FIFA.half_width),
            w2i(FIFA.half_length, FIFA.half_width),
            white,
            thick,
        )
        cv2.line(img, w2i(0, -FIFA.half_width), w2i(0, FIFA.half_width), white, thick)
        cv2.circle(img, w2i(0, 0), int(FIFA.CENTER_CIRCLE_RADIUS * scale), white, thick)
        cv2.circle(img, w2i(0, 0), 4, white, -1)

        pa_hw = FIFA.penalty_area_half_width
        cv2.rectangle(
            img,
            w2i(-FIFA.half_length, -pa_hw),
            w2i(-FIFA.half_length + FIFA.PENALTY_AREA_LENGTH, pa_hw),
            white,
            thick,
        )
        cv2.rectangle(
            img,
            w2i(FIFA.half_length - FIFA.PENALTY_AREA_LENGTH, -pa_hw),
            w2i(FIFA.half_length, pa_hw),
            white,
            thick,
        )

        ga_hw = FIFA.goal_area_half_width
        cv2.rectangle(
            img,
            w2i(-FIFA.half_length, -ga_hw),
            w2i(-FIFA.half_length + FIFA.GOAL_AREA_LENGTH, ga_hw),
            white,
            thick,
        )
        cv2.rectangle(
            img,
            w2i(FIFA.half_length - FIFA.GOAL_AREA_LENGTH, -ga_hw),
            w2i(FIFA.half_length, ga_hw),
            white,
            thick,
        )

        cv2.circle(
            img, w2i(-FIFA.half_length + FIFA.PENALTY_SPOT_DISTANCE, 0), 4, white, -1
        )
        cv2.circle(
            img, w2i(FIFA.half_length - FIFA.PENALTY_SPOT_DISTANCE, 0), 4, white, -1
        )

        arc_r = int(FIFA.PENALTY_ARC_RADIUS * scale)
        cv2.ellipse(
            img,
            w2i(-FIFA.half_length + FIFA.PENALTY_SPOT_DISTANCE, 0),
            (arc_r, arc_r),
            0,
            -53,
            53,
            white,
            thick,
        )
        cv2.ellipse(
            img,
            w2i(FIFA.half_length - FIFA.PENALTY_SPOT_DISTANCE, 0),
            (arc_r, arc_r),
            0,
            127,
            233,
            white,
            thick,
        )

        return img


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "D:/ITE/year5/graduation project/5th Dataset/examples photo/Screenshot 2026-02-02 210426.png"

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        sys.exit(1)

    config = {
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

    pipeline = SoccerHomographyPipeline(config)
    result = pipeline.process(img)

    pipeline.visualize(img, result, save_path="homography_result.png")

    print(f"\nLines: {len(result.lines)}")
    for l in result.lines:
        desc = PENALTY_SIDE_DESCRIPTIONS.get(l.line_type, l.line_type.name)
        print(f"  {desc}  conf={l.confidence:.2f}  len={l.length:.0f}")

    print(f"\nEllipses: {len(result.ellipses)}")
    print(f"Keypoints: {len(result.keypoints)}")
    print(f"Homography valid: {result.is_valid}")

    if result.is_valid:
        print(f"Inliers: {result.inlier_count}")
        print(f"Reprojection error: {result.reprojection_error:.3f} meters")
        print(f"\nHomography matrix:\n{result.homography}")

        h, w = img.shape[:2]
        wx, wy = pipeline.image_to_world(result.homography, w / 2, h / 2)
        print(f"\nImage center ({w/2:.0f}, {h/2:.0f}) → World ({wx:.1f}m, {wy:.1f}m)")
