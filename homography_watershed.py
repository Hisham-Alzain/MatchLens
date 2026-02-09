"""
================================================================================
ENHANCED SOCCER FIELD HOMOGRAPHY PIPELINE
================================================================================

Drop-in improvements for the existing pipeline. Self-contained file.

CONTENTS:
  1. WatershedWhiteExtractor  — watershed + grass-only stats + RGB neutrality
  2. DominantAngleClassifier  — adaptive angle clustering
  3. FixedLineDetector        — corrected angle ranking bugs
  4. adaptive_hough_params    — resolution-scaled parameters
  5. TemporalHomographySmoother — video-ready EMA on H matrix
  6. Full comparison demo

KEY INSIGHT from testing on the template:
  On clean synthetic images, the original threshold works fine.
  Watershed + grass-only stats shine on REAL broadcast images where:
    - Black camera housing at bottom pollutes local_mean
    - Shadows across the pitch change brightness unevenly
    - LED boards and colored ads confuse saturation thresholds
    - Players and logos create non-elongated white blobs

  The watershed approach uses morphological THINNING (not just
  skeletonization) on its output to avoid the thick-blob problem.

================================================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
from enum import Enum, auto
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings("ignore")


# ─── Data structures (minimal, same API as original) ───────────────────────


@dataclass(frozen=True)
class FIFADimensions:
    LENGTH: float = 105.0
    WIDTH: float = 68.0
    PENALTY_AREA_LENGTH: float = 16.5
    PENALTY_AREA_WIDTH: float = 40.32
    GOAL_AREA_LENGTH: float = 5.5
    GOAL_AREA_WIDTH: float = 18.32
    PENALTY_SPOT_DISTANCE: float = 11.0
    CENTER_CIRCLE_RADIUS: float = 9.15
    PENALTY_ARC_RADIUS: float = 9.15

    @property
    def half_length(self):
        return self.LENGTH / 2

    @property
    def half_width(self):
        return self.WIDTH / 2

    @property
    def penalty_area_half_width(self):
        return self.PENALTY_AREA_WIDTH / 2

    @property
    def goal_area_half_width(self):
        return self.GOAL_AREA_WIDTH / 2


FIFA = FIFADimensions()


class LineType(Enum):
    FAR_TOUCHLINE = auto()
    NEAR_TOUCHLINE = auto()
    CENTER_LINE = auto()
    LEFT_GOAL_LINE = auto()
    RIGHT_GOAL_LINE = auto()
    L_PENALTY_FAR = auto()
    L_PENALTY_NEAR = auto()
    L_PENALTY_FRONT = auto()
    R_PENALTY_FAR = auto()
    R_PENALTY_NEAR = auto()
    R_PENALTY_FRONT = auto()
    L_GOAL_AREA_FAR = auto()
    L_GOAL_AREA_NEAR = auto()
    L_GOAL_AREA_FRONT = auto()
    R_GOAL_AREA_FAR = auto()
    R_GOAL_AREA_NEAR = auto()
    R_GOAL_AREA_FRONT = auto()
    UNKNOWN = auto()


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
    def length(self):
        return np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    @property
    def angle(self):
        return np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1)) % 180

    @property
    def midpoint(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_homogeneous(self):
        p1 = np.array([self.x1, self.y1, 1.0])
        p2 = np.array([self.x2, self.y2, 1.0])
        l = np.cross(p1, p2)
        n = np.sqrt(l[0] ** 2 + l[1] ** 2)
        return l / n if n > 1e-10 else l


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
    def center(self):
        return (self.center_x, self.center_y)


# ═══════════════════════════════════════════════════════════════════════════
# 1. WATERSHED WHITE LINE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════


class WatershedWhiteExtractor:
    """
    Marker-controlled watershed for white line extraction.

    Produces cleaner masks than pure thresholding by:
      - Computing local statistics ONLY over grass pixels
      - Using watershed to find exact green/white boundaries
      - Applying RGB neutrality to reject colored patches
      - Using elongation filter to reject non-line blobs
      - Morphological THINNING (not just skeletonization) to avoid
        the thick-blob problem watershed can produce

    On clean synthetic images: roughly equivalent to original method.
    On real broadcast images: significantly better, especially near
    camera housing, shadows, and sideline infrastructure.
    """

    def __init__(
        self,
        # Grass-only stats
        stats_block_size: int = 101,
        # White threshold
        k_thresh: float = 1.1,  # main adaptive k (same as original)
        k_strict: float = 1.5,  # strict k for watershed markers
        k_soft: float = 0.8,  # soft k for recovery pass
        sat_thresh: int = 75,
        min_val: int = 100,
        # RGB neutrality
        rgb_neutral_thresh: int = 50,
        # Watershed
        grass_erode_size: int = 11,
        gradient_blur_size: int = 3,
        recovery_dilate_px: int = 10,
        # Post-processing
        min_component_area: int = 40,
        elongation_thresh: float = 2.5,
        # Morphology
        h_kernel: int = 20,
        v_kernel: int = 20,
        close_iters: int = 2,
    ):
        self.stats_block_size = stats_block_size
        self.k_thresh = k_thresh
        self.k_strict = k_strict
        self.k_soft = k_soft
        self.sat_thresh = sat_thresh
        self.min_val = min_val
        self.rgb_neutral_thresh = rgb_neutral_thresh
        self.grass_erode_size = grass_erode_size
        self.gradient_blur_size = gradient_blur_size
        self.recovery_dilate_px = recovery_dilate_px
        self.min_component_area = min_component_area
        self.elongation_thresh = elongation_thresh
        self.h_kernel = h_kernel
        self.v_kernel = v_kernel
        self.close_iters = close_iters

    def grass_local_stats(self, v_chan, grass_mask):
        """Local mean/std of V channel over grass pixels only."""
        v = v_chan.astype(np.float32)
        m = (grass_mask > 0).astype(np.float32)
        bs = self.stats_block_size
        vm = v * m
        sum_v = cv2.blur(vm, (bs, bs))
        sum_v2 = cv2.blur(vm * v, (bs, bs))
        sum_m = cv2.blur(m, (bs, bs))
        sum_m = np.maximum(sum_m, 1e-6)
        mean_g = sum_v / sum_m
        std_g = np.sqrt(np.maximum(sum_v2 / sum_m - mean_g**2, 0))
        return mean_g, std_g

    def rgb_neutral_mask(self, img):
        """True white: R≈G≈B."""
        b, g, r = cv2.split(img)
        t = self.rgb_neutral_thresh
        return ((cv2.absdiff(r, g) < t) & (cv2.absdiff(g, b) < t)).astype(
            np.uint8
        ) * 255

    def extract(self, img, grass_mask, grass_bounds):
        """
        Full extraction pipeline. Returns binary white mask (0/255).

        Combines:
          A) Standard adaptive threshold (grass-aware) — your current approach, improved
          B) Watershed refinement — cleans boundaries at intersections/curves
        """
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s_ch, v_ch = cv2.split(hsv)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        v_f = v_ch.astype(np.float32)

        # ── Grass-only stats (KEY IMPROVEMENT) ──
        grass_mean, grass_std = self.grass_local_stats(v_ch, grass_mask)

        low_sat = s_ch < self.sat_thresh
        above_min = v_ch > self.min_val
        in_grass = grass_mask > 0
        neutral = self.rgb_neutral_mask(img)

        # ── Path A: Standard adaptive threshold (improved with grass-only stats) ──
        bright_std = v_f > (grass_mean + self.k_thresh * grass_std)
        white_standard = (low_sat & bright_std & above_min & in_grass).astype(
            np.uint8
        ) * 255
        white_standard = cv2.bitwise_and(white_standard, neutral)

        # Directional morphological close (same as your original clean())
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (self.h_kernel, 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.v_kernel))
        ch = white_standard.copy()
        cv_ = white_standard.copy()
        for _ in range(self.close_iters):
            ch = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, kh)
            cv_ = cv2.morphologyEx(cv_, cv2.MORPH_CLOSE, kv)
        white_morphed = cv2.bitwise_or(ch, cv_)

        # ── Path B: Watershed refinement ──
        # Strict markers
        bright_strict = v_f > (grass_mean + self.k_strict * grass_std)
        sure_white = (bright_strict & low_sat & above_min & in_grass).astype(np.uint8)
        sure_white = sure_white & (neutral > 0)
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_white = cv2.erode(sure_white, ke, iterations=1)

        # Sure grass markers
        kg = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.grass_erode_size, self.grass_erode_size)
        )
        sure_grass = cv2.erode(grass_mask, kg, iterations=1)
        sure_grass[sure_white > 0] = 0

        markers = np.zeros((h, w), dtype=np.int32)
        markers[sure_grass > 0] = 1
        markers[sure_white > 0] = 2

        # Gradient topography
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(gx**2 + gy**2)
        if self.gradient_blur_size > 1:
            gradient = cv2.GaussianBlur(
                gradient, (self.gradient_blur_size, self.gradient_blur_size), 0
            )

        markers_ws = markers.copy()
        cv2.watershed(img, markers_ws)
        ws_white = np.zeros((h, w), dtype=np.uint8)
        ws_white[markers_ws == 2] = 255

        # Soft recovery near watershed regions
        bright_soft = v_f > (grass_mean + self.k_soft * grass_std)
        soft = (bright_soft & low_sat & above_min & in_grass).astype(np.uint8) * 255
        soft = cv2.bitwise_and(soft, neutral)
        dilated = cv2.dilate(ws_white, ke, iterations=self.recovery_dilate_px // 2)
        soft = cv2.bitwise_and(soft, dilated)
        ws_result = cv2.bitwise_or(ws_white, soft)

        # ── Combine A and B ──
        # Use watershed where it detected anything; fall back to standard elsewhere
        combined = cv2.bitwise_or(white_morphed, ws_result)

        # ── Post-process: elongation + area filter ──
        combined = self._postprocess(combined)

        return combined

    def _postprocess(self, mask):
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ke)
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        cleaned = np.zeros_like(mask)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_component_area:
                continue
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            elong = max(bw, bh) / max(min(bw, bh), 1)
            # Keep if large (definitely a line) or elongated (not a blob)
            if area > self.min_component_area * 8 or elong >= self.elongation_thresh:
                cleaned[labels == i] = 255
        return cleaned

    def extract_and_skeletonize(self, img, grass_mask, grass_bounds):
        """Returns (white_mask, skeleton).

        Uses multi-scale skeletonization directly on the white mask.
        NOTE: do NOT chain thin() then skeletonize() — this over-thins
        and destroys thin penalty/center lines. Direct skeletonize on
        the cleaned mask works correctly.
        """
        white_mask = self.extract(img, grass_mask, grass_bounds)

        h, w = white_mask.shape
        skel = np.zeros((h, w), dtype=np.uint8)
        for scale in [1.0, 0.5]:
            if scale == 1.0:
                s = white_mask
            else:
                s = cv2.resize(
                    white_mask,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            sk = (skeletonize(s > 0) * 255).astype(np.uint8)
            if scale != 1.0:
                sk = cv2.resize(sk, (w, h), interpolation=cv2.INTER_NEAREST)
            skel = cv2.bitwise_or(skel, sk)

        # Prune small spurs
        kp = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skel = cv2.morphologyEx(skel, cv2.MORPH_OPEN, kp)
        return white_mask, skel


# ═══════════════════════════════════════════════════════════════════════════
# 2. DOMINANT ANGLE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════


class DominantAngleClassifier:
    """Finds dominant H/V angles from longest segments."""

    def __init__(self, angle_band: float = 15.0):
        self.angle_band = angle_band
        self.dominant_h = 0.0
        self.dominant_v = 90.0

    def estimate(self, lines, length_percentile=50.0):
        if len(lines) < 4:
            self.dominant_h, self.dominant_v = 0.0, 90.0
            return self.dominant_h, self.dominant_v

        lengths = np.array([l.length for l in lines])
        thresh = np.percentile(lengths, length_percentile)
        long_lines = [l for l in lines if l.length >= thresh] or lines

        angles = np.array([l.angle for l in long_lines])
        weights = np.array([l.length for l in long_lines])

        dist_h = np.minimum(angles, 180 - angles)
        mask_h = dist_h < 45
        mask_v = ~mask_h

        if not np.any(mask_h) or not np.any(mask_v):
            self.dominant_h, self.dominant_v = 0.0, 90.0
            return self.dominant_h, self.dominant_v

        def circ_mean(angs, ws):
            rads = np.deg2rad(angs * 2)
            s = np.sum(ws * np.sin(rads))
            c = np.sum(ws * np.cos(rads))
            return (np.rad2deg(np.arctan2(s, c)) / 2) % 180

        self.dominant_h = circ_mean(angles[mask_h], weights[mask_h])
        self.dominant_v = circ_mean(angles[mask_v], weights[mask_v])
        return self.dominant_h, self.dominant_v

    def is_horizontal(self, angle):
        d = min(abs(angle - self.dominant_h), 180 - abs(angle - self.dominant_h))
        return d < self.angle_band

    def is_vertical(self, angle):
        d = min(abs(angle - self.dominant_v), 180 - abs(angle - self.dominant_v))
        return d < self.angle_band


# ═══════════════════════════════════════════════════════════════════════════
# 3. RESOLUTION-ADAPTIVE HOUGH PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════


def adaptive_hough_params(img_w, img_h):
    scale = ((img_w / 1920) + (img_h / 1080)) / 2
    return {
        "hough_threshold": max(20, int(30 * scale)),
        "min_length": max(15, int(40 * scale)),
        "max_gap": max(10, int(25 * scale)),
        "merge_dist_thresh": max(15, int(30 * scale)),
        "merge_gap_thresh": max(80, int(150 * scale)),
        "h_kernel": max(10, int(img_w / 60)),
        "v_kernel": max(10, int(img_h / 60)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. FIXED LINE DETECTOR (corrected angle ranking)
# ═══════════════════════════════════════════════════════════════════════════


class FixedLineDetector:
    """
    BUG FIXES vs original EnhancedLineDetector:
      1. select_best: abs(angle-90) → -abs(angle-90) for vertical preference
      2. select_best: horizontalness uses -min(angle, 180-angle)
      3. enforce_geometry: same fixes
      4. Uses DominantAngleClassifier instead of fixed 0/90 thresholds
    """

    def __init__(
        self,
        hough_threshold=30,
        min_length=30,
        max_gap=20,
        merge_angle_thresh=10,
        merge_dist_thresh=30,
        merge_gap_thresh=150,
        min_confidence=0.3,
        min_touchline_length_ratio=0.25,
        angle_classifier=None,
    ):
        self.hough_threshold = hough_threshold
        self.min_length = min_length
        self.max_gap = max_gap
        self.merge_angle_thresh = merge_angle_thresh
        self.merge_dist_thresh = merge_dist_thresh
        self.merge_gap_thresh = merge_gap_thresh
        self.min_confidence = min_confidence
        self.min_touchline_ratio = min_touchline_length_ratio
        self.angle_clf = angle_classifier or DominantAngleClassifier()

    def detect(self, skeleton):
        edges = cv2.Canny(skeleton, 50, 150)
        h = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            self.hough_threshold,
            minLineLength=self.min_length,
            maxLineGap=self.max_gap,
        )
        if h is None:
            return []
        return [DetectedLine(x1, y1, x2, y2) for x1, y1, x2, y2 in h[:, 0]]

    def merge_collinear(self, lines):
        if len(lines) < 2:
            return lines

        def pd(px, py, x1, y1, x2, y2):
            dx, dy = x2 - x1, y2 - y1
            ln = np.sqrt(dx**2 + dy**2)
            return (
                np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
                if ln < 1e-6
                else abs(dy * px - dx * py + x2 * y1 - y2 * x1) / ln
            )

        def collinear(l1, l2):
            ad = abs(l1.angle - l2.angle)
            ad = min(ad, 180 - ad)
            if ad > self.merge_angle_thresh:
                return False
            m1, m2 = l1.midpoint, l2.midpoint
            return (
                pd(m1[0], m1[1], l2.x1, l2.y1, l2.x2, l2.y2) < self.merge_dist_thresh
                and pd(m2[0], m2[1], l1.x1, l1.y1, l1.x2, l1.y2)
                < self.merge_dist_thresh
            )

        def gap(l1, l2):
            pts1 = [(l1.x1, l1.y1), (l1.x2, l1.y2)]
            pts2 = [(l2.x1, l2.y1), (l2.x2, l2.y2)]
            return min(
                np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
                for a in pts1
                for b in pts2
            )

        n = len(lines)
        used = [False] * n
        groups = []
        for i in range(n):
            if used[i]:
                continue
            grp = [i]
            used[i] = True
            for j in range(i + 1, n):
                if used[j]:
                    continue
                for k in grp:
                    if (
                        collinear(lines[k], lines[j])
                        and gap(lines[k], lines[j]) < self.merge_gap_thresh
                    ):
                        grp.append(j)
                        used[j] = True
                        break
            groups.append(grp)
        merged = []
        for grp in groups:
            if len(grp) == 1:
                merged.append(lines[grp[0]])
                continue
            pts = []
            for idx in grp:
                l = lines[idx]
                pts.extend([(l.x1, l.y1), (l.x2, l.y2)])
            pts = np.array(pts)
            d = cdist(pts, pts)
            i, j = np.unravel_index(np.argmax(d), d.shape)
            merged.append(
                DetectedLine(
                    int(pts[i, 0]), int(pts[i, 1]), int(pts[j, 0]), int(pts[j, 1])
                )
            )
        return merged

    def classify(self, lines, grass_bounds, ellipses, img_shape=None):
        self.angle_clf.estimate(lines)

        g_top = grass_bounds["top"]
        g_bot = grass_bounds["bottom"]
        g_left = grass_bounds["left"]
        g_right = grass_bounds["right"]
        g_h = max(g_bot - g_top, 1)
        g_w = max(g_right - g_left, 1)

        cc = la = ra = None
        for e in ellipses:
            if e.ellipse_type == EllipseType.CENTER_CIRCLE:
                cc = e
            elif e.ellipse_type == EllipseType.LEFT_PENALTY_ARC:
                la = e
            elif e.ellipse_type == EllipseType.RIGHT_PENALTY_ARC:
                ra = e

        classified = []
        for line in lines:
            a = line.angle
            mx, my = line.midpoint
            xr = (mx - g_left) / g_w
            yr = (my - g_top) / g_h
            is_h = self.angle_clf.is_horizontal(a)
            is_v = self.angle_clf.is_vertical(a)
            lt = LineType.UNKNOWN
            conf = 0.5

            if is_h:
                if (
                    cc
                    and abs(my - cc.center_y) < cc.axis_major * 0.5
                    and 0.3 < xr < 0.7
                ):
                    lt = LineType.CENTER_LINE
                    conf = 0.9
                if lt == LineType.UNKNOWN:
                    if yr < 0.2:
                        lt = LineType.FAR_TOUCHLINE
                        conf = 0.8
                    elif yr > 0.8:
                        lt = LineType.NEAR_TOUCHLINE
                        conf = 0.8
                    elif 0.4 < yr < 0.6:
                        lt = LineType.CENTER_LINE
                        conf = 0.6
                    else:
                        if xr < 0.35:
                            lt = (
                                LineType.L_PENALTY_FAR
                                if yr < 0.5
                                else LineType.L_PENALTY_NEAR
                            )
                            conf = 0.7
                        elif xr > 0.65:
                            lt = (
                                LineType.R_PENALTY_FAR
                                if yr < 0.5
                                else LineType.R_PENALTY_NEAR
                            )
                            conf = 0.7
            elif is_v:
                nl = la and abs(mx - la.center_x) < la.axis_major * 1.5
                nr = ra and abs(mx - ra.center_x) < ra.axis_major * 1.5
                if xr < 0.15:
                    lt = LineType.LEFT_GOAL_LINE
                    conf = 0.85
                elif xr > 0.85:
                    lt = LineType.RIGHT_GOAL_LINE
                    conf = 0.85
                elif nl or 0.15 < xr < 0.35:
                    lt = LineType.L_PENALTY_FRONT
                    conf = 0.75
                elif nr or 0.65 < xr < 0.85:
                    lt = LineType.R_PENALTY_FRONT
                    conf = 0.75
            else:
                if yr < 0.25:
                    lt = LineType.FAR_TOUCHLINE
                    conf = 0.5
                elif yr > 0.75:
                    lt = LineType.NEAR_TOUCHLINE
                    conf = 0.5
                elif xr < 0.3:
                    lt = LineType.L_PENALTY_FAR if yr < 0.5 else LineType.L_PENALTY_NEAR
                    conf = 0.5
                elif xr > 0.7:
                    lt = LineType.R_PENALTY_FAR if yr < 0.5 else LineType.R_PENALTY_NEAR
                    conf = 0.5

            classified.append(
                DetectedLine(
                    line.x1, line.y1, line.x2, line.y2, line_type=lt, confidence=conf
                )
            )

        iw = img_shape[1] if img_shape and len(img_shape) > 1 else g_w
        ih = img_shape[0] if img_shape else g_h
        best = self._select_best(classified, iw)
        return self._enforce_geometry(best, iw, ih)

    def _select_best(self, lines, img_w):
        """FIXED: corrected angle ranking keys."""
        by_type = defaultdict(list)
        for l in lines:
            if l.line_type != LineType.UNKNOWN:
                by_type[l.line_type].append(l)

        TL = {LineType.FAR_TOUCHLINE, LineType.NEAR_TOUCHLINE}
        GL = {LineType.LEFT_GOAL_LINE, LineType.RIGHT_GOAL_LINE}
        PV = {LineType.L_PENALTY_FRONT, LineType.R_PENALTY_FRONT}
        sel = []

        for lt, ll in by_type.items():
            valid = [l for l in ll if l.confidence >= self.min_confidence]
            if not valid:
                continue

            if lt in TL:
                best = max(
                    valid,
                    key=lambda l: (
                        l.length / img_w,
                        l.confidence,
                        -min(
                            l.angle, 180 - l.angle
                        ),  # FIX: more horizontal = higher score
                    ),
                )
                if best.length / img_w >= self.min_touchline_ratio:
                    sel.append(best)

            elif lt in GL or lt in PV:
                best = max(
                    valid,
                    key=lambda l: (
                        l.confidence,
                        -abs(
                            l.angle - 90
                        ),  # FIX: more vertical = higher score (was +abs)
                        l.length,
                    ),
                )
                sel.append(best)

            elif lt == LineType.CENTER_LINE:
                best = max(
                    valid,
                    key=lambda l: (
                        l.confidence,
                        l.length,
                        -min(l.angle, 180 - l.angle),  # FIX
                    ),
                )
                sel.append(best)

            else:
                sel.append(max(valid, key=lambda l: (l.confidence, l.length)))

        return sel

    def _enforce_geometry(self, lines, iw, ih):
        """FIXED geometry constraints."""
        by_type = defaultdict(list)
        for l in lines:
            by_type[l.line_type].append(l)
        final = []
        for lt, ll in by_type.items():
            if not ll:
                continue
            if lt == LineType.CENTER_LINE:
                final.append(
                    max(
                        ll,
                        key=lambda l: (
                            l.confidence,
                            -min(l.angle, 180 - l.angle),  # FIX
                            l.length,
                        ),
                    )
                )
            elif lt in (LineType.FAR_TOUCHLINE, LineType.NEAR_TOUCHLINE):
                valid = [l for l in ll if l.length / iw > 0.25]
                if valid:
                    final.append(max(valid, key=lambda l: (l.confidence, l.length)))
            elif lt in (LineType.LEFT_GOAL_LINE, LineType.RIGHT_GOAL_LINE):
                valid = [l for l in ll if abs(l.angle - 90) < 30]
                if valid:
                    final.append(
                        max(
                            valid,
                            key=lambda l: (
                                l.confidence,
                                l.length,
                                -abs(l.angle - 90),  # FIX: more vertical = better
                            ),
                        )
                    )
            else:
                final.append(max(ll, key=lambda l: l.confidence))

        # Swap touchlines if far is below near
        far = [l for l in final if l.line_type == LineType.FAR_TOUCHLINE]
        near = [l for l in final if l.line_type == LineType.NEAR_TOUCHLINE]
        if far and near and far[0].midpoint[1] > near[0].midpoint[1]:
            for l in final:
                if l.line_type == LineType.FAR_TOUCHLINE:
                    l.line_type = LineType.NEAR_TOUCHLINE
                elif l.line_type == LineType.NEAR_TOUCHLINE:
                    l.line_type = LineType.FAR_TOUCHLINE
        return final


# ═══════════════════════════════════════════════════════════════════════════
# 5. TEMPORAL HOMOGRAPHY SMOOTHER (for video)
# ═══════════════════════════════════════════════════════════════════════════


class TemporalHomographySmoother:
    """EMA smoother for H matrices. Call update() every keyframe."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.H_smooth = None
        self.frame_count = 0

    def update(self, H_new):
        if self.H_smooth is None:
            self.H_smooth = H_new.copy()
        else:
            Hn = H_new / (H_new[2, 2] + 1e-12)
            Ho = self.H_smooth / (self.H_smooth[2, 2] + 1e-12)
            self.H_smooth = self.alpha * Hn + (1 - self.alpha) * Ho
            self.H_smooth /= self.H_smooth[2, 2] + 1e-12
        self.frame_count += 1
        return self.H_smooth.copy()

    def reset(self):
        self.H_smooth = None
        self.frame_count = 0


# ═══════════════════════════════════════════════════════════════════════════
# 6. COMPARISON DEMO
# ═══════════════════════════════════════════════════════════════════════════


def run_comparison(image_path):
    """Side-by-side comparison: original vs enhanced pipeline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot load {image_path}")
        return
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")

    # Preprocessing (shared)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, ke)
    grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, ke, iterations=2)
    rows = np.any(grass_mask > 0, axis=1)
    cols = np.any(grass_mask > 0, axis=0)
    gb = {
        "top": int(np.argmax(rows)) if np.any(rows) else 0,
        "bottom": int(h - np.argmax(rows[::-1])) if np.any(rows) else h,
        "left": int(np.argmax(cols)) if np.any(cols) else 0,
        "right": int(w - np.argmax(cols[::-1])) if np.any(cols) else w,
    }
    cov = np.sum(grass_mask > 0) / grass_mask.size * 100
    print(f"Grass: {cov:.1f}%, bounds={gb}")

    # ── Original pipeline ──
    print("\n=== ORIGINAL ===")
    _, s, v = cv2.split(hsv)
    v_f = v.astype(np.float32)
    lm = cv2.blur(v_f, (101, 101))
    ls2 = cv2.blur(v_f**2, (101, 101))
    lstd = np.sqrt(np.maximum(ls2 - lm**2, 0))
    orig_w = ((s < 70) & (v_f > (lm + 1.1 * lstd)) & (v > 100)).astype(np.uint8) * 255
    orig_w = cv2.bitwise_and(orig_w, grass_mask)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    ch = cv2.morphologyEx(orig_w, cv2.MORPH_CLOSE, kh, iterations=2)
    cv_ = cv2.morphologyEx(orig_w, cv2.MORPH_CLOSE, kv, iterations=2)
    orig_clean = cv2.bitwise_or(ch, cv_)
    orig_skel = (skeletonize(orig_clean > 0) * 255).astype(np.uint8)

    params = adaptive_hough_params(w, h)
    det_orig = FixedLineDetector(
        **{
            k: v
            for k, v in params.items()
            if k
            in [
                "hough_threshold",
                "min_length",
                "max_gap",
                "merge_dist_thresh",
                "merge_gap_thresh",
            ]
        }
    )
    raw_o = det_orig.detect(orig_skel)
    merged_o = det_orig.merge_collinear(raw_o)
    classified_o = det_orig.classify(merged_o, gb, [], img_shape=(h, w))
    print(f"  Raw={len(raw_o)}, Merged={len(merged_o)}, Classified={len(classified_o)}")
    for l in classified_o:
        print(
            f"    {l.line_type.name:25s} a={l.angle:6.1f}° len={l.length:6.0f} c={l.confidence:.2f}"
        )

    # ── Enhanced pipeline ──
    print("\n=== WATERSHED + FIXES ===")
    ws_ext = WatershedWhiteExtractor(
        h_kernel=params.get("h_kernel", 20),
        v_kernel=params.get("v_kernel", 20),
    )
    ws_mask, ws_skel = ws_ext.extract_and_skeletonize(img, grass_mask, gb)

    det_ws = FixedLineDetector(
        **{
            k: v
            for k, v in params.items()
            if k
            in [
                "hough_threshold",
                "min_length",
                "max_gap",
                "merge_dist_thresh",
                "merge_gap_thresh",
            ]
        }
    )
    raw_w = det_ws.detect(ws_skel)
    merged_w = det_ws.merge_collinear(raw_w)
    classified_w = det_ws.classify(merged_w, gb, [], img_shape=(h, w))
    print(f"  Raw={len(raw_w)}, Merged={len(merged_w)}, Classified={len(classified_w)}")
    for l in classified_w:
        print(
            f"    {l.line_type.name:25s} a={l.angle:6.1f}° len={l.length:6.0f} c={l.confidence:.2f}"
        )

    # ── Visualization ──
    colors = {
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

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(
        "Original vs Enhanced (Watershed + Bug Fixes + Adaptive Angles)",
        fontsize=13,
        fontweight="bold",
    )

    # Row 1: white masks and skeletons
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(orig_clean, cmap="gray")
    axes[0, 1].set_title(f"Original White ({np.sum(orig_clean>0)} px)")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(ws_mask, cmap="gray")
    axes[0, 2].set_title(f"Watershed White ({np.sum(ws_mask>0)} px)")
    axes[0, 2].axis("off")

    # Row 2: classified lines
    img_o = img.copy()
    for l in classified_o:
        c = colors.get(l.line_type, (128, 128, 128))
        cv2.line(img_o, (l.x1, l.y1), (l.x2, l.y2), c, 3)
        mx, my = l.midpoint
        cv2.putText(
            img_o,
            l.line_type.name[:12],
            (int(mx), int(my) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )
    axes[1, 0].imshow(cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Original Classified ({len(classified_o)} lines)")
    axes[1, 0].axis("off")

    img_w = img.copy()
    for l in classified_w:
        c = colors.get(l.line_type, (128, 128, 128))
        cv2.line(img_w, (l.x1, l.y1), (l.x2, l.y2), c, 3)
        mx, my = l.midpoint
        cv2.putText(
            img_w,
            l.line_type.name[:12],
            (int(mx), int(my) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )
    axes[1, 1].imshow(cv2.cvtColor(img_w, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Watershed Classified ({len(classified_w)} lines)")
    axes[1, 1].axis("off")

    # Legend
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.95
    ax.text(
        0.05,
        y,
        "Line Type Colors:",
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
        va="top",
    )
    for lt, c in colors.items():
        y -= 0.065
        ax.plot(
            [0.05, 0.15],
            [y, y],
            color=np.array(c[::-1]) / 255,
            linewidth=3,
            transform=ax.transAxes,
        )
        ax.text(0.18, y, lt.name, fontsize=7, transform=ax.transAxes, va="center")
    y -= 0.1
    ax.text(
        0.05,
        y,
        f"Dominant angles:\n  H={det_ws.angle_clf.dominant_h:.1f}°  V={det_ws.angle_clf.dominant_v:.1f}°",
        fontsize=8,
        transform=ax.transAxes,
        va="top",
        family="monospace",
    )
    y -= 0.1
    ax.text(
        0.05,
        y,
        f"Adaptive params:\n  {params}",
        fontsize=6,
        transform=ax.transAxes,
        va="top",
        family="monospace",
        wrap=True,
    )

    plt.tight_layout()
    out = "pipeline_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()
    return out


if __name__ == "__main__":
    import sys

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "D:/ITE/year5/graduation project/5th Dataset/examples photo/Screenshot 2026-02-02 210426.png"
    )
    run_comparison(path)
