"""
TactixAI: Automatic Homography Calibration Module

This module provides automatic homography estimation from soccer footage
using classical computer vision techniques (no ML model required).

Pipeline:
1. Grass segmentation (adaptive green detection)
2. White line detection on grass mask
3. Hough Line Transform for straight lines
4. Hough Circle Transform / Ellipse fitting for center circle
5. Keypoint extraction from line intersections
6. Homography computation via RANSAC

Designed for TACTICAL CAMERAS:
- Works with partial pitch visibility (center + touchlines only)
- Handles goal lines appearing later in the video
- Smooth homography updates for slow camera motion

Author: TactixAI
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum


# =============================================================================
# PITCH TEMPLATE
# =============================================================================


class PitchTemplate:
    """
    Standard FIFA pitch template (105m x 68m).
    Provides keypoints in pitch coordinates for homography matching.

    Coordinate System:
        Origin (0, 0) = bottom-left corner
        X-axis: 0 to 105m (along the length)
        Y-axis: 0 to 68m (along the width)

        (0,68) -------- (52.5,68) -------- (105,68)
          |                |                  |
          |     LEFT       |      RIGHT       |
          |                |                  |
        (0,34) -------- (52.5,34) -------- (105,34)  <- center line
          |                |                  |
          |                |                  |
          |                |                  |
        (0,0) --------- (52.5,0) --------- (105,0)
    """

    def __init__(self, length: float = 105.0, width: float = 68.0):
        self.length = float(length)  # x-axis (meters)
        self.width = float(width)  # y-axis (meters)

        # Key reference points in pitch coordinates (x, y)
        self.keypoints = {
            # Center features
            "center_spot": (52.5, 34.0),
            "center_circle_center": (52.5, 34.0),
            "center_circle_radius": 9.15,
            # Halfway line intersections with touchlines
            "halfway_top": (52.5, 68.0),  # Top touchline (y = 68)
            "halfway_bottom": (52.5, 0.0),  # Bottom touchline (y = 0)
            # Goal lines (x = 0 and x = 105)
            "goal_line_left_top": (0.0, 68.0),
            "goal_line_left_bottom": (0.0, 0.0),
            "goal_line_right_top": (105.0, 68.0),
            "goal_line_right_bottom": (105.0, 0.0),
            # Left penalty area (16.5m from goal line, extends 40.32m)
            "penalty_left_top": (16.5, 54.16),
            "penalty_left_bottom": (16.5, 13.84),
            "penalty_left_top_corner": (0.0, 54.16),
            "penalty_left_bottom_corner": (0.0, 13.84),
            # Right penalty area
            "penalty_right_top": (88.5, 54.16),
            "penalty_right_bottom": (88.5, 13.84),
            "penalty_right_top_corner": (105.0, 54.16),
            "penalty_right_bottom_corner": (105.0, 13.84),
            # Left goal area (5.5m from goal line, extends 18.32m)
            "goal_area_left_top": (5.5, 43.16),
            "goal_area_left_bottom": (5.5, 24.84),
            # Right goal area
            "goal_area_right_top": (99.5, 43.16),
            "goal_area_right_bottom": (99.5, 24.84),
            # Penalty spots (11m from goal line)
            "penalty_spot_left": (11.0, 34.0),
            "penalty_spot_right": (94.0, 34.0),
            # Corner arcs (1m radius)
            "corner_bottom_left": (0.0, 0.0),
            "corner_bottom_right": (105.0, 0.0),
            "corner_top_left": (0.0, 68.0),
            "corner_top_right": (105.0, 68.0),
        }

    def get_center_circle_points(self, num_points: int = 32) -> np.ndarray:
        """Sample points on the center circle for matching."""
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        cx, cy = self.keypoints["center_circle_center"]
        r = self.keypoints["center_circle_radius"]

        points = np.array(
            [[cx + r * np.cos(a), cy + r * np.sin(a)] for a in angles], dtype=np.float32
        )
        return points

    def get_halfway_line_points(self, num_points: int = 16) -> np.ndarray:
        """Sample points on the halfway line."""
        y_vals = np.linspace(0, self.width, num_points)
        x_vals = np.full_like(y_vals, 52.5)
        return np.column_stack([x_vals, y_vals]).astype(np.float32)

    def get_pitch_outline(self) -> np.ndarray:
        """Get the four corners of the pitch for visualization."""
        return np.array(
            [
                [0.0, 0.0],
                [self.length, 0.0],
                [self.length, self.width],
                [0.0, self.width],
            ],
            dtype=np.float32,
        )

    def get_all_lines(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get all pitch lines as (start, end) point pairs for visualization."""
        lines = [
            # Pitch outline
            (np.array([0, 0]), np.array([105, 0])),  # Bottom touchline
            (np.array([105, 0]), np.array([105, 68])),  # Right goal line
            (np.array([105, 68]), np.array([0, 68])),  # Top touchline
            (np.array([0, 68]), np.array([0, 0])),  # Left goal line
            # Halfway line
            (np.array([52.5, 0]), np.array([52.5, 68])),
            # Left penalty area
            (np.array([0, 13.84]), np.array([16.5, 13.84])),
            (np.array([16.5, 13.84]), np.array([16.5, 54.16])),
            (np.array([16.5, 54.16]), np.array([0, 54.16])),
            # Right penalty area
            (np.array([105, 13.84]), np.array([88.5, 13.84])),
            (np.array([88.5, 13.84]), np.array([88.5, 54.16])),
            (np.array([88.5, 54.16]), np.array([105, 54.16])),
            # Left goal area
            (np.array([0, 24.84]), np.array([5.5, 24.84])),
            (np.array([5.5, 24.84]), np.array([5.5, 43.16])),
            (np.array([5.5, 43.16]), np.array([0, 43.16])),
            # Right goal area
            (np.array([105, 24.84]), np.array([99.5, 24.84])),
            (np.array([99.5, 24.84]), np.array([99.5, 43.16])),
            (np.array([99.5, 43.16]), np.array([105, 43.16])),
        ]
        return lines


# =============================================================================
# LINE REPRESENTATION
# =============================================================================


@dataclass
class Line:
    """
    Represents a line in various forms.

    Stores:
    - Endpoints (x1, y1, x2, y2) in pixel coordinates
    - Angle in degrees
    - Length in pixels
    - Normal form coefficients (ax + by + c = 0)
    """

    # Endpoints (pixel coordinates)
    x1: float
    y1: float
    x2: float
    y2: float

    # Computed properties
    angle: float = field(init=False)
    length: float = field(init=False)
    # Normal form: ax + by + c = 0
    a: float = field(init=False)
    b: float = field(init=False)
    c: float = field(init=False)

    def __post_init__(self):
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        self.length = np.sqrt(dx**2 + dy**2)
        self.angle = np.arctan2(dy, dx) * 180 / np.pi  # degrees

        # Convert to normal form: ax + by + c = 0
        self.a = dy
        self.b = -dx
        self.c = dx * self.y1 - dy * self.x1

        # Normalize coefficients
        norm = np.sqrt(self.a**2 + self.b**2)
        if norm > 1e-6:
            self.a /= norm
            self.b /= norm
            self.c /= norm

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def endpoints(self) -> np.ndarray:
        return np.array([[self.x1, self.y1], [self.x2, self.y2]])

    def is_vertical(self, threshold: float = 20) -> bool:
        """Check if line is near-vertical (within threshold degrees of 90)."""
        abs_angle = abs(self.angle)
        return (90 - threshold) <= abs_angle <= (90 + threshold)

    def is_horizontal(self, threshold: float = 20) -> bool:
        """Check if line is near-horizontal (within threshold degrees of 0 or 180)."""
        abs_angle = abs(self.angle)
        return abs_angle <= threshold or abs_angle >= (180 - threshold)


# =============================================================================
# PITCH LINE DETECTOR
# =============================================================================


class PitchLineDetector:
    """
    Detects pitch markings (lines and circles) using classical computer vision.

    Pipeline:
    1. Segment grass (green) to create field mask
    2. Detect white lines within the field mask
    3. Run Hough Line Transform for straight lines
    4. Run Hough Circle Transform for center circle
    5. Extract keypoints from intersections

    Designed for tactical cameras where:
    - Center circle + touchlines visible from frame 1
    - Goal lines may appear later
    - Camera motion is slow and smooth
    """

    def __init__(
        self,
        # Grass detection params
        hue_tolerance: int = 15,
        min_saturation: int = 30,
        min_value: int = 30,
        # Line detection params
        white_sat_max: int = 50,
        white_val_min: int = 180,
        # Hough params
        hough_threshold: int = 80,
        min_line_length: int = 100,
        max_line_gap: int = 20,
        # Circle params
        circle_dp: float = 1.5,
        circle_min_dist: int = 100,
        circle_param1: int = 100,
        circle_param2: int = 25,
        min_circle_radius: int = 30,
        max_circle_radius: int = 200,
    ):
        # Grass segmentation
        self.hue_tolerance = hue_tolerance
        self.min_saturation = min_saturation
        self.min_value = min_value

        # White line detection
        self.white_sat_max = white_sat_max
        self.white_val_min = white_val_min

        # Hough line params
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

        # Hough circle params
        self.circle_dp = circle_dp
        self.circle_min_dist = circle_min_dist
        self.circle_param1 = circle_param1
        self.circle_param2 = circle_param2
        self.min_circle_radius = min_circle_radius
        self.max_circle_radius = max_circle_radius

        # Cached results
        self.last_mask_field = None
        self.last_mask_lines = None

    def detect(self, frame: np.ndarray) -> Dict:
        """
        Main detection pipeline.

        Returns:
            Dict with keys:
                - mask_field: Binary mask of the playing field
                - mask_lines: Binary mask of white lines
                - lines: List of Line objects
                - circle: (cx, cy, radius) or None
                - ellipse: ((cx, cy), (ma, MA), angle) or None
                - keypoints: Dict mapping keypoint names to (x, y) image coords
        """
        # Step 1: Segment the grass/field
        mask_field = self._segment_grass(frame)
        self.last_mask_field = mask_field

        # Step 2: Detect white lines on the field
        mask_lines = self._detect_white_lines(frame, mask_field)
        self.last_mask_lines = mask_lines

        # Step 3: Extract straight lines using Hough
        lines = self._extract_lines(mask_lines)

        # Step 4: Cluster lines into vertical/horizontal groups
        vertical_lines, horizontal_lines = self._cluster_lines(lines)

        # Step 5: Find the halfway line first (needed to guide circle detection)
        halfway_line = self._find_halfway_line(vertical_lines, frame.shape[1], None)

        # Step 6: Detect center circle (pass halfway_line to help locate it)
        circle, ellipse = self._detect_circle(mask_lines, frame.shape, halfway_line)

        # Step 7: Find keypoints from intersections
        keypoints = self._find_keypoints(
            vertical_lines, horizontal_lines, circle, ellipse, mask_field
        )

        return {
            "mask_field": mask_field,
            "mask_lines": mask_lines,
            "lines": lines,
            "vertical_lines": vertical_lines,
            "horizontal_lines": horizontal_lines,
            "circle": circle,
            "ellipse": ellipse,
            "keypoints": keypoints,
        }

    def _segment_grass(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment the playing field using adaptive green detection.
        Returns a binary mask where 255 = grass pixels.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Find dominant green hue using histogram
        hist = cv2.calcHist([h], [0], None, [180], [0, 180])

        # Look for peak in green range (35-85 typically)
        green_range_hist = hist[35:85]
        peak_offset = np.argmax(green_range_hist)
        peak_hue = 35 + peak_offset

        # Define bounds around the peak
        h_low = max(peak_hue - self.hue_tolerance, 0)
        h_high = min(peak_hue + self.hue_tolerance, 179)

        # Create green mask
        lower = np.array(
            [int(h_low), int(self.min_saturation), int(self.min_value)], dtype=np.uint8
        )
        upper = np.array([int(h_high), 255, 255], dtype=np.uint8)
        mask_green = cv2.inRange(hsv, lower, upper)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_field = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        mask_field = cv2.morphologyEx(mask_field, cv2.MORPH_OPEN, kernel)

        # Keep only the largest connected component (the field)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_field)

        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_field = np.where(labels == largest_label, 255, 0).astype(np.uint8)

        return mask_field

    def _detect_white_lines(
        self, frame: np.ndarray, mask_field: np.ndarray
    ) -> np.ndarray:
        """
        Detect white pitch markings within the field mask.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply field mask to HSV
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_field)

        # White detection: high value, low saturation
        lower_white = np.array([0, 0, self.white_val_min])
        upper_white = np.array([180, self.white_sat_max, 255])
        mask_white = cv2.inRange(masked_hsv, lower_white, upper_white)

        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_lines = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
        mask_lines = cv2.morphologyEx(mask_lines, cv2.MORPH_OPEN, kernel)

        return mask_lines

    def _extract_lines(self, mask_lines: np.ndarray) -> List[Line]:
        """
        Extract straight lines using Canny + Hough Line Transform.
        """
        edges = cv2.Canny(mask_lines, 50, 150, apertureSize=3)

        hough_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        lines = []
        if hough_lines is not None:
            for line in hough_lines:
                x1, y1, x2, y2 = line[0]
                lines.append(Line(x1, y1, x2, y2))

        return lines

    def _cluster_lines(
        self, lines: List[Line], angle_threshold: float = 25
    ) -> Tuple[List[Line], List[Line]]:
        """
        Cluster lines into vertical and horizontal groups based on angle.
        """
        vertical = []
        horizontal = []

        for line in lines:
            if line.is_vertical(angle_threshold):
                vertical.append(line)
            elif line.is_horizontal(angle_threshold):
                horizontal.append(line)

        return vertical, horizontal

    def _detect_circle(
        self, mask_lines: np.ndarray, frame_shape: Tuple, halfway_line: "Line" = None
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple]]:
        """
        Detect the center circle using Hough Circle Transform.
        Falls back to ellipse fitting if circle detection fails.

        Args:
            mask_lines: Binary mask of detected white lines
            frame_shape: Shape of the frame (h, w, ...)
            halfway_line: If provided, only look for circles near this line

        Returns:
            circle: (cx, cy, radius) or None
            ellipse: ((cx, cy), (minor_axis, major_axis), angle) or None
        """
        h, w = frame_shape[:2]
        circle = None
        ellipse = None

        # Minimum radius based on image size (center circle should be at least ~3% of image height)
        min_r = max(self.min_circle_radius, int(h * 0.03))
        # Maximum radius (center circle should be at most ~25% of image height)
        max_r = min(self.max_circle_radius, int(h * 0.25))

        # Try Hough Circle Transform
        circles = cv2.HoughCircles(
            mask_lines,
            cv2.HOUGH_GRADIENT,
            dp=self.circle_dp,
            minDist=self.circle_min_dist,
            param1=self.circle_param1,
            param2=self.circle_param2,
            minRadius=min_r,
            maxRadius=max_r,
        )

        if circles is not None and len(circles[0]) > 0:
            # Filter circles: prefer ones near image center and near halfway line if known
            best_circle = None
            best_score = float("inf")

            for c in circles[0]:
                cx, cy, r = c

                # Score based on distance from image center
                img_center_x, img_center_y = w / 2, h / 2
                dist_to_img_center = np.sqrt(
                    (cx - img_center_x) ** 2 + (cy - img_center_y) ** 2
                )

                # Prefer circles near the halfway line if known
                if halfway_line is not None:
                    # Distance from circle center to halfway line
                    # Line goes from (x1, y1) to (x2, y2)
                    x1, y1 = halfway_line.x1, halfway_line.y1
                    x2, y2 = halfway_line.x2, halfway_line.y2

                    # Point-to-line distance
                    line_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if line_len > 0:
                        dist_to_line = (
                            abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1)
                            / line_len
                        )
                    else:
                        dist_to_line = float("inf")

                    # Heavy penalty for circles far from halfway line
                    score = dist_to_line * 3 + dist_to_img_center * 0.5
                else:
                    score = dist_to_img_center

                # Prefer circles with reasonable radius (not too small, not too big)
                expected_r = h * 0.08  # Expect ~8% of image height
                radius_penalty = abs(r - expected_r) / expected_r * 100
                score += radius_penalty

                if score < best_score:
                    best_score = score
                    best_circle = (float(cx), float(cy), float(r))

            circle = best_circle

        # Also try ellipse fitting on the region around the halfway line or image center
        if halfway_line is not None:
            # Focus on region around halfway line
            mid_x = int(halfway_line.midpoint[0])
            margin_x = w // 4
            x_start = max(0, mid_x - margin_x)
            x_end = min(w, mid_x + margin_x)
            y_start, y_end = 0, h  # Full height since line is vertical
        else:
            # Fall back to central region
            x_start, x_end = w // 3, 2 * w // 3
            y_start, y_end = h // 4, 3 * h // 4

        center_mask = np.zeros_like(mask_lines)
        center_mask[y_start:y_end, x_start:x_end] = mask_lines[
            y_start:y_end, x_start:x_end
        ]

        contours, _ = cv2.findContours(
            center_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                if len(contour) >= 5:
                    try:
                        fitted_ellipse = cv2.fitEllipse(contour)
                        (ecx, ecy), (ma, MA), angle = fitted_ellipse

                        # Check aspect ratio is reasonable for a circle/ellipse
                        aspect_ratio = ma / (MA + 1e-6)
                        if 0.3 < aspect_ratio < 3.0:
                            # Check size is reasonable
                            avg_axis = (ma + MA) / 2
                            if min_r < avg_axis / 2 < max_r:
                                ellipse = fitted_ellipse

                                if circle is None:
                                    r = avg_axis / 4
                                    circle = (float(ecx), float(ecy), float(r))
                                break
                    except cv2.error:
                        continue

        return circle, ellipse

    def _find_keypoints(
        self,
        vertical_lines: List[Line],
        horizontal_lines: List[Line],
        circle: Optional[Tuple[float, float, float]],
        ellipse: Optional[Tuple],
        mask_field: np.ndarray,
    ) -> Dict[str, any]:
        """
        Extract keypoints from detected lines and circle.

        For tactical cameras with perspective:
        - Touchlines may NOT be horizontal (they converge to vanishing point)
        - We find touchline intersections by extending the halfway line to field edges

        CRITICAL: The center circle center MUST be on the halfway line.
        If it's not, the circle detection is wrong and we should reject it.
        """
        keypoints = {}
        h, w = mask_field.shape[:2]

        # Minimum valid circle radius (to filter out noise)
        MIN_CIRCLE_RADIUS = 30

        # 1. Find the halfway line FIRST (we need this to validate the circle)
        halfway_line = self._find_halfway_line(vertical_lines, w, circle)

        if halfway_line is not None:
            keypoints["halfway_line"] = halfway_line.midpoint
            halfway_x = halfway_line.midpoint[0]

            # 2. Find touchline endpoints (top and bottom of halfway line)
            # METHOD A: Try traditional touchline detection first
            top_touchline, bottom_touchline = self._find_touchlines(horizontal_lines, h)

            if top_touchline is not None:
                intersection = self._line_intersection(halfway_line, top_touchline)
                if intersection and self._point_in_bounds(intersection, w, h):
                    keypoints["halfway_top"] = intersection

            if bottom_touchline is not None:
                intersection = self._line_intersection(halfway_line, bottom_touchline)
                if intersection and self._point_in_bounds(intersection, w, h):
                    keypoints["halfway_bottom"] = intersection

            # METHOD B: If touchline intersections not found, use field mask boundary
            if "halfway_top" not in keypoints or "halfway_bottom" not in keypoints:
                top_pt, bottom_pt = self._find_halfway_line_endpoints(
                    halfway_line, mask_field
                )
                if top_pt is not None and "halfway_top" not in keypoints:
                    keypoints["halfway_top"] = top_pt
                if bottom_pt is not None and "halfway_bottom" not in keypoints:
                    keypoints["halfway_bottom"] = bottom_pt

            # 3. Now validate and process the circle
            # The center circle center MUST be on (or very close to) the halfway line
            valid_circle = False

            if circle is not None:
                cx, cy, r = circle
                if (
                    all(
                        isinstance(v, (int, float, np.floating, np.integer))
                        for v in [cx, cy, r]
                    )
                    and r >= MIN_CIRCLE_RADIUS
                ):

                    # Check if circle center is on the halfway line (within tolerance)
                    # Tolerance: 20% of circle radius or 30px, whichever is larger
                    tolerance = max(r * 0.2, 30)
                    distance_to_halfway = abs(cx - halfway_x)

                    if distance_to_halfway <= tolerance:
                        # Also check circle is between top and bottom touchlines
                        if "halfway_top" in keypoints and "halfway_bottom" in keypoints:
                            top_y = keypoints["halfway_top"][1]
                            bottom_y = keypoints["halfway_bottom"][1]
                            # Circle center should be between top and bottom (with margin)
                            if top_y < cy < bottom_y:
                                keypoints["center_spot"] = (float(cx), float(cy))
                                keypoints["center_circle_radius"] = float(r)
                                valid_circle = True
                            else:
                                print(
                                    f"[Homography] Rejecting circle: center Y={cy:.0f} not between top={top_y:.0f} and bottom={bottom_y:.0f}"
                                )
                        else:
                            # No touchlines to validate against, accept if on halfway line
                            keypoints["center_spot"] = (float(cx), float(cy))
                            keypoints["center_circle_radius"] = float(r)
                            valid_circle = True
                    else:
                        print(
                            f"[Homography] Rejecting circle: center X={cx:.0f} is {distance_to_halfway:.0f}px from halfway line at X={halfway_x:.0f}"
                        )
                else:
                    if circle is not None:
                        _, _, r = circle
                        print(
                            f"[Homography] Rejecting circle: radius={r:.0f}px too small (min={MIN_CIRCLE_RADIUS})"
                        )

            # 4. If no valid circle but we have halfway_top and halfway_bottom,
            # estimate center_spot from halfway line midpoint
            if (
                not valid_circle
                and "halfway_top" in keypoints
                and "halfway_bottom" in keypoints
            ):
                top_x, top_y = keypoints["halfway_top"]
                bottom_x, bottom_y = keypoints["halfway_bottom"]

                # Center spot is at midpoint of halfway line
                center_x = (top_x + bottom_x) / 2
                center_y = (top_y + bottom_y) / 2

                keypoints["center_spot"] = (float(center_x), float(center_y))
                # Estimate radius from the visible portion of the pitch
                # The center circle is 9.15m radius, halfway line is 68m
                # So circle is about 27% of the halfway line length
                halfway_length = abs(bottom_y - top_y)
                estimated_radius = (
                    halfway_length * 0.27 / 2
                )  # Divide by 2 because it's radius not diameter
                if estimated_radius > MIN_CIRCLE_RADIUS:
                    keypoints["center_circle_radius"] = float(estimated_radius)
                    print(
                        f"[Homography] Estimated center_spot from halfway line midpoint: ({center_x:.0f}, {center_y:.0f}), radius≈{estimated_radius:.0f}px"
                    )

            # Store ellipse if available and valid
            if ellipse is not None and valid_circle:
                keypoints["ellipse"] = ellipse

        else:
            # No halfway line found - try to use circle alone (less reliable)
            if circle is not None:
                cx, cy, r = circle
                if (
                    all(
                        isinstance(v, (int, float, np.floating, np.integer))
                        for v in [cx, cy, r]
                    )
                    and r >= MIN_CIRCLE_RADIUS
                ):
                    keypoints["center_spot"] = (float(cx), float(cy))
                    keypoints["center_circle_radius"] = float(r)
                    if ellipse is not None:
                        keypoints["ellipse"] = ellipse
                    print(
                        f"[Homography] Using circle without halfway line validation (less reliable)"
                    )

        # 5. Find penalty area lines (if visible)
        penalty_keypoints = self._find_penalty_keypoints(
            vertical_lines, horizontal_lines, w, h
        )
        keypoints.update(penalty_keypoints)

        return keypoints

    def _find_halfway_line_endpoints(
        self,
        halfway_line: Line,
        mask_field: np.ndarray,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """
        Find where the halfway line intersects the field boundary.

        This works for perspective views where touchlines aren't horizontal.
        We extend the halfway line and find where it exits the grass mask.

        Returns:
            (top_point, bottom_point) - intersection points with field boundary
            top_point has smaller y (top of image)
            bottom_point has larger y (bottom of image)
        """
        h, w = mask_field.shape[:2]

        # Get line endpoints
        x1, y1 = halfway_line.x1, halfway_line.y1
        x2, y2 = halfway_line.x2, halfway_line.y2

        # Ensure consistent direction: from top to bottom (y1 < y2)
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        # Direction vector (from top to bottom)
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length < 1:
            return None, None

        # Normalize direction
        dx /= length
        dy /= length

        # Start from line center
        mid_x, mid_y = halfway_line.midpoint

        # Search UPWARD (toward top of image, decreasing y)
        top_point = None
        for step in range(1, max(h, w)):
            # Move opposite to direction (toward smaller y)
            test_x = int(mid_x - dx * step)
            test_y = int(mid_y - dy * step)

            # Check image bounds
            if test_y < 0 or test_y >= h or test_x < 0 or test_x >= w:
                # Hit image boundary - use last valid point
                last_x = mid_x - dx * (step - 1)
                last_y = mid_y - dy * (step - 1)
                if 0 <= last_x < w and 0 <= last_y < h:
                    top_point = (float(last_x), float(last_y))
                break

            # Check if we left the field mask
            if mask_field[test_y, test_x] == 0:
                # Found field edge - use last valid point inside field
                last_x = mid_x - dx * (step - 1)
                last_y = mid_y - dy * (step - 1)
                top_point = (float(last_x), float(last_y))
                break

        # Search DOWNWARD (toward bottom of image, increasing y)
        bottom_point = None
        for step in range(1, max(h, w)):
            # Move in direction (toward larger y)
            test_x = int(mid_x + dx * step)
            test_y = int(mid_y + dy * step)

            # Check image bounds
            if test_y < 0 or test_y >= h or test_x < 0 or test_x >= w:
                # Hit image boundary - use last valid point
                last_x = mid_x + dx * (step - 1)
                last_y = mid_y + dy * (step - 1)
                if 0 <= last_x < w and 0 <= last_y < h:
                    bottom_point = (float(last_x), float(last_y))
                break

            # Check if we left the field mask
            if mask_field[test_y, test_x] == 0:
                # Found field edge - use last valid point inside field
                last_x = mid_x + dx * (step - 1)
                last_y = mid_y + dy * (step - 1)
                bottom_point = (float(last_x), float(last_y))
                break

        # Double-check: top should have smaller y than bottom
        if top_point and bottom_point:
            if top_point[1] > bottom_point[1]:
                top_point, bottom_point = bottom_point, top_point

        return top_point, bottom_point

    def _find_halfway_line(
        self,
        vertical_lines: List[Line],
        image_width: int,
        circle: Optional[Tuple[float, float, float]],
    ) -> Optional[Line]:
        """
        Find the halfway line - longest vertical line near center.
        """
        if not vertical_lines:
            return None

        candidates = []
        center_x = image_width / 2

        for line in vertical_lines:
            mid_x, mid_y = line.midpoint
            center_dist = abs(mid_x - center_x)

            circle_bonus = 0
            if circle is not None:
                cx, cy, r = circle
                if abs(mid_x - cx) < r * 2:
                    circle_bonus = 100

            score = line.length - center_dist * 0.5 + circle_bonus
            candidates.append((score, line))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    def _find_touchlines(
        self, horizontal_lines: List[Line], image_height: int
    ) -> Tuple[Optional[Line], Optional[Line]]:
        """
        Find the top and bottom touchlines.

        FIXED: In OpenCV image coordinates:
        - y = 0 is the TOP of the image
        - y = image_height is the BOTTOM of the image

        So:
        - TOP touchline (pitch y=68) has SMALL image y values
        - BOTTOM touchline (pitch y=0) has LARGE image y values

        For tactical cameras, touchlines may be near frame edges.
        We use more generous thresholds.
        """
        if not horizontal_lines:
            return None, None

        sorted_lines = sorted(horizontal_lines, key=lambda l: l.midpoint[1])

        # Top touchline: upper 50% of image (more generous than 40%)
        # Filter for substantial lines only
        top_candidates = [
            l
            for l in sorted_lines
            if l.midpoint[1] < image_height * 0.5 and l.length > 50
        ]

        # Bottom touchline: lower 50% of image (more generous than 60%)
        bottom_candidates = [
            l
            for l in sorted_lines
            if l.midpoint[1] > image_height * 0.5 and l.length > 50
        ]

        # Pick the longest in each region, preferring lines closer to edges
        if top_candidates:
            # Score by length and how close to top edge
            top_touchline = min(
                top_candidates, key=lambda l: l.midpoint[1] - l.length * 0.5
            )
        else:
            top_touchline = None

        if bottom_candidates:
            # Score by length and how close to bottom edge
            bottom_touchline = max(
                bottom_candidates, key=lambda l: l.midpoint[1] + l.length * 0.5
            )
        else:
            bottom_touchline = None

        return top_touchline, bottom_touchline

    def _find_penalty_keypoints(
        self,
        vertical_lines: List[Line],
        horizontal_lines: List[Line],
        image_width: int,
        image_height: int,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Find penalty area corners if visible.
        """
        keypoints = {}

        left_verticals = [l for l in vertical_lines if l.midpoint[0] < image_width / 3]
        if left_verticals:
            penalty_line = max(left_verticals, key=lambda l: l.length)
            keypoints["penalty_left_line"] = penalty_line.midpoint

        right_verticals = [
            l for l in vertical_lines if l.midpoint[0] > 2 * image_width / 3
        ]
        if right_verticals:
            penalty_line = max(right_verticals, key=lambda l: l.length)
            keypoints["penalty_right_line"] = penalty_line.midpoint

        return keypoints

    @staticmethod
    def _line_intersection(line1: Line, line2: Line) -> Optional[Tuple[float, float]]:
        """Compute intersection of two lines in normal form."""
        d = line1.a * line2.b - line2.a * line1.b

        if abs(d) < 1e-6:
            return None

        x = (line1.b * line2.c - line2.b * line1.c) / d
        y = (line1.c * line2.a - line2.c * line1.a) / d

        return (x, y)

    @staticmethod
    def _point_in_bounds(
        point: Tuple[float, float], width: int, height: int, margin: int = 50
    ) -> bool:
        """Check if a point is within image bounds."""
        x, y = point
        return -margin < x < width + margin and -margin < y < height + margin


# =============================================================================
# HOMOGRAPHY ESTIMATOR
# =============================================================================


class HomographyEstimator:
    """
    Estimates homography matrix between image and pitch coordinates.
    """

    def __init__(
        self,
        pitch_template: Optional[PitchTemplate] = None,
        ransac_threshold: float = 5.0,
        min_keypoints: int = 4,
        smoothing_alpha: float = 0.3,
    ):
        self.pitch = pitch_template or PitchTemplate()
        self.ransac_threshold = ransac_threshold
        self.min_keypoints = min_keypoints
        self.smoothing_alpha = smoothing_alpha

        self.H_pitch_to_image: Optional[np.ndarray] = None
        self.H_image_to_pitch: Optional[np.ndarray] = None

        self.is_calibrated = False
        self.calibration_error: float = float("inf")
        self.num_inliers: int = 0

    def calibrate(self, detected_keypoints: Dict[str, any]) -> bool:
        """
        Calibrate homography from detected image keypoints.

        Returns:
            True if calibration successful, False otherwise
        """
        img_pts = []
        pitch_pts = []

        # Map detected keypoints to pitch coordinates
        keypoint_mapping = {
            "center_spot": self.pitch.keypoints["center_spot"],
            "halfway_top": self.pitch.keypoints["halfway_top"],
            "halfway_bottom": self.pitch.keypoints["halfway_bottom"],
        }

        # Debug: show what keypoints are available
        available_kps = [k for k in detected_keypoints.keys() if k not in ("ellipse",)]
        print(f"[Homography DEBUG] Available keypoints: {available_kps}")

        # Add basic keypoints
        for name, pitch_coord in keypoint_mapping.items():
            if name in detected_keypoints:
                kp = detected_keypoints[name]
                if isinstance(kp, (tuple, list)) and len(kp) >= 2:
                    img_pts.append((float(kp[0]), float(kp[1])))
                    pitch_pts.append(pitch_coord)
                    print(
                        f"[Homography DEBUG] {name}: image=({kp[0]:.0f}, {kp[1]:.0f}) → pitch={pitch_coord}"
                    )

        # Only add center circle points if we have a VALID detected circle
        # CRITICAL: Must validate that circle center is geometrically consistent with halfway line
        has_real_circle = False

        if (
            "center_spot" in detected_keypoints
            and "center_circle_radius" in detected_keypoints
            and "halfway_top" in detected_keypoints
            and "halfway_bottom" in detected_keypoints
        ):

            img_center = detected_keypoints.get("center_spot")
            img_radius = detected_keypoints.get("center_circle_radius")

            if (
                img_center is not None
                and isinstance(img_center, (tuple, list))
                and len(img_center) >= 2
                and img_radius is not None
                and isinstance(img_radius, (int, float, np.floating, np.integer))
            ):

                cx, cy = float(img_center[0]), float(img_center[1])
                r = float(img_radius)

                # Get halfway line endpoints
                top = detected_keypoints["halfway_top"]
                bottom = detected_keypoints["halfway_bottom"]
                top_x, top_y = float(top[0]), float(top[1])
                bot_x, bot_y = float(bottom[0]), float(bottom[1])

                # Expected center position (midpoint of halfway line)
                expected_cx = (top_x + bot_x) / 2
                expected_cy = (top_y + bot_y) / 2
                line_length = np.sqrt((bot_x - top_x) ** 2 + (bot_y - top_y) ** 2)

                # Tolerance checks
                # Check 1: Center X should be close to halfway line X (within 15% of line length)
                x_tolerance = max(line_length * 0.15, 40)
                x_ok = abs(cx - expected_cx) < x_tolerance

                # Check 2: Center Y should be between top and bottom (with 10% margin)
                y_margin = line_length * 0.1
                y_ok = (
                    (min(top_y, bot_y) - y_margin) < cy < (max(top_y, bot_y) + y_margin)
                )

                # Check 3: Radius should be reasonable (at least 15px, at most 40% of line length)
                r_ok = 15 < r < (line_length * 0.4)

                has_real_circle = x_ok and y_ok and r_ok

                if not has_real_circle:
                    print(
                        f"[Homography WARNING] Circle REJECTED - center=({cx:.0f}, {cy:.0f}), radius={r:.0f}"
                    )
                    print(
                        f"  Expected center near ({expected_cx:.0f}, {expected_cy:.0f})"
                    )
                    print(
                        f"  X offset: {abs(cx - expected_cx):.0f}px (tolerance: {x_tolerance:.0f}px) - {'OK' if x_ok else 'FAIL'}"
                    )
                    print(
                        f"  Y range [{min(top_y, bot_y) - y_margin:.0f}, {max(top_y, bot_y) + y_margin:.0f}]: {'OK' if y_ok else 'FAIL'}"
                    )
                    print(
                        f"  Radius range [15, {line_length * 0.4:.0f}]: {'OK' if r_ok else 'FAIL'}"
                    )

        if has_real_circle:
            img_center = detected_keypoints.get("center_spot")
            img_radius = detected_keypoints.get("center_circle_radius")
            ellipse = detected_keypoints.get("ellipse")

            valid_center = (
                img_center is not None
                and isinstance(img_center, (tuple, list))
                and len(img_center) >= 2
            )
            valid_radius = (
                img_radius is not None
                and isinstance(img_radius, (int, float, np.floating, np.integer))
                and float(img_radius) >= 30  # Minimum radius for real circle
            )

            if valid_center and valid_radius:
                print(
                    f"[Homography DEBUG] Adding 8 circle points (radius={float(img_radius):.0f}px)"
                )
                pitch_circle_pts = self.pitch.get_center_circle_points(num_points=8)

                for i, pitch_pt in enumerate(pitch_circle_pts):
                    angle = 2 * np.pi * i / 8

                    if ellipse is not None:
                        # Use ellipse for accurate sampling under perspective
                        (ecx, ecy), (ma, MA), ellipse_angle = ellipse
                        cos_a = np.cos(np.radians(ellipse_angle))
                        sin_a = np.sin(np.radians(ellipse_angle))
                        x_ellipse = (MA / 2) * np.cos(angle)
                        y_ellipse = (ma / 2) * np.sin(angle)
                        img_x = ecx + x_ellipse * cos_a - y_ellipse * sin_a
                        img_y = ecy + x_ellipse * sin_a + y_ellipse * cos_a
                        img_circle_pt = (float(img_x), float(img_y))
                    else:
                        # Fall back to circle approximation
                        img_circle_pt = (
                            float(img_center[0]) + float(img_radius) * np.cos(angle),
                            float(img_center[1]) + float(img_radius) * np.sin(angle),
                        )

                    img_pts.append(img_circle_pt)
                    pitch_pts.append(tuple(pitch_pt))
        else:
            print(
                f"[Homography DEBUG] Not adding circle points (no valid circle or estimated from halfway line)"
            )

        # If we only have 3 points (no circle), add extra points on the halfway line
        # to get enough constraints for homography (need at least 4)
        if not has_real_circle and len(img_pts) == 3:
            if (
                "halfway_top" in detected_keypoints
                and "halfway_bottom" in detected_keypoints
                and "center_spot" in detected_keypoints
            ):

                top = detected_keypoints["halfway_top"]
                bottom = detected_keypoints["halfway_bottom"]

                # Add point at 1/4 of the way along halfway line (from top)
                pt_quarter = (
                    top[0] + 0.25 * (bottom[0] - top[0]),
                    top[1] + 0.25 * (bottom[1] - top[1]),
                )
                # This corresponds to pitch Y = 68 - 0.25 * 68 = 51m
                img_pts.append(pt_quarter)
                pitch_pts.append((52.5, 51.0))

                # Add point at 3/4 of the way along halfway line (from top)
                pt_three_quarter = (
                    top[0] + 0.75 * (bottom[0] - top[0]),
                    top[1] + 0.75 * (bottom[1] - top[1]),
                )
                # This corresponds to pitch Y = 68 - 0.75 * 68 = 17m
                img_pts.append(pt_three_quarter)
                pitch_pts.append((52.5, 17.0))

                print(
                    f"[Homography DEBUG] Added 2 extra points on halfway line (total: {len(img_pts)})"
                )

        # If we only have 2 points (halfway_top and halfway_bottom, no center_spot)
        # Add center_spot from midpoint and extra points
        if len(img_pts) == 2:
            if (
                "halfway_top" in detected_keypoints
                and "halfway_bottom" in detected_keypoints
            ):

                top = detected_keypoints["halfway_top"]
                bottom = detected_keypoints["halfway_bottom"]

                # Add center_spot as midpoint
                center = ((top[0] + bottom[0]) / 2, (top[1] + bottom[1]) / 2)
                img_pts.append(center)
                pitch_pts.append((52.5, 34.0))  # Center of pitch
                print(
                    f"[Homography DEBUG] Added center_spot from midpoint: ({center[0]:.0f}, {center[1]:.0f})"
                )

                # Add 1/4 point
                pt_quarter = (
                    top[0] + 0.25 * (bottom[0] - top[0]),
                    top[1] + 0.25 * (bottom[1] - top[1]),
                )
                img_pts.append(pt_quarter)
                pitch_pts.append((52.5, 51.0))

                # Add 3/4 point
                pt_three_quarter = (
                    top[0] + 0.75 * (bottom[0] - top[0]),
                    top[1] + 0.75 * (bottom[1] - top[1]),
                )
                img_pts.append(pt_three_quarter)
                pitch_pts.append((52.5, 17.0))

                print(
                    f"[Homography DEBUG] Added extra points on halfway line (total: {len(img_pts)})"
                )

        print(f"[Homography DEBUG] Total correspondence points: {len(img_pts)}")

        if len(img_pts) < self.min_keypoints:
            print(
                f"[Homography] Not enough keypoints: {len(img_pts)} < {self.min_keypoints}"
            )
            return False

        # CRITICAL: Check if all points are collinear (on the same line)
        # With collinear points, we CANNOT compute a valid homography
        if len(img_pts) >= 3:
            pts_array = np.array(img_pts)
            # Check using cross product - if all points are collinear,
            # cross products will be near zero
            v1 = pts_array[1] - pts_array[0]
            max_cross = 0
            for i in range(2, len(pts_array)):
                v2 = pts_array[i] - pts_array[0]
                cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
                max_cross = max(max_cross, cross)

            # If max_cross is small relative to the point spread, points are collinear
            spread = np.max(np.ptp(pts_array, axis=0))  # Max range in x or y
            if (
                max_cross < spread * 5
            ):  # Threshold: cross product should be at least 5x the spread
                print(
                    f"[Homography] WARNING: All {len(img_pts)} points are nearly collinear (cross={max_cross:.0f}, spread={spread:.0f})"
                )
                print(
                    f"[Homography] Cannot compute valid homography with collinear points!"
                )
                print(
                    f"[Homography] Need center circle or penalty area for non-collinear constraints"
                )
                return False

        # Check if we have the essential 3 keypoints (center + top + bottom)
        essential_count = sum(
            1
            for name in ["center_spot", "halfway_top", "halfway_bottom"]
            if name in detected_keypoints
        )
        if essential_count < 3:
            print(
                f"[Homography WARNING] Only {essential_count}/3 essential keypoints - calibration may be unreliable"
            )

        img_pts = np.array(img_pts, dtype=np.float32)
        pitch_pts = np.array(pitch_pts, dtype=np.float32)

        try:
            H, mask = cv2.findHomography(
                pitch_pts, img_pts, cv2.RANSAC, self.ransac_threshold
            )
        except cv2.error as e:
            print(f"[Homography] findHomography error: {e}")
            return False

        if H is None:
            print("[Homography] findHomography returned None")
            return False

        self.num_inliers = int(np.sum(mask)) if mask is not None else len(img_pts)

        if self.num_inliers < self.min_keypoints:
            print(f"[Homography] Not enough inliers: {self.num_inliers}")
            return False

        reprojected = cv2.perspectiveTransform(pitch_pts.reshape(-1, 1, 2), H).reshape(
            -1, 2
        )
        errors = np.linalg.norm(reprojected - img_pts, axis=1)
        new_error = float(np.mean(errors))

        # VALIDATION: Reject if error is too high
        MAX_ACCEPTABLE_ERROR = 20.0  # pixels
        if new_error > MAX_ACCEPTABLE_ERROR:
            print(
                f"[Homography] Rejecting calibration: error {new_error:.1f}px > {MAX_ACCEPTABLE_ERROR}px threshold"
            )
            # Keep previous calibration if we have one
            if self.is_calibrated:
                print(
                    f"[Homography] Keeping previous calibration (error={self.calibration_error:.1f}px)"
                )
                return True
            return False

        # VALIDATION: Check geometric consistency
        # Project pitch center and compare to detected center_spot
        try:
            H_inv = np.linalg.inv(H)
            if "center_spot" in detected_keypoints:
                detected_center = detected_keypoints["center_spot"]
                projected_center = cv2.perspectiveTransform(
                    np.array([[[52.5, 34.0]]], dtype=np.float32), H
                )[0, 0]
                center_dist = np.sqrt(
                    (projected_center[0] - detected_center[0]) ** 2
                    + (projected_center[1] - detected_center[1]) ** 2
                )
                if center_dist > 50:  # More than 50px discrepancy
                    print(
                        f"[Homography] Warning: projected center is {center_dist:.0f}px from detected center"
                    )
        except:
            pass

        self.calibration_error = new_error

        if self.is_calibrated and self.H_pitch_to_image is not None:
            # Only apply smoothing if new error is reasonable
            H = (
                self.smoothing_alpha * H
                + (1 - self.smoothing_alpha) * self.H_pitch_to_image
            )

        self.H_pitch_to_image = H
        try:
            self.H_image_to_pitch = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("[Homography] Matrix inversion failed")
            return False

        self.is_calibrated = True

        print(
            f"[Homography] Calibrated with {self.num_inliers} inliers, error={self.calibration_error:.2f}px"
        )
        return True

    def pitch_to_image(self, pitch_points: np.ndarray) -> np.ndarray:
        """Transform points from pitch coordinates to image coordinates."""
        if self.H_pitch_to_image is None:
            raise ValueError("Homography not calibrated")

        pts = np.array(pitch_points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        transformed = cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2), self.H_pitch_to_image
        )
        return transformed.reshape(-1, 2)

    def image_to_pitch(self, image_points: np.ndarray) -> np.ndarray:
        """Transform points from image coordinates to pitch coordinates."""
        if self.H_image_to_pitch is None:
            raise ValueError("Homography not calibrated")

        pts = np.array(image_points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        transformed = cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2), self.H_image_to_pitch
        )
        return transformed.reshape(-1, 2)

    def project_player(
        self, bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        """
        Project a player bounding box to pitch coordinates.
        Uses the bottom center of the bbox (feet position).
        """
        x1, y1, x2, y2 = bbox
        feet_x = (x1 + x2) / 2
        feet_y = y2

        pitch_coords = self.image_to_pitch(np.array([[feet_x, feet_y]]))
        return float(pitch_coords[0, 0]), float(pitch_coords[0, 1])


# =============================================================================
# AUTO HOMOGRAPHY CALIBRATOR
# =============================================================================


class AutoHomographyCalibrator:
    """
    High-level class that combines detection and estimation
    for automatic homography calibration.
    """

    def __init__(
        self,
        pitch_template: Optional[PitchTemplate] = None,
        calibration_interval: int = 30,
        min_confidence: float = 0.5,
    ):
        self.pitch = pitch_template or PitchTemplate()
        self.detector = PitchLineDetector()
        self.estimator = HomographyEstimator(self.pitch)

        self.calibration_interval = calibration_interval
        self.min_confidence = min_confidence
        self.frame_count = 0
        self.last_detection = None

        self.seen_penalty_left = False
        self.seen_penalty_right = False

    def process_frame(self, frame: np.ndarray, force_calibrate: bool = False) -> Dict:
        """
        Process a frame for homography calibration.
        """
        self.frame_count += 1

        should_detect = (
            not self.estimator.is_calibrated
            or force_calibrate
            or (self.frame_count % self.calibration_interval == 0)
        )

        result = {
            "frame_count": self.frame_count,
            "is_calibrated": self.estimator.is_calibrated,
            "calibration_error": self.estimator.calibration_error,
            "detection": None,
        }

        if should_detect:
            detection = self.detector.detect(frame)
            self.last_detection = detection
            result["detection"] = detection

            # Check for new penalty area features
            if (
                "penalty_left_line" in detection["keypoints"]
                and not self.seen_penalty_left
            ):
                self.seen_penalty_left = True
                force_calibrate = True
                print("[Homography] New feature detected: left penalty area")

            if (
                "penalty_right_line" in detection["keypoints"]
                and not self.seen_penalty_right
            ):
                self.seen_penalty_right = True
                force_calibrate = True
                print("[Homography] New feature detected: right penalty area")

            if detection["keypoints"]:
                success = self.estimator.calibrate(detection["keypoints"])
                result["is_calibrated"] = success
                result["calibration_error"] = self.estimator.calibration_error

        return result

    def get_homography(self) -> Optional[np.ndarray]:
        """Get the current homography matrix (image -> pitch)."""
        return self.estimator.H_image_to_pitch

    def project_to_pitch(
        self, image_point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Project an image point to pitch coordinates."""
        if not self.estimator.is_calibrated:
            return None

        result = self.estimator.image_to_pitch(np.array([image_point]))
        return (float(result[0, 0]), float(result[0, 1]))

    def project_bbox_to_pitch(
        self, bbox: Tuple[float, float, float, float]
    ) -> Optional[Tuple[float, float]]:
        """Project a bounding box (player) to pitch coordinates."""
        if not self.estimator.is_calibrated:
            return None
        return self.estimator.project_player(bbox)

    def draw_debug(
        self, frame: np.ndarray, detection: Optional[Dict] = None
    ) -> np.ndarray:
        """Draw debug visualization on frame."""
        vis = frame.copy()
        detection = detection or self.last_detection

        if detection is None:
            return vis

        # Draw detected lines
        for line in detection.get("lines", []):
            color = (0, 255, 0) if line.is_vertical() else (255, 0, 0)
            cv2.line(
                vis,
                (int(line.x1), int(line.y1)),
                (int(line.x2), int(line.y2)),
                color,
                2,
            )

        # Draw detected circle
        circle = detection.get("circle")
        if circle is not None:
            cx, cy, r = circle
            cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 255), 2)
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 255), -1)

        # Draw detected ellipse
        ellipse = detection.get("ellipse")
        if ellipse is not None:
            cv2.ellipse(vis, ellipse, (255, 255, 0), 2)

        # Draw keypoints
        for name, value in detection.get("keypoints", {}).items():
            if name in ("center_circle_radius", "ellipse"):
                continue
            if not isinstance(value, (tuple, list)) or len(value) < 2:
                continue
            x, y = float(value[0]), float(value[1])
            cv2.circle(vis, (int(x), int(y)), 8, (255, 0, 255), -1)
            cv2.putText(
                vis,
                name[:12],
                (int(x) + 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
            )

        # Draw projected pitch grid if calibrated
        if self.estimator.is_calibrated:
            self._draw_pitch_overlay(vis)

        # Draw calibration status
        status_color = (0, 255, 0) if self.estimator.is_calibrated else (0, 0, 255)
        status_text = f"Calibrated: {self.estimator.is_calibrated} | Error: {self.estimator.calibration_error:.1f}px"
        cv2.putText(
            vis, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
        )

        return vis

    def _draw_pitch_overlay(self, frame: np.ndarray):
        """Draw the pitch template projected onto the frame."""
        if not self.estimator.is_calibrated:
            return

        for start, end in self.pitch.get_all_lines():
            try:
                img_pts = self.estimator.pitch_to_image(np.array([start, end]))
                pt1 = tuple(img_pts[0].astype(int))
                pt2 = tuple(img_pts[1].astype(int))
                cv2.line(frame, pt1, pt2, (0, 200, 200), 1, cv2.LINE_AA)
            except:
                continue

        try:
            circle_pts = self.pitch.get_center_circle_points(32)
            img_circle_pts = self.estimator.pitch_to_image(circle_pts).astype(int)
            for i in range(len(img_circle_pts)):
                pt1 = tuple(img_circle_pts[i])
                pt2 = tuple(img_circle_pts[(i + 1) % len(img_circle_pts)])
                cv2.line(frame, pt1, pt2, (0, 200, 200), 1, cv2.LINE_AA)
        except:
            pass


# =============================================================================
# MAIN (TEST)
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            sys.exit(1)

        calibrator = AutoHomographyCalibrator()

        for _ in range(100):
            cap.read()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = calibrator.process_frame(frame)
            vis = calibrator.draw_debug(frame, result.get("detection"))
            vis_resized = cv2.resize(vis, (1280, 720))

            cv2.imshow("Homography Calibration", vis_resized)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                result = calibrator.process_frame(frame, force_calibrate=True)
                print(f"Forced recalibration: {result['is_calibrated']}")

        cap.release()
        cv2.destroyAllWindows()

    else:
        print("Usage: python homography.py <video_path>")
