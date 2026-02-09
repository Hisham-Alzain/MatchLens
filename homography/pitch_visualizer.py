"""
================================================================================
TactixAI — Live 2D Pitch Visualization
================================================================================

Projects tracked players and ball from image space onto a top-down tactical
pitch view using a homography matrix.

Input:
    • Tracker outputs  (bounding boxes, track IDs, class IDs/names)
    • Homography matrix  H : image → world  (from SoccerHomographyPipeline)

Output:
    • A rendered 2D pitch image with player dots, ball, trails, and IDs

Coordinate convention  (world / pitch):
    Origin (0, 0) = center spot
    X axis: −52.5 (left goal) → +52.5 (right goal)
    Y axis: −34   (far touchline) → +34 (near touchline)

The canvas uses a shifted system where (0, 0) is the bottom-left corner of
the pitch so that all coordinates are positive for drawing:
    canvas_x = world_x + 52.5       (0 … 105)
    canvas_y = world_y + 34          (0 … 68)
================================================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ==============================================================================
# FIFA dimensions (duplicated here so the module is self-contained)
# ==============================================================================

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_WIDTH = 18.32
PENALTY_SPOT_DIST = 11.0
CENTER_CIRCLE_RADIUS = 9.15
PENALTY_ARC_RADIUS = 9.15
GOAL_WIDTH = 7.32

HALF_L = PITCH_LENGTH / 2  # 52.5
HALF_W = PITCH_WIDTH / 2  # 34.0
PA_HW = PENALTY_AREA_WIDTH / 2  # 20.16
GA_HW = GOAL_AREA_WIDTH / 2  # 9.16
GOAL_HW = GOAL_WIDTH / 2  # 3.66


# ==============================================================================
# Color schemes
# ==============================================================================


@dataclass
class PitchColors:
    grass: Tuple[int, int, int] = (34, 139, 34)
    lines: Tuple[int, int, int] = (255, 255, 255)
    team1: Tuple[int, int, int] = (255, 50, 50)  # Blue-ish (BGR)
    team2: Tuple[int, int, int] = (50, 50, 255)  # Red-ish (BGR)
    goalkeeper1: Tuple[int, int, int] = (255, 255, 0)  # Cyan
    goalkeeper2: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    ball: Tuple[int, int, int] = (0, 165, 255)  # Orange
    referee: Tuple[int, int, int] = (128, 128, 128)  # Gray
    unknown: Tuple[int, int, int] = (180, 180, 180)  # Light gray


# ==============================================================================
# Projection helpers
# ==============================================================================


def project_point(H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """Apply homography H (3×3) to an image point → world point."""
    pt = np.array([x, y, 1.0], dtype=np.float64)
    proj = H @ pt
    if abs(proj[2]) < 1e-12:
        return (np.nan, np.nan)
    return (proj[0] / proj[2], proj[1] / proj[2])


def foot_position(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Estimate the foot position from a bounding box (x1, y1, x2, y2).
    Uses the bottom-center of the box as the ground contact point.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


# ==============================================================================
# Pitch renderer (self-contained)
# ==============================================================================


class PitchRenderer:
    """
    Draws a clean 2D pitch template at a configurable resolution.

    All drawing uses *canvas coordinates* where (0, 0) is the bottom-left
    corner of the pitch.  The internal ``_p2c`` helper converts pitch-meters
    to pixel coordinates on the canvas (with Y flipped so the near touchline
    is at the bottom of the image).
    """

    def __init__(
        self,
        canvas_width: int = 1050,
        canvas_height: int = 680,
        margin: int = 50,
        colors: Optional[PitchColors] = None,
    ):
        self.colors = colors or PitchColors()
        self.margin = margin
        self.cw = canvas_width + 2 * margin
        self.ch = canvas_height + 2 * margin
        self.sx = canvas_width / PITCH_LENGTH
        self.sy = canvas_height / PITCH_WIDTH

        self._base = self._draw_pitch()

    # ── coordinate helpers ────────────────────────────────────────────

    def _p2c(self, px: float, py: float) -> Tuple[int, int]:
        """Pitch coords (0…105, 0…68) → canvas pixels.  Y is flipped."""
        cx = int(px * self.sx + self.margin)
        cy = int((PITCH_WIDTH - py) * self.sy + self.margin)
        return (cx, cy)

    def world_to_canvas(self, wx: float, wy: float) -> Tuple[int, int]:
        """World coords (origin=center spot) → canvas pixels."""
        px = wx + HALF_L  # shift so 0…105
        py = wy + HALF_W  # shift so 0…68
        return self._p2c(px, py)

    # ── empty pitch ───────────────────────────────────────────────────

    def _draw_pitch(self) -> np.ndarray:
        c = np.full((self.ch, self.cw, 3), self.colors.grass, dtype=np.uint8)
        w = self.colors.lines
        t = 2  # line thickness

        # Outline
        pts = np.array(
            [self._p2c(0, 0), self._p2c(105, 0), self._p2c(105, 68), self._p2c(0, 68)],
            np.int32,
        )
        cv2.polylines(c, [pts], True, w, t)

        # Halfway line
        cv2.line(c, self._p2c(52.5, 0), self._p2c(52.5, 68), w, t)

        # Center circle + spot
        center = self._p2c(52.5, 34)
        cv2.circle(c, center, int(CENTER_CIRCLE_RADIUS * self.sx), w, t)
        cv2.circle(c, center, 4, w, -1)

        # Penalty areas
        for xo in [0, 105]:
            sign = 1 if xo == 0 else -1
            pa = [
                self._p2c(xo, 34 - PA_HW),
                self._p2c(xo + sign * PENALTY_AREA_LENGTH, 34 - PA_HW),
                self._p2c(xo + sign * PENALTY_AREA_LENGTH, 34 + PA_HW),
                self._p2c(xo, 34 + PA_HW),
            ]
            cv2.polylines(c, [np.array(pa, np.int32)], False, w, t)

        # Goal areas
        for xo in [0, 105]:
            sign = 1 if xo == 0 else -1
            ga = [
                self._p2c(xo, 34 - GA_HW),
                self._p2c(xo + sign * GOAL_AREA_LENGTH, 34 - GA_HW),
                self._p2c(xo + sign * GOAL_AREA_LENGTH, 34 + GA_HW),
                self._p2c(xo, 34 + GA_HW),
            ]
            cv2.polylines(c, [np.array(ga, np.int32)], False, w, t)

        # Penalty spots
        cv2.circle(c, self._p2c(11, 34), 3, w, -1)
        cv2.circle(c, self._p2c(94, 34), 3, w, -1)

        # Penalty arcs
        r = int(PENALTY_ARC_RADIUS * self.sx)
        cv2.ellipse(c, self._p2c(11, 34), (r, r), 0, -53, 53, w, t)
        cv2.ellipse(c, self._p2c(94, 34), (r, r), 0, 127, 233, w, t)

        # Goals (drawn as thin rectangles behind the goal line)
        cv2.rectangle(c, self._p2c(-2, 34 - GOAL_HW), self._p2c(0, 34 + GOAL_HW), w, t)
        cv2.rectangle(
            c, self._p2c(105, 34 - GOAL_HW), self._p2c(107, 34 + GOAL_HW), w, t
        )

        return c

    def blank(self) -> np.ndarray:
        """Return a fresh copy of the empty pitch."""
        return self._base.copy()


# ==============================================================================
# Live tactical visualizer
# ==============================================================================


@dataclass
class TrackedObject:
    """One tracked object for a single frame."""

    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2  (image pixels)
    world_pos: Optional[Tuple[float, float]] = None  # filled after projection


class LivePitchVisualizer:
    """
    End-to-end: takes raw tracker outputs + homography → rendered 2D pitch.

    Usage::

        viz = LivePitchVisualizer()

        # Every frame:
        pitch_img = viz.update(
            tracks=tracks,          # list of TrackedObject or raw dicts
            H=cached_homography,    # 3×3 np.ndarray  (image → world)
        )
        cv2.imshow("Pitch", pitch_img)

    Class-name conventions (case-insensitive):
        "player"      → team dot  (team assignment via simple heuristic)
        "goalkeeper"  → keeper dot
        "referee"     → gray dot
        "ball"        → orange dot
    """

    def __init__(
        self,
        canvas_width: int = 1050,
        canvas_height: int = 680,
        margin: int = 50,
        trail_length: int = 30,
        colors: Optional[PitchColors] = None,
    ):
        self.colors = colors or PitchColors()
        self.renderer = PitchRenderer(canvas_width, canvas_height, margin, self.colors)
        self.trail_length = trail_length

        # Persistent state across frames
        self._trails: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self._team_cache: Dict[int, int] = {}  # track_id → 1 or 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        tracks: List,
        H: Optional[np.ndarray],
        team_assignments: Optional[Dict[int, int]] = None,
        show_trails: bool = True,
        show_ids: bool = True,
    ) -> np.ndarray:
        """
        Project tracks and render a new pitch frame.

        Parameters
        ----------
        tracks : list
            Each element is either a ``TrackedObject`` or a dict/tuple with
            keys  ``track_id, class_id, class_name, bbox``.
        H : np.ndarray or None
            3×3 homography  image → world.  If None, returns an empty pitch.
        team_assignments : dict, optional
            Mapping  track_id → team (1 or 2).  If not provided, a naive
            x-position heuristic is used.
        show_trails : bool
            Draw fading movement trails.
        show_ids : bool
            Draw track-ID numbers on dots.

        Returns
        -------
        np.ndarray
            Rendered pitch image (BGR), ready for cv2.imshow.
        """
        canvas = self.renderer.blank()

        if H is None:
            self._draw_status(canvas, "No homography", (0, 0, 255))
            return canvas

        # Normalise inputs → list[TrackedObject]
        objects = self._normalise_tracks(tracks)

        # Project every object
        ball_world = None
        players = []

        for obj in objects:
            foot = foot_position(obj.bbox)
            wx, wy = project_point(H, foot[0], foot[1])

            # Sanity: reject projections wildly outside the pitch
            if np.isnan(wx) or np.isnan(wy):
                continue
            if abs(wx) > HALF_L + 10 or abs(wy) > HALF_W + 10:
                continue

            obj.world_pos = (wx, wy)

            if obj.class_name.lower() == "ball":
                ball_world = (wx, wy)
            else:
                players.append(obj)

        # Assign teams if not provided
        if team_assignments is None:
            team_assignments = self._assign_teams(players)
        self._team_cache.update(team_assignments)

        # Update trails
        if show_trails:
            self._update_trails(players, ball_world)

        # ── Draw trails ──
        if show_trails:
            self._draw_trails(canvas)

        # ── Draw players ──
        for obj in players:
            if obj.world_pos is None:
                continue
            wx, wy = obj.world_pos
            cx, cy = self.renderer.world_to_canvas(wx, wy)
            team = self._team_cache.get(obj.track_id, 1)

            name_lower = obj.class_name.lower()
            if "goalkeeper" in name_lower:
                color = (
                    self.colors.goalkeeper1 if team == 1 else self.colors.goalkeeper2
                )
                radius = 9
            elif "referee" in name_lower:
                color = self.colors.referee
                radius = 6
            else:
                color = self.colors.team1 if team == 1 else self.colors.team2
                radius = 7

            # Filled dot with white outline
            cv2.circle(canvas, (cx, cy), radius, color, -1)
            cv2.circle(canvas, (cx, cy), radius, (255, 255, 255), 1)

            # ID label
            if show_ids:
                label = str(obj.track_id)
                cv2.putText(
                    canvas,
                    label,
                    (cx - 5 * len(label), cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                )

        # ── Draw ball ──
        if ball_world is not None:
            bx, by = self.renderer.world_to_canvas(*ball_world)
            cv2.circle(canvas, (bx, by), 6, self.colors.ball, -1)
            cv2.circle(canvas, (bx, by), 6, (0, 0, 0), 1)

        # Status bar
        n_proj = sum(1 for o in objects if o.world_pos is not None)
        self._draw_status(
            canvas,
            f"Tracked: {n_proj}/{len(objects)}  |  H active",
            (0, 255, 0),
        )

        return canvas

    def reset(self):
        """Clear all trails and team caches."""
        self._trails.clear()
        self._team_cache.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_tracks(tracks: List) -> List[TrackedObject]:
        """Accept TrackedObject, dicts, or tuples."""
        out = []
        for t in tracks:
            if isinstance(t, TrackedObject):
                out.append(t)
            elif isinstance(t, dict):
                out.append(
                    TrackedObject(
                        track_id=t["track_id"],
                        class_id=t.get("class_id", 0),
                        class_name=t.get("class_name", "player"),
                        bbox=tuple(t["bbox"]),
                    )
                )
            elif isinstance(t, (list, tuple)) and len(t) >= 6:
                # (x1, y1, x2, y2, track_id, class_id, [class_name])
                out.append(
                    TrackedObject(
                        track_id=int(t[4]),
                        class_id=int(t[5]),
                        class_name=str(t[6]) if len(t) > 6 else "player",
                        bbox=(float(t[0]), float(t[1]), float(t[2]), float(t[3])),
                    )
                )
        return out

    def _assign_teams(self, players: List[TrackedObject]) -> Dict[int, int]:
        """
        Naive team assignment: split by median world-x position.

        Override this with your own classifier (jersey color, etc.)
        and pass ``team_assignments`` to ``update()``.
        """
        if not players:
            return {}

        world_xs = []
        ids = []
        for p in players:
            if p.world_pos is not None and "referee" not in p.class_name.lower():
                world_xs.append(p.world_pos[0])
                ids.append(p.track_id)

        if not world_xs:
            return {}

        median_x = np.median(world_xs)
        assignments = {}
        for tid, wx in zip(ids, world_xs):
            # Preserve cached assignment to avoid flickering
            if tid in self._team_cache:
                assignments[tid] = self._team_cache[tid]
            else:
                assignments[tid] = 1 if wx < median_x else 2

        return assignments

    def _update_trails(
        self,
        players: List[TrackedObject],
        ball_world: Optional[Tuple[float, float]],
    ):
        """Append latest positions to trail buffers."""
        active_ids = set()

        for p in players:
            if p.world_pos is not None:
                self._trails[p.track_id].append(p.world_pos)
                if len(self._trails[p.track_id]) > self.trail_length:
                    self._trails[p.track_id] = self._trails[p.track_id][
                        -self.trail_length :
                    ]
                active_ids.add(p.track_id)

        if ball_world is not None:
            bid = -1  # special ID for ball
            self._trails[bid].append(ball_world)
            if len(self._trails[bid]) > self.trail_length:
                self._trails[bid] = self._trails[bid][-self.trail_length :]
            active_ids.add(bid)

        # Prune stale trails (objects that disappeared)
        stale = [k for k in self._trails if k not in active_ids]
        for k in stale:
            # Keep the trail for a few frames so it fades out
            if len(self._trails[k]) > 0:
                self._trails[k] = self._trails[k][1:]  # shrink
            if len(self._trails[k]) == 0:
                del self._trails[k]

    def _draw_trails(self, canvas: np.ndarray):
        """Draw fading polylines for every tracked object."""
        for tid, trail in self._trails.items():
            if len(trail) < 2:
                continue

            if tid == -1:
                base_color = self.colors.ball
            else:
                team = self._team_cache.get(tid, 1)
                base_color = self.colors.team1 if team == 1 else self.colors.team2

            n = len(trail)
            for i in range(n - 1):
                alpha = (i + 1) / n
                color = tuple(int(c * alpha) for c in base_color)
                pt1 = self.renderer.world_to_canvas(*trail[i])
                pt2 = self.renderer.world_to_canvas(*trail[i + 1])
                cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)

    def _draw_status(
        self,
        canvas: np.ndarray,
        text: str,
        color: Tuple[int, int, int],
    ):
        """Draw a status bar at the bottom of the canvas."""
        h = canvas.shape[0]
        cv2.putText(
            canvas,
            text,
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


# ==============================================================================
# Side-by-side display combiner
# ==============================================================================


class TacticalDisplay:
    """
    Combines the camera feed (left) with the 2D pitch (right) in one window.

    Usage::

        display = TacticalDisplay()
        viz = LivePitchVisualizer()

        while cap.isOpened():
            ...
            pitch_img = viz.update(tracks, H)
            combined = display.combine(annotated_frame, pitch_img)
            cv2.imshow("TactixAI", combined)
    """

    def __init__(
        self,
        video_width: int = 854,
        video_height: int = 480,
        pitch_width: int = 525,
        pitch_height: int = 480,
    ):
        self.vw = video_width
        self.vh = video_height
        self.pw = pitch_width
        self.ph = pitch_height

    def combine(
        self,
        video_frame: np.ndarray,
        pitch_frame: np.ndarray,
        info_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Stack video (left) and pitch (right) into one canvas.

        Both inputs are resized to the configured dimensions.
        """
        vid = cv2.resize(video_frame, (self.vw, self.vh))
        pit = cv2.resize(pitch_frame, (self.pw, self.ph))

        total_w = self.vw + self.pw
        total_h = max(self.vh, self.ph)
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # Place video
        y0 = (total_h - self.vh) // 2
        canvas[y0 : y0 + self.vh, : self.vw] = vid

        # Place pitch
        y1 = (total_h - self.ph) // 2
        canvas[y1 : y1 + self.ph, self.vw :] = pit

        # Divider
        cv2.line(canvas, (self.vw, 0), (self.vw, total_h), (80, 80, 80), 2)

        # Optional top-center info text
        if info_text:
            tw = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            cv2.putText(
                canvas,
                info_text,
                ((total_w - tw) // 2, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return canvas


# ==============================================================================
# Standalone test
# ==============================================================================

if __name__ == "__main__":
    import random

    viz = LivePitchVisualizer(canvas_width=1050, canvas_height=680)

    # Fake homography: identity-like (world coords = image coords for testing)
    # In reality this comes from SoccerHomographyPipeline
    H = np.eye(3, dtype=np.float64)

    # Simulate 30 frames of random movement
    base_positions = {}
    for i in range(1, 23):
        base_positions[i] = (
            random.uniform(-40, 40),
            random.uniform(-30, 30),
        )

    for frame_idx in range(60):
        tracks = []
        for tid, (bx, by) in base_positions.items():
            # Jitter
            wx = bx + random.gauss(0, 0.5)
            wy = by + random.gauss(0, 0.5)
            # For testing: bbox = world coords (since H=identity)
            tracks.append(
                TrackedObject(
                    track_id=tid,
                    class_id=0,
                    class_name="ball" if tid == 1 else "player",
                    bbox=(wx - 1, wy - 2, wx + 1, wy),
                )
            )

        pitch_img = viz.update(tracks, H, show_trails=True, show_ids=True)

        cv2.imshow("Pitch Test", pitch_img)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("LivePitchVisualizer test complete.")
