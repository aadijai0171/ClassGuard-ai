"""
detection_engine.py
───────────────────
Core computer-vision pipeline:
  • Person detection via YOLOv8
  • Teacher identity via DeepFace (replaces face_recognition/dlib)
  • Motion / behaviour analysis via OpenCV frame differencing
  • Noise-level simulation
"""

import cv2
import numpy as np
import time
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ── optional heavy imports (graceful fallback) ───────────────────────────────
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed – YOLO detection disabled")

try:
    from deepface import DeepFace
    _FACE_AVAILABLE = True
except ImportError:
    _FACE_AVAILABLE = False
    logger.warning("deepface not installed – face ID disabled")


# ── data classes ─────────────────────────────────────────────────────────────
@dataclass
class DetectionResult:
    timestamp: float = field(default_factory=time.time)

    person_count: int = 0
    teacher_present: bool = False
    teacher_name: Optional[str] = None
    teacher_confidence: float = 0.0

    motion_level: float = 0.0
    noise_level: float = 0.0
    group_formations: int = 0
    standing_count: int = 0
    chaos_score: float = 0.0

    annotated_frame: Optional[np.ndarray] = None

    @property
    def alert_needed(self) -> bool:
        return not self.teacher_present and self.chaos_score > 50

    @property
    def status_label(self) -> str:
        if self.teacher_present:
            return "✅ Teacher Present"
        if self.chaos_score > 70:
            return "🔴 HIGH ALERT"
        if self.chaos_score > 40:
            return "🟡 Monitoring"
        return "🟢 Calm"


# ── main engine class ─────────────────────────────────────────────────────────
class DetectionEngine:
    def __init__(self, yolo_model_size: str = "yolov8n.pt"):
        self.yolo: Optional[object] = None
        self._load_yolo(yolo_model_size)

        # DeepFace stores enrolled teachers as {name: image_path}
        self._enrolled: dict = {}          # name -> temp image path
        self._tmpdir = tempfile.mkdtemp()  # writable dir for face images

        self._prev_gray: Optional[np.ndarray] = None
        self._motion_history: list = []
        self._noise_history: list = []

        self._frame_idx = 0
        self._last_yolo_result = []

    # ── model loading ─────────────────────────────────────────────────────
    def _load_yolo(self, model_size: str):
        if not _YOLO_AVAILABLE:
            return
        try:
            self.yolo = YOLO(model_size)
            logger.info(f"YOLO loaded: {model_size}")
        except Exception as e:
            logger.error(f"YOLO load failed: {e}")

    # ── teacher enrollment ────────────────────────────────────────────────
    def enroll_teacher(self, image: np.ndarray, name: str) -> bool:
        """Save teacher image so DeepFace can verify against it."""
        if not _FACE_AVAILABLE:
            self._enrolled[name] = None
            return True
        try:
            # Check a face is actually present first
            DeepFace.extract_faces(img_path=image, detector_backend="opencv", enforce_detection=True)
            # Save the image to a temp file
            path = os.path.join(self._tmpdir, f"{name.replace(' ', '_')}.jpg")
            cv2.imwrite(path, image)
            self._enrolled[name] = path
            return True
        except Exception as e:
            logger.warning(f"Enroll failed for {name}: {e}")
            return False

    def remove_teacher(self, name: str):
        if name in self._enrolled:
            path = self._enrolled.pop(name)
            if path and os.path.exists(path):
                os.remove(path)

    @property
    def enrolled_teachers(self) -> list:
        return list(self._enrolled.keys())

    # ── main pipeline ─────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray, sensitivity: float = 50.0) -> DetectionResult:
        result = DetectionResult()
        annotated = frame.copy()
        self._frame_idx += 1

        boxes = self._detect_persons(frame)
        result.person_count = len(boxes)

        for (x1, y1, x2, y2, conf) in boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (59, 130, 246), 2)
            cv2.putText(annotated, f"Person {conf:.0%}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (59, 130, 246), 1)

        # Face identification (every 10 frames to keep it fast)
        if self._frame_idx % 10 == 1:
            teacher_name, teacher_conf = self._identify_teacher(frame)
            self._last_face_result = (teacher_name, teacher_conf)
        else:
            teacher_name, teacher_conf = getattr(self, "_last_face_result", (None, 0.0))

        result.teacher_present = teacher_name is not None
        result.teacher_name = teacher_name
        result.teacher_confidence = teacher_conf

        # Draw face label on frame
        if teacher_name:
            cv2.putText(annotated, f"✓ {teacher_name} ({teacher_conf:.0%})",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 197, 94), 2)

        motion, motion_map = self._compute_motion(frame)
        result.motion_level = min(motion * (sensitivity / 50.0), 100.0)

        if motion_map is not None:
            heat = cv2.applyColorMap(motion_map, cv2.COLORMAP_JET)
            annotated = cv2.addWeighted(annotated, 0.8, heat, 0.2, 0)

        result.group_formations = self._count_groups(boxes)
        result.standing_count = self._count_standing(boxes, frame.shape[0])
        result.noise_level = self._simulate_noise(result.motion_level)
        result.chaos_score = self._compute_chaos(result)
        result.annotated_frame = self._draw_hud(annotated, result)

        return result

    # ── person detection ──────────────────────────────────────────────────
    def _detect_persons(self, frame: np.ndarray) -> list:
        if self._frame_idx % 3 != 1 and self._last_yolo_result:
            return self._last_yolo_result

        if self.yolo is None:
            h, w = frame.shape[:2]
            self._last_yolo_result = [
                (int(w*0.1), int(h*0.3), int(w*0.2), int(h*0.9), 0.91),
                (int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.9), 0.87),
                (int(w*0.6), int(h*0.3), int(w*0.7), int(h*0.9), 0.83),
            ]
            return self._last_yolo_result

        try:
            results = self.yolo(frame, classes=[0], verbose=False)
            boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf > 0.4:
                        boxes.append((x1, y1, x2, y2, conf))
            self._last_yolo_result = boxes
            return boxes
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return []

    # ── face identification via DeepFace ──────────────────────────────────
    def _identify_teacher(self, frame: np.ndarray):
        if not _FACE_AVAILABLE or not self._enrolled:
            return None, 0.0

        for name, ref_path in self._enrolled.items():
            if ref_path is None:
                continue
            try:
                result = DeepFace.verify(
                    img1_path=frame,
                    img2_path=ref_path,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False,
                )
                if result.get("verified", False):
                    distance = result.get("distance", 1.0)
                    threshold = result.get("threshold", 0.4)
                    confidence = max(0.0, 1.0 - (distance / threshold))
                    return name, confidence
            except Exception as e:
                logger.debug(f"DeepFace verify error for {name}: {e}")
                continue

        return None, 0.0

    # ── motion analysis ───────────────────────────────────────────────────
    def _compute_motion(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0, None

        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        motion_pixels = float(np.sum(thresh > 0))
        total_pixels = float(thresh.size)
        motion_ratio = (motion_pixels / total_pixels) * 100.0 * 5

        self._motion_history.append(motion_ratio)
        if len(self._motion_history) > 30:
            self._motion_history.pop(0)

        smoothed = float(np.mean(self._motion_history))
        motion_map = cv2.resize(thresh, (frame.shape[1], frame.shape[0]))
        return smoothed, motion_map

    # ── group / standing ──────────────────────────────────────────────────
    def _count_groups(self, boxes: list) -> int:
        if len(boxes) < 2:
            return 0
        centers = np.array([(int((x1+x2)/2), int((y1+y2)/2)) for (x1,y1,x2,y2,_) in boxes])
        groups = 0
        visited = set()
        for i, c in enumerate(centers):
            if i in visited:
                continue
            cluster = [i]
            for j, c2 in enumerate(centers):
                if j != i and j not in visited:
                    if np.linalg.norm(c - c2) < 120:
                        cluster.append(j)
            if len(cluster) >= 2:
                groups += 1
                visited.update(cluster)
        return groups

    def _count_standing(self, boxes: list, frame_h: int) -> int:
        count = 0
        for (x1, y1, x2, y2, _) in boxes:
            h = y2 - y1
            w = x2 - x1
            if w > 0 and (h / w) > 1.8:
                count += 1
        return count

    # ── noise simulation ──────────────────────────────────────────────────
    def _simulate_noise(self, motion: float) -> float:
        base = motion * 0.6 + np.random.normal(10, 5)
        self._noise_history.append(max(0, min(100, base)))
        if len(self._noise_history) > 20:
            self._noise_history.pop(0)
        return float(np.mean(self._noise_history))

    # ── chaos score ───────────────────────────────────────────────────────
    def _compute_chaos(self, r: DetectionResult) -> float:
        motion_w = r.motion_level * 0.35
        noise_w = r.noise_level * 0.30
        group_w = min(r.group_formations * 15, 30) * 0.20
        stand_w = min(r.standing_count * 10, 20) * 0.15
        score = motion_w + noise_w + group_w + stand_w
        return float(min(score, 100.0))

    # ── HUD overlay ───────────────────────────────────────────────────────
    def _draw_hud(self, frame: np.ndarray, r: DetectionResult) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        cv2.rectangle(overlay, (0, 0), (w, 50), (15, 23, 42), cv2.FILLED)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        color = (34, 197, 94) if r.teacher_present else \
                (239, 68, 68) if r.chaos_score > 70 else (234, 179, 8)
        cv2.putText(frame, r.status_label, (10, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        chaos_txt = f"Chaos: {r.chaos_score:.0f}%"
        cv2.putText(frame, chaos_txt, (w - 180, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        info = (f"Persons: {r.person_count}  |  "
                f"Motion: {r.motion_level:.0f}%  |  "
                f"Noise: {r.noise_level:.0f}%  |  "
                f"Groups: {r.group_formations}")
        cv2.rectangle(frame, (0, h - 30), (w, h), (15, 23, 42), cv2.FILLED)
        cv2.putText(frame, info, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame
