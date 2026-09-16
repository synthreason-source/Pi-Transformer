"""
algorithm.py

Continuous camera + continuous YOLO + human-gated robot decision loop.
Includes visual rendering of the simulated robot arm vectors.

Install:
    pip install opencv-python numpy ultralytics

Run:
    python algorithm.py
    python algorithm.py --model yolov8n.pt --device cpu
    python algorithm.py --auto
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

REQUESTED_EXPOSURE_US = 500
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640

WINDOW_NAME = "Continuous YOLO Robot Agent"

WORLD_FILE = Path("robot_world_model.json")
SCRATCHPAD_FILE = Path("robot_scratchpad.json")


# ============================================================
# HELPERS
# ============================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ns() -> int:
    return time.perf_counter_ns()


# ============================================================
# CONTINUOUS CAMERA
# ============================================================

class ContinuousCamera:
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
    ):
        self.index = index
        self.width = width
        self.height = height

        self.cap: Optional[cv2.VideoCapture] = None
        self.backend = "synthetic"
        self.real_camera = False

        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_metadata: dict = {}
        self.frame_number = 0

        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self._open()

        self.thread = threading.Thread(
            target=self._loop,
            name="camera-capture",
            daemon=True,
        )
        self.thread.start()

    def _open(self):
        if sys.platform.startswith("win"):
            candidates = [
                (cv2.CAP_DSHOW, "DSHOW"),
                (cv2.CAP_MSMF, "MSMF"),
            ]
        else:
            candidates = [
                (cv2.CAP_ANY, "ANY"),
            ]

        for backend_id, backend_name in candidates:
            cap = cv2.VideoCapture(
                self.index,
                backend_id,
            )

            if not cap.isOpened():
                cap.release()
                continue

            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                self.width,
            )
            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                self.height,
            )

            cap.set(
                cv2.CAP_PROP_AUTO_EXPOSURE,
                0.25,
            )

            cap.set(
                cv2.CAP_PROP_EXPOSURE,
                REQUESTED_EXPOSURE_US / 1_000_000.0,
            )

            ok, frame = cap.read()

            if ok and frame is not None:
                self.cap = cap
                self.real_camera = True

                try:
                    self.backend = cap.getBackendName()
                except Exception:
                    self.backend = backend_name

                return

            cap.release()

    def _loop(self):
        while not self.stop_event.is_set():
            if self.cap is None:
                frame = self._synthetic_frame()
                driver_exposure = None
            else:
                ok, frame = self.cap.read()

                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                try:
                    driver_exposure = float(
                        self.cap.get(
                            cv2.CAP_PROP_EXPOSURE
                        )
                    )
                except Exception:
                    driver_exposure = None

            self.frame_number += 1

            metadata = {
                "frame_number": self.frame_number,
                "timestamp_ns": ns(),
                "requested_exposure_us":
                    REQUESTED_EXPOSURE_US,
                "driver_exposure":
                    driver_exposure,
                "backend": self.backend,
                "real_camera": self.real_camera,
            }

            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_metadata = metadata

    def _synthetic_frame(self):
        h = self.height
        w = self.width
        t = time.monotonic()

        frame = np.full(
            (h, w, 3),
            35,
            dtype=np.uint8,
        )

        x1 = int(
            w * 0.30
            + 110 * math.sin(t * 0.7)
        )
        y1 = int(
            h * 0.40
            + 70 * math.cos(t * 0.6)
        )

        x2 = int(
            w * 0.67
            + 120 * math.cos(t * 0.5)
        )
        y2 = int(
            h * 0.50
            + 60 * math.sin(t * 0.8)
        )

        cv2.rectangle(
            frame,
            (x1 - 75, y1 - 55),
            (x1 + 75, y1 + 55),
            (180, 180, 180),
            3,
        )

        cv2.circle(
            frame,
            (x2, y2),
            65,
            (180, 180, 180),
            3,
        )

        cv2.putText(
            frame,
            "SYNTHETIC CAMERA",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

        return frame

    def latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, None

            return (
                self.latest_frame.copy(),
                dict(self.latest_metadata),
            )

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=1.5)

        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ============================================================
# YOLO DATA & REGISTRY
# ============================================================

@dataclass
class RecognizedObject:
    shared_id: str
    label: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    area_px: int
    uncertainty: float
    frame_number: int
    source: str = "YOLO"

    def to_dict(self):
        return asdict(self)


@dataclass
class RegionTrack:
    shared_id: str
    label: str
    class_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    last_frame: int


class SharedRegionRegistry:
    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_missed_frames: int = 15,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.tracks: dict[str, RegionTrack] = {}

    @staticmethod
    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        inter = iw * ih

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter

        return 0.0 if union <= 0 else inter / union

    def update(
        self,
        detections: list[dict],
        frame_number: int,
    ) -> list[RecognizedObject]:

        possible = []

        for di, det in enumerate(detections):
            for track_id, track in self.tracks.items():
                if track.label != det["label"]:
                    continue

                score = self.iou(
                    det["bbox"],
                    track.bbox,
                )

                if score >= self.iou_threshold:
                    possible.append(
                        (score, di, track_id)
                    )

        possible.sort(reverse=True)

        assigned_detections = set()
        assigned_tracks = set()
        assignments = {}

        for score, di, track_id in possible:
            if di in assigned_detections:
                continue
            if track_id in assigned_tracks:
                continue

            assignments[di] = track_id
            assigned_detections.add(di)
            assigned_tracks.add(track_id)

        objects = []

        for di, det in enumerate(detections):
            shared_id = assignments.get(
                di,
                "region:" + uuid.uuid4().hex[:12],
            )

            x1, y1, x2, y2 = det["bbox"]

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            confidence = clamp(
                float(det["confidence"]),
                0.0,
                1.0,
            )

            obj = RecognizedObject(
                shared_id=shared_id,
                label=det["label"],
                class_id=int(det["class_id"]),
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center=(
                    (x1 + x2) // 2,
                    (y1 + y2) // 2,
                ),
                area_px=width * height,
                uncertainty=1.0 - confidence,
                frame_number=frame_number,
            )

            objects.append(obj)

            self.tracks[shared_id] = RegionTrack(
                shared_id=shared_id,
                label=obj.label,
                class_id=obj.class_id,
                bbox=obj.bbox,
                confidence=obj.confidence,
                last_frame=frame_number,
            )

        stale = [
            track_id
            for track_id, track in self.tracks.items()
            if frame_number - track.last_frame
            > self.max_missed_frames
        ]

        for track_id in stale:
            del self.tracks[track_id]

        return objects


# ============================================================
# CONTINUOUS YOLO WORKER
# ============================================================

class ContinuousYOLO:
    def __init__(
        self,
        camera: ContinuousCamera,
        registry: SharedRegionRegistry,
        model_path: str,
        confidence: float,
        iou: float,
        image_size: int,
        device: Optional[str],
    ):
        self.camera = camera
        self.registry = registry

        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou
        self.image_size = image_size
        self.device = device

        self.model = None
        self.names = {}

        self.lock = threading.Lock()

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_metadata = {}
        self.latest_objects: list[RecognizedObject] = []

        self.last_camera_frame = -1
        self.inference_count = 0
        self.last_inference_ms = 0.0
        self.error = None

        self.stop_event = threading.Event()
        self.thread = None

    def load(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed.\n"
                "Run:\n"
                "    pip install ultralytics"
            ) from exc

        print(
            f"Loading YOLO model: {self.model_path}"
        )

        self.model = YOLO(
            self.model_path
        )

        names = getattr(
            self.model,
            "names",
            {},
        )

        if isinstance(names, dict):
            self.names = {
                int(k): str(v)
                for k, v in names.items()
            }
        elif isinstance(names, list):
            self.names = {
                i: str(v)
                for i, v in enumerate(names)
            }

    def start(self):
        if self.model is None:
            self.load()

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="continuous-yolo",
        )
        self.thread.start()

    def _loop(self):
        while not self.stop_event.is_set():
            frame, metadata = (
                self.camera.latest()
            )

            if frame is None:
                time.sleep(0.005)
                continue

            frame_number = int(
                metadata.get(
                    "frame_number",
                    0,
                )
            )

            if frame_number == self.last_camera_frame:
                time.sleep(0.001)
                continue

            self.last_camera_frame = frame_number

            started = time.perf_counter()

            try:
                kwargs = {
                    "source": frame,
                    "conf": self.confidence,
                    "iou": self.iou_threshold,
                    "imgsz": self.image_size,
                    "verbose": False,
                }

                if self.device is not None:
                    kwargs["device"] = self.device

                results = self.model.predict(
                    **kwargs
                )

                detections = self._parse(
                    results
                )

                objects = self.registry.update(
                    detections,
                    frame_number,
                )

                elapsed_ms = (
                    time.perf_counter()
                    - started
                ) * 1000.0

                with self.lock:
                    self.latest_frame = frame.copy()
                    self.latest_metadata = dict(metadata)
                    self.latest_objects = list(objects)
                    self.inference_count += 1
                    self.last_inference_ms = elapsed_ms
                    self.error = None

            except Exception as exc:
                self.error = str(exc)
                time.sleep(0.01)

    def _parse(self, results):
        detections = []

        if not results:
            return detections

        boxes = getattr(
            results[0],
            "boxes",
            None,
        )

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        conf = (
            boxes.conf
            .cpu()
            .numpy()
        )

        classes = (
            boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, score, class_id in zip(
            xyxy,
            conf,
            classes,
        ):
            x1, y1, x2, y2 = [
                int(round(v))
                for v in box
            ]

            detections.append(
                {
                    "label": self.names.get(
                        int(class_id),
                        str(int(class_id)),
                    ),
                    "class_id": int(class_id),
                    "confidence": float(score),
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                }
            )

        return detections

    def latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, [], {}

            return (
                self.latest_frame.copy(),
                list(self.latest_objects),
                dict(self.latest_metadata),
            )

    def annotate(self, frame, objects, robot_position=None):
        output = frame.copy()

        # Draw detected objects
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (235, 235, 235),
                2,
            )

            cv2.putText(
                output,
                (
                    f"{obj.label} "
                    f"{obj.confidence:.2f} "
                    f"u={obj.uncertainty:.2f}"
                ),
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                output,
                obj.shared_id,
                (
                    x1,
                    min(
                        output.shape[0] - 10,
                        y2 + 20,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        # Draw visual representation of the simulated robot vector P = [x, y, z]
        if robot_position is not None:
            h, w = output.shape[:2]
            cx, cy = w // 2, h // 2

            # Project robot cartesian vector back onto pixel space
            rx = int(cx + (robot_position["x"] / 0.20) * w)
            ry = int(cy + (robot_position["y"] / 0.20) * h)

            # Draw robot end-effector indicator vector node
            cv2.circle(output, (rx, ry), 12, (0, 255, 255), 2)
            cv2.line(output, (rx - 15, ry), (rx + 15, ry), (0, 255, 255), 2)
            cv2.line(output, (rx, ry - 15), (rx, ry + 15), (0, 255, 255), 2)

            # Draw connecting vector line from center to robot position
            cv2.line(output, (cx, cy), (rx, ry), (0, 255, 255), 1, cv2.LINE_AA)

            # Text label for robot position vector
            vec_text = f"Robot P: [{robot_position['x']:.3f}, {robot_position['y']:.3f}, {robot_position['z']:.3f}]m"
            cv2.putText(
                output,
                vec_text,
                (rx + 15, ry - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return output

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2.0)


# ============================================================
# RELATIONS + ONTOLOGY + MODELS
# ============================================================

@dataclass
class RegionRelation:
    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    score: float
    explanation: str

    def to_dict(self):
        return asdict(self)


@dataclass
class OntologyStatement:
    subject: str
    predicate: str
    object: str
    confidence: float
    source: str
    timestamp_ns: int

    def to_dict(self):
        return asdict(self)


def vertical_overlap(a, b):
    return max(
        0,
        min(a[3], b[3])
        - max(a[1], b[1]),
    )


def horizontal_overlap(a, b):
    return max(
        0,
        min(a[2], b[2])
        - max(a[0], b[0]),
    )


def compatible_border_score(
    a,
    b,
    tolerance=12,
):
    horizontal_gap = min(
        abs(a[2] - b[0]),
        abs(b[2] - a[0]),
    )

    y_overlap = vertical_overlap(a, b)

    if (
        horizontal_gap <= tolerance
        and y_overlap > 0
    ):
        span = max(
            1,
            min(
                a[3] - a[1],
                b[3] - b[1],
            ),
        )

        return clamp(
            y_overlap / span,
            0.0,
            1.0,
        )

    vertical_gap = min(
        abs(a[3] - b[1]),
        abs(b[3] - a[1]),
    )

    x_overlap = horizontal_overlap(a, b)

    if (
        vertical_gap <= tolerance
        and x_overlap > 0
    ):
        span = max(
            1,
            min(
                a[2] - a[0],
                b[2] - b[0],
            ),
        )

        return clamp(
            x_overlap / span,
            0.0,
            1.0,
        )

    return 0.0


def build_relations(objects):
    relations = []

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a = objects[i]
            b = objects[j]

            score = compatible_border_score(
                a.bbox,
                b.bbox,
            )

            if score <= 0:
                continue

            relations.append(
                RegionRelation(
                    relation_id=(
                        "border:"
                        + uuid.uuid4().hex[:10]
                    ),
                    subject_id=a.shared_id,
                    predicate="CompatibleBorder",
                    object_id=b.shared_id,
                    score=score,
                    explanation=(
                        "The two detected regions "
                        "have adjacent image-space "
                        "bounding borders."
                    ),
                )
            )

    return relations


def build_ontology(
    objects,
    relations,
    metadata,
):
    result = []
    t = ns()

    for obj in objects:
        result.append(
            OntologyStatement(
                subject=obj.shared_id,
                predicate="Perception",
                object=obj.label,
                confidence=obj.confidence,
                source="YOLO",
                timestamp_ns=t,
            )
        )

        result.append(
            OntologyStatement(
                subject=obj.shared_id,
                predicate="Uncertain",
                object=f"{obj.uncertainty:.6f}",
                confidence=1.0,
                source="YOLO-confidence",
                timestamp_ns=t,
            )
        )

        result.append(
            OntologyStatement(
                subject=obj.shared_id,
                predicate="ObservedBy",
                object="camera",
                confidence=1.0,
                source="camera",
                timestamp_ns=t,
            )
        )

        result.append(
            OntologyStatement(
                subject=obj.shared_id,
                predicate="Exposure",
                object=(
                    f"{REQUESTED_EXPOSURE_US}"
                    "us-requested"
                ),
                confidence=1.0,
                source="camera",
                timestamp_ns=t,
            )
        )

    for relation in relations:
        result.append(
            OntologyStatement(
                subject=relation.subject_id,
                predicate=relation.predicate,
                object=relation.object_id,
                confidence=relation.score,
                source="image-geometry",
                timestamp_ns=t,
            )
        )

    return result


@dataclass
class ScratchpadEntry:
    timestamp_ns: int
    frame_number: int
    shared_id: str
    observation: str
    uncertainty: float
    proposed_action: str
    decision: str

    def to_dict(self):
        return asdict(self)


class Scratchpad:
    def __init__(self):
        self.lock = threading.Lock()
        self.entries = []

    def add(self, entry):
        with self.lock:
            self.entries.append(entry)

    def save(self):
        with self.lock:
            data = [
                e.to_dict()
                for e in self.entries
            ]

        SCRATCHPAD_FILE.write_text(
            json.dumps(
                data,
                indent=2,
            ),
            encoding="utf-8",
        )


class WorldModel:
    def __init__(self):
        self.lock = threading.Lock()

        self.frame_number = 0
        self.current_objects = []
        self.objects_by_id = {}
        self.relations = []
        self.ontology = []

    def update(
        self,
        frame_number,
        objects,
        relations,
        ontology,
    ):
        with self.lock:
            self.frame_number = frame_number
            self.current_objects = list(objects)

            for obj in objects:
                self.objects_by_id[
                    obj.shared_id
                ] = obj

            self.relations = list(relations)
            self.ontology = list(ontology)

    def save(self):
        with self.lock:
            data = {
                "frame_number":
                    self.frame_number,
                "objects_by_id": {
                    k: v.to_dict()
                    for k, v in self.objects_by_id.items()
                },
                "current_objects": [
                    x.to_dict()
                    for x in self.current_objects
                ],
                "relations": [
                    x.to_dict()
                    for x in self.relations
                ],
                "ontology": [
                    x.to_dict()
                    for x in self.ontology
                ],
            }

        WORLD_FILE.write_text(
            json.dumps(
                data,
                indent=2,
            ),
            encoding="utf-8",
        )


# ============================================================
# DECISION CONTROLLER & SIMULATED ROBOT
# ============================================================

class DecisionController:
    def __init__(self, auto=False):
        self.auto = auto
        self.lock = threading.Lock()
        self.paused = False
        self.pending_decision: Optional[str] = None

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused
            return self.paused

    def is_paused(self):
        with self.lock:
            return self.paused

    def set_decision(self, decision: str):
        with self.lock:
            self.pending_decision = decision

    def get_and_clear_decision(self) -> Optional[str]:
        with self.lock:
            d = self.pending_decision
            self.pending_decision = None
            return d


class SimulatedRobot:
    def __init__(self, agent):
        self.agent = agent
        self.lock = threading.Lock()
        self.position = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
        self.paused = False

    def set_paused(self, paused):
        with self.lock:
            self.paused = paused

    def get_position(self):
        with self.lock:
            return dict(self.position)

    def move_to(
        self,
        x,
        y,
        z,
        duration=1.0,
    ):
        start_time = time.perf_counter()

        with self.lock:
            start = dict(self.position)

        while True:
            with self.lock:
                paused = self.paused

            elapsed = (
                time.perf_counter()
                - start_time
            )

            progress = clamp(
                elapsed / duration,
                0.0,
                1.0,
            )

            if not paused:
                smooth = (
                    progress
                    * progress
                    * (
                        3.0
                        - 2.0
                        * progress
                    )
                )

                with self.lock:
                    self.position["x"] = (
                        start["x"]
                        + (
                            x - start["x"]
                        )
                        * smooth
                    )

                    self.position["y"] = (
                        start["y"]
                        + (
                            y - start["y"]
                        )
                        * smooth
                    )

                    self.position["z"] = (
                        start["z"]
                        + (
                            z - start["z"]
                        )
                        * smooth
                    )

                    px = self.position["x"]
                    py = self.position["y"]
                    pz = self.position["z"]

                self.agent.set_status(
                    "ROBOT MOVING",
                    (
                        f"x={px:.3f} "
                        f"y={py:.3f} "
                        f"z={pz:.3f}"
                    ),
                )

                if progress >= 1.0:
                    break

            else:
                self.agent.set_status(
                    "ROBOT PAUSED",
                    "camera + YOLO still running",
                )

            time.sleep(0.01)

        self.agent.set_status(
            "ACTION COMPLETE",
            (
                f"x={self.position['x']:.3f} "
                f"y={self.position['y']:.3f} "
                f"z={self.position['z']:.3f}"
            ),
        )


# ============================================================
# ROBOT AGENT & MAIN UI THREAD
# ============================================================

class RobotAgent:
    def __init__(
        self,
        camera,
        yolo,
        world,
        scratchpad,
        auto=False,
        min_confidence=DEFAULT_CONF,
    ):
        self.camera = camera
        self.yolo = yolo
        self.world = world
        self.scratchpad = scratchpad

        self.min_confidence = min_confidence
        self.controller = DecisionController(auto=auto)
        self.robot = SimulatedRobot(self)

        self.status = "STARTING"
        self.detail = ""
        self.lock = threading.Lock()

        self.stop_event = threading.Event()
        self.current_action_text = ""
        self.current_target = None

        self.thread = threading.Thread(
            target=self._decision_loop,
            daemon=True,
            name="robot-decision-agent",
        )
        self.thread.start()

    def set_status(self, status, detail=""):
        with self.lock:
            self.status = status
            self.detail = detail

    def get_status(self):
        with self.lock:
            return self.status, self.detail

    def update_world(self):
        frame, objects, metadata = (
            self.yolo.latest()
        )

        if frame is None:
            return None

        frame_number = int(
            metadata.get(
                "frame_number",
                0,
            )
        )

        relations = build_relations(
            objects
        )

        ontology = build_ontology(
            objects,
            relations,
            metadata,
        )

        self.world.update(
            frame_number,
            objects,
            relations,
            ontology,
        )

        return (
            frame,
            objects,
            metadata,
        )

    def select_target(self, objects):
        valid = [
            obj
            for obj in objects
            if obj.confidence
            >= self.min_confidence
        ]

        if not valid:
            return None

        return max(
            valid,
            key=lambda obj: (
                obj.confidence,
                obj.area_px,
            ),
        )

    def target_to_cartesian(self, obj):
        width = max(
            1,
            self.camera.width,
        )
        height = max(
            1,
            self.camera.height,
        )

        x = (
            obj.center[0] / width
            - 0.5
        ) * 0.20

        y = (
            obj.center[1] / height
            - 0.5
        ) * 0.20

        z = 0.10

        return x, y, z

    def _decision_loop(self):
        self.set_status(
            "LIVE",
            "camera + YOLO continuously running",
        )

        while not self.stop_event.is_set():
            latest = self.update_world()

            if latest is None:
                time.sleep(0.02)
                continue

            frame, objects, metadata = latest
            target = self.select_target(objects)

            if target is None:
                self.set_status(
                    "LIVE / NO TARGET",
                    "YOLO continues running",
                )
                time.sleep(0.05)
                continue

            x, y, z = self.target_to_cartesian(target)

            action = (
                f"Move simulated robot toward "
                f"{target.label} "
                f"{target.shared_id} "
                f"at "
                f"({x:.3f}, {y:.3f}, {z:.3f}) m"
            )

            with self.lock:
                self.current_action_text = action
                self.current_target = target

            self.set_status(
                "WAITING FOR APPROVAL",
                (
                    f"{target.label} "
                    f"{target.shared_id} "
                    f"conf={target.confidence:.2f}"
                ),
            )

            entry = ScratchpadEntry(
                timestamp_ns=ns(),
                frame_number=target.frame_number,
                shared_id=target.shared_id,
                observation=(
                    f"YOLO recognized "
                    f"{target.label}; "
                    f"confidence="
                    f"{target.confidence:.3f}"
                ),
                uncertainty=target.uncertainty,
                proposed_action=action,
                decision="pending",
            )

            self.scratchpad.add(entry)

            if self.controller.auto:
                time.sleep(0.1)
                decision = "approved"
            else:
                while not self.stop_event.is_set():
                    d = self.controller.get_and_clear_decision()
                    if d is not None:
                        decision = d
                        break
                    time.sleep(0.04)
                else:
                    break

            if decision == "quit":
                entry.decision = "quit"
                self.stop_event.set()
                break

            if decision == "skipped":
                entry.decision = "skipped"
                self.set_status(
                    "SKIPPED",
                    target.shared_id,
                )
                time.sleep(0.05)
                continue

            if decision == "reperceive":
                entry.decision = "reperceive"
                self.set_status(
                    "REPERCEIVING",
                    "camera and YOLO never stopped",
                )
                time.sleep(0.02)
                continue

            if decision == "approved":
                entry.decision = "approved"
                self.set_status(
                    "APPROVED",
                    target.shared_id,
                )
                self.robot.move_to(x, y, z, duration=1.0)

        self.stop_event.set()


# ============================================================
# OPTIONAL PLANNER
# ============================================================

def beam_subset_sum(values, target, max_width=2000):
    states = {0: None}

    for index, value in enumerate(values):
        new_states = dict(states)

        for total, parent in states.items():
            candidate = total + value

            if candidate > target:
                continue

            if candidate not in new_states:
                new_states[candidate] = (
                    index,
                    parent,
                )

            if candidate == target:
                node = (
                    index,
                    parent,
                )
                answer = []

                while node is not None:
                    move_index, node = node
                    answer.append(move_index)

                answer.reverse()
                return answer

        if len(new_states) > max_width:
            keys = sorted(
                new_states,
                key=lambda x: abs(
                    target - x
                ),
            )[:max_width]

            states = {
                key: new_states[key]
                for key in keys
            }
        else:
            states = new_states

    return None


def planner_demo():
    import random

    random.seed(4)

    values = [
        random.randint(1, 8)
        for _ in range(80)
    ]

    target = 100

    answer = beam_subset_sum(
        values,
        target,
    )

    print()
    print("Planner demonstration")
    print("-" * 50)
    print(f"Target: {target}")

    if answer is None:
        print("No subset found.")
    else:
        print(
            "Indices:",
            answer,
        )
        print(
            "Sum:",
            sum(values[i] for i in answer),
        )


# ============================================================
# ARGUMENTS & MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Continuous camera + continuous YOLO "
            "robot decision simulator with visual vectors"
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO model path/name",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help="YOLO confidence threshold",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help="YOLO NMS IoU threshold",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="YOLO inference image size",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="YOLO device, e.g. cpu, 0, cuda:0",
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically approve simulated actions",
    )

    parser.add_argument(
        "--no-plan",
        action="store_true",
        help="Skip planner demonstration",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 78)
    print(
        "CONTINUOUS CAMERA + CONTINUOUS YOLO ROBOT AGENT"
    )
    print("=" * 78)
    print(
        f"Requested exposure: "
        f"{REQUESTED_EXPOSURE_US} us"
    )
    print(
        f"YOLO model: {args.model}"
    )
    print(
        f"Camera: {args.camera}"
    )

    if not args.no_plan:
        planner_demo()

    camera = ContinuousCamera(
        index=args.camera,
    )

    print(
        f"Camera backend: {camera.backend}"
    )
    print(
        f"Physical camera: {camera.real_camera}"
    )

    registry = SharedRegionRegistry()

    yolo = ContinuousYOLO(
        camera=camera,
        registry=registry,
        model_path=args.model,
        confidence=args.conf,
        iou=args.iou,
        image_size=args.imgsz,
        device=args.device,
    )

    try:
        yolo.load()
    except Exception as exc:
        camera.stop()
        raise SystemExit(
            str(exc)
        )

    yolo.start()

    world = WorldModel()
    scratchpad = Scratchpad()

    agent = RobotAgent(
        camera=camera,
        yolo=yolo,
        world=world,
        scratchpad=scratchpad,
        auto=args.auto,
        min_confidence=args.conf,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 800)

    print()
    print("CAMERA: RUNNING CONTINUOUSLY")
    print("YOLO:   RUNNING CONTINUOUSLY")
    print("ROBOT:  WAITING FOR DECISION")
    print()
    print(
        "A/ENTER approve | "
        "S skip | "
        "R reperceive | "
        "P pause | "
        "Q/ESC quit"
    )

    try:
        while not agent.stop_event.is_set():
            frame, objects, metadata = yolo.latest()

            if frame is None:
                frame, metadata = camera.latest()
                objects = []

            if frame is None:
                time.sleep(0.01)
                continue

            # Fetch the current simulated robot vector position P = [x, y, z]
            robot_pos = agent.robot.get_position()

            # Pass the robot vector position into annotate to render it visually
            output = yolo.annotate(frame, objects, robot_position=robot_pos)
            
            status, detail = agent.get_status()

            frame_number = metadata.get("frame_number", 0)
            driver_exposure = metadata.get("driver_exposure")

            lines = [
                f"LIVE CAMERA | frame={frame_number}",
                (
                    f"YOLO={len(objects)} objects | "
                    f"inferences={yolo.inference_count} | "
                    f"last={yolo.last_inference_ms:.1f} ms"
                ),
                (
                    f"Exposure request={REQUESTED_EXPOSURE_US} us | "
                    f"driver={driver_exposure}"
                ),
                f"STATE={status}" + (f" | {detail}" if detail else ""),
                (
                    "A/ENTER approve | "
                    "S skip | "
                    "R reperceive | "
                    "P pause | "
                    "Q/ESC quit"
                ),
            ]

            y = 28
            for line in lines:
                cv2.putText(
                    output,
                    line,
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (245, 245, 245),
                    2,
                    cv2.LINE_AA,
                )
                y += 25

            if yolo.error:
                cv2.putText(
                    output,
                    "YOLO ERROR: " + yolo.error[:150],
                    (15, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (245, 245, 245),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, output)

            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                agent.controller.set_decision("quit")
                break
            elif key in (13, 10, ord("a"), ord("A")):
                agent.controller.set_decision("approved")
            elif key in (ord("s"), ord("S")):
                agent.controller.set_decision("skipped")
            elif key in (ord("r"), ord("R")):
                agent.controller.set_decision("reperceive")
            elif key in (ord("p"), ord("P")):
                paused = agent.controller.toggle_pause()
                print("ROBOT PAUSED" if paused else "ROBOT RESUMED")

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        agent.stop_event.set()
        yolo.stop()
        camera.stop()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        world.save()
        scratchpad.save()

    print()
    print(f"World model: {WORLD_FILE}")
    print(f"Scratchpad:  {SCRATCHPAD_FILE}")


if __name__ == "__main__":
    main()
