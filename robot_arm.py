"""
algorithm.py

YOLO-based vision-guided robotic-agent simulation.

Pipeline:

    Camera
       |
       v
    0.5 ms exposure request
       |
       v
    YOLO object recognition
       |
       v
    persistent shared region IDs
       |
       +---- uncertainty
       |
       +---- compatible border relationships
       |
       +---- ontology statements
       |
       +---- scratchpad
       |
       v
    action proposal
       |
       v
    human intervention gate
       |
       v
    simulated robot action


Install:

    pip install opencv-python numpy ultralytics

Run:

    python algorithm.py

Specify another YOLO model:

    python algorithm.py --model yolov8n.pt

CPU:

    python algorithm.py --device cpu

GPU:

    python algorithm.py --device 0

Automatic simulated approval:

    python algorithm.py --auto


IMPORTANT ABOUT 0.5 ms EXPOSURE
--------------------------------

The program requests 500 microseconds from OpenCV.

A generic webcam does NOT necessarily interpret CAP_PROP_EXPOSURE
in seconds. Camera drivers frequently use device-specific or logarithmic
units.

Therefore this program distinguishes:

    requested_exposure_us
    driver_exposure_value

It does not falsely claim that a generic webcam achieved exactly 500 us.

For guaranteed 500 us hardware exposure, a camera/vendor SDK supporting
manual exposure or hardware triggering is required.


ROBOT SAFETY
------------

This implementation does not command a physical robot.

The robot movement is visualized/simulated.

Human approval is required before each simulated action unless --auto
is explicitly supplied.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sys
import time
import uuid

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import freeze_support
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

AXES = ("x", "y", "z")

REQUESTED_EXPOSURE_US = 500
REQUESTED_EXPOSURE_S = REQUESTED_EXPOSURE_US / 1_000_000.0

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMAGE_SIZE = 640

WINDOW_NAME = "YOLO Vision-Guided Robot Agent"

SAVE_DIR = Path("simulation_frames")
WORLD_FILE = Path("robot_world_model.json")
SCRATCHPAD_FILE = Path("robot_scratchpad.json")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clamp(
    value: float,
    low: float,
    high: float,
) -> float:
    return max(low, min(high, value))


def timestamp_ns() -> int:
    return time.perf_counter_ns()


def safe_float(value) -> Optional[float]:
    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return None


# ============================================================
# CAMERA
# ============================================================

class VisionCamera:
    """
    OpenCV camera wrapper.

    The camera is configured to request a 500 us exposure.

    Exact exposure control is backend/device dependent.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        requested_exposure_us: int = REQUESTED_EXPOSURE_US,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.requested_exposure_us = requested_exposure_us

        self.cap: Optional[cv2.VideoCapture] = None

        self.backend_name = "unknown"
        self.real_camera_active = False

        self._open()

    # --------------------------------------------------------

    def _open(self) -> None:

        candidates = []

        if sys.platform.startswith("win"):

            candidates.append(
                (
                    self.camera_index,
                    cv2.CAP_DSHOW,
                )
            )

            candidates.append(
                (
                    self.camera_index,
                    cv2.CAP_MSMF,
                )
            )

        else:

            candidates.append(
                (
                    self.camera_index,
                    cv2.CAP_ANY,
                )
            )

        for index, backend in candidates:

            cap = cv2.VideoCapture(
                index,
                backend,
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

            # Ask for manual exposure where supported.
            cap.set(
                cv2.CAP_PROP_AUTO_EXPOSURE,
                0.25,
            )

            # Request 500 us.
            #
            # NOTE:
            # This value is not guaranteed to mean seconds on every
            # camera/backend.
            cap.set(
                cv2.CAP_PROP_EXPOSURE,
                REQUESTED_EXPOSURE_S,
            )

            ret, frame = cap.read()

            if (
                ret
                and frame is not None
                and frame.size > 0
            ):
                self.cap = cap
                self.real_camera_active = True

                try:
                    self.backend_name = cap.getBackendName()
                except Exception:
                    self.backend_name = str(backend)

                return

            cap.release()

        self.cap = None
        self.real_camera_active = False
        self.backend_name = "synthetic"

    # --------------------------------------------------------

    def read(self):
        capture_start = timestamp_ns()

        if self.cap is not None:

            ret, frame = self.cap.read()

            if (
                ret
                and frame is not None
                and frame.size > 0
            ):
                capture_end = timestamp_ns()

                driver_exposure = safe_float(
                    self.cap.get(
                        cv2.CAP_PROP_EXPOSURE
                    )
                )

                metadata = {
                    "capture_start_ns": capture_start,
                    "capture_end_ns": capture_end,
                    "capture_elapsed_us": (
                        capture_end - capture_start
                    ) / 1000.0,
                    "requested_exposure_us":
                        self.requested_exposure_us,
                    "driver_exposure_value":
                        driver_exposure,
                    "backend":
                        self.backend_name,
                    "real_camera":
                        True,
                }

                return True, frame, metadata

        frame = self.synthetic_frame()

        capture_end = timestamp_ns()

        metadata = {
            "capture_start_ns": capture_start,
            "capture_end_ns": capture_end,
            "capture_elapsed_us": (
                capture_end - capture_start
            ) / 1000.0,
            "requested_exposure_us":
                self.requested_exposure_us,
            "driver_exposure_value":
                None,
            "backend":
                "synthetic",
            "real_camera":
                False,
        }

        return True, frame, metadata

    # --------------------------------------------------------

    def synthetic_frame(self) -> np.ndarray:

        h = self.height
        w = self.width

        frame = np.full(
            (h, w, 3),
            38,
            dtype=np.uint8,
        )

        cv2.rectangle(
            frame,
            (100, 120),
            (330, 390),
            (150, 150, 150),
            3,
        )

        cv2.rectangle(
            frame,
            (450, 180),
            (700, 440),
            (170, 170, 170),
            3,
        )

        cv2.circle(
            frame,
            (930, 300),
            100,
            (160, 160, 160),
            3,
        )

        cv2.putText(
            frame,
            "NO CAMERA",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "Connect a camera for YOLO recognition",
            (40, h - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (190, 190, 190),
            2,
            cv2.LINE_AA,
        )

        return frame

    # --------------------------------------------------------

    def close(self):

        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ============================================================
# YOLO OBJECT REPRESENTATION
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

    source: str = "yolo"

    def to_dict(self):

        data = asdict(self)

        data["bbox"] = list(self.bbox)
        data["center"] = list(self.center)

        return data


# ============================================================
# YOLO RECOGNIZER
# ============================================================

class YOLORecognizer:
    """
    Ultralytics YOLO detector.

    This replaces the previous HSV/red contour detector completely.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        device: Optional[str] = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ):

        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.image_size = image_size

        self.model = None

        self.names = {}

    # --------------------------------------------------------

    def load(self):

        try:
            from ultralytics import YOLO

        except ImportError as exc:

            raise RuntimeError(
                "\nUltralytics is not installed.\n\n"
                "Install it with:\n\n"
                "    pip install ultralytics\n"
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

        print(
            f"YOLO classes loaded: {len(self.names)}"
        )

    # --------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[dict]:

        if self.model is None:
            self.load()

        kwargs = {
            "source": frame,
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.image_size,
            "verbose": False,
        }

        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(
            **kwargs
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        boxes = getattr(
            result,
            "boxes",
            None,
        )

        if boxes is None:
            return detections

        xyxy = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        confs = (
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

        for box, conf, cls_id in zip(
            xyxy,
            confs,
            classes,
        ):

            x1, y1, x2, y2 = [
                int(round(v))
                for v in box
            ]

            confidence = clamp(
                float(conf),
                0.0,
                1.0,
            )

            label = self.names.get(
                int(cls_id),
                str(int(cls_id)),
            )

            detections.append(
                {
                    "label": label,
                    "class_id": int(cls_id),
                    "confidence": confidence,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                }
            )

        return detections

    # --------------------------------------------------------

    def annotate(
        self,
        frame: np.ndarray,
        objects: Sequence[RecognizedObject],
    ) -> np.ndarray:

        output = frame.copy()

        for obj in objects:

            x1, y1, x2, y2 = obj.bbox

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (230, 230, 230),
                2,
            )

            text = (
                f"{obj.label} "
                f"{obj.confidence:.2f} "
                f"u={obj.uncertainty:.2f}"
            )

            cv2.putText(
                output,
                text,
                (
                    x1,
                    max(
                        25,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (240, 240, 240),
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
                        y2 + 22,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )

        return output


# ============================================================
# BOUNDING BOX GEOMETRY
# ============================================================

def bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0,
        ix2 - ix1,
    )

    ih = max(
        0,
        iy2 - iy1,
    )

    intersection = iw * ih

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# SHARED REGION REGISTRY
# ============================================================

@dataclass
class RegionTrack:

    shared_id: str

    label: str

    class_id: int

    bbox: tuple[int, int, int, int]

    confidence: float

    last_seen_frame: int


class SharedRegionRegistry:
    """
    Maintains identifiers across camera frames.

    Instead of creating a completely new UUID for every detection,
    matching detections reuse the same shared identifier.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_missed_frames: int = 12,
    ):

        self.iou_threshold = iou_threshold

        self.max_missed_frames = (
            max_missed_frames
        )

        self.tracks = {}

    # --------------------------------------------------------

    def update(
        self,
        detections: Sequence[dict],
        frame_number: int,
    ) -> list[RecognizedObject]:

        candidates = []

        for detection in detections:

            candidates.append(
                {
                    "label":
                        detection["label"],
                    "class_id":
                        int(
                            detection["class_id"]
                        ),
                    "confidence":
                        float(
                            detection["confidence"]
                        ),
                    "bbox":
                        detection["bbox"],
                }
            )

        pairs = []

        for det_index, detection in enumerate(
            candidates
        ):

            for track_id, track in self.tracks.items():

                if track.label != detection["label"]:
                    continue

                score = bbox_iou(
                    detection["bbox"],
                    track.bbox,
                )

                if score >= self.iou_threshold:

                    pairs.append(
                        (
                            score,
                            det_index,
                            track_id,
                        )
                    )

        pairs.sort(
            reverse=True
        )

        assigned_detections = set()
        assigned_tracks = set()

        matched = {}

        for score, det_index, track_id in pairs:

            if det_index in assigned_detections:
                continue

            if track_id in assigned_tracks:
                continue

            assigned_detections.add(
                det_index
            )

            assigned_tracks.add(
                track_id
            )

            matched[det_index] = track_id

        objects = []

        for det_index, detection in enumerate(
            candidates
        ):

            if det_index in matched:

                shared_id = matched[
                    det_index
                ]

            else:

                shared_id = (
                    "region:"
                    + uuid.uuid4().hex[:12]
                )

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            area = (
                max(0, x2 - x1)
                * max(0, y2 - y1)
            )

            center = (
                (x1 + x2) // 2,
                (y1 + y2) // 2,
            )

            confidence = clamp(
                detection["confidence"],
                0.0,
                1.0,
            )

            obj = RecognizedObject(
                shared_id=shared_id,
                label=detection["label"],
                class_id=detection["class_id"],
                confidence=confidence,
                bbox=detection["bbox"],
                center=center,
                area_px=area,
                uncertainty=1.0 - confidence,
            )

            objects.append(obj)

            self.tracks[shared_id] = RegionTrack(
                shared_id=shared_id,
                label=detection["label"],
                class_id=detection["class_id"],
                bbox=detection["bbox"],
                confidence=confidence,
                last_seen_frame=frame_number,
            )

        stale = [
            track_id
            for track_id, track in self.tracks.items()
            if (
                frame_number
                - track.last_seen_frame
                > self.max_missed_frames
            )
        ]

        for track_id in stale:
            del self.tracks[track_id]

        return objects


# ============================================================
# BORDER RELATIONSHIPS
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


def horizontal_overlap(
    a,
    b,
) -> int:

    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b

    return max(
        0,
        min(ax2, bx2)
        - max(ax1, bx1),
    )


def vertical_overlap(
    a,
    b,
) -> int:

    _, ay1, _, ay2 = a
    _, by1, _, by2 = b

    return max(
        0,
        min(ay2, by2)
        - max(ay1, by1),
    )


def border_relation_score(
    a,
    b,
    tolerance: int = 12,
) -> float:

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    horizontal_gap = min(
        abs(ax2 - bx1),
        abs(bx2 - ax1),
    )

    overlap_y = vertical_overlap(
        a,
        b,
    )

    if (
        horizontal_gap <= tolerance
        and overlap_y > 0
    ):

        span = max(
            1,
            min(
                ay2 - ay1,
                by2 - by1,
            ),
        )

        return clamp(
            overlap_y / span,
            0.0,
            1.0,
        )

    vertical_gap = min(
        abs(ay2 - by1),
        abs(by2 - ay1),
    )

    overlap_x = horizontal_overlap(
        a,
        b,
    )

    if (
        vertical_gap <= tolerance
        and overlap_x > 0
    ):

        span = max(
            1,
            min(
                ax2 - ax1,
                bx2 - bx1,
            ),
        )

        return clamp(
            overlap_x / span,
            0.0,
            1.0,
        )

    return 0.0


def build_border_relations(
    objects: Sequence[RecognizedObject],
) -> list[RegionRelation]:

    relations = []

    for i in range(len(objects)):

        for j in range(
            i + 1,
            len(objects),
        ):

            a = objects[i]
            b = objects[j]

            score = border_relation_score(
                a.bbox,
                b.bbox,
            )

            if score <= 0:
                continue

            relation = RegionRelation(
                relation_id=(
                    "border:"
                    + uuid.uuid4().hex[:10]
                ),
                subject_id=a.shared_id,
                predicate="CompatibleBorder",
                object_id=b.shared_id,
                score=score,
                explanation=(
                    f"{a.shared_id} and "
                    f"{b.shared_id} have "
                    f"adjacent image-space "
                    f"bounding borders."
                ),
            )

            relations.append(
                relation
            )

    return relations


# ============================================================
# ONTOLOGY
# ============================================================

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


def make_ontology(
    objects,
    relations,
    camera_metadata,
):

    statements = []

    ts = timestamp_ns()

    for obj in objects:

        statements.append(
            OntologyStatement(
                subject=obj.shared_id,
                predicate="Perception",
                object=obj.label,
                confidence=obj.confidence,
                source="YOLO",
                timestamp_ns=ts,
            )
        )

        statements.append(
            OntologyStatement(
                subject=obj.shared_id,
                predicate="Uncertain",
                object=str(
                    round(
                        obj.uncertainty,
                        6,
                    )
                ),
                confidence=1.0,
                source="uncertainty-model",
                timestamp_ns=ts,
            )
        )

        statements.append(
            OntologyStatement(
                subject=obj.shared_id,
                predicate="ObservedBy",
                object="camera",
                confidence=1.0,
                source="camera",
                timestamp_ns=ts,
            )
        )

        statements.append(
            OntologyStatement(
                subject=obj.shared_id,
                predicate="Exposure",
                object=(
                    f"{camera_metadata['requested_exposure_us']}"
                    "us-requested"
                ),
                confidence=1.0,
                source="camera",
                timestamp_ns=ts,
            )
        )

    for relation in relations:

        statements.append(
            OntologyStatement(
                subject=relation.subject_id,
                predicate=relation.predicate,
                object=relation.object_id,
                confidence=relation.score,
                source="spatial-geometry",
                timestamp_ns=ts,
            )
        )

    return statements


# ============================================================
# TWO-SYSTEM UNCERTAINTY AGREEMENT
# ============================================================

@dataclass
class UncertaintyAgreement:

    shared_id: str

    perception_confidence: float

    planner_confidence: float

    perception_uncertainty: float

    planner_uncertainty: float

    agreement: bool

    tolerance: float

    def to_dict(self):
        return asdict(self)


def compute_uncertainty_agreement(
    obj: RecognizedObject,
    planner_confidence: float,
    tolerance: float = 0.25,
) -> UncertaintyAgreement:

    planner_confidence = clamp(
        planner_confidence,
        0.0,
        1.0,
    )

    perception_uncertainty = (
        1.0 - obj.confidence
    )

    planner_uncertainty = (
        1.0 - planner_confidence
    )

    agreement = (
        abs(
            perception_uncertainty
            - planner_uncertainty
        )
        <= tolerance
    )

    return UncertaintyAgreement(
        shared_id=obj.shared_id,
        perception_confidence=obj.confidence,
        planner_confidence=planner_confidence,
        perception_uncertainty=(
            perception_uncertainty
        ),
        planner_uncertainty=(
            planner_uncertainty
        ),
        agreement=agreement,
        tolerance=tolerance,
    )


# ============================================================
# SCRATCHPAD
# ============================================================

@dataclass
class ScratchpadEntry:

    timestamp_ns: int

    shared_id: str

    observation: str

    uncertainty: float

    proposed_action: str

    decision: str = "pending"

    def to_dict(self):
        return asdict(self)


class Scratchpad:

    def __init__(self):

        self.entries = []

    def add(
        self,
        entry: ScratchpadEntry,
    ):

        self.entries.append(
            entry
        )

    def save(
        self,
        path: Path = SCRATCHPAD_FILE,
    ):

        path.write_text(
            json.dumps(
                [
                    entry.to_dict()
                    for entry in self.entries
                ],
                indent=2,
            ),
            encoding="utf-8",
        )


# ============================================================
# SHARED WORLD
# ============================================================

class SharedWorld:

    def __init__(self):

        self.frame_number = 0

        self.objects = {}

        self.relations = []

        self.ontology = []

        self.agreements = []

    def update(
        self,
        objects,
        relations,
        ontology,
    ):

        self.frame_number += 1

        for obj in objects:
            self.objects[
                obj.shared_id
            ] = obj

        self.relations = list(
            relations
        )

        self.ontology = list(
            ontology
        )

    def save(
        self,
        path: Path = WORLD_FILE,
    ):

        data = {
            "frame_number":
                self.frame_number,

            "objects": [
                obj.to_dict()
                for obj in self.objects.values()
            ],

            "relations": [
                relation.to_dict()
                for relation in self.relations
            ],

            "ontology": [
                statement.to_dict()
                for statement in self.ontology
            ],

            "agreements": [
                agreement.to_dict()
                for agreement in self.agreements
            ],
        }

        path.write_text(
            json.dumps(
                data,
                indent=2,
            ),
            encoding="utf-8",
        )


# ============================================================
# CARTESIAN MOTION
# ============================================================

@dataclass(frozen=True)
class CartesianTarget:

    x_m: float

    y_m: float

    z_m: float

    precision_ticks: int = 0

    precision_unit_m: float = 0.001

    def coordinate(
        self,
        axis: str,
    ) -> float:

        if axis == "x":
            return self.x_m

        if axis == "y":
            return self.y_m

        if axis == "z":
            return self.z_m

        raise ValueError(
            f"Unknown axis: {axis}"
        )


@dataclass(frozen=True)
class CartesianMove:

    move_id: int

    dx_m: float = 0.0

    dy_m: float = 0.0

    dz_m: float = 0.0

    precision_ticks: int = 0

    def displacement(
        self,
        axis: str,
    ) -> float:

        if axis == "x":
            return self.dx_m

        if axis == "y":
            return self.dy_m

        if axis == "z":
            return self.dz_m

        raise ValueError(
            f"Unknown axis: {axis}"
        )


# ============================================================
# SUBSET-SUM BEAM SEARCH
# ============================================================

def _materialize(
    node,
) -> list[int]:

    result = []

    while node is not None:

        move_index, parent = node

        result.append(
            move_index
        )

        node = parent

    result.reverse()

    return result


def equation_beam_search(
    nums: Sequence[int],
    target: int,
    r: float,
    c: float = 2.0,
    max_beam_width: int = 2000,
):

    n = len(nums)

    if target == 0:
        return [], -1

    if n == 0:
        return None, 0

    cap = min(
        max(
            1,
            n ** max(
                1,
                int(c),
            ),
        ),
        max(
            1,
            max_beam_width,
        ),
    )

    # partial_sum ->
    # (last_index, parent_node)
    beam = {
        0: (
            -1,
            None,
        )
    }

    for k, value in enumerate(nums):

        current_items = list(
            beam.items()
        )

        seen = dict(
            beam
        )

        for partial, state in current_items:

            new_partial = (
                partial + value
            )

            if new_partial > target:
                continue

            if new_partial not in seen:

                seen[new_partial] = (
                    k,
                    state_to_node(
                        state
                    ),
                )

            if new_partial == target:

                terminal = (
                    k,
                    state_to_node(
                        state
                    ),
                )

                return (
                    _materialize(
                        terminal
                    ),
                    k,
                )

        log_width = (
            n * math.log(2.0)
            + k
            * math.log(
                max(
                    1e-12,
                    1.0 - r,
                )
            )
        )

        if (
            log_width
            > math.log(
                max(1, cap)
            )
        ):

            width = cap

        else:

            width = max(
                1,
                min(
                    cap,
                    int(
                        math.ceil(
                            math.exp(
                                log_width
                            )
                        )
                    ),
                ),
            )

        if len(seen) <= width:

            beam = seen

        else:

            selected = heapq.nsmallest(
                width,
                seen.items(),
                key=lambda item: (
                    abs(
                        target
                        - item[0]
                    ),
                    item[0],
                ),
            )

            beam = dict(
                selected
            )

    return None, n


def state_to_node(
    state,
):
    """
    Converts the stored beam state into the linked-node form used by
    _materialize.
    """

    if state is None:
        return None

    index, parent = state

    if index == -1:
        return None

    return (
        index,
        state_to_node(parent),
    )


def measure_r(
    nums: Sequence[int],
    sample_size: int = 24,
    subset_fraction: int = 5,
    seed: Optional[int] = None,
) -> float:

    if not nums:
        return 0.5

    rng = random.Random(
        seed
    )

    sample = list(
        nums[
            :min(
                len(nums),
                sample_size,
            )
        ]
    )

    if len(sample) < 2:
        return 0.5

    node_count = 0

    for _ in range(
        max(
            1,
            subset_fraction,
        )
    ):

        target = sum(
            rng.choice(
                [0, value]
            )
            for value in sample
        )

        stack = [
            (
                0,
                0,
            )
        ]

        while stack:

            index, total = stack.pop()

            node_count += 1

            if index >= len(sample):
                continue

            stack.append(
                (
                    index + 1,
                    total,
                )
            )

            new_total = (
                total
                + sample[index]
            )

            if new_total <= target:

                stack.append(
                    (
                        index + 1,
                        new_total,
                    )
                )

    node_count = max(
        1,
        node_count,
    )

    denominator = max(
        1.0,
        2.0 ** len(sample),
    )

    ratio = (
        node_count
        / denominator
    )

    r = (
        1.0
        - math.exp(
            math.log(
                max(
                    1e-12,
                    ratio,
                )
            )
            / max(
                1,
                len(sample),
            )
        )
    )

    return clamp(
        r,
        0.0,
        0.999,
    )


@dataclass
class AxisJob:

    limb: str

    axis: str

    moves: list[CartesianMove]

    target: CartesianTarget

    is_approach_axis: bool = False

    c: float = 2.0

    max_beam_width: int = 2000

    r: Optional[float] = None

    seed: Optional[int] = None

    coordinate_unit_m: float = 0.001

    precision_weight: int = 1


@dataclass
class AxisResult:

    limb: str

    axis: str

    found: bool

    move_ids_used: list[int] = field(
        default_factory=list
    )

    displacement_m: float = 0.0

    explored_moves: int = 0


def target_units(
    job: AxisJob,
) -> int:

    base = round(
        job.target.coordinate(
            job.axis
        )
        / job.coordinate_unit_m
    )

    return (
        base
        + job.target.precision_ticks
        * job.precision_weight
    )


def move_units(
    job: AxisJob,
    move: CartesianMove,
) -> int:

    base = round(
        move.displacement(
            job.axis
        )
        / job.coordinate_unit_m
    )

    return (
        base
        + move.precision_ticks
        * job.precision_weight
    )


def solve_one_axis(
    job: AxisJob,
) -> AxisResult:

    values = [
        move_units(
            job,
            move,
        )
        for move in job.moves
    ]

    target = target_units(
        job
    )

    r = (
        job.r
        if job.r is not None
        else measure_r(
            values,
            seed=job.seed,
        )
    )

    positions, explored = (
        equation_beam_search(
            values,
            target,
            r=r,
            c=job.c,
            max_beam_width=job.max_beam_width,
        )
    )

    if positions is None:

        return AxisResult(
            limb=job.limb,
            axis=job.axis,
            found=False,
            explored_moves=explored,
        )

    move_ids = [
        job.moves[position].move_id
        for position in positions
    ]

    displacement = sum(
        job.moves[position].displacement(
            job.axis
        )
        for position in positions
    )

    return AxisResult(
        limb=job.limb,
        axis=job.axis,
        found=True,
        move_ids_used=move_ids,
        displacement_m=displacement,
        explored_moves=explored,
    )


def solve_all_limbs(
    jobs: Sequence[AxisJob],
    max_workers: Optional[int] = None,
) -> list[AxisResult]:

    if not jobs:
        return []

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = [
            executor.submit(
                solve_one_axis,
                job,
            )
            for job in jobs
        ]

        return [
            future.result()
            for future in futures
        ]


# ============================================================
# ROBOT VISUALIZER
# ============================================================

class VisionGuidedArmOverlay:

    def __init__(
        self,
        camera: VisionCamera,
        recognizer: YOLORecognizer,
        width: int = 1000,
        height: int = 750,
        save_frames: bool = True,
    ):

        self.camera = camera
        self.recognizer = recognizer

        self.width = width
        self.height = height

        self.save_frames = save_frames

        self.step_counter = 0

        SAVE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            width,
            height,
        )

    # --------------------------------------------------------

    def show_decision_snapshot(
        self,
        frame,
        objects,
        metadata,
        proposed_action,
    ):

        annotated = self.recognizer.annotate(
            frame,
            objects,
        )

        lines = [
            "DECISION SNAPSHOT",
            (
                "Requested exposure: "
                f"{metadata['requested_exposure_us']} us"
            ),
            (
                "Driver exposure: "
                f"{metadata['driver_exposure_value']}"
            ),
            (
                "Backend: "
                f"{metadata['backend']}"
            ),
            (
                "Real camera: "
                f"{metadata['real_camera']}"
            ),
            (
                "Objects: "
                f"{len(objects)}"
            ),
            (
                "Action: "
                f"{proposed_action}"
            ),
            (
                "ENTER/A approve | "
                "S skip | "
                "R reperceive | "
                "P pause | "
                "Q quit"
            ),
        ]

        y = 28

        for line in lines:

            cv2.putText(
                annotated,
                line,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

            y += 25

        cv2.imshow(
            WINDOW_NAME,
            annotated,
        )

        if self.save_frames:

            path = (
                SAVE_DIR
                / (
                    f"decision_"
                    f"{self.step_counter:04d}_"
                    f"{REQUESTED_EXPOSURE_US}us.png"
                )
            )

            cv2.imwrite(
                str(path),
                annotated,
            )

        self.step_counter += 1

    # --------------------------------------------------------

    def draw_arm(
        self,
        frame,
        axis,
        displacement_m,
        status,
    ):

        output = frame.copy()

        h, w = output.shape[:2]

        origin = (
            w // 2,
            h - 120,
        )

        scale = min(
            w,
            h,
        ) * 1.2

        dx = (
            displacement_m
            / 0.25
            * scale
        )

        dy = dx

        if axis == "x":

            elbow = (
                int(
                    origin[0]
                    + dx * 0.45
                ),
                origin[1] - 110,
            )

            end = (
                int(
                    origin[0]
                    + dx
                ),
                origin[1] - 170,
            )

        elif axis == "y":

            elbow = (
                origin[0] + 90,
                int(
                    origin[1]
                    - 110
                    - dy * 0.45
                ),
            )

            end = (
                origin[0] + 130,
                int(
                    origin[1]
                    - 160
                    - dy
                ),
            )

        else:

            elbow = (
                origin[0] + 90,
                origin[1] - 110,
            )

            end = (
                origin[0] + 90,
                int(
                    origin[1]
                    - 110
                    - dy
                ),
            )

        cv2.line(
            output,
            origin,
            elbow,
            (210, 210, 210),
            12,
            cv2.LINE_AA,
        )

        cv2.line(
            output,
            elbow,
            end,
            (235, 235, 235),
            10,
            cv2.LINE_AA,
        )

        cv2.circle(
            output,
            origin,
            18,
            (245, 245, 245),
            -1,
        )

        cv2.circle(
            output,
            elbow,
            16,
            (220, 220, 220),
            -1,
        )

        cv2.circle(
            output,
            end,
            13,
            (255, 255, 255),
            -1,
        )

        cv2.putText(
            output,
            (
                f"axis={axis} "
                f"displacement="
                f"{displacement_m:.4f} m"
            ),
            (25, h - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            status,
            (25, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

        return output

    # --------------------------------------------------------

    def show_action(
        self,
        frame,
        axis,
        displacement_m,
        status,
    ):

        output = self.draw_arm(
            frame,
            axis,
            displacement_m,
            status,
        )

        cv2.imshow(
            WINDOW_NAME,
            output,
        )

        if self.save_frames:

            path = (
                SAVE_DIR
                / (
                    f"action_"
                    f"{self.step_counter:04d}.png"
                )
            )

            cv2.imwrite(
                str(path),
                output,
            )

        self.step_counter += 1

    # --------------------------------------------------------

    def close(self):

        try:
            cv2.destroyWindow(
                WINDOW_NAME
            )
        except cv2.error:
            pass


# ============================================================
# HUMAN INTERVENTION
# ============================================================

class HumanIntervention:

    """
    Human gate before simulated action.

    ENTER / A = approve
    S         = skip
    R         = reperceive
    P         = pause
    Q / ESC   = quit
    """

    def __init__(
        self,
        auto: bool = False,
    ):

        self.auto = auto

    def decision(self):

        if self.auto:
            return "approved"

        while True:

            key = cv2.waitKey(50) & 0xFF

            if key in (
                13,
                10,
                ord("a"),
                ord("A"),
            ):
                return "approved"

            if key in (
                ord("s"),
                ord("S"),
            ):
                return "skipped"

            if key in (
                ord("r"),
                ord("R"),
            ):
                return "reperceive"

            if key in (
                ord("p"),
                ord("P"),
            ):

                print(
                    "PAUSED - "
                    "ENTER/A approve, "
                    "Q quit"
                )

                while True:

                    paused_key = (
                        cv2.waitKey(50)
                        & 0xFF
                    )

                    if paused_key in (
                        13,
                        10,
                        ord("a"),
                        ord("A"),
                    ):
                        return "approved"

                    if paused_key in (
                        ord("q"),
                        ord("Q"),
                    ):
                        return "quit"

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                return "quit"


# ============================================================
# ROBOT AGENT
# ============================================================

class VisionRobotAgent:

    def __init__(
        self,
        camera: VisionCamera,
        recognizer: YOLORecognizer,
        visualizer: VisionGuidedArmOverlay,
        auto: bool = False,
        minimum_confidence: float = DEFAULT_CONFIDENCE,
    ):

        self.camera = camera
        self.recognizer = recognizer
        self.visualizer = visualizer

        self.intervention = (
            HumanIntervention(
                auto=auto
            )
        )

        self.minimum_confidence = (
            minimum_confidence
        )

        self.registry = (
            SharedRegionRegistry()
        )

        self.world = SharedWorld()

        self.scratchpad = Scratchpad()

        self.running = True

    # --------------------------------------------------------

    def perceive(self):

        ok, frame, metadata = (
            self.camera.read()
        )

        if not ok:
            return (
                None,
                [],
                metadata,
            )

        detections = (
            self.recognizer.detect(
                frame
            )
        )

        objects = (
            self.registry.update(
                detections,
                self.world.frame_number + 1,
            )
        )

        relations = (
            build_border_relations(
                objects
            )
        )

        ontology = make_ontology(
            objects,
            relations,
            metadata,
        )

        self.world.update(
            objects,
            relations,
            ontology,
        )

        return (
            frame,
            objects,
            metadata,
        )

    # --------------------------------------------------------

    def choose_object(
        self,
        objects,
    ):

        usable = [
            obj
            for obj in objects
            if (
                obj.confidence
                >= self.minimum_confidence
            )
        ]

        if not usable:
            return None

        return max(
            usable,
            key=lambda obj: (
                obj.confidence,
                obj.area_px,
            ),
        )

    # --------------------------------------------------------

    def proposed_action(
        self,
        obj,
    ):

        x, y = obj.center

        if x < 400:

            direction = "x-left"

        elif x > 850:

            direction = "x-right"

        else:

            direction = "center"

        return (
            f"inspect {obj.label} "
            f"{obj.shared_id} "
            f"at pixel ({x},{y}); "
            f"direction={direction}"
        )

    # --------------------------------------------------------

    def run_perception_cycle(self):

        (
            frame,
            objects,
            metadata,
        ) = self.perceive()

        if frame is None:
            return "quit"

        if not objects:

            display = frame.copy()

            cv2.putText(
                display,
                "YOLO: no objects detected",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "R = retry    Q = quit",
                (25, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            if self.intervention.auto:

                time.sleep(
                    0.25
                )

                return "reperceive"

            while True:

                key = (
                    cv2.waitKey(50)
                    & 0xFF
                )

                if key in (
                    ord("r"),
                    ord("R"),
                ):
                    return "reperceive"

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    return "quit"

        selected = (
            self.choose_object(
                objects
            )
        )

        if selected is None:
            return "reperceive"

        action = (
            self.proposed_action(
                selected
            )
        )

        # The planner's confidence is represented independently from
        # YOLO's perception confidence.
        #
        # For now this is a conservative direct correspondence.
        # A physical implementation can replace it with a genuine
        # motion-feasibility model.
        planner_confidence = clamp(
            selected.confidence,
            0.0,
            1.0,
        )

        agreement = (
            compute_uncertainty_agreement(
                selected,
                planner_confidence,
            )
        )

        self.world.agreements.append(
            agreement
        )

        self.scratchpad.add(
            ScratchpadEntry(
                timestamp_ns=timestamp_ns(),
                shared_id=selected.shared_id,
                observation=(
                    f"YOLO recognized "
                    f"{selected.label} "
                    f"with confidence "
                    f"{selected.confidence:.3f}."
                ),
                uncertainty=(
                    selected.uncertainty
                ),
                proposed_action=action,
            )
        )

        # ----------------------------------------------------
        # CRITICAL:
        #
        # The image is displayed BEFORE the decision.
        # No simulated action occurs before this gate.
        # ----------------------------------------------------

        self.visualizer.show_decision_snapshot(
            frame,
            objects,
            metadata,
            action,
        )

        decision = (
            self.intervention.decision()
        )

        self.scratchpad.entries[
            -1
        ].decision = decision

        if decision == "approved":

            self.execute_simulated_action(
                frame,
                selected,
                action,
            )

            return "continue"

        if decision == "reperceive":
            return "reperceive"

        if decision == "skipped":
            return "continue"

        return "quit"

    # --------------------------------------------------------

    def execute_simulated_action(
        self,
        frame,
        obj,
        action,
    ):

        x1, y1, x2, y2 = obj.bbox

        image_width = max(
            1,
            frame.shape[1],
        )

        normalized_x = (
            (
                x1 + x2
            )
            * 0.5
            / image_width
        )

        displacement = (
            normalized_x
            - 0.5
        ) * 0.20

        axis = "x"

        self.visualizer.show_action(
            frame,
            axis,
            displacement,
            f"APPROVED: {action}",
        )

        if self.intervention.auto:

            time.sleep(
                0.15
            )

        else:

            cv2.waitKey(
                150
            )

    # --------------------------------------------------------

    def run(
        self,
        max_cycles: int = 0,
    ):

        cycles = 0

        try:

            while self.running:

                result = (
                    self.run_perception_cycle()
                )

                if result == "quit":
                    break

                cycles += 1

                if (
                    max_cycles > 0
                    and cycles >= max_cycles
                ):
                    break

        finally:

            self.world.save()

            self.scratchpad.save()

            self.visualizer.close()

            self.camera.close()


# ============================================================
# DEMO SUBSET-SUM JOBS
# ============================================================

def make_demo_jobs():

    random.seed(4)

    target = CartesianTarget(
        x_m=0.250,
        y_m=0.180,
        z_m=0.187,
        precision_ticks=2,
    )

    jobs = []

    for axis in AXES:

        moves = []

        for i in range(149):

            amount_mm = random.randint(
                1,
                8,
            )

            if axis == "x":

                move = CartesianMove(
                    move_id=i,
                    dx_m=(
                        amount_mm
                        / 1000.0
                    ),
                )

            elif axis == "y":

                move = CartesianMove(
                    move_id=i,
                    dy_m=(
                        amount_mm
                        / 1000.0
                    ),
                )

            else:

                move = CartesianMove(
                    move_id=i,
                    dz_m=(
                        amount_mm
                        / 1000.0
                    ),
                )

            moves.append(
                move
            )

        jobs.append(
            AxisJob(
                limb="left_arm",
                axis=axis,
                moves=moves,
                target=target,
                is_approach_axis=(
                    axis == "z"
                ),
                c=2.0,
                max_beam_width=2000,
                seed=(
                    4
                    + AXES.index(axis)
                ),
            )
        )

    return jobs


def print_plan(
    results,
):

    print()
    print(
        "SUBSET-SUM PLAN"
    )
    print(
        "=" * 70
    )

    for result in results:

        print(
            f"{result.limb:12s} "
            f"axis={result.axis} "
            f"found={result.found} "
            f"displacement="
            f"{result.displacement_m:.4f} m "
            f"moves="
            f"{len(result.move_ids_used)}"
        )

        if result.found:

            preview = (
                result.move_ids_used[
                    :20
                ]
            )

            suffix = (
                " ..."
                if len(
                    result.move_ids_used
                ) > 20
                else ""
            )

            print(
                f"    move IDs: "
                f"{preview}{suffix}"
            )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "YOLO image-recognition "
            "robot-agent simulation"
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Ultralytics YOLO model. "
            f"Default: {DEFAULT_MODEL}"
        ),
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
        default=DEFAULT_CONFIDENCE,
        help=(
            "YOLO confidence threshold"
        ),
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
        default=DEFAULT_IMAGE_SIZE,
        help="YOLO inference image size",
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "YOLO device: cpu, 0, "
            "cuda:0, etc."
        ),
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Automatically approve "
            "simulated actions"
        ),
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help=(
            "Maximum perception cycles. "
            "0 = continuous."
        ),
    )

    parser.add_argument(
        "--no-plan",
        action="store_true",
        help=(
            "Skip subset-sum planning "
            "demonstration"
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print()
    print(
        "=" * 78
    )
    print(
        "YOLO VISION-GUIDED ROBOT AGENT"
    )
    print(
        "=" * 78
    )

    print(
        f"YOLO model:         {args.model}"
    )

    print(
        f"YOLO confidence:    {args.conf}"
    )

    print(
        f"YOLO IoU:           {args.iou}"
    )

    print(
        f"YOLO image size:    {args.imgsz}"
    )

    print(
        f"Requested exposure: "
        f"{REQUESTED_EXPOSURE_US} us"
    )

    print(
        "Human gate:         "
        + (
            "OFF (--auto)"
            if args.auto
            else "ON"
        )
    )

    # --------------------------------------------------------
    # Motion planner demonstration
    # --------------------------------------------------------

    if not args.no_plan:

        print()
        print(
            "Solving Cartesian "
            "subset-sum jobs..."
        )

        jobs = make_demo_jobs()

        try:

            results = (
                solve_all_limbs(
                    jobs
                )
            )

            print_plan(
                results
            )

        except Exception as exc:

            print(
                "Planner demonstration "
                f"failed: {exc}"
            )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera = VisionCamera(
        camera_index=args.camera,
        requested_exposure_us=(
            REQUESTED_EXPOSURE_US
        ),
    )

    print()
    print(
        f"Camera backend:     "
        f"{camera.backend_name}"
    )

    print(
        f"Real camera active: "
        f"{camera.real_camera_active}"
    )

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    recognizer = YOLORecognizer(
        model_path=args.model,
        confidence=args.conf,
        iou=args.iou,
        device=args.device,
        image_size=args.imgsz,
    )

    try:

        recognizer.load()

    except Exception as exc:

        camera.close()

        raise SystemExit(
            str(exc)
        )

    # --------------------------------------------------------
    # Visualizer
    # --------------------------------------------------------

    visualizer = (
        VisionGuidedArmOverlay(
            camera=camera,
            recognizer=recognizer,
        )
    )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    agent = VisionRobotAgent(
        camera=camera,
        recognizer=recognizer,
        visualizer=visualizer,
        auto=args.auto,
        minimum_confidence=args.conf,
    )

    print()
    print(
        "CONTROLS"
    )
    print(
        "  ENTER / A : approve"
    )
    print(
        "  S         : skip"
    )
    print(
        "  R         : capture another image"
    )
    print(
        "  P         : pause"
    )
    print(
        "  Q / ESC   : quit"
    )
    print()

    agent.run(
        max_cycles=args.cycles
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"  {WORLD_FILE}"
    )

    print(
        f"  {SCRATCHPAD_FILE}"
    )

    print(
        f"  {SAVE_DIR}/"
    )


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    freeze_support()

    main()
