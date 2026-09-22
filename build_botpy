"""
offline_vision_3d_vessel_agent.py

Continuous Webcam + Canny Edge Detector + 3D Cartesian & Euler Orientation Estimation 
+ Object Count Reward & 3D Vessel Steering + Multi-Axis 3D Rotation HUD.

Install:
    pip install opencv-python numpy
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

DEFAULT_MIN_CONTOUR_AREA = 150
COLLAGE_DIR = Path("3d_geometry_collages")
WINDOW_NAME = "3D Vessel Navigation & Multi-Axis Rotation HUD Agent"


# ============================================================
# CONTINUOUS WEBCAM & 3D EDGE DETECTOR INFRASTRUCTURE
# ============================================================

class ContinuousCamera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.index = index
        self.width = width
        self.height = height
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.stop_event = threading.Event()
        self._open()
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_ANY)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ok, frame = cap.read()
            if ok and frame is not None:
                self.cap = cap
                return
            cap.release()

    def _loop(self):
        while not self.stop_event.is_set():
            if self.cap is None:
                frame = self._synthetic_frame()
            else:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue
            with self.lock:
                self.latest_frame = frame.copy()

    def _synthetic_frame(self):
        h, w = self.height, self.width
        t = time.monotonic()
        frame = np.full((h, w, 3), 35, dtype=np.uint8)
        for i in range(3):
            offset_x = int(150 * math.sin(t + i))
            x1 = int(w * (0.2 + 0.2 * i) + offset_x)
            y1 = int(h * 0.4 + 50 * math.cos(t * 0.5 + i))
            cv2.rectangle(frame, (x1 - 50, y1 - 40), (x1 + 50, y1 + 40), (150 + i * 35, 100, 200), 3)
        cv2.putText(frame, "SYNTHETIC 3D MULTI-OBJECT FEED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
        return frame

    def latest(self):
        with self.lock:
            if self.latest_frame is None: return None
            return self.latest_frame.copy()

    def stop(self):
        self.stop_event.set()
        if self.cap: self.cap.release()


@dataclass
class RecognizedObject3D:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    coord_3d: tuple[float, float, float]  # (X, Y, Z) in 3D space
    euler_angles: tuple[float, float, float]  # (Roll, Pitch, Yaw) in degrees
    track_id: int = 1


class Cartesian3DCoordLearner:
    """Estimates 3D spatial coordinates $(X, Y, Z)$ and 3D rotational Euler angles $(\text{Roll}, \text{Pitch}, \text{Yaw})$."""
    def __init__(self):
        self.spatial_memory: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {}

    def update(self, key: str, contour, bbox: Tuple[int, int, int, int], frame_w: int, frame_h: int):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        # Normalized 3D Cartesian coordinates
        norm_x = (cx - (frame_w / 2.0)) / (frame_w / 2.0)
        norm_y = ((frame_h / 2.0) - cy) / (frame_h / 2.0)
        
        box_area = max(1.0, (x2 - x1) * (y2 - y1))
        frame_area = max(1.0, float(frame_w * frame_h))
        depth_z = math.sqrt(frame_area / box_area)
        
        coord_3d = (round(norm_x, 2), round(norm_y, 2), round(depth_z, 2))

        # 3D Rotational Pose Estimation via Min Area Rect & Contour Moments
        rect = cv2.minAreaRect(contour)
        box_angle = rect[2]
        
        # Estimate 3D Euler angles: Roll, Pitch, Yaw (in degrees)
        # Using aspect ratio and contour elongation to simulate 3D tilt
        w_box, h_box = rect[1]
        aspect = w_box / h_box if h_box > 0 else 1.0
        pitch = round((aspect - 1.0) * 45.0, 1)
        roll = round(box_angle if box_angle <= 90 else box_angle - 90, 1)
        yaw = round(norm_x * 30.0, 1)  # View angle offset relative to camera center
        
        euler_angles = (roll, pitch, yaw)

        # Temporal smoothing
        if key in self.spatial_memory:
            prev_coord, prev_euler = self.spatial_memory[key]
            alpha = 0.6
            sm_coord = tuple(round(alpha * c + (1 - alpha) * p, 2) for c, p in zip(coord_3d, prev_coord))
            sm_euler = tuple(round(alpha * c + (1 - alpha) * p, 2) for c, p in zip(euler_angles, prev_euler))
            self.spatial_memory[key] = (sm_coord, sm_euler)
        else:
            self.spatial_memory[key] = (coord_3d, euler_angles)

        return self.spatial_memory[key]


class EdgeDetectorRecognizer3D:
    def __init__(self, camera, min_area: int = DEFAULT_MIN_CONTOUR_AREA):
        self.camera = camera
        self.min_area = min_area
        self.coord_learner = Cartesian3DCoordLearner()
        self.lock = threading.Lock()
        self._objects = []
        self.stop_event = threading.Event()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop_event.is_set():
            frame = self.camera.latest()
            if frame is None:
                time.sleep(0.005)
                continue
            try:
                h, w, _ = frame.shape
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                detections = []
                idx = 1
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > self.min_area:
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        aspect_ratio = float(bw) / bh if bh > 0 else 1.0
                        if 0.8 <= aspect_ratio <= 1.2:
                            shape_name = "cube_square"
                        elif aspect_ratio > 1.2:
                            shape_name = "prism_wide"
                        else:
                            shape_name = "cylinder_tall"

                        unique_key = f"{shape_name}_id{idx}"
                        coord_3d, euler = self.coord_learner.update(unique_key, cnt, (x, y, x+bw, y+bh), w, h)

                        confidence = min(1.0, area / 10000.0)
                        detections.append(RecognizedObject3D(
                            label=shape_name,
                            confidence=confidence,
                            bbox=(x, y, x + bw, y + bh),
                            coord_3d=coord_3d,
                            euler_angles=euler,
                            track_id=idx
                        ))
                        idx += 1

                with self.lock:
                    self._objects = detections
            except Exception:
                time.sleep(0.01)

    def get_latest_objects(self):
        with self.lock:
            return list(self._objects)

    def stop(self):
        self.stop_event.set()


# ============================================================
# 3D VESSEL NAVIGATION & MULTI-AXIS ROTATION HUD CONTROLLER
# ============================================================

class Vessel3DHUDController:
    """
    Steers a 3D simulated robot vessel using object count reward, generates 3D-aligned 
    collages, and computes multi-axis rotation HUD telemetry ($\Delta \text{Roll}, \Delta \text{Pitch}, \Delta \text{Yaw}$).
    """
    def __init__(self):
        COLLAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.vessel_heading_3d = (0.0, 0.0, 0.0)  # (dX, dY, dZ)
        self.reward_score = 0.0
        self.rotation_telemetry_3d: Dict[str, Tuple[float, float, float]] = {}

    def evaluate_and_steer(self, frame: np.ndarray, objects: List[RecognizedObject3D]) -> Tuple[str, Optional[np.ndarray]]:
        if frame is None or not objects:
            self.rotation_telemetry_3d.clear()
            return "SEARCHING FOR 3D CLUSTERS (REWARD: 0.0)", None

        h, w, _ = frame.shape
        self.reward_score = float(len(objects))

        # Compute 3D center of mass for vessel heading telemetry
        mean_x = sum(obj.coord_3d[0] for obj in objects) / len(objects)
        mean_y = sum(obj.coord_3d[1] for obj in objects) / len(objects)
        mean_z = sum(obj.coord_3d[2] for obj in objects) / len(objects)
        self.vessel_heading_3d = (round(mean_x, 2), round(mean_y, 2), round(mean_z, 2))

        # Calculate multi-axis rotational adjustments needed to normalize subgeometries to target frame axes ($0^\circ, 0^\circ, 0^\circ$)
        self.rotation_telemetry_3d.clear()
        for idx, obj in enumerate(objects):
            roll_adj = -obj.euler_angles[0]
            pitch_adj = -obj.euler_angles[1]
            yaw_adj = -obj.euler_angles[2]
            key = f"Obj#{idx+1} ({obj.label})"
            self.rotation_telemetry_3d[key] = (round(roll_adj, 1), round(pitch_adj, 1), round(yaw_adj, 1))

        if len(objects) >= 2:
            status = f"3D CLUSTER REACHED [Reward: {self.reward_score}] -> 3D ROTATION HUD & COLLAGE ACTIVE"
            collage = self._generate_3d_geometric_collage(frame, objects)
            return status, collage
        else:
            status = f"STEERING 3D VESSEL (Heading X,Y,Z: {self.vessel_heading_3d})"
            return status, None

    def _generate_3d_geometric_collage(self, frame: np.ndarray, objects: List[RecognizedObject3D]) -> np.ndarray:
        h, w, _ = frame.shape
        base_canvas = frame.copy()
        alpha = 0.6
        
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crop_resized = cv2.resize(crop, (300, 200))
                
                # Apply 3D affine transformation matrix using Roll angle for multi-axis alignment
                center = (150, 100)
                roll_angle = obj.euler_angles[0]
                M = cv2.getRotationMatrix2D(center, roll_angle, 1.0)
                aligned_crop = cv2.warpAffine(crop_resized, M, (300, 200))

                ch, cw, _ = aligned_crop.shape
                if y1 + ch <= h and x1 + cw <= w:
                    roi = base_canvas[y1:y1+ch, x1:x1+cw]
                    blended = cv2.addWeighted(roi, 1.0 - alpha, aligned_crop, alpha, 0)
                    base_canvas[y1:y1+ch, x1:x1+cw] = blended

        timestamp = time.strftime("%H%M%S")
        collage_path = COLLAGE_DIR / f"3d_aligned_collage_{timestamp}.jpg"
        cv2.imwrite(str(collage_path), base_canvas)
        return base_canvas


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_CONTOUR_AREA)
    args = parser.parse_args()

    print("=" * 78)
    print("3D VESSEL NAVIGATION & MULTI-AXIS ROTATION HUD AGENT")
    print("=" * 78)

    camera = ContinuousCamera(index=args.camera)
    recognizer = EdgeDetectorRecognizer3D(camera, min_area=args.min_area)
    recognizer.start()

    controller = Vessel3DHUDController()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print(f"\n3D Collages saving to: ./{COLLAGE_DIR.name}/")
    print("Running 3D agent. Press ESC or 'q' to exit.\n")

    try:
        while True:
            frame = camera.latest()
            if frame is None:
                time.sleep(0.01)
                continue

            objects = recognizer.get_latest_objects()
            status_text, collage = controller.evaluate_and_steer(frame, objects)
            
            # Draw 3D bounding boxes, coordinates, and Euler angles
            for obj in objects:
                x1, y1, x2, y2 = obj.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                tag = f"{obj.label} | XYZ{obj.coord_3d}"
                euler_tag = f"R:{obj.euler_angles[0]} P:{obj.euler_angles[1]} Y:{obj.euler_angles[2]}"
                cv2.putText(frame, tag, (x1, max(20, y1 - 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                cv2.putText(frame, euler_tag, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Display Status & 3D Vessel Telemetry
            cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"3D Vessel Steering Vector (X,Y,Z): {controller.vessel_heading_3d}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            # Draw Multi-Axis 3D Rotation HUD Panel
            hud_y = 105
            cv2.putText(frame, "--- 3D HUD: MULTI-AXIS ROTATION TELEMETRY ($\Delta$Roll, Pitch, Yaw) ---", (20, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 255), 2, cv2.LINE_AA)
            for obj_key, (r_adj, p_adj, y_adj) in controller.rotation_telemetry_3d.items():
                hud_y += 26
                hud_txt = f"{obj_key} -> $\Delta$Roll:{r_adj:+.1f}° | $\Delta$Pitch:{p_adj:+.1f}° | $\Delta$Yaw:{y_adj:+.1f}°"
                cv2.putText(frame, hud_txt, (35, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)

            if collage is not None:
                cv2.imshow("3D Aligned Subgeometry Collage", collage)

            if cv2.waitKey(30) & 0xFF in (ord('q'), ord('Q'), 27):
                break
    except KeyboardInterrupt:
        print("\nStopping agent...")
    finally:
        recognizer.stop()
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
