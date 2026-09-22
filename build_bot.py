"""
offline_vision_3d_ergonomic_agent.py

Continuous Webcam + Canny Edge Detector + 3D Cartesian & Euler Pose Estimation 
+ Object Count Reward & 3D Vessel Steering + Ergonomic Fit Evaluation & Contour-Masked Transparent Collage 
+ Feature-Matched Unique Connectors HUD.

Install:
    pip install opencv-python numpy
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

DEFAULT_MIN_CONTOUR_AREA = 65
COLLAGE_DIR = Path("3d_ergonomic_collages")
WINDOW_NAME = "3D Vessel Navigation & Ergonomic Fit HUD Agent"


# ============================================================
# ERGONOMIC FIT ENGINE
# ============================================================

class ErgonomicFitEvaluator:
    """
    Evaluates 3D spatial coordinates and rotational poses against human ergonomic 
    standards (reach zones, comfortable viewing/operation tilt angles, and clearance).
    """
    def __init__(self):
        self.target_optimal_depth = 1.5  # Ideal reach distance
        self.target_comfort_pitch = 50.0 # Ideal ergonomic tilt angle in degrees

    def evaluate_object(self, obj: RecognizedObject3D) -> Tuple[float, Dict[str, float]]:
        x, y, z = obj.coord_3d
        roll, pitch, yaw = obj.euler_angles

        # 1. Depth / Reach Ergonomics
        depth_diff = abs(z - self.target_optimal_depth)
        reach_score = max(0.0, 1.0 - (depth_diff / 2.0))

        # 2. Angular / Posture Ergonomics
        pitch_diff = abs(pitch - self.target_comfort_pitch)
        posture_score = max(0.0, 1.0 - (pitch_diff / 45.0))

        # Combined Ergonomic Fit Score
        ergonomic_score = round((reach_score * 0.5) + (posture_score * 0.5), 2)

        reach_adjustment = round(self.target_optimal_depth - z, 2)
        posture_adjustment = round(self.target_comfort_pitch - pitch, 2)

        adjustments = {
            "reach_adj_z": reach_adjustment,
            "posture_adj_pitch": posture_adjustment,
            "score": ergonomic_score
        }
        return ergonomic_score, adjustments


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
        cv2.putText(frame, "SYNTHETIC ERGONOMIC FEED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
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
    coord_3d: tuple[float, float, float]
    euler_angles: tuple[float, float, float]
    ergonomic_score: float = 0.0
    track_id: int = 1
    contour_mask: Optional[np.ndarray] = field(default=None, repr=False)


class Cartesian3DCoordLearner:
    def __init__(self):
        self.spatial_memory: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {}

    def update(self, key: str, contour, bbox: Tuple[int, int, int, int], frame_w: int, frame_h: int):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        norm_x = (cx - (frame_w / 2.0)) / (frame_w / 2.0)
        norm_y = ((frame_h / 2.0) - cy) / (frame_h / 2.0)
        
        box_area = max(1.0, (x2 - x1) * (y2 - y1))
        frame_area = max(1.0, float(frame_w * frame_h))
        depth_z = math.sqrt(frame_area / box_area)
        
        coord_3d = (round(norm_x, 2), round(norm_y, 2), round(depth_z, 2))

        rect = cv2.minAreaRect(contour)
        box_angle = rect[2]
        
        w_box, h_box = rect[1]
        aspect = w_box / h_box if h_box > 0 else 1.0
        pitch = round((aspect - 1.0) * 45.0, 1)
        roll = round(box_angle if box_angle <= 90 else box_angle - 90, 1)
        yaw = round(norm_x * 30.0, 1)
        
        euler_angles = (roll, pitch, yaw)

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
        self.ergonomic_evaluator = ErgonomicFitEvaluator()
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
                            shape_name = "module_square"
                        elif aspect_ratio > 1.2:
                            shape_name = "panel_wide"
                        else:
                            shape_name = "lever_tall"

                        # Build precise local binary mask for matched feature contour
                        local_cnt = cnt - np.array([x, y], dtype=np.int32)
                        local_mask = np.zeros((bh, bw), dtype=np.uint8)
                        cv2.drawContours(local_mask, [local_cnt], -1, 255, thickness=cv2.FILLED)

                        unique_key = f"{shape_name}_id{idx}"
                        coord_3d, euler = self.coord_learner.update(unique_key, cnt, (x, y, x+bw, y+bh), w, h)

                        temp_obj = RecognizedObject3D(
                            label=shape_name, confidence=1.0, bbox=(x, y, x+bw, y+bh),
                            coord_3d=coord_3d, euler_angles=euler, track_id=idx,
                            contour_mask=local_mask
                        )
                        ergo_score, _ = self.ergonomic_evaluator.evaluate_object(temp_obj)
                        temp_obj.ergonomic_score = ergo_score

                        detections.append(temp_obj)
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
# VESSEL NAVIGATION & ERGONOMIC HUD CONTROLLER
# ============================================================

class VesselErgonomicHUDController:
    """
    Steers the simulated vessel using object count reward, generates precise contour-masked 
    transparent collages, connects matched features, and provides HUD telemetry.
    """
    def __init__(self):
        COLLAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.vessel_heading_3d = (0.0, 0.0, 0.0)
        self.reward_score = 0.0
        self.ergonomic_telemetry: Dict[str, Dict[str, float]] = {}
        self.ergo_evaluator = ErgonomicFitEvaluator()

    def evaluate_and_steer(self, frame: np.ndarray, objects: List[RecognizedObject3D]) -> Tuple[str, Optional[np.ndarray]]:
        if frame is None or not objects:
            self.ergonomic_telemetry.clear()
            return "SEARCHING FOR ERGONOMIC CLUSTERS (REWARD: 0.0)", None

        h, w, _ = frame.shape
        self.reward_score = float(len(objects))

        mean_x = sum(obj.coord_3d[0] for obj in objects) / len(objects)
        mean_y = sum(obj.coord_3d[1] for obj in objects) / len(objects)
        mean_z = sum(obj.coord_3d[2] for obj in objects) / len(objects)
        self.vessel_heading_3d = (round(mean_x, 2), round(mean_y, 2), round(mean_z, 2))

        self.ergonomic_telemetry.clear()
        for idx, obj in enumerate(objects):
            _, adj = self.ergo_evaluator.evaluate_object(obj)
            key = f"Obj#{idx+1} ({obj.label})"
            self.ergonomic_telemetry[key] = {
                "score": obj.ergonomic_score,
                "reach_z": adj["reach_adj_z"],
                "pitch_deg": adj["posture_adj_pitch"]
            }

        if len(objects) >= 2:
            status = f"CLUSTER REACHED [Reward: {self.reward_score}] -> FEATURE-MATCHED COLLAGE"
            collage = self._generate_ergonomic_transparent_collage(frame, objects)
            return status, collage
        else:
            status = f"STEERING TO ERGONOMIC CLUSTER (Heading X,Y,Z: {self.vessel_heading_3d})"
            return status, None

    def _generate_ergonomic_transparent_collage(self, frame: np.ndarray, objects: List[RecognizedObject3D]) -> np.ndarray:
        h, w, _ = frame.shape
        base_canvas = frame.copy()
        overlay_alpha = 0.45

        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crop_resized = cv2.resize(crop, (300, 200))

                if obj.contour_mask is not None and obj.contour_mask.size > 0:
                    mask_resized = cv2.resize(obj.contour_mask, (300, 200), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = np.ones((200, 300), dtype=np.uint8) * 255

                center = (150, 100)
                ergonomic_angle = obj.euler_angles[0]
                M = cv2.getRotationMatrix2D(center, ergonomic_angle, 1.0)
                
                aligned_crop = cv2.warpAffine(crop_resized, M, (300, 200), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                aligned_mask = cv2.warpAffine(mask_resized, M, (300, 200), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                ch, cw, _ = aligned_crop.shape
                if y1 + ch <= h and x1 + cw <= w:
                    roi = base_canvas[y1:y1+ch, x1:x1+cw]
                    mask_3ch = cv2.cvtColor(aligned_mask, cv2.COLOR_GRAY2BGR) / 255.0

                    blended_region = (roi * (1.0 - (mask_3ch * overlay_alpha))) + (aligned_crop * (mask_3ch * overlay_alpha))
                    base_canvas[y1:y1+ch, x1:x1+cw] = blended_region.astype(np.uint8)

        timestamp = time.strftime("%H%M%S")
        collage_path = COLLAGE_DIR / f"feature_matched_collage_{timestamp}.jpg"
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
    print("3D VESSEL NAVIGATION & ERGONOMIC FIT HUD AGENT")
    print("=" * 78)

    camera = ContinuousCamera(index=args.camera)
    recognizer = EdgeDetectorRecognizer3D(camera, min_area=args.min_area)
    recognizer.start()

    controller = VesselErgonomicHUDController()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print(f"\nErgonomic Collages saving to: ./{COLLAGE_DIR.name}/")
    print("Running ergonomic agent. Press ESC or 'q' to exit.\n")

    try:
        while True:
            frame = camera.latest()
            if frame is None:
                time.sleep(0.01)
                continue

            objects = recognizer.get_latest_objects()
            status_text, collage = controller.evaluate_and_steer(frame, objects)
            
            # Find and connect genuinely matching feature pairs (based on label affinity and spatial proximity)
            matched_pairs = set()
            if len(objects) >= 2:
                for i, obj_a in enumerate(objects):
                    best_match_idx = -1
                    min_cost = float('inf')
                    for j, obj_b in enumerate(objects):
                        if i == j:
                            continue
                        
                        # Calculate matching cost: prioritize same label type, then spatial distance
                        label_penalty = 0.0 if obj_a.label == obj_b.label else 500.0
                        
                        pt_a = np.array([(obj_a.bbox[0] + obj_a.bbox[2]) / 2, (obj_a.bbox[1] + obj_a.bbox[3]) / 2])
                        pt_b = np.array([(obj_b.bbox[0] + obj_b.bbox[2]) / 2, (obj_b.bbox[1] + obj_b.bbox[3]) / 2])
                        spatial_dist = np.linalg.norm(pt_a - pt_b)
                        
                        cost = label_penalty + spatial_dist
                        if cost < min_cost:
                            min_cost = cost
                            best_match_idx = j

                    if best_match_idx != -1:
                        pair = tuple(sorted([i, best_match_idx]))
                        matched_pairs.add(pair)

                # Draw unique feature-matched connector lines
                for idx_a, idx_b in matched_pairs:
                    obj_a = objects[idx_a]
                    obj_b = objects[idx_b]
                    pt_a = int((obj_a.bbox[0] + obj_a.bbox[2]) / 2), int((obj_a.bbox[1] + obj_a.bbox[3]) / 2)
                    pt_b = int((obj_b.bbox[0] + obj_b.bbox[2]) / 2), int((obj_b.bbox[1] + obj_b.bbox[3]) / 2)

                    match_color = (0, 255, 255) if obj_a.label == obj_b.label else (255, 165, 0)
                    cv2.line(frame, pt_a, pt_b, match_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, pt_a, 4, (255, 255, 255), -1)
                    cv2.circle(frame, pt_b, 4, (255, 255, 255), -1)

            for obj in objects:
                x1, y1, x2, y2 = obj.bbox
                box_color = (0, 255, 0) if obj.ergonomic_score >= 0.6 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                tag = f"{obj.label} | ErgoScore: {obj.ergonomic_score}"
                cv2.putText(frame, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

            cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"3D Vessel Steering Vector (X,Y,Z): {controller.vessel_heading_3d}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            hud_y = 105
            cv2.putText(frame, "--- ERGONOMIC FIT HUD: REACH & POSTURE TELEMETRY ---", (20, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 255), 2, cv2.LINE_AA)
            for obj_key, data in controller.ergonomic_telemetry.items():
                hud_y += 26
                hud_txt = f"{obj_key} -> FitScore:{data['score']} | ReachAdjZ:{data['reach_z']:+.2f} | PitchAdj:{data['pitch_deg']:+.1f}°"
                cv2.putText(frame, hud_txt, (35, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)

            if collage is not None:
                cv2.imshow("Feature-Matched Connector Collage", collage)

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
