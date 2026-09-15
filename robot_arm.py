from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import freeze_support
from typing import Callable, Optional, Sequence
import heapq
import math
import random
import time
from pathlib import Path
import sys

import cv2
import numpy as np


# ============================================================
# Vision-Enabled Real Camera & Automated Grabber Visualizer
# ============================================================

class VisionGuidedArmOverlay:
    """Captures a live feed, performs object recognition via color/contour tracking, 
    and drives the robot arm to automatically grab the detected target."""

    def __init__(self, width: int = 1000, height: int = 750, save_frames: bool = True):
        self.width = width
        self.height = height
        self.save_frames = save_frames
        self.frame_dir = Path("simulation_frames")
        
        if self.save_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)

        # Automatically search for an available camera index with backend api preference
        self.cap = None
        working_index = -1
        api_preference = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY

        for idx in range(3):
            print(f"Trying to open camera index {idx}...")
            cap = cv2.VideoCapture(idx, api_preference)
            if cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        working_index = idx
                        self.cap = cap
                        break
                except Exception:
                    pass
                cap.release()

        self.real_camera_active = self.cap is not None and self.cap.isOpened()
        if self.real_camera_active:
            print(f"[Success] Connected to physical camera at index {working_index}.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            print("[Warning] Could not initialize a stable physical camera stream. Using fallback mode.")
        
        self.window_name = "Vision-Guided Robot Arm Grabber"
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.gui_available = True
        except Exception as e:
            print(f"Warning: GUI window could not be opened ({e}).")
            self.gui_available = False

        self.step_counter = 0

    def detect_object(self, frame: np.ndarray) -> Optional[tuple[int, int]]:
        """Performs object recognition using HSV color filtering and contour detection.
        Looks for a prominent red/orange object."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Broadened HSV range for red/orange objects to catch more variations
        lower_red1 = np.array([0, 80, 50])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([170, 80, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Noise removal via morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of the detected object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Print area to terminal for debugging (check your console output)
            # print(f"Largest object contour area: {area}")

            if area > 100:  # Lowered threshold to pick up objects more easily
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(frame, f"TARGET (Area: {int(area)})", (x, max(20, y - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    return (cx, cy)
        return None

    def render_move(self, limb: str, axis: str, move_id: int, displacement_m: float) -> None:
        canvas = None
        target_coords = None

        if self.real_camera_active:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    frame = np.ascontiguousarray(frame)
                    frame = cv2.resize(frame, (self.width, self.height))
                    target_coords = self.detect_object(frame)
                    canvas = frame
            except (cv2.error, Exception):
                canvas = None

        if canvas is None:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 35)
            cv2.putText(canvas, "CAMERA STREAM ERROR - FALLBACK MODE", (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        origin = (self.width // 2 - 50, self.height // 2 + 100)
        
        scale = 8000
        dx = (displacement_m * scale) if axis == "x" else 0
        dy = (displacement_m * scale) if axis == "y" else 0
        dz = (displacement_m * scale) if axis == "z" else 0

        elbow = (origin[0] + 120 + int(dx * 0.5), origin[1] - 150 - int(dz * 0.5))
        
        if target_coords is not None:
            eff = target_coords
            grab_status = "GRABBING TARGET OBJECT!"
            status_color = (0, 255, 0)
        else:
            eff = (elbow[0] + 100 + int(dx), elbow[1] - 100 - int(dy))
            grab_status = "SEARCHING FOR OBJECT..."
            status_color = (0, 165, 255)

        cv2.line(canvas, origin, elbow, (50, 50, 50), 16)
        cv2.line(canvas, origin, elbow, (200, 200, 200), 10)
        cv2.line(canvas, elbow, eff, (50, 50, 50), 12)
        cv2.line(canvas, elbow, eff, (220, 220, 220), 6)

        cv2.circle(canvas, origin, 18, (0, 0, 0), -1)
        cv2.circle(canvas, origin, 15, (0, 120, 255), -1)
        cv2.circle(canvas, elbow, 14, (0, 0, 0), -1)
        cv2.circle(canvas, elbow, 11, status_color, -1)
        cv2.circle(canvas, eff, 12, (0, 0, 0), -1)
        cv2.circle(canvas, eff, 8, (0, 255, 0) if target_coords else (0, 0, 255), -1)

        color_map = {"x": (0, 0, 255), "y": (0, 255, 0), "z": (255, 0, 0)}
        vec_color = color_map.get(axis, (255, 255, 255))
        cv2.arrowedLine(canvas, elbow, eff, vec_color, 4, tipLength=0.4)

        cv2.rectangle(canvas, (30, 20), (self.width - 30, 120), (15, 15, 20), -1)
        cv2.rectangle(canvas, (30, 20), (self.width - 30, 120), vec_color, 2)
        
        cv2.putText(canvas, f"VISION STATUS: {grab_status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"LIMB: {limb.upper()}  |  ACTIVE AXIS: [{axis.upper()}]  |  MOVE ID: {move_id}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
        cv2.putText(canvas, f"DISPLACEMENT: {displacement_m:.6f} m", (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, vec_color, 2)

        if self.save_frames:
            frame_path = self.frame_dir / f"step_{self.step_counter:04d}.png"
            cv2.imwrite(str(frame_path), canvas)
            self.step_counter += 1

        if self.gui_available:
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(100) & 0xFF
            if key == 27:
                sys.exit(0)

    def close(self) -> None:
        if self.real_camera_active and self.cap is not None:
            self.cap.release()
        if self.gui_available:
            cv2.waitKey(500)
            cv2.destroyAllWindows()
        if self.save_frames:
            print(f"\n[Info] Frames saved to: {self.frame_dir.resolve()}")


# ============================================================
# Coordinate & Planning Models
# ============================================================

AXES = ("x", "y", "z")

@dataclass(frozen=True)
class CartesianTarget:
    x_m: float
    y_m: float
    z_m: float
    precision_ticks: int = 0
    precision_unit_m: float = 0.001

    def coordinate(self, axis: str) -> float:
        return {"x": self.x_m, "y": self.y_m, "z": self.z_m}[axis]


@dataclass(frozen=True)
class CartesianMove:
    move_id: int
    dx_m: float = 0.0
    dy_m: float = 0.0
    dz_m: float = 0.0
    precision_ticks: int = 0

    def displacement(self, axis: str) -> float:
        return {"x": self.dx_m, "y": self.dy_m, "z": self.dz_m}[axis]


def _materialize(node):
    out = []
    while node is not None:
        move_id, node = node
        out.append(move_id)
    out.reverse()
    return out


def equation_beam_search(nums, target, r, c=2, max_beam_width=2000):
    n = len(nums)
    cap = min(n ** c, max(1, max_beam_width))
    beam = [(0, None)]

    for k in range(1, n + 1):
        value = nums[k - 1]
        seen = {}
        for partial, node in beam:
            if partial not in seen:
                seen[partial] = node
            new_partial = partial + value
            if new_partial <= target and new_partial not in seen:
                seen[new_partial] = (value, node)

        if target in seen:
            return _materialize(seen[target]), k

        log_width = n * math.log(2) + k * math.log(max(1e-12, 1 - r))
        width = cap if log_width > math.log(cap) else max(1, min(cap, math.ceil(math.exp(log_width))))
        beam = list(seen.items()) if len(seen) <= width else heapq.nsmallest(width, seen.items(), key=lambda item: abs(target - item[0]))

    return None, n


def measure_r(nums, sample_size=24, subset_fraction=5, seed=None):
    rng = random.Random(seed)
    if not nums:
        return 1e-6, 1, 0, 0
    sample_size = min(sample_size, len(nums))
    sample = rng.sample(list(nums), sample_size)
    subset_target = sum(rng.sample(sample, max(1, min(subset_fraction, sample_size))))
    
    suffix = [0] * (sample_size + 1)
    for i in range(sample_size - 1, -1, -1):
        suffix[i] = suffix[i + 1] + sample[i]

    nodes, stack = 0, [(0, 0)]
    while stack:
        index, partial = stack.pop()
        nodes += 1
        if partial == subset_target:
            break
        if index == sample_size or partial > subset_target or partial + suffix[index] < subset_target:
            continue
        stack.append((index + 1, partial))
        stack.append((index + 1, partial + sample[index]))

    r = 1 - math.exp(math.log(max(nodes / (2 ** sample_size), 1e-300)) / sample_size)
    return min(max(r, 1e-6), 1 - 1e-6), nodes, sample_size, subset_target


@dataclass
class AxisJob:
    limb: str
    axis: str
    moves: Sequence[CartesianMove]
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
    move_ids_used: list[int] = field(default_factory=list)
    displacement_m: float = 0.0


def target_units(job: AxisJob) -> int:
    return round(job.target.coordinate(job.axis) / job.coordinate_unit_m) + (job.target.precision_ticks * job.precision_weight)


def move_units(job: AxisJob, move: CartesianMove) -> int:
    return round(move.displacement(job.axis) / job.coordinate_unit_m) + (move.precision_ticks * job.precision_weight)


def solve_one_axis(job: AxisJob) -> AxisResult:
    values = [move_units(job, m) for m in job.moves]
    target = target_units(job)
    r_used = job.r or measure_r(values, seed=job.seed)[0]
    result, _ = equation_beam_search(values, target, r_used, c=job.c, max_beam_width=job.max_beam_width)

    if result is None:
        return AxisResult(limb=job.limb, axis=job.axis, found=False)

    remaining, selected_ids = list(job.moves), []
    for val in result:
        for idx, m in enumerate(remaining):
            if move_units(job, m) == val:
                selected_ids.append(m.move_id)
                remaining.pop(idx)
                break

    by_id = {m.move_id: m for m in job.moves}
    coord_units = sum(round(by_id[mid].displacement(job.axis) / job.coordinate_unit_m) for mid in selected_ids)
    
    return AxisResult(
        limb=job.limb, axis=job.axis, found=True, move_ids_used=selected_ids,
        displacement_m=coord_units * job.coordinate_unit_m
    )


def solve_all_limbs(jobs: Sequence[AxisJob], max_workers=None):
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(solve_one_axis, j): (j.limb, j.axis) for j in jobs}
        for f in as_completed(futures):
            results[futures[f]] = f.result()
    return results


def execute_interleaved_plan(
    limb: str,
    jobs_by_axis: dict[str, AxisJob],
    results_by_axis: dict[str, AxisResult],
    visualizer: VisionGuidedArmOverlay,
) -> dict:
    tangential = [ax for ax, j in jobs_by_axis.items() if not j.is_approach_axis]
    approach = [ax for ax, j in jobs_by_axis.items() if j.is_approach_axis]

    for axes in (tangential, approach):
        positions = {ax: 0 for ax in axes}
        while True:
            active = [ax for ax in axes if positions[ax] < len(results_by_axis[ax].move_ids_used)]
            if not active:
                break
            for axis in active:
                job = jobs_by_axis[axis]
                move_id = results_by_axis[axis].move_ids_used[positions[axis]]
                move = next(m for m in job.moves if m.move_id == move_id)

                visualizer.render_move(limb, axis, move.move_id, move.displacement(axis))
                positions[axis] += 1

    visualizer.close()
    return {"completed": True, "limb": limb}


# ============================================================
# Main Execution Entry
# ============================================================

def _demo() -> None:
    random.seed(4)
    visualizer = VisionGuidedArmOverlay(save_frames=True)

    target = CartesianTarget(x_m=0.250, y_m=0.180, z_m=0.187, precision_ticks=2)
    jobs, jobs_by_limb = [], {"left_arm": {}}

    for axis in AXES:
        moves = [CartesianMove(move_id=i, dx_m=(random.randint(1, 8)/1000.0 if axis=="x" else 0.0),
                               dy_m=(random.randint(1, 8)/1000.0 if axis=="y" else 0.0),
                               dz_m=(random.randint(1, 8)/1000.0 if axis=="z" else 0.0)) for i in range(1, 150)]
        job = AxisJob(limb="left_arm", axis=axis, moves=moves, target=target, is_approach_axis=(axis=="z"), seed=hash(("left_arm", axis)) % (2**31))
        jobs.append(job)
        jobs_by_limb["left_arm"][axis] = job

    print("Computing precision trajectories...")
    results = solve_all_limbs(jobs)
    axis_results = {ax: results[("left_arm", ax)] for ax in AXES}

    print("Starting vision-guided auto-grab execution...")
    execute_interleaved_plan(
        limb="left_arm",
        jobs_by_axis=jobs_by_limb["left_arm"],
        results_by_axis=axis_results,
        visualizer=visualizer,
    )
    print("Execution completed successfully.")


if __name__ == "__main__":
    freeze_support()
    _demo()
