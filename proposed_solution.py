# merged_controller_opencv.py
#
# This controller integrates:
# 1. Advanced Navigation (Rendezvous, Exploration) from proposed_solution.py
# 2. Real OpenCV-based "Detect Red" logic.
# 3. Real sensor methods from rosbot_sensors_actuators_example.py
# 4. Supervisor reporting from simple_rosbot_sar 4.py (modified to not stop)
#
# BEHAVIOR:
# - Robot explores using advanced logic.
# - When it detects a red blob:
#   - It calculates the victim's world coordinates.
#   - It immediately reports the victim to the supervisor (send_decision_request).
#   - It immediately starts navigating toward the victim.

import json
import math
import os
import random
import time
import uuid  # Added for unique victim IDs
from controller import Robot
import numpy as np
from PIL import Image
from enum import Enum
from typing import Tuple, List

# --- NEW: Added OpenCV dependency ---
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    cv2 = None
    OPENCV_AVAILABLE = False


def now_ts():
    return time.time()


def safe_json_dumps(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        def convert(x):
            try:
                json.dumps(x)
                return x
            except Exception:
                return str(x)
        return json.dumps(_recursively_map(obj, convert), ensure_ascii=False)


def _recursively_map(o, fn):
    if isinstance(o, dict):
        return {k: _recursively_map(v, fn) for k, v in o.items()}
    if isinstance(o, list):
        return [_recursively_map(v, fn) for v in o]
    return fn(o)


class RobotState(Enum):
    """Robot operational states (Only EXPLORING is used now)"""
    EXPLORING = "exploring"
    WAITING_APPROVAL = "waiting_approval"


class Rosbot:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        try:
            self.name = self.robot.getName()
        except Exception:
            self.name = getattr(self.robot, "getName", lambda: "rosbot")()
            
        self.robot_id = self.name
        
        # State logic is kept for handle_supervisor_response, but not used to stop
        self.state = RobotState.EXPLORING
        self.action_pending = False

        self.last_decision_time = 0.0
        self.decision_interval = 2.0

        self.last_victim_report_time = 0.0
        self.victim_report_cooldown = 10.0

        self.debug_dir = os.path.join(os.getcwd(), "rgb_camera_outputs")
        os.makedirs(self.debug_dir, exist_ok=True)

        self.map_file = os.path.join(self.debug_dir, "map.json")
        self.map_history_file = os.path.join(self.debug_dir, "map_history.jsonl")

        # victim persistence
        self.victim_file = os.path.join(self.debug_dir, "victims.json")
        self.victims = {}  # id -> info
        self._load_victims()

        # actively targetable red victims (deduped & tracked)
        self.red_victims = {}     # vid -> {"id","world","pixel","first_seen","last_seen","visited":bool}
        self.visited_victims = set()
        self.current_victim_id = None
        self.post_detection_backoff_steps = 0
        self.post_detection_backoff_duration = 28
        self.victim_target_tol_m = 0.95

        # nodes (local map of discovered node centroids)
        self.nodes = {}
        try:
            if os.path.isfile(self.map_file):
                with open(self.map_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self.nodes = {n["id"]: n["centroid"] for n in saved.get("nodes", [])}
                print(f"[{self.name}] Loaded map with {len(self.nodes)} nodes from {self.map_file}")
        except Exception as ex:
            print(f"[{self.name}] Warning: failed to load map file ({ex})")

        # motors
        self.motors = {}
        for k, name in (("fl", "fl_wheel_joint"), ("fr", "fr_wheel_joint"),
                        ("rl", "rl_wheel_joint"), ("rr", "rr_wheel_joint")):
            try:
                m = self.robot.getDevice(name)
                m.setPosition(float("inf"))
                m.setVelocity(0.0)
                self.motors[k] = m
            except Exception as ex:
                self.motors[k] = None
                print(f"[{self.name}] Warning: Motor '{name}' not found ({ex})")

        # wheel sensors
        self.wheel_sensors = {}
        for k, name in (("fl", "front left wheel motor sensor"),
                        ("fr", "front right wheel motor sensor"),
                        ("rl", "rear left wheel motor sensor"),
                        ("rr", "rear right wheel motor sensor")):
            try:
                s = self.robot.getDevice(name)
                s.enable(self.timestep)
                self.wheel_sensors[k] = s
            except Exception as ex:
                self.wheel_sensors[k] = None
                print(f"[{self.name}] Warning: Wheel sensor '{name}' not found ({ex})")

        # cameras
        try:
            self.camera_rgb = self.robot.getDevice("camera rgb")
            self.camera_rgb.enable(self.timestep)
        except Exception:
            self.camera_rgb = None
        try:
            self.camera_depth = self.robot.getDevice("camera depth")
            self.camera_depth.enable(self.timestep)
        except Exception:
            self.camera_depth = None

        # lidar
        try:
            self.lidar = self.robot.getDevice("laser")
            self.lidar.enable(self.timestep)
            try:
                self.lidar.enablePointCloud()
                self.lidar_pointcloud_supported = True
            except Exception:
                self.lidar_pointcloud_supported = False
        except Exception as ex:
            self.lidar = None
            self.lidar_pointcloud_supported = False

        # GPS/IMU
        try:
            self.gps = self.robot.getDevice("gps")
            self.gps.enable(self.timestep)
        except Exception:
            self.gps = None
        try:
            self.compass = self.robot.getDevice("imu compass")
            self.compass.enable(self.timestep)
        except Exception:
            self.compass = None
        try:
            self.inertial_unit = self.robot.getDevice("imu inertial_unit")
            self.inertial_unit.enable(self.timestep)
        except Exception:
            self.inertial_unit = None

        # short-range range sensors
        self.range_sensors = {}
        for name in ("fl_range", "fr_range", "rl_range", "rr_range"):
            try:
                s = self.robot.getDevice(name)
                s.enable(self.timestep)
                self.range_sensors[name] = s
            except Exception:
                self.range_sensors[name] = None

        # comms
        try:
            self.supervisor_receiver = self.robot.getDevice("supervisor receiver")
            self.supervisor_receiver.enable(self.timestep)
            self.supervisor_emitter = self.robot.getDevice("supervisor emitter")
        except Exception:
            self.supervisor_receiver = None
            self.supervisor_emitter = None

        try:
            self.squad_receiver = self.robot.getDevice("robot to robot receiver")
            self.squad_receiver.enable(self.timestep)
            self.squad_emitter = self.robot.getDevice("robot to robot emitter")
        except Exception:
            self.squad_receiver = None
            self.squad_emitter = None

        # exploration/navigation
        self.free_cells = 0
        self.current_task = None
        self.task_target = None
        self.step_count = 0
        self.node_discovery_times = {}
        self.assignment_delay = 3.0

        # speed / safety (tune as needed)
        self.max_wheel_speed = 200.0
        self.yaw_inversion = 0.1
        self.d_min = 0.030
        self.cbf_alpha = 5.0
        self.omega_gain = 2.5

        # short-range safety thresholds (meters)
        self.sr_stop_front = 0.05
        self.sr_stop_rear = 0.05
        self.sr_slow_front = 0.07
        self.sr_slow_rear = 0.07

        # safety trigger and percentile smoothing
        self.cbf_trigger_distance = 0.1
        self.cbf_allowed_percentile = 47
        
        self.victim_confidence_threshold = 0.001 

        # stuck/low-speed detection & recovery (tune)
        self.stuck_history_len = 12
        self.stuck_front_history = []
        self.stuck_backoff_steps = 0
        self.emergency_backoff_steps = 0
        self.low_speed_count = 0
        self.v_low_thresh = 0.05
        self.low_speed_timeout_steps = 4
        self.emergency_backoff_duration = 8

        self.debug_rate = 10

        # --- HARD BARRIER (position-based anti-stuck, 2s) ---
        self.stuck_hard_window_s = 2.0
        self.stuck_hard_disp_thresh = 0.15
        self.stuck_hard_cooldown_s = 3.0
        self.stuck_last_moved_ts = now_ts()
        self.stuck_last_pos = self.get_position()
        self.stuck_last_trigger_ts = 0.0

        # --- REPULSIVE mode parameters (safe defaults) ---
        self.repulsive_steps = 0
        self.repulsive_duration = 4
        self.repulsive_influence = 1.0
        self.repulsive_max_v = 10.0
        self.repulsive_max_omega = 10.0
        self.repulsive_k_v = 4.0
        self.repulsive_k_omega = 10

        # victim detection config
        self.victim_check_interval = 2  # check every N steps
        self.victim_area_thresh = 300   # pixels (from proposed_solution)
        self.victim_pixel_dedupe = 50   # px (from proposed_solution)
        self.victim_world_dedupe = 0.8  # m (from proposed_solution)
        self.victim_next_id = 1

        # pose broadcast / received poses
        self.pose_bcast_interval_s = 0.5
        try:
            self.pose_bcast_steps = max(1, int(self.pose_bcast_interval_s * 1000.0 / float(self.timestep)))
        except Exception:
            self.pose_bcast_steps = 15
        self.last_pose_bcast = 0
        self.received_poses = {}  # robot_name -> (x,y,ts)

        # Startup / rendezvous window (first N seconds we do rendezvous)
        self.start_time = now_ts()
        self.rendezvous_window_s = 200.0
        self.rendezvous_reached = False

        # smoothing / angular limits
        self.omega_scale = 1
        self.omega_limit = 4
        self.omega_smooth_alpha = 0.5 # (0.0 = instant, 1.0 = no change). 0.8 is smooth.
        self.prev_omega = 0.0
        self.heading_deadzone_rad = math.radians(6.0)

        self.v_smooth_alpha = 0.8  # (0.0 = instant, 1.0 = no change). 0.8 is smooth.
        self.prev_v = 0.0

        # GREEDY pixel-centering pursuit (fallback when no depth/world)
        self.greedy_active = False
        self.greedy_target_px = None
        self.greedy_last_seen_ts = 0.0
        self.greedy_timeout_s = 4.0

        # Narrow HSV ranges for RED (from proposed_solution)
        self.hsv_narrow_ranges = [
            (np.array([0,   200, 40], dtype=np.uint8),  np.array([6,   255, 130], dtype=np.uint8)),
            (np.array([175, 200, 40], dtype=np.uint8),  np.array([179, 255, 130], dtype=np.uint8)),
        ]

        # --- NEW: Check for OpenCV ---
        if not OPENCV_AVAILABLE:
            print(f"[{self.name}] CRITICAL: OpenCV not available. Red detection will be disabled.")
        else:
            print(f"[{self.name}] OpenCV is available. Red detection enabled.")
        # --- END NEW ---

        # If GPS available, add our own pose as initial
        pos = self.get_position()
        if pos is not None:
            self.received_poses[self.name] = (float(pos[0]), float(pos[1]), now_ts())

        # -------------------------------
        # Exploration-sharing parameters
        # -------------------------------
        self.cell_size = 1
        self.explore_mark_radius = 20
        self.explored_cells = {}
        self.remote_explored = {}
        self.explored_ttl = 1000.0
        self.explored_bcast_interval_s = 1.0
        self.last_explored_bcast = 0.0
        self.explored_bcast_max = 600
        
        print(f"[{self.robot_id}] Initialized - Ready for search and rescue mission")


    # ---------- Victim persistence ----------
    def _load_victims(self):
        try:
            if os.path.isfile(self.victim_file):
                with open(self.victim_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.victims = data.get("victims", {})
                print(f"[{self.name}] Loaded {len(self.victims)} victims from {self.victim_file}")
        except Exception as ex:
            print(f"[{self.name}] Warning: failed to load victims file ({ex})")
            self.victims = {}

    def persist_victims(self):
        try:
            payload = {"updated_at": now_ts(), "robot": self.name, "victims": self.victims}
            with open(self.victim_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as ex:
            print(f"[{self.name}] Error persisting victims: {ex}")

    # ---------- Map persistence ----------
    def _persist_map(self):
        try:
            payload = {"updated_at": now_ts(), "robot": self.name,
                       "nodes": [{"id": nid, "centroid": c} for nid, c in self.nodes.items()]}
            with open(self.map_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as ex:
            print(f"[{self.name}] Error persisting map: {ex}")

    def _append_map_history(self, event_type, details):
        try:
            entry = {"ts": now_ts(), "robot": self.name, "event": event_type, "details": details}
            with open(self.map_history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as ex:
            print(f"[{self.name}] Error appending map history: {ex}")

    def merge_graph_summary(self, graph_msg):
        try:
            nodes = graph_msg.get("nodes", [])
            added = updated = 0
            for n in nodes:
                nid = n.get("id")
                c = n.get("centroid")
                if not nid:
                    continue
                if nid not in self.nodes:
                    self.nodes[nid] = c
                    self.node_discovery_times[nid] = now_ts()
                    added += 1
                else:
                    if self.nodes[nid] != c:
                        self.nodes[nid] = c
                        updated += 1
            if added or updated:
                self._persist_map()
                self._append_map_history("merge_graph_summary", {"added": added, "updated": updated})
                print(f"[{self.name}] Merged GRAPH_SUMMARY added={added} updated={updated}")
        except Exception as ex:
            print(f"[{self.name}] Error merging graph summary: {ex}")

    # ---------- Comms helper ----------
    def send_squad_message(self, payload):
        try:
            if not self.squad_emitter:
                return False
            raw = safe_json_dumps(payload)
            if hasattr(self.squad_emitter, "sendString"):
                self.squad_emitter.sendString(raw)
            else:
                try:
                    self.squad_emitter.send(raw.encode("utf-8"))
                except Exception:
                    try:
                        self.squad_emitter.send(raw)
                    except Exception:
                        pass
            return True
        except Exception as ex:
            print(f"[{self.name}] Error send_squad_message: {ex}")
            return False

    def send_supervisor_message(self, payload):
        try:
            if not self.supervisor_emitter:
                return False
            raw = safe_json_dumps(payload)
            if hasattr(self.supervisor_emitter, "sendString"):
                self.supervisor_emitter.sendString(raw)
            else:
                try:
                    self.supervisor_emitter.send(raw.encode("utf-8"))
                except Exception:
                    try:
                        self.supervisor_emitter.send(raw)
                    except Exception:
                        pass
            return True
        except Exception as ex:
            print(f"[{self.name}] Error send_supervisor_message: {ex}")
            return False

    # ---------- Explored-area helpers ----------
    def world_to_cell(self, world_xy):
        try:
            ix = int(math.floor(world_xy[0] / float(self.cell_size)))
            iy = int(math.floor(world_xy[1] / float(self.cell_size)))
            return (ix, iy)
        except Exception:
            return None

    def cell_to_world_center(self, cell):
        ix, iy = cell
        cx = (ix + 0.5) * self.cell_size
        cy = (iy + 0.5) * self.cell_size
        return (cx, cy)

    def mark_explored_world_area(self, world_xy, radius=None):
        try:
            if world_xy is None:
                return
            if radius is None:
                radius = self.explore_mark_radius
            cx = world_xy[0]
            cy = world_xy[1]
            r = float(radius)
            min_ix = int(math.floor((cx - r) / self.cell_size))
            max_ix = int(math.floor((cx + r) / self.cell_size))
            min_iy = int(math.floor((cy - r) / self.cell_size))
            max_iy = int(math.floor((cy + r) / self.cell_size))
            ts = now_ts()
            for ix in range(min_ix, max_ix + 1):
                for iy in range(min_iy, max_iy + 1):
                    cell_center = self.cell_to_world_center((ix, iy))
                    if math.hypot(cell_center[0] - cx, cell_center[1] - cy) <= r:
                        self.explored_cells[(ix, iy)] = ts
        except Exception as ex:
            print(f"[{self.name}] Error mark_explored_world_area: {ex}")

    def expire_old_explored(self):
        try:
            now = now_ts()
            to_del = [c for c,ts in self.explored_cells.items() if (now - ts) > self.explored_ttl]
            for c in to_del:
                del self.explored_cells[c]
            # expire remote ones too
            for rname in list(self.remote_explored.keys()):
                for c, ts in list(self.remote_explored[rname].items()):
                    if now - ts > self.explored_ttl:
                        try:
                            del self.remote_explored[rname][c]
                        except Exception:
                            pass
                if not self.remote_explored[rname]:
                    try:
                        del self.remote_explored[rname]
                    except Exception:
                        pass
        except Exception as ex:
            print(f"[{self.name}] Error expire_old_explored: {ex}")

    def get_combined_explored(self):
        combined = set(self.explored_cells.keys())
        for rmap in self.remote_explored.values():
            combined.update(rmap.keys())
        return combined

    def find_nearest_unexplored_cell(self, max_radius_cells=50):
        pos = self.get_position()
        if pos is None:
            return None
        center_cell = self.world_to_cell(pos)
        if center_cell is None:
            return None
        combined = self.get_combined_explored()
        if center_cell not in combined:
            return center_cell
        cx, cy = center_cell
        for r in range(1, max_radius_cells+1):
            for dx in range(-r, r+1):
                for dy in (-r, r):
                    cell = (cx + dx, cy + dy)
                    if cell not in combined:
                        return cell
            for dy in range(-r+1, r):
                for dx in (-r, r):
                    cell = (cx + dx, cy + dy)
                    if cell not in combined:
                        return cell
        return None

    def broadcast_explored_update(self):
        try:
            now = now_ts()
            items = sorted(self.explored_cells.items(), key=lambda kv: kv[1], reverse=True)
            items = items[:self.explored_bcast_max]
            cells_payload = [{"cell": [int(k[0]), int(k[1])], "ts": float(v)} for k, v in items]
            payload = {"type": "EXPLORED_UPDATE", "robot": self.name, "cells": cells_payload, "ts": now}
            self.send_squad_message(payload)
            self.last_explored_bcast = now
            if self.step_count % (self.debug_rate*5) == 0:
                print(f"[{self.name}] Broadcasted {len(cells_payload)} explored cells")
            return True
        except Exception as ex:
            print(f"[{self.name}] Error broadcast_explored_update: {ex}")
            return False

    def merge_remote_explored(self, robot_name, cells_list):
        try:
            if robot_name not in self.remote_explored:
                self.remote_explored[robot_name] = {}
            changed = 0
            for c in cells_list:
                cell = tuple(int(x) for x in c.get("cell", []))
                ts = float(c.get("ts", now_ts()))
                prev = self.remote_explored[robot_name].get(cell)
                if (prev is None) or (ts > prev):
                    self.remote_explored[robot_name][cell] = ts
                    changed += 1
            if changed and self.step_count % (self.debug_rate*5) == 0:
                print(f"[{self.name}] Merged {changed} remote explored cells from {robot_name}")
            return True
        except Exception as ex:
            print(f"[{self.name}] Error merge_remote_explored: {ex}")
            return False

    # ---------- Motors ----------
    def set_wheel_speeds(self, fl=0.0, fr=0.0, rl=0.0, rr=0.0):
        if self.motors.get("fl"):
            self.motors["fl"].setVelocity(fl)
        if self.motors.get("fr"):
            self.motors["fr"].setVelocity(fr)
        if self.motors.get("rl"):
            self.motors["rl"].setVelocity(rl)
        if self.motors.get("rr"):
            self.motors["rr"].setVelocity(rr)
            
    def stop(self):
        """Stop robot movement"""
        self.set_wheel_speeds(0, 0, 0, 0)

    # ---------- Wheel mapping helper (damping & scaling) ----------
    def _apply_wheel_cmds(self, v_safe, omega_safe):
        """Convert v_safe and omega_safe into left/right wheel velocities with smoothing and limits."""
        try:
            # --- This is your existing angular smoothing ---
            omega_scaled = float(omega_safe) * self.omega_scale
            omega_scaled = self._clip(omega_scaled, -self.omega_limit, self.omega_limit)
            omega_cmd = self.prev_omega * self.omega_smooth_alpha + omega_scaled * (1.0 - self.omega_smooth_alpha)
            self.prev_omega = omega_cmd

            # --- ADD THIS BLOCK: Linear (forward) smoothing ---
            v_cmd = self.prev_v * self.v_smooth_alpha + v_safe * (1.0 - self.v_smooth_alpha)
            self.prev_v = v_cmd
            # --- END ADD ---

            # --- MODIFIED: Use v_cmd instead of v_safe ---
            left = self._clip(v_cmd - omega_cmd, -self.max_wheel_speed, self.max_wheel_speed)
            right = self._clip(v_cmd + omega_cmd, -self.max_wheel_speed, self.max_wheel_speed)
            # --- END MODIFIED ---
            
            self.set_wheel_speeds(left, right, left, right)
        except Exception as ex:
            print(f"[{self.name}] Error _apply_wheel_cmds: {ex}")
            try:
                self.set_wheel_speeds(0, 0, 0, 0)
            except Exception:
                pass

    # ---------- Sensors shortcuts ----------

    # --- NEW: Added from rosbot_sensors_actuators_example.py ---
    def get_rgb_image(self):
        if not self.camera_rgb:
            return None
        buf = self.camera_rgb.getImage()
        w, h = self.camera_rgb.getWidth(), self.camera_rgb.getHeight()
        # Webots stores camera images as BGRA
        img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        rgb = img[:, :, :3][:, :, ::-1]  # convert BGRA->RGB
        return rgb

    def get_depth_image(self):
        if not self.camera_depth:
            return None
        w, h = self.camera_depth.getWidth(), self.camera_depth.getHeight()
        depth = np.array(self.camera_depth.getRangeImage(), dtype=np.float32).reshape(
            (h, w)
        )
        return depth
    # --- END NEW ---

    def get_lidar_ranges(self):
        if not self.lidar:
            return None
        try:
            return self.lidar.getRangeImage()
        except Exception:
            return None

    def get_depth_scan_ranges(self, num_rows_to_average=5):
        """
        Gets a 1D scan from the depth camera by sampling the middle rows.
        Returns a 1D numpy array of distances, or None.
        """
        if not self.camera_depth:
            return None
        
        depth_image = self.get_depth_image() # This is (h, w)
        if depth_image is None:
            return None
        
        h, w = depth_image.shape
        
        # --- START MODIFICATION ---
        # Select ONLY the middle row. This is less likely to see the floor
        # than taking the 'min' of a wide band.
        mid_row = h // 2
        
        if h > 0:
            horizontal_slice = depth_image[mid_row, :]
        else:
            return None
        # --- END MODIFICATION ---
        
        # Return this 1D array
        return horizontal_slice
    
    def get_position(self):
        if not self.gps:
            return None
        try:
            vals = self.gps.getValues()
            return (float(vals[0]), float(vals[2]))
        except Exception:
            return None

    def get_inertial_quaternion(self):
        if not self.inertial_unit:
            return None
        try:
            return self.inertial_unit.getQuaternion()
        except Exception:
            return None

    def get_yaw(self):
        raw_yaw = None
        if self.compass:
            try:
                c = self.compass.getValues()
                raw_yaw = math.atan2(c[0], c[2])
            except Exception:
                raw_yaw = None
        if raw_yaw is None:
            q = self.get_inertial_quaternion()
            if q and len(q) == 4:
                try:
                    w, x, y, z = q
                    raw_yaw = math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
                except Exception:
                    try:
                        x, y, z, w = q
                        raw_yaw = math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
                    except Exception:
                        raw_yaw = None
        if raw_yaw is None:
            return None
        return raw_yaw * self.yaw_inversion

    def calibrate_yaw_sign(self, rotate_speed=1.2, steps=6):
        try:
            initial = self.get_yaw()
            if initial is None:
                print(f"[{self.name}] calibrate_yaw_sign: no yaw reading available, skipping")
                return
            self.set_wheel_speeds(-rotate_speed, rotate_speed, -rotate_speed, rotate_speed)
            for _ in range(steps):
                self.robot.step(self.timestep)
            after = self.get_yaw()
            self.set_wheel_speeds(0, 0, 0, 0)
            if after is None:
                print(f"[{self.name}] calibrate_yaw_sign: no after yaw, skipping")
                return
            delta = self._angle_diff(initial, after)
            self.yaw_inversion = 1 if delta > 0 else -1
            print(f"[{self.name}] yaw_inversion={self.yaw_inversion} (delta={delta:.3f})")
        except Exception as ex:
            print(f"[{self.name}] Error calibrate_yaw_sign: {ex}")

    # ---------- Helpers ----------
    def _clip(self, v, lo, hi):
        return max(lo, min(hi, v))

    def _angle_diff(self, a, b):
        d = b - a
        while d > math.pi:
            d -= 2*math.pi
        while d < -math.pi:
            d += 2*math.pi
        return d

    def _lidar_angles(self):
        if not self.lidar:
            return None
        try:
            n = self.lidar.getHorizontalResolution()
            fov = self.lidar.getFov()
            return np.linspace(-fov/2.0, fov/2.0, n)
        except Exception:
            return None

    def _depth_camera_angles(self):
        """Get the horizontal angles for each pixel column of the depth camera."""
        if not self.camera_depth:
            return None
        try:
            n = self.camera_depth.getWidth()
            fov = self.camera_depth.getFov()
            # Create an array of angles, just like for the lidar
            return np.linspace(-fov/2.0, fov/2.0, n)
        except Exception:
            return None
        
    # ---------- Short-range helper ----------
    def get_short_range_values(self):
        vals = {}
        name_map = {"fl": "fl_range", "fr": "fr_range", "rl": "rl_range", "rr": "rr_range"}
        for k, devname in name_map.items():
            s = self.range_sensors.get(devname)
            try:
                v = float(s.getValue()) if s else float("inf")
                if not np.isfinite(v) or v <= 0:
                    v = float("inf")
            except Exception:
                v = float("inf")
            vals[k] = v
        return vals

    # ---------- Pixel -> World (approx) ----------
    def _pixel_to_world_approx(self, px, py, depth_val):
        try:
            cam = self.camera_rgb
            if cam is None or depth_val is None or not np.isfinite(depth_val):
                return None
            w = cam.getWidth()
            fov = cam.getFov()
            cx = w / 2.0
            bearing = (px - cx) / float(w) * fov
            yaw = self.get_yaw()
            pos = self.get_position()
            if yaw is None or pos is None:
                return None
            world_heading = yaw + bearing
            r = float(depth_val)
            # Add small offset to not target *inside* the wall
            r_safe = r - 0.15 
            wx = pos[0] + r_safe * math.cos(world_heading)
            wy = pos[1] + r_safe * math.sin(world_heading)
            return (float(wx), float(wy))
        except Exception:
            return None

    # --- START: Supervisor comms logic from simple_rosbot_sar 4.py ---
    def handle_supervisor_response(self):
        """Handle response from supervisor"""
        while self.supervisor_receiver.getQueueLength() > 0:
            message = ""
            try:
                if hasattr(self.supervisor_receiver, "getString"):
                    message = self.supervisor_receiver.getString()
                else:
                    message_bytes = self.supervisor_receiver.getData()
                    message = message_bytes.decode("utf-8")
            except Exception as e:
                print(f"[{self.robot_id}] Error reading supervisor message: {e}")
                self.supervisor_receiver.nextPacket()
                continue
                
            self.supervisor_receiver.nextPacket()

            try:
                response = json.loads(message)
                approved = response.get("approved", False)
                message_target = response.get("robot_id", None)
                
                if message_target and message_target != self.robot_id:
                    print(
                        f"Ignoring message for a different robot {message_target, self.robot_id}"
                    )
                    continue

                if approved:
                    print(f"[{self.robot_id}] Supervisor APPROVED action.")
                    self.action_pending = False
                    self.state = RobotState.EXPLORING
                    return True # Approved
                else:
                    print(f"[{self.robot_id}] Supervisor REJECTED action.")
                    self.action_pending = False
                    self.state = RobotState.EXPLORING
                    # --- MODIFICATION: If rejected, add emergency backoff ---
                    self.emergency_backoff_steps = max(self.emergency_backoff_steps, self.emergency_backoff_duration)
                    # --- END MODIFICATION ---
                    return False # Rejected

            except json.JSONDecodeError:
                print(f"[{self.robot_id}] Error decoding supervisor response: {message}")
            except Exception as e:
                print(f"[{self.robot_id}] Error processing supervisor response: {e}")

        return None  # No response processed
    
    def generate_explanation(self, action: str) -> str:
        """Generate human-readable explanation for the intended action"""
        pos = self.get_position()
        pos_str = "unknown"
        if pos:
             pos_str = f"{pos[0]:.1f}, {pos[1]:.1f}"

        explanations = {
            "investigate_victim": [
                f"Moving to investigate potential victim detected near {pos_str}",
                f"Approaching suspected victim location for detailed assessment",
                f"Detected red object, investigating potential victim.",
            ],
            "explore_forward": [
                f"Moving forward to explore uncharted territory at coordinates {pos_str}",
                "Proceeding ahead to systematically search for victims in this area",
                "Continuing forward exploration to maximize coverage of the search zone",
            ],            "return_to_rendezvous": [
                f"Returning to rendezvous point to regroup with squad",
                f"Heading back to rendezvous location for team coordination",
                f"Navigating to rendezvous point to ensure team safety and communication",
            ],
        }
        if action in explanations:
            explanation = random.choice(explanations[action])
        else:
            explanation = f"Executing {action} to continue search and rescue mission"
        return explanation

    # --- MODIFIED: Added wait_for_approval flag ---
    def send_decision_request(
        self,
        action: str,
        reason: str,
        victim_detected: bool = False,
        confidence: float = 0.0,
        wait_for_approval: bool = True  # <-- NEW FLAG
    ):
        """Send decision request to supervisor"""
        if not self.supervisor_emitter:
            print(f"[{self.robot_id}] Warning: Cannot send request - no emitter")
            if wait_for_approval:
                self.state = RobotState.WAITING_APPROVAL
            return

        pos = self.get_position()
        pos_list = [pos[0], 0.0, pos[1]] if pos else [0.0, 0.0, 0.0] 

        request = {
            "timestamp": self.robot.getTime(),
            "robot_id": self.robot_id,
            "position": pos_list,
            "intended_action": action,
            "reason": reason,
            "victim_found": victim_detected,
            "victim_confidence": confidence,
        }

        message = json.dumps(request)
        
        try:
            if hasattr(self.supervisor_emitter, "sendString"):
                self.supervisor_emitter.sendString(message)
            else:
                self.supervisor_emitter.send(message.encode("utf-8"))
        except Exception as e:
             print(f"[{self.robot_id}] Error sending decision request: {e}")

        # --- MODIFIED: Only wait if flag is True ---
        if wait_for_approval:
            self.action_pending = True
            self.state = RobotState.WAITING_APPROVAL
        # --- END MODIFICATION ---

        print(f"[{self.robot_id}] REQUEST: {action} - {reason}")
        if victim_detected:
            print(f"[{self.robot_id}] VICTIM ALERT: Confidence {confidence:.1%}")
    # --- END: Supervisor comms logic ---


    def send_mission_report(self, reason="status_update", rejected=False, supervisor_msg=None):
        # This function is from proposed_solution, it's fine
        try:
            pos = self.get_position()
            victims_list = []
            for vid, info in list(self.victims.items()):
                victims_list.append({
                    "id": vid,
                    "world": info.get("world"),
                    "first_seen": info.get("first_seen"),
                    "last_seen": info.get("last_seen"),
                })
            nodes_list = [{"id": nid, "centroid": c} for nid, c in list(self.nodes.items())][:50]
            
            metrics_payload = {
                "victims_seen": len(self.red_victims),
                "victims_confirmed": len(self.visited_victims),
                "false_positives": 0, # This logic isn't fully implemented
                "distance_m": 0.0,
                "avg_detection_time_s": None,
            }
            
            report = {
                "type": "MISSION_REPORT",
                "robot": self.name,
                "ts": now_ts(),
                "position": [float(pos[0]), float(pos[1])] if pos else None,
                "current_task": self.current_task,
                "task_target": self.task_target,
                "victims": victims_list,
                "nodes_count": len(self.nodes),
                "nodes_sample": nodes_list,
                "rejected": bool(rejected),
                "reason": reason,
                "supervisor_msg": supervisor_msg,
                "metrics": metrics_payload
            }

            self._append_map_history("mission_report", report)
            if self.supervisor_emitter:
                try:
                    raw = safe_json_dumps(report)
                    if hasattr(self.supervisor_emitter, "sendString"):
                        self.supervisor_emitter.sendString(raw)
                    else:
                        self.supervisor_emitter.send(raw.encode("utf-8"))
                    # print(f"[{self.name}] Sent mission report to supervisor.")
                    return True
                except Exception:
                    pass
            try:
                self.send_squad_message(report)
                # print(f"[{self.name}] Broadcasted mission report to squad (fallback).")
                return True
            except Exception:
                pass
            return False
        except Exception as ex:
            print(f"[{self.name}] Error send_mission_report: {ex}")
            return False

    # --- NEW: Replaced detect_victim with OpenCV version ---
    # --- NEW: Replaced detect_victim with OpenCV version ---
    def detect_victim(self):
        """
        Detects RED victims using OpenCV, estimates world position.
        Updates self.red_victims internally.
        SAVES a copy of the detection image to self.debug_dir.
        Returns a list of detection dicts for use in the main loop.
        """
        if not OPENCV_AVAILABLE or self.camera_rgb is None or self.camera_depth is None:
            return []

        rgb_image = self.get_rgb_image()
        depth_image = self.get_depth_image()
        if rgb_image is None or depth_image is None:
            return []

        h, w, _ = rgb_image.shape
        detections_list = [] # For return value

        try:
            # 1. Convert to HSV
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

            # 2. Create mask for red (using ranges from __init__)
            mask = cv2.inRange(hsv, self.hsv_narrow_ranges[0][0], self.hsv_narrow_ranges[0][1])
            mask2 = cv2.inRange(hsv, self.hsv_narrow_ranges[1][0], self.hsv_narrow_ranges[1][1])
            red_mask = cv2.bitwise_or(mask, mask2)

            # 3. Find contours
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return []

            # 4. Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area < self.victim_area_thresh: # Use threshold from __init__
                return [] # No significant blob found

            # 5. Get centroid
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return []
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # --- START: SAVE DETECTION IMAGE (NEW CODE) ---
            try:
                # 1. Create a BGR copy for drawing (OpenCV expects BGR)
                bgr_image = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2BGR)
                
                # 2. Draw a green circle on the detected centroid
                cv2.circle(bgr_image, (cx, cy), 10, (0, 255, 0), 2) # (0, 255, 0) is green in BGR
                
                # 3. Convert back to RGB for saving with PIL
                rgb_to_save = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                
                # 4. Convert to PIL Image
                pil_img = Image.fromarray(rgb_to_save)
                
                # 5. Create a unique filename and save
                ts = int(now_ts() * 1000)
                filename = f"detection_{self.name}_{ts}_at_({cx},{cy}).png"
                path = os.path.join(self.debug_dir, filename)
                pil_img.save(path)
                
            except Exception as e:
                print(f"[{self.name}] Failed to save detection image: {e}")
            # --- END: SAVE DETECTION IMAGE (NEW CODE) ---


            # 6. Get depth and world coordinates
            cx = np.clip(cx, 0, w - 1)
            cy = np.clip(cy, 0, h - 1)

            depth_val = float(depth_image[cy, cx])
            
            world_coords = None
            if np.isfinite(depth_val) and depth_val > self.camera_depth.getMinRange():
                world_coords = self._pixel_to_world_approx(cx, cy, depth_val)

            # 7. Add to self.red_victims (with deduplication)
            now = now_ts()
            is_new = True
            found_vid = None
            
            if world_coords:
                for vid, v_info in self.red_victims.items():
                    if v_info.get("world"):
                        dist = math.hypot(v_info["world"][0] - world_coords[0], v_info["world"][1] - world_coords[1])
                        if dist < self.victim_world_dedupe:
                            is_new = False
                            found_vid = vid
                            break
            else:
                for vid, v_info in self.red_victims.items():
                    if v_info.get("pixel"):
                        pix_dist = math.hypot(v_info["pixel"]["x"] - cx, v_info["pixel"]["y"] - cy)
                        if pix_dist < self.victim_pixel_dedupe:
                            is_new = False
                            found_vid = vid
                            break
            
            detection_info = {
                "pixel": {"x": cx, "y": cy},
                "world": world_coords,
                "last_seen": now,
                "confidence": 0.9
            }

            if is_new:
                new_vid = str(uuid.uuid4())
                self.red_victims[new_vid] = {
                    "id": new_vid,
                    "world": world_coords,
                    "pixel": {"x": cx, "y": cy},
                    "first_seen": now,
                    "last_seen": now,
                    "visited": False,
                    "confidence": 0.9
                }
                detection_info["id"] = new_vid
                print(f"[{self.name}] New RED victim {new_vid} detected at world={world_coords} pixel=({cx},{cy})")
            else:
                self.red_victims[found_vid].update(detection_info)
                detection_info["id"] = found_vid
            
            detections_list.append(detection_info)

        except Exception as e:
            print(f"[{self.name}] Error in detect_victim: {e}")
        
        return detections_list
    # --- END NEW ---
    
    def poll_squad_messages(self):
        msgs = []
        try:
            if not getattr(self, "squad_receiver", None):
                return msgs
            while self.squad_receiver.getQueueLength() > 0:
                try:
                    if hasattr(self.squad_receiver, "getString"):
                        raw = self.squad_receiver.getString()
                    else:
                        raw_bytes = self.squad_receiver.getData()
                        try:
                            raw = raw_bytes.decode("utf-8")
                        except Exception:
                            raw = str(raw_bytes)
                    try:
                        parsed = json.loads(raw)
                        msgs.append(parsed)
                    except Exception:
                        msgs.append(raw)
                except Exception as ex:
                    print(f"[{getattr(self,'name','robot')}] Error reading squad message: {ex}")
                finally:
                    try:
                        self.squad_receiver.nextPacket()
                    except Exception:
                        pass
        except Exception as ex:
            print(f"[{getattr(self,'name','robot')}] poll_squad_messages outer error: {ex}")
        return msgs
        
    def compute_repulsive_command(self, ranges, angles):
        try:
            if ranges is None or angles is None or len(ranges) == 0:
                return 0.0, 0.0
            ranges = np.array(ranges, dtype=float)
            angles = np.array(angles, dtype=float)
            ranges = np.where(np.isfinite(ranges), ranges, 1e3)

            sum_x = 0.0
            sum_y = 0.0
            eps = 1e-6
            r0 = self.repulsive_influence

            for r, theta in zip(ranges, angles):
                if r <= 0 or r > r0:
                    continue
                w = (r0 - r) / (r + eps)
                rx = math.cos(theta)
                ry = math.sin(theta)
                sum_x += (-w) * rx
                sum_y += (-w) * ry

            if abs(sum_x) < eps and abs(sum_y) < eps:
                v_cmd = 0.0
                omega_cmd = self._clip(self.repulsive_k_omega * 0.6, -self.repulsive_max_omega, self.repulsive_max_omega)
                return v_cmd, omega_cmd

            desired_heading = math.atan2(sum_y, sum_x)
            abs_heading = abs(desired_heading)

            if abs_heading > (math.pi * 0.5):
                v_cmd = -self.repulsive_max_v * min(1.0, (abs_heading - math.pi/2) / (math.pi/2) + 0.3)
            else:
                v_cmd = self.repulsive_k_v * self.repulsive_max_v * max(0.2, (1.0 - abs_heading / (math.pi)))

            omega_cmd = self.repulsive_k_omega * desired_heading
            v_cmd = self._clip(v_cmd, -self.repulsive_max_v, self.repulsive_max_v)
            omega_cmd = self._clip(omega_cmd, -self.repulsive_max_omega, self.repulsive_max_omega)
            return v_cmd, omega_cmd
        except Exception as ex:
            print(f"[{self.name}] Error compute_repulsive_command: {ex}")
            return 0.0, 0.0

    def compute_safe_controls(self, v_des, omega_des, lidar_ranges):
        """
        Computes safe controls, using LIDAR data only.
        """
        v = float(v_des)
        omega = float(omega_des)

        # --- (All backoff/repulsive/short-range sensor logic remains the same) ---
        if self.emergency_backoff_steps > 0:
            self.emergency_backoff_steps -= 1
            v_cmd = -1.0
            omega_cmd = self._clip(1.4, -self.omega_limit, self.omega_limit)
            if self.step_count % self.debug_rate == 0:
                print(f"[{self.name}] EMERGENCY_BACKOFF remaining={self.emergency_backoff_steps} v={v_cmd:.2f} omega={omega_cmd:.2f}")
            return v_cmd, omega_cmd

        if self.post_detection_backoff_steps > 0:
            self.post_detection_backoff_steps -= 1
            v_cmd = -1.3
            omega_cmd = self._clip(1.0, -self.omega_limit, self.omega_limit)
            if self.step_count % self.debug_rate == 0:
                print(f"[{self.name}] POST_DET_BACKOFF remaining={self.post_detection_backoff_steps}")
            return v_cmd, omega_cmd

        if self.repulsive_steps > 0:
            try:
                # Repulsive uses LIDAR data passed in
                angles = np.array(self._lidar_angles()) if self._lidar_angles() is not None else None
                v_rep, w_rep = self.compute_repulsive_command(lidar_ranges, angles)
                self.repulsive_steps -= 1
                if self.step_count % self.debug_rate == 0:
                    print(f"[{self.name}] REPULSIVE active steps_left={self.repulsive_steps} v_rep={v_rep:.2f} w_rep={w_rep:.2f}")
                w_rep = self._clip(w_rep, -self.omega_limit, self.omega_limit)
                return v_rep, w_rep
            except Exception as ex:
                print(f"[{self.name}] Error in repulsive mode: {ex}")

        sr = self.get_short_range_values()
        fl, fr, rl, rr = sr["fl"], sr["fr"], sr["rl"], sr["rr"]
        front_min_sr = min(fl, fr)
        rear_min_sr = min(rl, rr)

        if v > 0:
            if front_min_sr < self.sr_stop_front:
                print(f"[{self.name}] SR FRONT STOP ({front_min_sr:.2f}m) -> EMERGENCY_BACKOFF")
                self.emergency_backoff_steps = max(self.emergency_backoff_steps, self.emergency_backoff_duration)
                return -1.6, self._clip(1.4 * np.sign(fr - fl), -self.omega_limit, self.omega_limit)
            elif front_min_sr < self.sr_slow_front:
                slow_scale = max(0.15, (front_min_sr - self.sr_stop_front) / max(1e-6, (self.sr_slow_front - self.sr_stop_front)))
                v *= slow_scale
                omega += 1.1 * np.sign(fr - fl) * (1.0 - slow_scale)

        if v < 0:
            if rear_min_sr < self.sr_stop_rear:
                print(f"[{self.name}] SR REAR STOP ({rear_min_sr:.2f}m) -> EMERGENCY_BACKOFF")
                self.emergency_backoff_steps = max(self.emergency_backoff_steps, self.emergency_backoff_duration)
                return 1.6, self._clip(1.4 * np.sign(rr - rl), -self.omega_limit, self.omega_limit)
            elif rear_min_sr < self.sr_slow_rear:
                slow_scale = max(0.15, (rear_min_sr - self.sr_stop_rear) / max(1e-6, (self.sr_slow_rear - self.sr_stop_rear)))
                v *= slow_scale
                omega += 1.1 * np.sign(rr - rl) * (1.0 - slow_scale)

        # --- START: REVERTED SENSOR LOGIC ---
        
        # If no lidar available, return clipped requested commands
        if lidar_ranges is None or self.lidar is None:
            v_ret = self._clip(v, -self.max_wheel_speed, self.max_wheel_speed)
            omega_ret = self._clip(omega, -self.omega_limit, self.omega_limit)
            return v_ret, omega_ret
        
        # --- END: REVERTED SENSOR LOGIC ---

        try:
            # The 'ranges' variable is now always from lidar
            ranges = np.array(lidar_ranges, dtype=float)
            ranges = np.where(np.isfinite(ranges), ranges, 1e3)
            min_range = float(np.min(ranges))

            if min_range > self.cbf_trigger_distance:
                # Nothing close in mid-field — allow requested command
                v_ret = self._clip(v, -self.max_wheel_speed, self.max_wheel_speed)
                omega_ret = self._clip(omega, -self.omega_limit, self.omega_limit)
                return v_ret, omega_ret

            # angles for lidar beams
            angles = np.array(self._lidar_angles()) if self._lidar_angles() is not None else np.linspace(-math.pi/2, math.pi/2, len(ranges))

            # CBF logic
            allowed_vs = []
            for r, theta in zip(ranges, angles):
                c = math.cos(theta)
                if c <= 1e-6:
                    continue
                margin = r - self.d_min
                v_allowed = (self.cbf_alpha * margin) / (c + 1e-9)
                if v_allowed < -5.0 * self.max_wheel_speed:
                    continue
                allowed_vs.append(v_allowed)

            if allowed_vs:
                v_max_allowed = float(np.percentile(np.array(allowed_vs), self.cbf_allowed_percentile))
                v_max_allowed = self._clip(v_max_allowed, -self.max_wheel_speed, self.max_wheel_speed)
            else:
                v_max_allowed = self.max_wheel_speed

            # Central sector check
            center_idx = len(ranges)//2
            sector_width = max(5, len(ranges) // 10) # 10% of sensor width
            sector = ranges[max(0, center_idx - sector_width): center_idx + sector_width + 1]
            front_min = float(np.min(sector)) if sector.size else float(np.min(ranges))

            if front_min < (self.d_min * 0.5):
                self.emergency_backoff_steps = self.emergency_backoff_duration
                print(f"[{self.name}] VERY CLOSE OBSTACLE -> EMERGENCY_BACKOFF front_min={front_min:.3f}")
                return -1.6, self._clip(1.4, -self.omega_limit, self.omega_limit)

            if v_max_allowed < 0.0:
                v = min(v, max(v_max_allowed, -1.8))
            else:
                v = min(v, v_max_allowed, self.max_wheel_speed)

            # Angular avoidance
            d_thresh = max(self.d_min*3.0, 1.0)
            angular_sum = 0.0
            angular_weight = 0.0
            for r, theta in zip(ranges, angles):
                if r < d_thresh:
                    w = (d_thresh - r) / (d_thresh + 1e-9)
                    w *= (1.0 - (abs(theta) / (math.pi/2.0)))
                    angular_sum += w * (-math.copysign(1.0, theta))
                    angular_weight += abs(w)

            if angular_weight > 1e-6:
                omega_avoid = self.omega_gain * (angular_sum / (angular_weight + 1e-9))
                front_mask = (np.abs(angles) < math.radians(30))
                front_close = np.any((ranges[front_mask] < (self.d_min * 1.5))) if front_mask.size > 0 else False
                if front_close:
                    v = min(v, 0.25)
                    omega = 0.65 * omega_avoid + 0.35 * omega
                else:
                    omega = 0.3 * omega_avoid + 0.7 * omega
            
            # Stuck/low-speed logic
            if abs(v) < self.v_low_thresh:
                self.low_speed_count += 1
            else:
                self.low_speed_count = 0
            if self.low_speed_count >= self.low_speed_timeout_steps:
                self.repulsive_steps = self.repulsive_duration
                self.low_speed_count = 0
                print(f"[{self.name}] REPULSIVE triggered due to prolonged low v -> repulsive_steps={self.repulsive_steps}")
                # Use the current sensor data for the repulsive command
                return self.compute_repulsive_command(ranges, angles)

            self.stuck_front_history.append(front_min)
            if len(self.stuck_front_history) > self.stuck_history_len:
                self.stuck_front_history.pop(0)
            if len(self.stuck_front_history) == self.stuck_history_len and np.mean(self.stuck_front_history) < (self.d_min * 0.9):
                self.repulsive_steps = max(self.repulsive_steps, int(self.repulsive_duration))
                self.stuck_front_history = []
                print(f"[{self.name}] STUCK_HISTORY triggered -> repulsive_steps={self.repulsive_steps}")

            if self.step_count % self.debug_rate == 0:
                sensor_name = "LIDAR"
                print(f"[{self.name}] NAV ({sensor_name}) min_range={min_range:.3f} front_min={front_min:.3f} v_allowed={v_max_allowed:.3f} v_cmd={v:.3f} omega={omega:.3f}")

            omega = self._clip(omega, -self.omega_limit, self.omega_limit)
            v = self._clip(v, -self.max_wheel_speed, self.max_wheel_speed)
            return v, omega

        except Exception as ex:
            print(f"[{self.name}] Error compute_safe_controls: {ex}")
            return 0.0, 0.0

    # ---------- Navigation ----------
    def navigate_to(self, target_centroid, linear_speed=3.0, angle_k=2.0, dist_tol=0.6, debug=False):
        pos = self.get_position()
        yaw = self.get_yaw()
        v_des = linear_speed
        omega_des = 0.0

        if pos is None:
            lidar_ranges = self.get_lidar_ranges()
            v_safe, omega_safe = self.compute_safe_controls(v_des*0.6, 0.0, lidar_ranges)
            self._apply_wheel_cmds(v_safe, omega_safe)
            return False

        tx, ty = float(target_centroid[0]), float(target_centroid[1])
        dx = tx - pos[0]
        dy = ty - pos[1]
        distance = math.hypot(dx, dy)
        if distance <= dist_tol:
            self.set_wheel_speeds(0, 0, 0, 0)
            return True

        desired_heading = math.atan2(dy, dx)
        if yaw is None:
            lidar_ranges = self.get_lidar_ranges()
            v_safe, omega_safe = self.compute_safe_controls(v_des*0.5, 0.0, lidar_ranges)
            self._apply_wheel_cmds(v_safe, omega_safe)
            return False

        angle_err = self._angle_diff(yaw, desired_heading)
        omega_des = 0.0 if abs(angle_err) < self.heading_deadzone_rad else self._clip(angle_k * angle_err, -self.omega_limit, self.omega_limit)
        v_des = linear_speed * max(0.2, (1.0 - abs(angle_err)/math.pi))

        lidar_ranges = self.get_lidar_ranges()
        v_safe, omega_safe = self.compute_safe_controls(v_des, omega_des, lidar_ranges)

        if debug and (self.step_count % (self.debug_rate) == 0):
            print(f"[{self.name}] NAV_DEBUG dist={distance:.2f} yaw={yaw:.2f} angle_err={angle_err:.2f} v_des={v_des:.2f} v_safe={v_safe:.2f} omega_safe={omega_safe:.2f}")

        self._apply_wheel_cmds(v_safe, omega_safe)
        return False

    def compute_inverted_command(self, linear_speed=4.0, angle_k=2.0):
        pos = self.get_position()
        yaw = self.get_yaw()
        if pos is None or yaw is None:
            return linear_speed, 0.0

        sum_x = 0.0
        sum_y = 0.0
        count = 0
        now = now_ts()
        for name, (x, y, ts) in list(self.received_poses.items()):
            if name == self.name:
                continue
            if now - float(ts) > 10.0:
                continue
            dx = pos[0] - float(x)
            dy = pos[1] - float(y)
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            w = min(1.0, 1.0 / (dist + 1e-3))
            sum_x += w * (dx / dist)
            sum_y += w * (dy / dist)
            count += 1

        if count == 0 or (abs(sum_x) < 1e-6 and abs(sum_y) < 1e-6):
            return linear_speed, 0.0

        away_heading = math.atan2(sum_y, sum_x)
        for dist_m in (2.0, 4.0, 6.0, 8.0, 10.0):
            cand_x = pos[0] + dist_m * math.cos(away_heading)
            cand_y = pos[1] + dist_m * math.sin(away_heading)
            cell = self.world_to_cell((cand_x, cand_y))
            combined = self.get_combined_explored()
            if cell is None:
                continue
            if cell not in combined:
                desired_heading = math.atan2(cand_y - pos[1], cand_x - pos[0])
                angle_err = self._angle_diff(yaw, desired_heading)
                omega_des = 0.0 if abs(angle_err) < self.heading_deadzone_rad else self._clip(angle_k * angle_err, -self.omega_limit, self.omega_limit)
                v_des = linear_speed * max(0.2, (1.0 - abs(angle_err)/math.pi))
                return v_des, omega_des

        angle_err = self._angle_diff(yaw, away_heading)
        omega_des = 0.0 if abs(angle_err) < self.heading_deadzone_rad else self._clip(angle_k * angle_err, -self.omega_limit, self.omega_limit)
        v_des = linear_speed * max(0.2, (1.0 - abs(angle_err)/math.pi))
        return v_des, omega_des

    # ---------- Consensus helpers (simplified rendezvous) ----------
    def broadcast_pose(self):
        pos = self.get_position()
        if pos is None:
            return False
        payload = {"type": "POSE", "robot": self.name, "pos": [float(pos[0]), float(pos[1])], "ts": now_ts()}
        try:
            self.send_squad_message(payload)
            return True
        except Exception:
            return False

    def update_received_pose(self, msg):
        try:
            if not isinstance(msg, dict):
                return
            if msg.get("type") != "POSE":
                return
            name = msg.get("robot")
            pos = msg.get("pos")
            ts = msg.get("ts", now_ts())
            if name is None or pos is None:
                return
            self.received_poses[name] = (float(pos[0]), float(pos[1]), float(ts))
        except Exception:
            pass

    def compute_consensus_target(self, freshness_s=6.0, min_robots=2):
        try:
            now = now_ts()
            poses = []
            mypos = self.get_position()
            if mypos is not None:
                poses.append((float(mypos[0]), float(mypos[1])))
            for name, (x, y, ts) in list(self.received_poses.items()):
                if now - ts > freshness_s:
                    try:
                        del self.received_poses[name]
                    except Exception:
                        pass
                    continue
                poses.append((float(x), float(y)))
            if len(poses) < min_robots:
                return None
            sx = sum(p[0] for p in poses) / float(len(poses))
            sy = sum(p[1] for p in poses) / float(len(poses))
            return (sx, sy)
        except Exception as ex:
            print(f"[{self.name}] Error compute_consensus_target: {ex}")
            return None

    # ---------- Victim mission helpers ----------
    def _nearest_unvisited_red_victim(self):
        pos = self.get_position()
        if pos is None:
            return None
        best = None
        best_d = float("inf")
        for vid, info in self.red_victims.items():
            if info.get("visited"):
                continue
            w = info.get("world")
            if not w:
                continue
            d = math.hypot(w[0]-pos[0], w[1]-pos[1])
            if d < best_d:
                best_d = d
                best = (vid, w)
        return best

    def _mark_victim_visited(self, vid):
        self.visited_victims.add(vid)
        if vid in self.red_victims:
            self.red_victims[vid]["visited"] = True
        if vid in self.victims:
            self.victims[vid]["visited"] = True
        self.persist_victims()

    def _notify_supervisor_detected(self, vid, world_xy=None):
        # This function is from proposed_solution and sends a *different*
        # message type ("VICTIM_DETECTED") than the simple supervisor expects.
        # We will use send_decision_request instead.
        # This function can be kept for a more advanced supervisor.
        
        vinfo = self.victims.get(vid, {})
        pos = self.get_position()
        dist = None
        if pos and world_xy:
            try:
                dist = float(math.hypot(pos[0]-world_xy[0], pos[1]-world_xy[1]))
            except Exception:
                dist = None
    
        if dist is None:
            print(f"[{self.name}] NOT notifying supervisor: distance unknown for victim {vid} (world={world_xy})")
            return False
    
        # The simple supervisor checks proximity, so we don't need this check
        # if not (dist < 1.0):
        #     print(f"[{self.name}] NOT notifying supervisor: distance {dist:.3f} >= 1.0m for victim {vid}")
        #     return False
    
        payload = {
            "type": "VICTIM_DETECTED", # This is the complex message type
            "robot": self.name,
            "victim_id": vid,
            "world": list(world_xy) if world_xy else None,
            "color": "red",
            "victim_confidence": float(vinfo.get("confidence", 0.0)),
            "distance_to_victim": float(dist),
            "robot_pos": [float(pos[0]), float(pos[1])],
            "ts": now_ts()
        }
        # We send this *in addition* to the simple request
        ok = self.send_supervisor_message(payload)
        print(f"[{self.name}] Supervisor notify (COMPLEX) vid={vid} dist={dist:.3f} sent={ok}")
        self._append_map_history("victim_detected_confirmed", payload)
        return ok

    # greedy pixel-centering control (fallback when no world/depth)
    def greedy_pursue_pixel(self, cx, w_img, base_speed=2.0):
        center = w_img / 2.0
        ex = (cx - center) / float(w_img)
        omega_des = 4.0 * ex
        v_des = base_speed * max(0.3, 1.0 - abs(ex))
        lidar_ranges = self.get_lidar_ranges()
        v_safe, omega_safe = self.compute_safe_controls(v_des, omega_des, lidar_ranges)
        self._apply_wheel_cmds(v_safe, omega_safe)

    def save_rgb_frame(self, step_count):
        rgb = self.get_rgb_image()
        if rgb is None: return
        Image.fromarray(rgb).save(os.path.join(self.debug_dir, f"rgb_step{step_count}.png"))

    def save_depth_frame(self, step_count):
        depth = self.get_depth_image()
        if depth is None: return
        dmin = self.camera_depth.getMinRange()
        dmax = self.camera_depth.getMaxRange()
        norm = ((np.clip(depth, dmin, dmax) - dmin) / (dmax - dmin) * 255).astype(np.uint8)
        Image.fromarray(norm).save(os.path.join(self.debug_dir, f"depth_step{step_count}.png"))

    def read_sensors(self):
        gps_val = self.gps.getValues() if self.gps else None
        imu_val = self.get_inertial_quaternion()
        print(f"[{self.name}] SENSORS gps={gps_val} imu={imu_val}")


# --- Main loop ---
def main():
    bot = Rosbot()
    print("has poll_squad_messages?", hasattr(bot, "poll_squad_messages"))

    try:
        bot.calibrate_yaw_sign()
    except Exception:
        pass

    step = 0
    forward_speed = 6.0
    print(f"[{bot.name}] START")

    while bot.robot.step(bot.timestep) != -1:
        step += 1
        bot.step_count = step

        # --- MODIFIED: Removed the "stop-and-wait" logic ---
        # We only poll for responses, but we don't stop
        if bot.action_pending:
            approval_status = bot.handle_supervisor_response() 
            # We don't do anything with the status, just clear the flag
        
        # If we ARE in a wait state (e.g. from an old request),
        # clear it immediately. We no longer wait.
        if bot.state == RobotState.WAITING_APPROVAL:
             bot.state = RobotState.EXPLORING
             bot.action_pending = False
        # --- END MODIFICATION ---

        # broadcast pose
        if step % bot.pose_bcast_steps == 0:
            try:
                bot.broadcast_pose()
            except Exception:
                pass

        # process squad messages
        msgs = bot.poll_squad_messages()
        for m in msgs:
            try:
                if isinstance(m, dict) and m.get("type") == "GRAPH_SUMMARY":
                    bot.merge_graph_summary(m)
                if isinstance(m, dict) and m.get("type") == "POSE":
                    bot.update_received_pose(m)
                if isinstance(m, dict) and m.get("type") == "EXPLORED_UPDATE":
                    bot.merge_remote_explored(m.get("robot", "unknown"), m.get("cells", []))
            except Exception as ex:
                print(f"[{bot.name}] msg err {ex}")

        # expire old explored entries
        if step % (bot.pose_bcast_steps * 4) == 0:
            bot.expire_old_explored()

        # periodically broadcast explored cells
        if (now_ts() - bot.last_explored_bcast) >= bot.explored_bcast_interval_s:
            try:
                bot.broadcast_explored_update()
            except Exception:
                pass

        if step % 10 == 0:
            bot.read_sensors()

        if step == 1:
            bot.save_rgb_frame(0)
            bot.save_depth_frame(0)

        # HARD BARRIER position-based anti-stuck
        pos = bot.get_position()
        if pos is not None:
            bot.mark_explored_world_area(pos, radius=bot.explore_mark_radius)
            if bot.stuck_last_pos is None:
                bot.stuck_last_pos = pos
                bot.stuck_last_moved_ts = now_ts()
            else:
                disp = math.hypot(pos[0]-bot.stuck_last_pos[0], pos[1]-bot.stuck_last_pos[1])
                if disp >= bot.stuck_hard_disp_thresh:
                    bot.stuck_last_pos = pos
                    bot.stuck_last_moved_ts = now_ts()
                else:
                    if (now_ts() - bot.stuck_last_moved_ts) > bot.stuck_hard_window_s:
                        if (now_ts() - bot.stuck_last_trigger_ts) > bot.stuck_hard_cooldown_s:
                            bot.stuck_last_trigger_ts = now_ts()
                            bot.stuck_last_moved_ts = now_ts()
                            bot.stuck_last_pos = pos
                            bot.emergency_backoff_steps = max(bot.emergency_backoff_steps, bot.emergency_backoff_duration)
                            bot.repulsive_steps = max(bot.repulsive_steps, bot.repulsive_duration)
                            print(f"[{bot.name}] HARD_BARRIER: pos unchanged > {bot.stuck_hard_window_s:.1f}s -> EMERGENCY_BACKOFF + REPULSIVE")

        
        # --- NEW: Periodic OpenCV RED victim detection ---
        detections = [] # Reset detections list
        if step % bot.victim_check_interval == 0:
            try:
                # This function now uses OpenCV, updates self.red_victims,
                # and returns a list of detections for greedy-pixel updates
                detections = bot.detect_victim()
            except Exception as ex:
                print(f"[{bot.name}] red victim detect err: {ex}")
        # --- END NEW ---
        
        current_time = bot.robot.getTime()
        if (current_time - bot.last_decision_time > bot.decision_interval) and not bot.action_pending:
            # Only send a periodic request if we are NOT currently on a victim mission
            if bot.current_victim_id is None:
                action_name = "explore_forward" 
                
                # Check the bot's actual current task
                if bot.current_task == "navigate:consensus":
                    action_name = "return_to_rendezvous"
                elif bot.current_task == "navigate:explore_cell":
                    action_name = "explore_forward"
                elif bot.current_task is None and not in_rendezvous_window:
                    # If no task and after rendezvous, we are in "inverted rendezvous"
                    action_name = "explore_forward" # We'll just call this "explore"
                
                reason = bot.generate_explanation(action_name)
                bot.send_decision_request(
                    action_name,  # <-- Use the honest action_name
                    reason, 
                    victim_detected=False, 
                    confidence=0.0, 
                    wait_for_approval=False
                )
                bot.last_decision_time = current_time

        # --- MODIFIED: Re-activated VICTIM-FIRST logic ---
        # ---------------------------
        # VICTIM-FIRST: preempt EVERYTHING
        # ---------------------------
        victim_available = bool(bot.red_victims)
        if victim_available:
            if bot.current_task and bot.current_task.startswith("navigate"):
                bot.current_task = None
                bot.task_target = None

            if bot.current_victim_id is None:
                nv = bot._nearest_unvisited_red_victim()
                if nv is not None:
                    bot.current_victim_id, bot.task_target = nv[0], nv[1]
                    bot.current_task = "navigate:victim"
                    print(f"[{bot.name}] VICTIM mission start -> {bot.current_victim_id} @ {bot.task_target}")

                    # --- NEW: Report to Supervisor (as requested) ---
                    try:
                        v_info = bot.red_victims.get(bot.current_victim_id, {})
                        v_conf = v_info.get("confidence", 0.9) 
                        reason = bot.generate_explanation("investigate_victim")
                        
                        # Send the request, but DO NOT wait for approval
                        bot.send_decision_request(
                            "investigate_victim", reason, True, v_conf, wait_for_approval=False
                        )
                        bot.last_victim_report_time = bot.robot.getTime()
                    except Exception as e:
                        print(f"[{bot.name}] Error reporting victim to supervisor: {e}")
                    # --- END NEW ---
                
                else:
                    # no world coords yet — use greedy on freshest pixel
                    freshest_id = None
                    freshest_ts = -1
                    px_center = None
                    w_img = bot.camera_rgb.getWidth() if bot.camera_rgb else None
                    for vid, info in bot.red_victims.items():
                        if info.get("visited"):
                            continue
                        ts = info.get("last_seen", 0)
                        if ts > freshest_ts and info.get("pixel"):
                            freshest_ts = ts
                            freshest_id = vid
                            px_center = info["pixel"]["x"]
                    if freshest_id is not None and w_img is not None:
                        bot.current_victim_id = freshest_id
                        bot.current_task = "navigate:victim_greedy"
                        bot.greedy_active = True
                        bot.greedy_target_px = px_center
                        bot.greedy_last_seen_ts = now_ts()
                        print(f"[{bot.name}] VICTIM mission (GREEDY) -> {bot.current_victim_id} pixel_x={px_center}")

        # If we are on a victim mission
        if bot.current_victim_id is not None:
            rv = bot.red_victims.get(bot.current_victim_id, {})
            
            # Refresh greedy pixel if a new detection came
            if detections:
                for det in detections:
                    vid = det.get("id")
                    if vid == bot.current_victim_id and det.get("pixel"):
                        bot.greedy_target_px = det["pixel"]["x"]
                        bot.greedy_last_seen_ts = now_ts()

            # If we have a world target -> navigate gently and confirm within tol
            if rv.get("world"):
                bot.task_target = rv["world"]
                bot.current_task = "navigate:victim"
                reached = bot.navigate_to(bot.task_target, linear_speed=3.2, angle_k=2.0,
                                          dist_tol=bot.victim_target_tol_m, debug=True)

                if reached:
                    # --- MODIFIED: Use _notify_supervisor_detected ---
                    # This sends the "VICTIM_DETECTED" message for complex supervisors
                    bot._notify_supervisor_detected(bot.current_victim_id, bot.task_target)
                    # --- END MODIFICATION ---
                    bot._mark_victim_visited(bot.current_victim_id)
                    bot.post_detection_backoff_steps = bot.post_detection_backoff_duration
                    bot.current_victim_id = None
                    bot.current_task = None
                    bot.task_target = None
                    lidar_ranges = bot.get_lidar_ranges()
                    v_safe, w_safe = bot.compute_safe_controls(0.0, 0.0, lidar_ranges)
                    bot._apply_wheel_cmds(v_safe, w_safe)
                    continue
                continue
            else:
                # GREEDY pixel-centering pursuit
                if bot.greedy_active and bot.greedy_target_px is not None and bot.camera_rgb is not None:
                    w_img = bot.camera_rgb.getWidth()
                    bot.greedy_pursue_pixel(bot.greedy_target_px, w_img, base_speed=2.2)
                    if (now_ts() - bot.greedy_last_seen_ts) > bot.greedy_timeout_s:
                        print(f"[{bot.name}] GREEDY timeout -> releasing victim lock")
                        bot.greedy_active = False
                        bot.current_victim_id = None
                        bot.current_task = None
                        bot.task_target = None
                    continue
                else:
                    bot.current_victim_id = None
                    bot.current_task = None
                    bot.task_target = None
        # --- END VICTIM-FIRST BLOCK ---


        # -------------------------
        # Rendezvous (first window) -> Inverted rendezvous after timeout or reach
        # -------------------------
        in_rendezvous_window = (now_ts() - bot.start_time) <= bot.rendezvous_window_s
        if in_rendezvous_window:
            centroid = bot.compute_consensus_target(freshness_s=6.0, min_robots=2)
            if centroid:
                cell = bot.world_to_cell(centroid)
                combined = bot.get_combined_explored()
                if cell is not None and cell in combined:
                    target_cell = bot.find_nearest_unexplored_cell(max_radius_cells=40)
                    if target_cell is not None:
                        target_world = bot.cell_to_world_center(target_cell)
                        bot.current_task = "navigate:explore_cell"
                        bot.task_target = target_world
                        reached = bot.navigate_to(bot.task_target, linear_speed=forward_speed, angle_k=2.0, dist_tol=0.8, debug=True)
                        if reached:
                            bot.mark_explored_world_area(target_world, radius=bot.explore_mark_radius)
                            bot.current_task = None
                            bot.task_target = None
                        continue
                    else:
                        lidar_ranges = bot.get_lidar_ranges()
                        v_safe, omega_safe = bot.compute_safe_controls(forward_speed, 0.0, lidar_ranges)
                        bot._apply_wheel_cmds(v_safe, omega_safe)
                        continue
                else:
                    bot.current_task = "navigate:consensus"
                    bot.task_target = centroid
                    reached = bot.navigate_to(bot.task_target, linear_speed=4.0, angle_k=2.0, dist_tol=0.8, debug=True)
                    if reached:
                        bot.rendezvous_reached = True
                        bot.current_task = None
                        bot.task_target = None
                        bot._append_map_history("rendezvous_reached", {"target": centroid})
                    continue
            else:
                target_cell = bot.find_nearest_unexplored_cell(max_radius_cells=40)
                if target_cell is not None:
                    target_world = bot.cell_to_world_center(target_cell)
                    bot.current_task = "navigate:explore_cell"
                    bot.task_target = target_world
                    reached = bot.navigate_to(bot.task_target, linear_speed=3.6, angle_k=2.0, dist_tol=0.8, debug=True)
                    if reached:
                        bot.mark_explored_world_area(target_world, radius=bot.explore_mark_radius)
                        bot.current_task = None
                        bot.task_target = None
                    continue
                else:
                    lidar_ranges = bot.get_lidar_ranges()
                    v_safe, omega_safe = bot.compute_safe_controls(forward_speed, 0.0, lidar_ranges)
                    bot._apply_wheel_cmds(v_safe, omega_safe)
                    continue
        else:
            # After rendezvous window -> inverted rendezvous mode (move away)
            v_des, omega_des = bot.compute_inverted_command(linear_speed=4.0, angle_k=2.0)
            lidar_ranges = bot.get_lidar_ranges()
            v_safe, omega_safe = bot.compute_safe_controls(v_des, omega_des, lidar_ranges)
            bot._apply_wheel_cmds(v_safe, omega_safe)
            continue

        # default (shouldn't reach here) safe forward
        lidar_ranges = bot.get_lidar_ranges()
        v_safe, omega_safe = bot.compute_safe_controls(forward_speed, 0.0, lidar_ranges)
        bot._apply_wheel_cmds(v_safe, omega_safe)

    print(f"[{bot.name}] EXIT")


if __name__ == "__main__":
    main()
