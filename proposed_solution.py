# proposed_solution_fixed_stuck_with_victim_detection_consensus_timecycle.py
# Revised controller:
# - Victim-first policy: any red-victim sighting instantly cancels consensus and starts a victim mission
# - Greedy pursuit fallback if depth/world is unavailable (pixel-centering drive)
# - Time-based consensus cycle preserved, but preempted by victim missions
# - Tight HSV for ONLY the provided dark red swatch (h≈2°, high S, low V)
# - Preserves damping, smoothing, mapping, GRAPH_SUMMARY merging, short-range avoidance,
#   hard barrier anti-stuck, repulsive recovery, and supervisor notification/backoff.

import json
import math
import os
import time
import uuid
from controller import Robot
import numpy as np
from PIL import Image

# optional OpenCV for robust RED blob detection/annotating
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


class Rosbot:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        try:
            self.name = self.robot.getName()
        except Exception:
            self.name = getattr(self.robot, "getName", lambda: "rosbot")()

        self.debug_dir = os.path.join(os.getcwd(), "example_camera_outputs")
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
        self.victim_target_tol_m = 1.0

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
        self.max_wheel_speed = 12.0
        self.yaw_inversion = 1
        self.d_min = 0.3
        self.cbf_alpha = 30.5
        self.omega_gain = -1.5

        # short-range safety thresholds (meters)
        self.sr_stop_front = 0.28
        self.sr_stop_rear = 0.28
        self.sr_slow_front = 0.45
        self.sr_slow_rear = 0.45

        # stuck/low-speed detection & recovery (tune)
        self.stuck_history_len = 12
        self.stuck_front_history = []
        self.stuck_backoff_steps = 0
        self.emergency_backoff_steps = 0
        self.low_speed_count = 0
        self.v_low_thresh = 0.18
        self.low_speed_timeout_steps = 4
        self.emergency_backoff_duration = 28

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
        self.repulsive_duration = 28
        self.repulsive_influence = 0.2
        self.repulsive_max_v = 20.5
        self.repulsive_max_omega = 10.2
        self.repulsive_k_v = 10.0
        self.repulsive_k_omega = 20.0

        # victim detection config
        self.victim_check_interval = 5  # check every N steps
        self.victim_area_thresh = 300   # pixels
        self.victim_pixel_dedupe = 50   # px
        self.victim_world_dedupe = 0.8  # m
        self.victim_next_id = 1

        # --- Consensus / pose broadcast parameters ---
        self.pose_bcast_interval_s = 0.5
        try:
            self.pose_bcast_steps = max(1, int(self.pose_bcast_interval_s * 1000.0 / float(self.timestep)))
        except Exception:
            self.pose_bcast_steps = 15
        self.last_pose_bcast = 0
        self.received_poses = {}  # robot_name -> (x,y,ts)

        # Startup window
        self.start_time = now_ts()
        self.startup_window_s = 15.0
        self.startup_window_end = self.start_time + self.startup_window_s
        self.startup_consensus_done = False

        # consensus defaults
        self.consensus_target = None
        self.consensus_tolerance = 0.6
        self.consensus_min_robots = 2
        self.consensus_active = False
        self.consensus_timeout_s = 12.0
        self.consensus_started_ts = None

        # if GPS available, add our own pose
        pos = self.get_position()
        if pos is not None:
            self.received_poses[self.name] = (float(pos[0]), float(pos[1]), now_ts())

        # --- angular scaling / smoothing ---
        self.omega_scale = 0.6
        self.omega_limit = 4.0
        self.omega_smooth_alpha = 0.65
        self.prev_omega = 0.0
        self.heading_deadzone_rad = math.radians(6.0)

        # consensus cycle mode (None / 'forward' / 'reverse')
        self.consensus_motion_mode = None
        self.consensus_cycle_total = 34.0   # 15 + 2 + 15 + 2 = 34s cycle
        self.consensus_inactive_grace_s = 2.0
        self.consensus_inactive_since = None

        # GREEDY pixel-centering pursuit (fallback when no depth/world)
        self.greedy_active = False
        self.greedy_target_px = None
        self.greedy_last_seen_ts = 0.0
        self.greedy_timeout_s = 4.0  # give up if not updated

        # ---- Narrow HSV ranges for ONLY the given dark red swatch ----
        # Image analysis of swatch (HSV, OpenCV scale H:0..179) -> H≈2, S≈241, V≈74.
        # We set tight windows with a little tolerance and wrap handling.
        self.hsv_narrow_ranges = [
            (np.array([0,   200, 40], dtype=np.uint8),  np.array([6,   255, 130], dtype=np.uint8)),   # near H=2
            (np.array([175, 200, 40], dtype=np.uint8),  np.array([179, 255, 130], dtype=np.uint8)),   # wrap end, narrow
        ]

        if not OPENCV_AVAILABLE:
            print(f"[{self.name}] Warning: OpenCV not available — red victim detection disabled. Install opencv-python to enable.")

    # ---------- Consensus cycle helper ----------
    def update_consensus_cycle(self):
        """
        Cycle: 0-15s -> forward_active
               15-17s -> inactive (short)
               17-32s -> reverse_active
               32-34s -> inactive (short)
        Repeats every 34s. (active most of the time)
        """
        try:
            elapsed = now_ts() - self.start_time
            cycle = self.consensus_cycle_total
            t = elapsed % cycle
            if t < 12:
                return "forward_active"
            elif t < 17.0:
                return "inactive"
            elif t < 20.0:
                return "reverse_active"
            else:
                return "inactive"
        except Exception:
            return "inactive"


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

    # ---------- Wheel mapping helper (damping & scaling) ----------
    def _apply_wheel_cmds(self, v_safe, omega_safe):
        """Convert v_safe and omega_safe into left/right wheel velocities with smoothing and limits."""
        try:
            omega_scaled = float(omega_safe) * self.omega_scale
            omega_scaled = self._clip(omega_scaled, -self.omega_limit, self.omega_limit)
            omega_cmd = self.prev_omega * self.omega_smooth_alpha + omega_scaled * (1.0 - self.omega_smooth_alpha)
            self.prev_omega = omega_cmd

            left = self._clip(v_safe - omega_cmd, -self.max_wheel_speed, self.max_wheel_speed)
            right = self._clip(v_safe + omega_cmd, -self.max_wheel_speed, self.max_wheel_speed)
            self.set_wheel_speeds(left, right, left, right)
        except Exception as ex:
            print(f"[{self.name}] Error _apply_wheel_cmds: {ex}")
            try:
                self.set_wheel_speeds(0, 0, 0, 0)
            except Exception:
                pass

    # ---------- Sensors shortcuts ----------
    def get_lidar_ranges(self):
        if not self.lidar:
            return None
        try:
            return self.lidar.getRangeImage()
        except Exception:
            return None

    def get_position(self):
        if not self.gps:
            return None
        try:
            vals = self.gps.getValues()  # [x, y, z] - use x and z as ground-plane coords
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

    # ---------- Short-range helper ----------
    def get_short_range_values(self):
        """Returns dict with distances (meters) or np.inf if sensor absent."""
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
            fov = cam.getFov()  # horizontal FOV (rad)
            cx = w / 2.0
            # bearing relative to camera center
            bearing = (px - cx) / float(w) * fov
            yaw = self.get_yaw()
            pos = self.get_position()
            if yaw is None or pos is None:
                return None
            world_heading = yaw + bearing
            r = float(depth_val)
            wx = pos[0] + r * math.cos(world_heading)
            wy = pos[1] + r * math.sin(world_heading)
            return (float(wx), float(wy))
        except Exception:
            return None

    def handle_supervisor_response(self, msg):
        """
        Process a parsed supervisor message `msg`.
        If the supervisor explicitly rejects an action for this robot
        (approved == False and robot matches self.name), we:
          - log the rejection in map history
          - abort the current task / consensus navigation
          - trigger emergency backoff so the robot moves away safely
          - send a mission report explaining the state
        If approved, we log approval.
        """
        try:
            if not isinstance(msg, dict):
                return
            # message may use "robot" or "robot_id"
            target = msg.get("robot") or msg.get("robot_id")
            # only act on messages addressed to this robot (or broadcast with no robot specified)
            if target and str(target) != str(self.name):
                return
            approved = msg.get("approved", None)
            if approved is False:
                # supervisor denied an action — ensure we safely abort and report
                self._append_map_history("supervisor_reject", {"msg": msg})
                # abort tasks (consensus + navigation)
                if self.current_task:
                    self._append_map_history("abort_task", {"task": self.current_task})
                self.current_task = None
                self.task_target = None
                # cancel consensus if any
                self.consensus_active = False
                self.consensus_target = None
                self.consensus_motion_mode = None
                # trigger emergency backoff to avoid dangerous follow-through
                self.emergency_backoff_steps = max(self.emergency_backoff_steps, self.emergency_backoff_duration)
                # persist & broadcast a mission report describing the rejection
                try:
                    reason = msg.get("reason", "supervisor_rejected_action")
                    self.send_mission_report(reason=reason, rejected=True, supervisor_msg=msg)
                except Exception:
                    pass
                # also inform teammates briefly
                try:
                    self.send_squad_message({"type":"SUPERVISOR_REJECT","robot":self.name,"reason":str(msg.get("reason","")),"ts": now_ts()})
                except Exception:
                    pass
                print(f"[{self.name}] Supervisor rejected action -> aborting tasks, backoff set({self.emergency_backoff_steps})")
            elif approved is True:
                self._append_map_history("supervisor_approve", {"msg": msg})
                # optionally, you could resume or mark the last proposal accepted.
                print(f"[{self.name}] Supervisor approved action.")
            else:
                # generic supervisor message (not approval/rejection) -> just log it
                self._append_map_history("supervisor_message", {"msg": msg})
        except Exception as ex:
            print(f"[{self.name}] Error handle_supervisor_response: {ex}")

    def send_mission_report(self, reason="status_update", rejected=False, supervisor_msg=None):
        """
        Compile a concise mission report and send/persist it.
        Contains: robot name, timestamp, position, current task, victims (brief), known nodes (brief), rejected flag, supervisor_msg (if any).
        """
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
                "supervisor_msg": supervisor_msg
            }
            # persist small report in history file
            self._append_map_history("mission_report", report)
            # send to supervisor/emitter if available, otherwise to squad as fallback
            if self.supervisor_emitter:
                try:
                    raw = safe_json_dumps(report)
                    if hasattr(self.supervisor_emitter, "sendString"):
                        self.supervisor_emitter.sendString(raw)
                    else:
                        self.supervisor_emitter.send(raw.encode("utf-8"))
                    print(f"[{self.name}] Sent mission report to supervisor.")
                    return True
                except Exception:
                    pass
            # fallback to squad broadcast
            try:
                self.send_squad_message(report)
                print(f"[{self.name}] Broadcasted mission report to squad (fallback).")
                return True
            except Exception:
                pass
            return False
        except Exception as ex:
            print(f"[{self.name}] Error send_mission_report: {ex}")
            return False

    # ---------- RED Victim detection (OpenCV, very narrow range) ----------
    def detect_red_victims(self, step_count):
        """
        Returns list of detections: [(id, cx, cy, bbox(x,y,w,h), world_xy_or_None, 'red')]
        dedupes with existing victims. Uses tight HSV windows around the provided red swatch.
        """
        if not OPENCV_AVAILABLE or self.camera_rgb is None:
            return []

        try:
            buf = self.camera_rgb.getImage()
            w, h = self.camera_rgb.getWidth(), self.camera_rgb.getHeight()
            img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
            bgr = img[:, :, :3].copy()  # BGRA -> BGR
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            # Combine the two tight windows (with wrap)
            masks = []
            for lo, hi in self.hsv_narrow_ranges:
                masks.append(cv2.inRange(hsv, lo, hi))
            mask = masks[0]
            for m in masks[1:]:
                mask = cv2.bitwise_or(mask, m)

            # Morphology cleanup (light touch to keep small blobs)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            annotated = bgr.copy()

            depth_arr = None
            if self.camera_depth:
                try:
                    d = np.array(self.camera_depth.getRangeImage(), dtype=np.float32)
                    depth_arr = d.reshape((self.camera_depth.getHeight(), self.camera_depth.getWidth()))
                except Exception:
                    depth_arr = None

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.victim_area_thresh:
                    continue
                x, y, wbox, hbox = cv2.boundingRect(cnt)
                cx = int(x + wbox / 2.0)
                cy = int(y + hbox / 2.0)

                world_pos = None
                depth_val = None
                if depth_arr is not None:
                    try:
                        dh, dw = depth_arr.shape
                        sx = int(cx * (dw / float(bgr.shape[1])))
                        sy = int(cy * (dh / float(bgr.shape[0])))
                        sx = np.clip(sx, 0, dw-1)
                        sy = np.clip(sy, 0, dh-1)
                        depth_val = float(depth_arr[sy, sx])
                        if np.isfinite(depth_val) and depth_val > 0.05:
                            world_pos = self._pixel_to_world_approx(cx, cy, depth_val)
                    except Exception:
                        world_pos = None

                # Dedupe with existing victims
                found_same = None
                for vid, info in self.victims.items():
                    ipx = info.get("pixel", {}).get("x")
                    ipy = info.get("pixel", {}).get("y")
                    if ipx is not None and ipy is not None:
                        if math.hypot(ipx - cx, ipy - cy) < self.victim_pixel_dedupe:
                            found_same = vid
                            break
                    if world_pos and info.get("world"):
                        try:
                            wx, wy = info["world"]
                            if math.hypot(wx - world_pos[0], wy - world_pos[1]) < self.victim_world_dedupe:
                                found_same = vid
                                break
                        except Exception:
                            pass

                if found_same:
                    self.victims[found_same]["last_seen"] = now_ts()
                    self.victims[found_same]["color"] = "red"
                    if world_pos and not self.victims[found_same].get("world"):
                        self.victims[found_same]["world"] = world_pos
                    cv2.rectangle(annotated, (x, y), (x + wbox, y + hbox), (0, 165, 255), 2)
                    cv2.putText(annotated, f"victim:{found_same[:6]}", (x, y-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    detections.append((found_same, cx, cy, (x, y, wbox, hbox),
                                       self.victims[found_same].get("world"), "red"))
                    vid_to_use = found_same
                else:
                    vid = f"rv-{uuid.uuid4().hex[:8]}"
                    info = {
                        "id": vid,
                        "first_seen": now_ts(),
                        "last_seen": now_ts(),
                        "pixel": {"x": int(cx), "y": int(cy), "w": int(wbox), "h": int(hbox)},
                        "color": "red",
                    }
                    if world_pos:
                        info["world"] = (float(world_pos[0]), float(world_pos[1]))
                    self.victims[vid] = info
                    cv2.rectangle(annotated, (x, y), (x + wbox, y + hbox), (0, 0, 255), 2)
                    cv2.putText(annotated, f"NEW:{vid[:6]}", (x, y-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    detections.append((vid, cx, cy, (x, y, wbox, hbox), info.get("world"), "red"))
                    vid_to_use = vid

                # keep a red-victim list for missions
                vinfo = self.victims.get(vid_to_use, {})
                if vinfo.get("color") == "red":
                    rv = self.red_victims.get(vid_to_use, {
                        "id": vid_to_use,
                        "first_seen": vinfo.get("first_seen", now_ts()),
                        "visited": False
                    })
                    rv["last_seen"] = now_ts()
                    rv["pixel"] = vinfo.get("pixel")
                    if vinfo.get("world"):
                        rv["world"] = vinfo.get("world")
                    self.red_victims[vid_to_use] = rv

            if detections:
                out_path = os.path.join(self.debug_dir, f"red_victim_step{step_count}.png")
                try:
                    cv2.imwrite(out_path, annotated)
                except Exception:
                    try:
                        Image.fromarray(annotated[:, :, ::-1]).save(out_path)
                    except Exception:
                        pass

            if detections:
                self.persist_victims()

                # announce sightings
                for det in detections:
                    vid, _, _, _, world, _ = det
                    det_payload = {
                        "type": "VICTIM_SEEN",
                        "robot": self.name,
                        "victim_id": vid,
                        "pixel": self.victims[vid]["pixel"],
                        "world": world,
                        "color": "red",
                        "ts": now_ts()
                    }
                    try:
                        self._append_map_history("victim_seen", det_payload)
                    except Exception:
                        pass
                    try:
                        self.send_squad_message(det_payload)
                    except Exception:
                        pass

            return detections

        except Exception as ex:
            print(f"[{self.name}] Error detect_red_victims: {ex}")
            return []

    # ---------- REPULSIVE control ----------
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

    # ---------- CBF safety + recovery (with short-range integration) ----------
    def compute_safe_controls(self, v_des, omega_des, lidar_ranges):
        v = float(v_des)
        omega = float(omega_des)

        # high-priority emergency backoff
        if self.emergency_backoff_steps > 0:
            self.emergency_backoff_steps -= 1
            v_cmd = -1.6
            omega_cmd = self._clip(1.4, -self.omega_limit, self.omega_limit)
            if self.step_count % self.debug_rate == 0:
                print(f"[{self.name}] EMERGENCY_BACKOFF remaining={self.emergency_backoff_steps} v={v_cmd:.2f} omega={omega_cmd:.2f}")
            return v_cmd, omega_cmd

        # special backoff after a victim confirmation
        if self.post_detection_backoff_steps > 0:
            self.post_detection_backoff_steps -= 1
            v_cmd = -1.3
            omega_cmd = self._clip(1.0, -self.omega_limit, self.omega_limit)
            if self.step_count % self.debug_rate == 0:
                print(f"[{self.name}] POST_DET_BACKOFF remaining={self.post_detection_backoff_steps}")
            return v_cmd, omega_cmd

        # repulsive recovery mode
        if self.repulsive_steps > 0:
            try:
                angles = np.array(self._lidar_angles()) if self._lidar_angles() is not None else None
                v_rep, w_rep = self.compute_repulsive_command(lidar_ranges, angles)
                self.repulsive_steps -= 1
                if self.step_count % self.debug_rate == 0:
                    print(f"[{self.name}] REPULSIVE active steps_left={self.repulsive_steps} v_rep={v_rep:.2f} w_rep={w_rep:.2f}")
                w_rep = self._clip(w_rep, -self.omega_limit, self.omega_limit)
                return v_rep, w_rep
            except Exception as ex:
                print(f"[{self.name}] Error in repulsive mode: {ex}")

        # --- short-range near-field avoidance ---
        sr = self.get_short_range_values()
        fl, fr, rl, rr = sr["fl"], sr["fr"], sr["rl"], sr["rr"]
        front_min_sr = min(fl, fr)
        rear_min_sr = min(rl, rr)

        # forward motion gating
        if v > 0:
            if front_min_sr < self.sr_stop_front:
                print(f"[{self.name}] SR FRONT STOP ({front_min_sr:.2f}m) -> EMERGENCY_BACKOFF")
                self.emergency_backoff_steps = max(self.emergency_backoff_steps, self.emergency_backoff_duration)
                return -1.6, self._clip(1.4 * np.sign(fr - fl), -self.omega_limit, self.omega_limit)
            elif front_min_sr < self.sr_slow_front:
                slow_scale = max(0.15, (front_min_sr - self.sr_stop_front) / max(1e-6, (self.sr_slow_front - self.sr_stop_front)))
                v *= slow_scale
                omega += 1.1 * np.sign(fr - fl) * (1.0 - slow_scale)

        # reverse motion gating
        if v < 0:
            if rear_min_sr < self.sr_stop_rear:
                print(f"[{self.name}] SR REAR STOP ({rear_min_sr:.2f}m) -> EMERGENCY_BACKOFF")
                self.emergency_backoff_steps = max(self.emergency_backoff_steps, self.emergency_backoff_duration)
                return 1.6, self._clip(1.4 * np.sign(rr - rl), -self.omega_limit, self.omega_limit)
            elif rear_min_sr < self.sr_slow_rear:
                slow_scale = max(0.15, (rear_min_sr - self.sr_stop_rear) / max(1e-6, (self.sr_slow_rear - self.sr_stop_rear)))
                v *= slow_scale
                omega += 1.1 * np.sign(rr - rl) * (1.0 - slow_scale)

        # CBF mid-field safety
        if lidar_ranges is None or self.lidar is None:
            v_ret = self._clip(v, -self.max_wheel_speed, self.max_wheel_speed)
            omega_ret = self._clip(omega, -self.omega_limit, self.omega_limit)
            return v_ret, omega_ret

        try:
            ranges = np.array(lidar_ranges, dtype=float)
            ranges = np.where(np.isfinite(ranges), ranges, 1e3)
            angles = np.array(self._lidar_angles())
            if angles is None or len(angles) != len(ranges):
                angles = np.linspace(-math.pi/4, math.pi/4, len(ranges))

            allowed_vs = []
            for r, theta in zip(ranges, angles):
                c = math.cos(theta)
                if c <= 1e-6:
                    continue
                margin = r - self.d_min
                v_allowed = (self.cbf_alpha * margin) / (c + 1e-9)
                if v_allowed < -2.0 * self.max_wheel_speed:
                    continue
                allowed_vs.append(v_allowed)

            if allowed_vs:
                v_max_allowed = min(allowed_vs)
                v_max_allowed = self._clip(v_max_allowed, -self.max_wheel_speed, self.max_wheel_speed)
            else:
                v_max_allowed = self.max_wheel_speed

            center_idx = len(ranges)//2
            sector = ranges[max(0, center_idx-4): center_idx+5]
            front_min = float(np.min(sector)) if sector.size else float(np.min(ranges))

            if front_min < (self.d_min * 0.6):
                self.emergency_backoff_steps = self.emergency_backoff_duration
                print(f"[{self.name}] STUCK: front_min={front_min:.3f} < {self.d_min*0.6:.3f} -> EMERGENCY_BACKOFF({self.emergency_backoff_steps})")
                omega_cmd = self._clip(1.4, -self.omega_limit, self.omega_limit)
                return -1.6, omega_cmd

            if v_max_allowed < 0.0:
                v = min(v, max(v_max_allowed, -1.8))
            else:
                v = min(v, v_max_allowed, self.max_wheel_speed)

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

            # low-speed -> repulsive
            if abs(v) < self.v_low_thresh:
                self.low_speed_count += 1
            else:
                self.low_speed_count = 0
            if self.low_speed_count >= self.low_speed_timeout_steps:
                self.repulsive_steps = self.repulsive_duration
                self.low_speed_count = 0
                print(f"[{self.name}] REPULSIVE triggered due to prolonged low v (v_max_allowed={v_max_allowed:.3f}) -> repulsive_steps={self.repulsive_steps}")
                return self.compute_repulsive_command(ranges, angles)

            # stuck-history -> repulsive
            self.stuck_front_history.append(front_min)
            if len(self.stuck_front_history) > self.stuck_history_len:
                self.stuck_front_history.pop(0)
            if len(self.stuck_front_history) == self.stuck_history_len and np.mean(self.stuck_front_history) < (self.d_min * 0.9):
                self.repulsive_steps = max(self.repulsive_steps, int(self.repulsive_duration))
                self.stuck_front_history = []
                print(f"[{self.name}] STUCK_HISTORY triggered -> repulsive_steps={self.repulsive_steps}")

            if self.stuck_backoff_steps > 0:
                self.stuck_backoff_steps -= 1
                print(f"[{self.name}] RECOVERING stuck_backoff left={self.stuck_backoff_steps}")
                return -1.4, 1.2

            if self.step_count % self.debug_rate == 0:
                print(f"[{self.name}] LIDAR front_min={front_min:.3f} v_allowed={v_max_allowed:.3f} v_cmd={v:.3f} omega={omega:.3f}")

            omega = self._clip(omega, -self.omega_limit, self.omega_limit)
            v = self._clip(v, -self.max_wheel_speed, self.max_wheel_speed)
            return v, omega

        except Exception as ex:
            print(f"[{self.name}] Error compute_safe_controls: {ex}")
            return 0.0, 0.0

    # ---------- Navigation ----------
    def navigate_to(self, target_centroid, linear_speed=3.0, angle_k=2.0, dist_tol=0.6, debug=False, consensus_mode=None):
        """
        consensus_mode: None / 'forward' / 'reverse'
        """
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

        # Apply consensus cycle velocity rules EXACTLY as specified
        if consensus_mode == 'forward':
            v_des = -linear_speed * max(50.0, (1.0 - abs(angle_err)/math.pi))
            omega_des = 0.0 if abs(angle_err) < self.heading_deadzone_rad else self._clip(angle_k * angle_err, -self.omega_limit, self.omega_limit)
        elif consensus_mode == 'reverse':
            v_des = linear_speed * max(50.0, (1.0 - abs(angle_err)/math.pi))
            omega_des = 0.0 if abs(angle_err) < self.heading_deadzone_rad else self._clip(angle_k * angle_err, -self.omega_limit, self.omega_limit)
        else:
            omega_des = 0.0 if abs(angle_err) < self.heading_deadzone_rad else self._clip(angle_k * angle_err, -self.omega_limit, self.omega_limit)
            v_des = -linear_speed * max(50, (1.0 - abs(angle_err)/math.pi))

        lidar_ranges = self.get_lidar_ranges()
        v_safe, omega_safe = self.compute_safe_controls(v_des, omega_des, lidar_ranges)

        if debug and (self.step_count % (self.debug_rate) == 0):
            print(f"[{self.name}] NAV_DEBUG dist={distance:.2f} yaw={yaw:.2f} angle_err={angle_err:.2f} v_des={v_des:.2f} v_safe={v_safe:.2f} omega_safe={omega_safe:.2f} consensus_mode={consensus_mode}")

        self._apply_wheel_cmds(v_safe, omega_safe)
        return False

    # ---------- Consensus helpers ----------
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

    def compute_consensus_target(self, freshness_s=6.0, force=False):
        """Compute centroid of currently-known poses."""
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
            if len(poses) < self.consensus_min_robots and not force:
                return None
            if len(poses) == 0:
                return None
            sx = sum(p[0] for p in poses) / float(len(poses))
            sy = sum(p[1] for p in poses) / float(len(poses))
            return (sx, sy)
        except Exception as ex:
            print(f"[{self.name}] Error compute_consensus_target: {ex}")
            return None

    # ---------- Victim mission helpers ----------
    def _nearest_unvisited_red_victim(self):
        """Return (vid, world_xy) of closest unvisited red victim with a known world position."""
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
        payload = {
            "type": "VICTIM_DETECTED",
            "robot": self.name,
            "victim_id": vid,
            "world": list(world_xy) if world_xy else None,
            "color": "red",
            "ts": now_ts()
        }
        self.send_supervisor_message(payload)
        self._append_map_history("victim_detected_confirmed", payload)
        print(f"[{self.name}] Supervisor notified: {payload}")

    # greedy pixel-centering control (fallback when no world/depth)
    def greedy_pursue_pixel(self, cx, w_img, base_speed=2.0):
        # pixel error from center -> yaw command; small forward speed under safety
        center = w_img / 2.0
        ex = (cx - center) / float(w_img)  # -0.5..0.5
        omega_des = 4.0 * ex  # turn toward blob
        v_des = base_speed * max(0.3, 1.0 - abs(ex))
        lidar_ranges = self.get_lidar_ranges()
        v_safe, omega_safe = self.compute_safe_controls(v_des, omega_des, lidar_ranges)
        self._apply_wheel_cmds(v_safe, omega_safe)

    # rest of controller helpers (sensors, comms, map code)
    def poll_squad_messages(self):
        msgs = []
        if not self.squad_receiver:
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
                print(f"[{self.name}] Error reading squad message: {ex}")
            finally:
                try:
                    self.squad_receiver.nextPacket()
                except Exception:
                    pass
        return msgs

    def poll_supervisor_messages(self):
        msgs = []
        if not self.supervisor_receiver:
            return msgs
        while self.supervisor_receiver.getQueueLength() > 0:
            try:
                if hasattr(self.supervisor_receiver, "getString"):
                    raw = self.supervisor_receiver.getString()
                else:
                    raw_bytes = self.supervisor_receiver.getData()
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
                print(f"[{self.name}] Error reading supervisor msg: {ex}")
            finally:
                try:
                    self.supervisor_receiver.nextPacket()
                except Exception:
                    pass
        return msgs

    def save_rgb_frame(self, step_count):
        if not self.camera_rgb:
            return
        buf = self.camera_rgb.getImage()
        w, h = self.camera_rgb.getWidth(), self.camera_rgb.getHeight()
        img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        rgb = img[:, :, :3][:, :, ::-1]
        Image.fromarray(rgb).save(os.path.join(self.debug_dir, f"rgb_step{step_count}.png"))

    def save_depth_frame(self, step_count):
        if not self.camera_depth:
            return
        depth = np.array(self.camera_depth.getRangeImage(), dtype=np.float32).reshape(
            (self.camera_depth.getHeight(), self.camera_depth.getWidth()))
        dmin = self.camera_depth.getMinRange()
        dmax = self.camera_depth.getMaxRange()
        norm = ((np.clip(depth, dmin, dmax) - dmin) / (dmax - dmin) * 255).astype(np.uint8)
        Image.fromarray(norm).save(os.path.join(self.debug_dir, f"depth_step{step_count}.png"))

    def read_sensors(self):
        gps = self.gps.getValues() if self.gps else None
        imu = self.get_inertial_quaternion()
        print(f"[{self.name}] SENSORS gps={gps} imu={imu}")


# --- Main loop ---
def main():
    bot = Rosbot()
    try:
        bot.calibrate_yaw_sign()
    except Exception:
        pass

    step = 0
    forward_speed = 6.0  # nominal forward command (tune)
    print(f"[{bot.name}] START")

    while bot.robot.step(bot.timestep) != -1:
        step += 1
        bot.step_count = step

        # process squad messages
        msgs = bot.poll_squad_messages()
        for m in msgs:
            try:
                if isinstance(m, dict) and m.get("type") == "GRAPH_SUMMARY":
                    bot.merge_graph_summary(m)
                if isinstance(m, dict) and m.get("type") == "POSE":
                    bot.update_received_pose(m)
            except Exception as ex:
                print(f"[{bot.name}] msg err {ex}")

        # supervisor messages (debug)
                # supervisor messages: handle approvals/rejections properly
        sv = bot.poll_supervisor_messages()
        for sm in sv:
            try:
                bot.handle_supervisor_response(sm)
            except Exception as ex:
                print(f"[{bot.name}] SUP handle err {ex}")


        if step % 10 == 0:
            bot.read_sensors()

        if step == 1:
            bot.save_rgb_frame(0)
            bot.save_depth_frame(0)

        # --- HARD BARRIER position-based anti-stuck (2s) ---
        pos = bot.get_position()
        if pos is not None:
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

        # periodic RED victim detection
        detections = []
        if step % bot.victim_check_interval == 0:
            try:
                detections = bot.detect_red_victims(step)
                if detections:
                    print(f"[{bot.name}] Red victim detections: {len(detections)}")
                    for det in detections:
                        vid, px, py, bbox, world, color = det
                        print(f" - {vid} px=({px},{py}) bbox={bbox} color={color} world={world}")
            except Exception as ex:
                print(f"[{bot.name}] red victim detect err: {ex}")

        # ---------------------------
        # VICTIM-FIRST: preempt EVERYTHING
        # ---------------------------
        # If any red victim exists, start/continue mission immediately.
        # 1) If we have a current victim with world -> navigate there.
        # 2) Else pick nearest with world. If none has world, go GREEDY to the freshest pixel.
        victim_available = bool(bot.red_victims)
        if victim_available:
            # cancel any consensus
            if bot.consensus_active or bot.current_task == "navigate:consensus":
                print(f"[{bot.name}] VICTIM preempts consensus -> cancel current consensus attempt")
                bot.consensus_active = False
                bot.consensus_target = None
                bot.consensus_motion_mode = None
                bot.current_task = None
                bot.task_target = None

            # lock onto a specific victim if not already
            if bot.current_victim_id is None:
                nv = bot._nearest_unvisited_red_victim()
                if nv is not None:
                    bot.current_victim_id, bot.task_target = nv[0], nv[1]
                    bot.current_task = "navigate:victim"
                    print(f"[{bot.name}] VICTIM mission start -> {bot.current_victim_id} @ {bot.task_target}")
                else:
                    # no world coords yet — use greedy on the most recent pixel
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
            # refresh greedy pixel if a new detection came
            if detections:
                for det in detections:
                    vid, px, py, bbox, world, color = det
                    if vid == bot.current_victim_id and rv.get("pixel"):
                        bot.greedy_target_px = rv["pixel"]["x"]
                        bot.greedy_last_seen_ts = now_ts()

            # If we have a world target -> navigate gently and confirm within 1m
            if rv.get("world"):
                bot.task_target = rv["world"]
                bot.current_task = "navigate:victim"
                reached = bot.navigate_to(bot.task_target, linear_speed=3.2, angle_k=2.0,
                                          dist_tol=bot.victim_target_tol_m, debug=True, consensus_mode=None)

                if reached:
                    # Confirm & notify supervisor
                    bot._notify_supervisor_detected(bot.current_victim_id, bot.task_target)
                    bot._mark_victim_visited(bot.current_victim_id)
                    # Back off to clear space and continue search
                    bot.post_detection_backoff_steps = bot.post_detection_backoff_duration
                    # clear victim mission
                    bot.current_victim_id = None
                    bot.current_task = None
                    bot.task_target = None
                    # Produce a command this step (safety/backoff handled in compute_safe_controls)
                    lidar_ranges = bot.get_lidar_ranges()
                    v_safe, w_safe = bot.compute_safe_controls(0.0, 0.0, lidar_ranges)
                    bot._apply_wheel_cmds(v_safe, w_safe)
                    continue
            else:
                # GREEDY pixel-centering pursuit
                if bot.greedy_active and bot.greedy_target_px is not None and bot.camera_rgb is not None:
                    w_img = bot.camera_rgb.getWidth()
                    bot.greedy_pursue_pixel(bot.greedy_target_px, w_img, base_speed=2.2)
                    # timeout if not updated
                    if (now_ts() - bot.greedy_last_seen_ts) > bot.greedy_timeout_s:
                        print(f"[{bot.name}] GREEDY timeout -> releasing victim lock")
                        bot.greedy_active = False
                        bot.current_victim_id = None
                        bot.current_task = None
                        bot.task_target = None
                    continue
                else:
                    # no pixel to pursue; release lock
                    bot.current_victim_id = None
                    bot.current_task = None
                    bot.task_target = None

        # -------------------------
        # Consensus cycle management (only if not in victim mission)
        # -------------------------
                # Only run consensus logic when there is no active victim
        if bot.current_victim_id is None:
            phase = bot.update_consensus_cycle()

            if phase in ("forward_active", "reverse_active"):
                # reset inactive timer (we are active)
                bot.consensus_inactive_since = None

                bot.consensus_motion_mode = 'forward' if phase == "forward_active" else 'reverse'
                if not bot.consensus_active:
                    cand = bot.compute_consensus_target(freshness_s=bot.startup_window_s + 1.0, force=True)
                    if cand is not None:
                        bot.consensus_target = cand
                        bot.current_task = "navigate:consensus"
                        bot.task_target = cand
                        bot.consensus_active = True
                        bot.consensus_started_ts = now_ts()
                        bot._append_map_history("consensus_start", {"target": cand, "participants": len(bot.received_poses), "phase": phase})
                        print(f"[{bot.name}] CONSENSUS cycle START ({phase}) -> target={cand}")
                # else: already active — keep running

            else:  # phase == "inactive"
                # Start or update the inactive timer when we first see an inactive phase
                if bot.consensus_active:
                    if bot.consensus_inactive_since is None:
                        bot.consensus_inactive_since = now_ts()
                        # keep consensus running for the grace period
                    else:
                        # if inactive persisted beyond grace period, cancel consensus
                        if now_ts() - bot.consensus_inactive_since > bot.consensus_inactive_grace_s:
                            print(f"[{bot.name}] CONSENSUS cycle CANCEL (inactive persisted > {bot.consensus_inactive_grace_s}s)")
                            bot.consensus_active = False
                            bot.consensus_target = None
                            bot.consensus_motion_mode = None
                            bot.consensus_inactive_since = None
                            if bot.current_task == "navigate:consensus":
                                bot.current_task = None
                                bot.task_target = None
                # if consensus not active, nothing to cancel — keep waiting


        # Behavior manager: navigation to task targets (including consensus)
        if bot.current_victim_id is None and bot.current_task and bot.current_task.startswith("navigate:") and bot.task_target:
            tol = bot.consensus_tolerance if bot.current_task == "navigate:consensus" else 0.6
            consensus_mode = bot.consensus_motion_mode if bot.current_task == "navigate:consensus" else None
            reached = bot.navigate_to(bot.task_target, linear_speed=forward_speed, angle_k=2.0,
                                      dist_tol=tol, debug=True, consensus_mode=consensus_mode)
            if reached:
                nid = bot.current_task.split(":", 1)[1]
                bot._append_map_history("arrived_node", {"node": nid})
                if nid == "consensus" or bot.current_task == "navigate:consensus":
                    try:
                        payload = {"type": "GRAPH_SUMMARY", "robot": bot.name,
                                   "nodes": [{"id": n, "centroid": c} for n, c in bot.nodes.items()]}
                        bot.send_squad_message(payload)
                    except Exception:
                        pass
                    bot._append_map_history("consensus_arrived", {"target": bot.task_target})
                    bot.consensus_active = False
                    bot.consensus_target = None
                    bot.consensus_motion_mode = None
                bot.current_task = None
                bot.task_target = None
            else:
                if bot.consensus_active and bot.consensus_started_ts and now_ts() - bot.consensus_started_ts > bot.consensus_timeout_s:
                    print(f"[{bot.name}] CONSENSUS timeout, aborting this attempt")
                    bot.consensus_active = False
                    bot.consensus_target = None
                    bot.current_task = None
                    bot.task_target = None
            # proceed to next loop
            continue

        # avoid exiting area if such routine exists
        def_exited = False
        try:
            def_exited = bot.avoid_exiting_and_face_nearest_node(bot.get_lidar_ranges()) if hasattr(bot, "avoid_exiting_and_face_nearest_node") else False
        except Exception:
            def_exited = False
        if def_exited:
            continue

        # assign to nearest node eventually (existing logic)
        if bot.current_victim_id is None and bot.current_task is None and bot.nodes:
            pos2 = bot.get_position()
            if pos2:
                nearest = None
                nearest_d = float("inf")
                for nid, c in bot.nodes.items():
                    dx = float(c[0]) - pos2[0]
                    dy = float(c[1]) - pos2[1]
                    d = math.hypot(dx, dy)
                    if d < nearest_d:
                        nearest_d = d
                        nearest = (nid, c)
                if nearest:
                    nid, centroid = nearest
                    disc = bot.node_discovery_times.get(nid, 0)
                    if now_ts() - disc >= bot.assignment_delay and nearest_d > 0.8:
                        bot.current_task = f"navigate:{nid}"
                        bot.task_target = centroid
                        bot._append_map_history("assign_navigate", {"node": nid})
                        continue

        # default exploration under safety layer (forward)
        lidar_ranges = bot.get_lidar_ranges()
        v_safe, omega_safe = bot.compute_safe_controls(forward_speed, 0.0, lidar_ranges)
        bot._apply_wheel_cmds(v_safe, omega_safe)

    print(f"[{bot.name}] EXIT")


if __name__ == "__main__":
    main()
