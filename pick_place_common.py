"""Shared pieces of the SO-101 pick-and-place walkthrough.

Everything here is plain MuJoCo + NumPy and runs on the host (CPU):

* ``RobotSpec`` / ``SO101``      - the robot profile: scene layout and MJWarp capacities
* ``resolve_pick_place_scene()`` - fetch the Menagerie arm and write ``scene_pick_place.xml``
* ``PickPlaceController``        - waypoints + damped-least-squares inverse kinematics
* ``stack_check()``              - the two numbers every gate is compared on

The stage scripts (02 to 05) import from this module, so the only thing that
changes between them is *who advances the physics*: MuJoCo on the CPU or
MJWarp on the GPU.
"""

from __future__ import annotations

import dataclasses
import pathlib
import urllib.request
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
GENERATED = ROOT / ".generated"

# Menagerie assets change over time, so pin a known-good commit.
MENAGERIE_COMMIT = "4d038b3feae26ec82b46a4d586379114012a8ac7"
MENAGERIE_RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie"

# Stack success thresholds for 44 mm cubes (see the article, Gate 1).
XY_TOL = 0.015
DZ_MIN, DZ_MAX = 0.035, 0.055


@dataclasses.dataclass(frozen=True)
class RobotSpec:
  """Robot profile: where things are in the scene and how big MJWarp buffers must be."""

  name: str
  menagerie_dir: str
  robot_xml: str

  arm_joints: tuple[str, ...]
  gripper_actuator: str
  grasp_site: str

  table_pos: tuple[float, float, float]
  table_size: tuple[float, float, float]
  red_cube_pos: tuple[float, float, float]
  blue_cube_pos: tuple[float, float, float]
  cube_half: float

  gripper_open: float  # actuator target [rad] with the jaws wide enough to straddle a cube
  gripper_closed: float  # actuator target [rad] that squeezes a cube
  approach_pitch: float  # finger angle below horizontal while grasping [rad]

  # MJWarp allocates contact / constraint buffers up front (Gate 3).
  nconmax: int = 128
  njmax: int = 300


SO101 = RobotSpec(
  name="so101",
  menagerie_dir="robotstudio_so101",
  robot_xml="so101.xml",
  arm_joints=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
  gripper_actuator="gripper",
  grasp_site="gripperframe",
  table_pos=(0.35, -0.04, 0.012),
  table_size=(0.16, 0.26, 0.012),
  red_cube_pos=(0.33, -0.13, 0.046),
  blue_cube_pos=(0.33, 0.06, 0.046),
  cube_half=0.022,
  gripper_open=1.2,
  gripper_closed=0.15,
  approach_pitch=np.deg2rad(55.0),
)

ROBOTS = {SO101.name: SO101}

# The scene from the article. Nothing here is MJWarp-specific: an arm, a table
# and two cubes written as ordinary MJCF. Box sizes are half-extents.
_SCENE_TEMPLATE = """\
<mujoco model="{name}_pick_place">
  <include file="{robot_xml}"/>

  <worldbody>
    <light pos="0.3 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05"/>

    <geom name="table" type="box" pos="{table_pos}"
          size="{table_size}" rgba="0.32 0.32 0.32 1"
          friction="1 0.005 0.0005" condim="3"/>

    <body name="red_cube" pos="{red_cube_pos}">
      <freejoint name="red_cube_joint"/>
      <geom type="box" size="{cube_size}" mass="0.08"
            rgba="0.85 0.05 0.04 1" friction="1.2 0.005 0.0005" condim="3"/>
    </body>

    <body name="blue_cube" pos="{blue_cube_pos}">
      <freejoint name="blue_cube_joint"/>
      <geom type="box" size="{cube_size}" mass="0.08"
            rgba="0.05 0.20 0.90 1" friction="1.2 0.005 0.0005" condim="3"/>
    </body>
  </worldbody>
</mujoco>
"""


def _fmt(values) -> str:
  return " ".join(f"{v:g}" for v in values)


def _download(url: str, dst: pathlib.Path) -> None:
  dst.parent.mkdir(parents=True, exist_ok=True)
  tmp = dst.with_name(dst.name + ".part")
  with urllib.request.urlopen(url, timeout=60) as response:
    tmp.write_bytes(response.read())
  tmp.replace(dst)


def _fetch_robot(spec: RobotSpec, out_dir: pathlib.Path) -> None:
  """Copy the Menagerie arm (MJCF + meshes) into ``out_dir``, skipping files already there."""
  base = f"{MENAGERIE_RAW}/{MENAGERIE_COMMIT}/{spec.menagerie_dir}"
  robot_xml = out_dir / spec.robot_xml
  if not robot_xml.exists():
    print(f"fetching {spec.menagerie_dir} from MuJoCo Menagerie @ {MENAGERIE_COMMIT[:8]} ...")
    _download(f"{base}/{spec.robot_xml}", robot_xml)
    _download(f"{base}/LICENSE", out_dir / "LICENSE")

  root = ET.parse(robot_xml).getroot()
  meshdir = root.find("compiler").get("meshdir", "")
  for mesh in root.iter("mesh"):
    file = mesh.get("file")
    if file and not (out_dir / meshdir / file).exists():
      _download(f"{base}/{meshdir}/{file}", out_dir / meshdir / file)


def resolve_pick_place_scene(spec: RobotSpec = SO101) -> pathlib.Path:
  """Generate ``.generated/<robot>/scene_pick_place.xml`` and return its path."""
  out_dir = GENERATED / spec.name
  _fetch_robot(spec, out_dir)

  scene = out_dir / "scene_pick_place.xml"
  scene.write_text(
    _SCENE_TEMPLATE.format(
      name=spec.name,
      robot_xml=spec.robot_xml,
      table_pos=_fmt(spec.table_pos),
      table_size=_fmt(spec.table_size),
      red_cube_pos=_fmt(spec.red_cube_pos),
      blue_cube_pos=_fmt(spec.blue_cube_pos),
      cube_size=_fmt((spec.cube_half,) * 3),
    )
  )
  return scene


def stack_check(mjm: mujoco.MjModel, mjd: mujoco.MjData) -> tuple[float, float, bool]:
  """Return (xy_err, dz, success) for the red cube stacked on the blue cube.

  Reads ``mjd.xpos``, so call ``mujoco.mj_forward`` first if ``qpos`` was
  written from somewhere else (e.g. copied back from MJWarp).
  """
  delta = mjd.body("red_cube").xpos - mjd.body("blue_cube").xpos
  xy_err = float(np.linalg.norm(delta[:2]))
  dz = float(delta[2])
  return xy_err, dz, bool(xy_err <= XY_TOL and DZ_MIN <= dz <= DZ_MAX)


def _smoothstep(s: float) -> float:
  s = min(max(s, 0.0), 1.0)
  return s * s * (3.0 - 2.0 * s)


class PickPlaceController:
  """Scripted pick-and-place: Cartesian waypoints tracked with damped-least-squares IK.

  ``step()`` is called once per control frame and returns position targets
  for all actuators. It never steps physics itself, which is why the same
  controller drives the MuJoCo CPU rollout (Gate 1) and the MJWarp rollout
  (Gate 2) unchanged.

  The IK runs on a private kinematic copy of the arm (``self.plan``): each
  frame the planned joint angles take one damped-least-squares step towards
  the current waypoint and are sent to the position servos. The measured
  state in ``mjd`` is only read for what cannot be planned: where the cubes
  are, and how the red cube sits in the gripper once it is grasped.

  The SO-101 has 5 arm joints, so a gripper pose has 5 controllable degrees
  of freedom: position (3), finger pitch (1) and wrist roll (1). The IK asks
  for a position plus an orientation that is consistent with that: fingers
  pitched down in the vertical plane through the shoulder, jaws closing
  horizontally. In the gripper body frame the fingers point along -z and the
  jaws close along x.
  """

  # (phase name, duration [s]); anything after the last phase holds position.
  PHASES = (
    ("pregrasp", 1.6),
    ("descend", 1.0),
    ("close", 0.7),
    ("lift", 1.0),
    ("transport", 1.6),
    ("lower", 1.2),
    ("release", 0.7),
    ("retreat", 1.0),
  )

  HOVER = 0.07  # clearance above a cube before descending / after lifting [m]
  JAW_CLEARANCE = 0.014  # sideways offset so the fixed jaw clears the cube corner [m]
  PLACE_GAP = 0.003  # release the red cube this far above the blue one [m]
  ROT_WEIGHT = 0.1  # orientation error weight relative to position [m/rad]
  DAMPING = 0.02  # DLS damping: trades tracking accuracy for smooth joint motion
  GAIN = 0.5  # fraction of the IK step applied per control frame
  MAX_JOINT_SPEED = 2.0  # [rad/s]

  def __init__(self, spec: RobotSpec = SO101, debug: bool = False):
    self.spec = spec
    self.debug = debug
    self.time = 0.0
    self.phase = None
    self._ready = False

  def _setup(self, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> None:
    spec = self.spec
    self.site_id = mjm.site(spec.grasp_site).id
    self.hand_id = mjm.site_bodyid[self.site_id]
    joint_ids = [mjm.joint(name).id for name in spec.arm_joints]
    self.qadr = mjm.jnt_qposadr[joint_ids]
    self.dadr = mjm.jnt_dofadr[joint_ids]
    self.qrange = mjm.jnt_range[joint_ids]
    self.arm_act = np.array([mjm.actuator(name).id for name in spec.arm_joints])
    self.grip_act = mjm.actuator(spec.gripper_actuator).id
    self.pan_axis_xy = mjd.xanchor[joint_ids[0], :2].copy()

    self.plan = mujoco.MjData(mjm)  # kinematics-only scratch copy for the IK
    self.q_cmd = mjd.qpos[self.qadr].copy()
    self.grip_cmd = float(mjd.ctrl[self.grip_act])
    self.site_offset = mjm.site_pos[self.site_id].copy()  # grasp site in the hand frame

    # Waypoints are laid out once from where the cubes are at the start.
    self.red0 = mjd.body("red_cube").xpos.copy()
    self.blue0 = mjd.body("blue_cube").xpos.copy()
    self._ready = True

  def _hand_frame(self, target: np.ndarray) -> np.ndarray:
    """Desired gripper orientation (columns x, y, z) for a control point at ``target``."""
    heading = target[:2] - self.pan_axis_xy
    heading = np.append(heading / np.linalg.norm(heading), 0.0)
    up = np.array([0.0, 0.0, 1.0])
    c, s = np.cos(self.spec.approach_pitch), np.sin(self.spec.approach_pitch)
    fingers = c * heading - s * up
    y_axis = s * heading + c * up  # camera side of the gripper stays on top
    return np.column_stack((np.cross(heading, up), y_axis, -fingers))

  def _plan_pose(self, mjm: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics of the planned joint angles: control point and hand rotation."""
    self.plan.qpos[self.qadr] = self.q_cmd
    mujoco.mj_kinematics(mjm, self.plan)
    mujoco.mj_comPos(mjm, self.plan)  # needed by mj_jac
    hand_mat = self.plan.xmat[self.hand_id].reshape(3, 3)
    return self.plan.xpos[self.hand_id] + hand_mat @ self.offset, hand_mat

  def _begin_phase(self, name: str, mjm: mujoco.MjModel, mjd: mujoco.MjData) -> None:
    spec = self.spec
    self.phase = name

    # The control point is the point of the hand that the waypoints steer.
    # While the red cube is held it is the cube centre, as measured in the
    # hand frame; otherwise it is the grasp site between the jaws.
    if name in ("lift", "transport", "lower"):
      hand_mat = mjd.xmat[self.hand_id].reshape(3, 3)
      self.offset = hand_mat.T @ (mjd.body("red_cube").xpos - mjd.xpos[self.hand_id])
    else:
      self.offset = self.site_offset
    start, _ = self._plan_pose(mjm)

    jaws = self._hand_frame(self.red0)[:, 0]  # jaw closing direction at the grasp
    grasp = self.red0 - self.JAW_CLEARANCE * jaws
    stacked = self.blue0 + np.array([0.0, 0.0, 2.0 * spec.cube_half + self.PLACE_GAP])
    hover = np.array([0.0, 0.0, self.HOVER])

    grip = self.grip_cmd
    if name == "pregrasp":
      goal, grip = grasp + hover, spec.gripper_open
    elif name == "descend":
      goal = grasp
    elif name == "close":
      goal, grip = start, spec.gripper_closed
    elif name == "lift":
      goal = start + hover
    elif name == "transport":
      goal = stacked + hover
    elif name == "lower":
      goal = stacked
    elif name == "release":
      # Open, and slide the fixed jaw off the cube so it is not dragged along.
      goal = start - 0.5 * self.JAW_CLEARANCE * self._hand_frame(start)[:, 0]
      grip = spec.gripper_open
    elif name == "retreat":
      goal = start + hover
    else:  # hold
      goal = start

    self.start, self.goal = start, goal
    self.grip_from, self.grip_to = self.grip_cmd, grip

  def _schedule(self) -> tuple[str, float]:
    """Return (phase name, progress in [0, 1]) for the current time."""
    t = self.time
    for name, duration in self.PHASES:
      if t < duration:
        return name, t / duration
      t -= duration
    return "hold", 1.0

  def step(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, frame_dt: float) -> np.ndarray:
    if not self._ready:
      self._setup(mjm, mjd)

    name, progress = self._schedule()
    if name != self.phase:
      self._begin_phase(name, mjm, mjd)
    blend = _smoothstep(progress)
    target = self.start + blend * (self.goal - self.start)
    self.grip_cmd = self.grip_from + blend * (self.grip_to - self.grip_from)

    # Task-space error: position of the control point, orientation of the hand.
    point, hand_mat = self._plan_pose(mjm)
    pos_err = target - point
    hand_quat, want_quat, hand_inv, diff = np.empty(4), np.empty(4), np.empty(4), np.empty(4)
    mujoco.mju_mat2Quat(hand_quat, hand_mat.ravel())
    mujoco.mju_mat2Quat(want_quat, self._hand_frame(target).ravel())
    mujoco.mju_negQuat(hand_inv, hand_quat)
    mujoco.mju_mulQuat(diff, want_quat, hand_inv)
    rot_err = np.empty(3)
    mujoco.mju_quat2Vel(rot_err, diff, 1.0)

    # Jacobian of the control point, restricted to the arm joints.
    jacp, jacr = np.empty((3, mjm.nv)), np.empty((3, mjm.nv))
    mujoco.mj_jac(mjm, self.plan, jacp, jacr, point, self.hand_id)
    jac = np.vstack((jacp[:, self.dadr], self.ROT_WEIGHT * jacr[:, self.dadr]))
    err = np.concatenate((pos_err, self.ROT_WEIGHT * rot_err))

    # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 err
    dq = jac.T @ np.linalg.solve(jac @ jac.T + self.DAMPING**2 * np.eye(6), err)
    max_dq = self.MAX_JOINT_SPEED * frame_dt
    self.q_cmd = np.clip(self.q_cmd + np.clip(self.GAIN * dq, -max_dq, max_dq), *self.qrange.T)

    ctrl = np.zeros(mjm.nu)
    ctrl[self.arm_act] = self.q_cmd
    ctrl[self.grip_act] = self.grip_cmd

    if self.debug and name != "hold" and round(self.time / frame_dt) % 10 == 0:
      print(
        f"  t={self.time:5.2f}s {name:<9} |pos_err|={1e3 * np.linalg.norm(pos_err):6.1f} mm "
        f"|rot_err|={np.rad2deg(np.linalg.norm(rot_err)):5.1f} deg grip={self.grip_cmd:4.2f}"
      )

    self.time += frame_dt
    return ctrl
