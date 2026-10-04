"""Everything about *looking* at the simulation rather than computing it.

* ``run_viewer()``   - interactive window that replays the task until closed
* ``show_text()``    - live readout in the corner of the window
* ``LiveFigure``     - live line plot inside the window
* ``draw_worlds()``  - draw many worlds side by side in one window
* ``Recorder`` / ``save_result()`` - write rollouts to ``results/`` for ``report.py``

None of this is part of the Warp / MJWarp lesson; the stage scripts call into
it so that their own code stays about the physics.
"""

from __future__ import annotations

import json
import math
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np

from pick_place_common import ROOT

RESULTS = ROOT / "results"

KEY_SPACE = 32
PAUSE_SECONDS = 2.0  # hold the final state this long before the next episode

# Marker colours for per-world status (always shown together with a text label).
PENDING = (0.55, 0.55, 0.53, 1.0)
GOOD = (0.05, 0.64, 0.05, 1.0)
BAD = (0.82, 0.23, 0.23, 1.0)

# Line colours for LiveFigure (the first two series colours of report.py, dark-surface steps).
SERIES_1 = (0.22, 0.53, 0.90)
SERIES_2 = (0.85, 0.35, 0.15)


def run_viewer(mjm, mjd, simulate_frame, reset, frame_dt, episode_frames=600, draw=None, camera=None):
  """Replay the task in an interactive viewer until the window is closed.

  ``simulate_frame()`` advances one control frame, ``reset()`` restarts the
  episode, ``draw(viewer)`` refreshes overlays. Space pauses and resumes.
  The mouse orbits (left drag), pans (right drag) and zooms (scroll).
  """
  paused = False

  def on_key(key: int) -> None:
    nonlocal paused
    if key == KEY_SPACE:
      paused = not paused

  total = episode_frames + round(PAUSE_SECONDS / frame_dt)
  with mujoco.viewer.launch_passive(mjm, mjd, key_callback=on_key, show_left_ui=False, show_right_ui=False) as viewer:
    lookat, distance, azimuth, elevation = camera or ((0.25, -0.03, 0.1), 1.0, 140.0, -25.0)
    viewer.cam.lookat[:] = lookat
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = distance, azimuth, elevation

    frame = 0
    while viewer.is_running():
      start = time.perf_counter()
      if not paused:
        if frame < episode_frames:
          simulate_frame()
        frame += 1
        if frame == total:
          reset()
          frame = 0
      if draw is not None:
        draw(viewer)
      viewer.sync()
      time.sleep(max(0.0, frame_dt - (time.perf_counter() - start)))


def show_text(viewer, rows: dict[str, str]) -> None:
  """Show a two-column readout in the top-left corner of the viewer."""
  viewer.set_texts(
    (
      mujoco.mjtFontScale.mjFONTSCALE_150,
      mujoco.mjtGridPos.mjGRID_TOPLEFT,
      "\n".join(rows),
      "\n".join(rows.values()),
    )
  )


class LiveFigure:
  """A line plot drawn inside the viewer window, updated while the task runs."""

  MAX_POINTS = 1000  # per line, a limit of MuJoCo's figure buffers

  def __init__(self, title: str, lines: dict[str, tuple[float, float, float]], xmax: float, ymax: float,
               xlabel: str = "time [s]"):
    self.fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(self.fig)
    self.fig.title = title
    self.fig.xlabel = xlabel
    self.fig.flg_legend = 1
    self.fig.flg_extend = 1  # grow the axes if the data leaves the initial range
    self.fig.gridsize[:] = (4, 4)
    self.fig.figurergba[:] = (0.0, 0.0, 0.0, 0.6)
    # Leave headroom above the data: the legend is drawn in the top-right corner.
    self.fig.range[:] = ((0.0, xmax), (0.0, ymax))
    for i, (name, rgb) in enumerate(lines.items()):
      self.fig.linename[i] = name
      self.fig.linergb[i] = rgb
    self.nline = len(lines)

  def clear(self) -> None:
    self.fig.linepnt[: self.nline] = 0

  def add(self, x: float, values) -> None:
    for i, value in enumerate(values):
      n = self.fig.linepnt[i]
      if n < self.MAX_POINTS:
        self.fig.linedata[i, 2 * n : 2 * n + 2] = (x, value)
        self.fig.linepnt[i] = n + 1

  def show(self, viewer) -> None:
    """Draw the figure in the bottom-right third of the window."""
    viewport = viewer.viewport
    if viewport is None:
      return
    width, height = viewport.width // 3, viewport.height // 3
    viewer.set_figures((mujoco.MjrRect(viewport.width - width, 0, width, height), self.fig))


def grid_offsets(nworld: int, spacing=(0.8, 0.7)) -> np.ndarray:
  """World-frame offsets that lay ``nworld`` worlds out on a square-ish grid."""
  cols = math.ceil(math.sqrt(nworld))
  index = np.arange(nworld)
  return np.column_stack((spacing[0] * (index // cols), spacing[1] * (index % cols), np.zeros(nworld)))


def grid_camera(offsets: np.ndarray):
  """A camera (lookat, distance, azimuth, elevation) that frames the whole grid."""
  centre = offsets.mean(axis=0) + np.array([0.25, -0.03, 0.1])
  extent = float(np.linalg.norm(offsets.max(axis=0) - offsets.min(axis=0)))
  return centre, 1.2 + 1.1 * extent, 140.0, -35.0


def draw_worlds(viewer, mjm, datas, offsets, markers=None) -> None:
  """Draw a batch of worlds in one viewer, each shifted to its grid cell.

  The viewer itself draws ``datas[0]`` (the ``MjData`` it was launched with).
  The other worlds are added to the viewer's user scene: MuJoCo generates the
  geoms of each host ``MjData`` and they are moved by that world's offset.
  ``markers`` is an optional list of (label, rgba) shown above each world.
  """
  scene = viewer.user_scn
  with viewer.lock():
    scene.ngeom = 0
    for i, (data, offset) in enumerate(zip(datas, offsets)):
      if i > 0:
        first = scene.ngeom
        mujoco.mjv_addGeoms(mjm, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_ALL, scene)
        for geom in scene.geoms[first : scene.ngeom]:
          if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.rgba[3] = 0.0  # the floor is infinite: world 0 already draws it
          else:
            geom.pos[:] += offset
      if markers is not None:
        label, rgba = markers[i]
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
          geom,
          mujoco.mjtGeom.mjGEOM_SPHERE,
          np.array([0.015, 0.0, 0.0]),
          offset + np.array([0.33, -0.04, 0.32]),
          np.eye(3).ravel(),
          np.array(rgba, dtype=np.float32),
        )
        geom.label = label
        scene.ngeom += 1


def save_result(name: str, payload: dict) -> pathlib.Path:
  """Write ``results/<name>.json`` (read by report.py)."""
  RESULTS.mkdir(exist_ok=True)
  path = RESULTS / f"{name}.json"
  path.write_text(json.dumps(payload))
  return path


def load_result(name: str) -> dict | None:
  path = RESULTS / f"{name}.json"
  return json.loads(path.read_text()) if path.exists() else None


class Recorder:
  """Per-frame log of one rollout: where the cubes are and how busy the solver is."""

  def __init__(self):
    self.series = {key: [] for key in ("t", "phase", "red", "blue", "contacts", "constraints")}

  def frame(self, mjd: mujoco.MjData, t: float, phase: str, contacts: int, constraints: int) -> None:
    series = self.series
    series["t"].append(round(t, 4))
    series["phase"].append(phase)
    series["red"].append([round(float(v), 5) for v in mjd.body("red_cube").xpos])
    series["blue"].append([round(float(v), 5) for v in mjd.body("blue_cube").xpos])
    series["contacts"].append(int(contacts))
    series["constraints"].append(int(constraints))

  def save(self, name: str, **summary) -> pathlib.Path:
    return save_result(name, {"summary": summary, "series": self.series})
