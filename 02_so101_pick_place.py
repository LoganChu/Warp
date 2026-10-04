"""Stage 2 (Gate 1) - establish a MuJoCo CPU baseline.

An SO-101 arm picks up the red cube and stacks it on the blue one. Physics is
ordinary single-world MuJoCo on the CPU; nothing here knows about Warp. The
run ends with the two numbers every later gate is compared against:

    stack check: xy_err=... dz=...

Usage:
    uv run python 02_so101_pick_place.py                             # interactive viewer
    uv run python 02_so101_pick_place.py --headless-steps 600 --debug

The viewer replays the task until you close it (Space pauses). The headless
run also records the rollout to results/ for report.py.
"""

import argparse
import time

import mujoco

from pick_place_common import ROBOTS, PickPlaceController, resolve_pick_place_scene, stack_check
from viewing import Recorder, run_viewer, show_text

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", default="so101", choices=sorted(ROBOTS))
parser.add_argument("--headless-steps", type=int, default=0, help="run this many control frames without a viewer")
parser.add_argument("--debug", action="store_true", help="print controller progress")
args = parser.parse_args()

spec = ROBOTS[args.robot]
mjm = mujoco.MjModel.from_xml_path(str(resolve_pick_place_scene(spec)))
mjd = mujoco.MjData(mjm)

fps = 50  # controller rate
sim_substeps = 10  # physics steps per control frame
frame_dt = 1.0 / fps
mjm.opt.timestep = frame_dt / sim_substeps  # 50 Hz x 10 substeps -> 0.002 s

controller = None


def reset() -> None:
  global controller
  mujoco.mj_resetData(mjm, mjd)
  mujoco.mj_forward(mjm, mjd)  # fill mjd.xpos etc. before the controller reads them
  controller = PickPlaceController(spec=spec, debug=args.debug)  # waypoints + damped-least-squares IK


def simulate_frame() -> None:
  # Compute controls once per frame, step physics sim_substeps times.
  ctrl = controller.step(mjm, mjd, frame_dt)
  for _ in range(sim_substeps):
    mjd.ctrl[: mjm.nu] = ctrl
    mujoco.mj_step(mjm, mjd)


def draw(viewer) -> None:
  xy_err, dz, ok = stack_check(mjm, mjd)
  show_text(
    viewer,
    {
      "backend": "MuJoCo (CPU), 1 world",
      "time": f"{controller.time:5.2f} s",
      "phase": controller.phase or "-",
      "contacts": f"{mjd.ncon}",
      "constraints": f"{mjd.nefc}",
      "xy_err": f"{1e3 * xy_err:5.1f} mm  (<= 15)",
      "dz": f"{1e3 * dz:5.1f} mm  (35..55)",
      "stacked": "YES" if ok else "no",
    },
  )


reset()
if args.headless_steps:
  recorder = Recorder()
  t0 = time.perf_counter()
  for _ in range(args.headless_steps):
    simulate_frame()
    recorder.frame(mjd, controller.time, controller.phase, mjd.ncon, mjd.nefc)
  elapsed = time.perf_counter() - t0
  steps = args.headless_steps * sim_substeps
  print(f"{steps} physics steps in {elapsed:.2f} s ({steps / elapsed:,.0f} steps/second, 1 world, CPU)")
else:
  run_viewer(mjm, mjd, simulate_frame, reset, frame_dt, draw=draw)

xy_err, dz, ok = stack_check(mjm, mjd)
print(f"stack check: xy_err={xy_err:.4f} dz={dz:.4f} -> {'SUCCESS' if ok else 'FAIL'}")

if args.headless_steps:
  path = recorder.save("stage2_cpu", backend="MuJoCo (CPU)", xy_err=xy_err, dz=dz, ok=ok, steps_per_second=steps / elapsed)
  print(f"recorded to {path.relative_to(path.parents[1])} (see report.py)")
