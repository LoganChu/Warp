"""Stage 3 (Gates 2 + 3) - one-world MJWarp parity, and contact / constraint capacity.

The same task, scene and controller as Gate 1, but the physics step now runs
in MJWarp on the GPU. The host stays in the loop on purpose: the controller,
the viewer and the stack check keep reading an ordinary ``MjData``, which is
refreshed from the device after every step. That makes this a
task-validation path, not a benchmark.

Usage:
    uv run python 03_so101_mjwarp.py                       # interactive viewer
    uv run python 03_so101_mjwarp.py --headless-steps 600

The viewer replays the task until you close it (Space pauses) and plots the
contact / constraint buffers filling up live. The headless run also records
the rollout to results/ for report.py.
"""

import argparse
import time

import mujoco
import warp as wp

from pick_place_common import ROBOTS, PickPlaceController, resolve_pick_place_scene, stack_check
from viewing import SERIES_1, SERIES_2, LiveFigure, Recorder, run_viewer, show_text

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", default="so101", choices=sorted(ROBOTS))
parser.add_argument("--headless-steps", type=int, default=0, help="run this many control frames without a viewer")
parser.add_argument("--debug", action="store_true", help="print controller progress")
args = parser.parse_args()

wp.config.log_level = wp.LOG_WARNING  # hide the per-module "load on device" lines (Stage 1 shows them)
wp.init()
import mujoco_warp as mjw  # noqa: E402

print("first run compiles MJWarp's kernels and can take a minute; later runs load them from cache")

spec = ROBOTS[args.robot]
mjm = mujoco.MjModel.from_xml_path(str(resolve_pick_place_scene(spec)))
mjd = mujoco.MjData(mjm)

fps = 50  # controller rate
sim_substeps = 10  # physics steps per control frame
frame_dt = 1.0 / fps
# Set the timestep BEFORE put_model: the device model is a snapshot of mjm.
mjm.opt.timestep = frame_dt / sim_substeps

# --- Gate 2: upload the model and allocate batched state.
device = wp.get_device()

m = mjw.put_model(mjm)  # raises if the model uses features MJWarp does not support
# MJWarp prints a warning whenever a bit is set in Data.overflow. Keep the
# buffer-overflow warnings, but mute "solver stopped at its iteration cap":
# that one is reported once at the end instead of on every step it happens.
solver_notes = mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS
m.opt.warn_overflow = int(mjw.OverflowType.ALL & ~solver_notes)
d = mjw.make_data(mjm, nworld=1, nconmax=spec.nconmax, njmax=spec.njmax)

controller = None

# --- Gate 3: track how much of the contact / constraint budget the task really uses.
peak_contacts = 0
peak_constraints = 0
overflow_bits = 0
frame_contacts = 0  # peaks within the latest control frame, for the live plot
frame_constraints = 0

usage_plot = LiveFigure(
  "buffer use (MJWarp)",
  {f"constraints (njmax {spec.njmax})": SERIES_2, f"contacts (nconmax {spec.nconmax})": SERIES_1},
  xmax=12.0,
  ymax=150.0,
)


def reset() -> None:
  """Start an episode: initialise the host state, then seed the device state from it."""
  global controller
  mujoco.mj_resetData(mjm, mjd)
  mujoco.mj_forward(mjm, mjd)
  controller = PickPlaceController(spec=spec, debug=args.debug)
  usage_plot.clear()

  mjw.reset_data(m, d)
  # Every device array has a leading world dimension: (nq,) on the host is (1, nq) here.
  wp.copy(d.qpos, wp.array(mjd.qpos[None, :], dtype=wp.float32, device=device))
  wp.copy(d.qvel, wp.array(mjd.qvel[None, :], dtype=wp.float32, device=device))
  wp.copy(d.ctrl, wp.array(mjd.ctrl[None, :], dtype=wp.float32, device=device))
  mjw.forward(m, d)


def simulate_frame() -> None:
  global peak_contacts, peak_constraints, overflow_bits, frame_contacts, frame_constraints
  frame_contacts = frame_constraints = 0
  ctrl = controller.step(mjm, mjd, frame_dt)
  for _ in range(sim_substeps):
    mjd.ctrl[: mjm.nu] = ctrl
    wp.copy(d.ctrl, wp.array(mjd.ctrl[None, :], dtype=wp.float32, device=device))
    mjw.step(m, d)
    # .numpy() synchronizes and copies device -> host. Fine for validation,
    # far too slow for a benchmark (Gate 4 removes it).
    mjd.qpos[:] = d.qpos.numpy()[0]
    mjd.qvel[:] = d.qvel.numpy()[0]

    frame_contacts = max(frame_contacts, int(d.nacon.numpy()[0]))
    frame_constraints = max(frame_constraints, int(d.nefc.numpy()[0]))
    overflow_bits |= int(d.overflow.numpy()[0])
  peak_contacts = max(peak_contacts, frame_contacts)
  peak_constraints = max(peak_constraints, frame_constraints)
  usage_plot.add(controller.time, (frame_constraints, frame_contacts))
  # qpos/qvel changed behind MuJoCo's back: refresh xpos, site_xpos, ... before
  # the controller, the viewer or the stack check read them.
  mujoco.mj_forward(mjm, mjd)


def draw(viewer) -> None:
  xy_err, dz, ok = stack_check(mjm, mjd)
  show_text(
    viewer,
    {
      "backend": f"MJWarp ({device}), 1 world",
      "time": f"{controller.time:5.2f} s",
      "phase": controller.phase or "-",
      "contacts": f"{frame_contacts} now, peak {peak_contacts} / {spec.nconmax}",
      "constraints": f"{frame_constraints} now, peak {peak_constraints} / {spec.njmax}",
      "xy_err": f"{1e3 * xy_err:5.1f} mm  (<= 15)",
      "dz": f"{1e3 * dz:5.1f} mm  (35..55)",
      "stacked": "YES" if ok else "no",
    },
  )
  usage_plot.show(viewer)


reset()
if args.headless_steps:
  recorder = Recorder()
  t0 = time.perf_counter()
  for _ in range(args.headless_steps):
    simulate_frame()
    recorder.frame(mjd, controller.time, controller.phase, frame_contacts, frame_constraints)
  elapsed = time.perf_counter() - t0
  steps = args.headless_steps * sim_substeps
  print(f"{steps} physics steps in {elapsed:.2f} s ({steps / elapsed:,.0f} steps/second, 1 world, {device})")
  print("  (slower than the CPU on purpose: one world, with a host round-trip every step)")
else:
  run_viewer(mjm, mjd, simulate_frame, reset, frame_dt, draw=draw)

xy_err, dz, ok = stack_check(mjm, mjd)
print(f"stack check: xy_err={xy_err:.4f} dz={dz:.4f} -> {'SUCCESS' if ok else 'FAIL'}")

# Data.overflow is a per-world bitmask. Buffer overflows invalidate the
# rollout; solver iteration caps are only a note that the solver stopped early.
overflow = mjw.OverflowType(overflow_bits)
capacity = overflow & ~solver_notes
print(f"capacity: peak contacts {peak_contacts} / nconmax {spec.nconmax}, "
      f"peak constraints {peak_constraints} / njmax {spec.njmax}")
if capacity:
  print(f"  ** OVERFLOW {capacity!r}: raise the limits and rerun before trusting this rollout **")
else:
  print("  no buffer overflow: the rollout is valid")
if overflow & solver_notes:
  print(f"  note: the constraint solver hit an iteration cap on some steps ({overflow & solver_notes!r})")

if args.headless_steps:
  path = recorder.save(
    "stage3_mjwarp", backend=f"MJWarp ({device})", xy_err=xy_err, dz=dz, ok=ok, steps_per_second=steps / elapsed,
    peak_contacts=peak_contacts, peak_constraints=peak_constraints, nconmax=spec.nconmax, njmax=spec.njmax,
    buffer_overflow=bool(capacity),
  )
  print(f"recorded to {path.relative_to(path.parents[1])} (see report.py)")
