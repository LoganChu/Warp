"""Stage 5 - watch a batch of GPU worlds.

Stage 4 stepped thousands of identical worlds and never looked at them. This
stage keeps the batch small enough to draw, gives every world a different
cube layout, and runs the whole pick-and-place in all of them at once: one
``mjw.step`` advances every world, and one viewer window shows them side by
side.

It also shows the pattern a training loop uses. Per control frame:

    host:   one action per world            -> d.ctrl   (one upload)
    device: sim_substeps batched steps      (CUDA graph replay)
    host:   d.qpos / d.qvel for all worlds  <- one download

Here the "policy" is the scripted controller from Stage 2, one instance per
world, running on the host.

Usage:
    uv run python 05_view_batch.py                         # 16 worlds, interactive viewer
    uv run python 05_view_batch.py --worlds 64 --jitter 0.03
    uv run python 05_view_batch.py --headless-steps 600    # no viewer, prints per-world results
"""

import argparse
import time

import mujoco
import numpy as np
import warp as wp

from pick_place_common import ROBOTS, PickPlaceController, resolve_pick_place_scene, stack_check
from viewing import BAD, GOOD, PENDING, draw_worlds, grid_camera, grid_offsets, run_viewer, save_result, show_text

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", default="so101", choices=sorted(ROBOTS))
parser.add_argument("--worlds", type=int, default=16, help="number of worlds to simulate and draw")
parser.add_argument("--jitter", type=float, default=0.02, help="cube start positions vary by +- this much [m]")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--headless-steps", type=int, default=0, help="run this many control frames without a viewer")
args = parser.parse_args()

wp.config.log_level = wp.LOG_WARNING
wp.init()
import mujoco_warp as mjw  # noqa: E402

device = wp.get_device()
if not device.is_cuda:
  raise SystemExit("CUDA graph capture needs a CUDA device")

spec = ROBOTS[args.robot]
mjm = mujoco.MjModel.from_xml_path(str(resolve_pick_place_scene(spec)))

fps = 50
sim_substeps = 10
frame_dt = 1.0 / fps
episode_frames = 600
mjm.opt.timestep = frame_dt / sim_substeps

nworld = args.worlds
# One host MjData per world. They never step physics: they hold the state
# downloaded from the device so the controllers and the viewer can read it.
datas = [mujoco.MjData(mjm) for _ in range(nworld)]
controllers: list[PickPlaceController] = []
red_adr = mjm.jnt_qposadr[mjm.joint("red_cube_joint").id]
blue_adr = mjm.jnt_qposadr[mjm.joint("blue_cube_joint").id]
rng = np.random.default_rng(args.seed)

m = mjw.put_model(mjm)
solver_notes = mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS
m.opt.warn_overflow = int(mjw.OverflowType.ALL & ~solver_notes)
d = mjw.make_data(mjm, nworld=nworld, nconmax=spec.nconmax, njmax=spec.njmax)

# The graph replays launches against the buffers of `m` and `d`. Resets and
# new controls are written INTO those buffers, so one capture lasts forever.
with wp.ScopedCapture() as capture:
  mjw.step(m, d)
step_graph = capture.graph

ctrl = np.zeros((nworld, mjm.nu), dtype=np.float32)
frame = 0
physics_ms = 0.0
start_offsets = np.zeros((nworld, 2, 2))  # per world: (red, blue) xy offset from the nominal layout


def reset() -> None:
  """Start an episode with a fresh random cube layout in every world."""
  global controllers, frame
  frame = 0
  controllers = [PickPlaceController(spec=spec) for _ in range(nworld)]
  for i, data in enumerate(datas):
    mujoco.mj_resetData(mjm, data)
    # Per-world randomization: every world gets its own row of qpos.
    start_offsets[i] = rng.uniform(-args.jitter, args.jitter, size=(2, 2))
    data.qpos[red_adr : red_adr + 2] += start_offsets[i, 0]
    data.qpos[blue_adr : blue_adr + 2] += start_offsets[i, 1]
    mujoco.mj_forward(mjm, data)

  mjw.reset_data(m, d)
  wp.copy(d.qpos, wp.array(np.stack([data.qpos for data in datas]), dtype=wp.float32, device=device))
  wp.copy(d.qvel, wp.array(np.stack([data.qvel for data in datas]), dtype=wp.float32, device=device))
  wp.copy(d.ctrl, wp.array(np.stack([data.ctrl for data in datas]), dtype=wp.float32, device=device))
  mjw.forward(m, d)


def simulate_frame() -> None:
  global frame, physics_ms
  # Host: one action per world.
  for i, (controller, data) in enumerate(zip(controllers, datas)):
    ctrl[i] = controller.step(mjm, data, frame_dt)

  # Device: upload all actions, advance every world, download all states.
  t0 = time.perf_counter()
  wp.copy(d.ctrl, wp.array(ctrl, dtype=wp.float32, device=device))
  for _ in range(sim_substeps):
    wp.capture_launch(step_graph)
  qpos = d.qpos.numpy()  # one synchronizing copy for the whole batch, once per frame
  qvel = d.qvel.numpy()
  physics_ms = 1e3 * (time.perf_counter() - t0)

  # Host: refresh body / geom poses for the controllers and the viewer.
  for i, data in enumerate(datas):
    data.qpos[:] = qpos[i]
    data.qvel[:] = qvel[i]
    mujoco.mj_kinematics(mjm, data)
  frame += 1


def results() -> list[tuple[float, float, bool]]:
  return [stack_check(mjm, data) for data in datas]


offsets = grid_offsets(nworld)


def draw(viewer) -> None:
  checks = results()
  done = frame >= episode_frames
  markers = []
  for i, (_, _, ok) in enumerate(checks):
    if ok:
      markers.append((f"{i} stacked", GOOD))
    elif done:
      markers.append((f"{i} failed", BAD))
    else:
      markers.append((f"{i}", PENDING))
  draw_worlds(viewer, mjm, datas, offsets, markers)

  stacked = sum(ok for _, _, ok in checks)
  show_text(
    viewer,
    {
      "backend": f"MJWarp ({device}), {nworld} worlds",
      "time": f"{frame * frame_dt:5.2f} s",
      "phase": controllers[0].phase or "-",
      "stacked": f"{stacked} / {nworld}",
      "physics": f"{physics_ms:.1f} ms per frame ({sim_substeps} batched steps + state download)",
      "cube jitter": f"+-{1e3 * args.jitter:.0f} mm, new layout every episode",
    },
  )


reset()
if args.headless_steps:
  for _ in range(args.headless_steps):
    simulate_frame()
  checks = results()
  print(f"{'world':>5} {'red start offset [mm]':>24} {'xy_err':>8} {'dz':>8}  result")
  for i, (xy_err, dz, ok) in enumerate(checks):
    dx, dy = 1e3 * start_offsets[i, 0]
    print(f"{i:>5} {f'({dx:+.0f}, {dy:+.0f})':>24} {xy_err:>8.4f} {dz:>8.4f}  {'stacked' if ok else 'FAILED'}")
  stacked = sum(ok for _, _, ok in checks)
  overflow = mjw.OverflowType(int(np.bitwise_or.reduce(d.overflow.numpy()))) & ~solver_notes
  print(f"\n{stacked} / {nworld} worlds stacked the cube; buffer overflow: {overflow!r}")
  path = save_result(
    "stage5_batch",
    {
      "summary": {"nworld": nworld, "jitter": args.jitter, "seed": args.seed, "stacked": stacked},
      "worlds": [
        {"world": i, "red_offset": start_offsets[i, 0].tolist(), "blue_offset": start_offsets[i, 1].tolist(),
         "xy_err": xy_err, "dz": dz, "ok": ok}
        for i, (xy_err, dz, ok) in enumerate(checks)
      ],
    },
  )
  print(f"recorded to {path.relative_to(path.parents[1])} (see report.py)")
else:
  run_viewer(mjm, datas[0], simulate_frame, reset, frame_dt, episode_frames, draw=draw, camera=grid_camera(offsets))
