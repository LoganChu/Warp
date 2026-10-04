"""Stage 4 (Gates 4 + 5) - scale to thousands of worlds, then measure correctly.

Reallocate the MJWarp state for a batch of worlds, replicate the initialized
MuJoCo state across it, capture one ``mjw.step`` as a CUDA graph and replay
it. Nothing crosses the PCIe bus per step, so this - unlike Stage 3 - is a
throughput path.

For each batch size the script reports ms per batched step, aggregate
world-steps per second, and the speedup over single-world MuJoCo on the CPU.

Usage:
    uv run python 04_scaling_study.py
    uv run python 04_scaling_study.py --worlds 1 64 1024 2048 8192 --steps 100
    uv run python 04_scaling_study.py --warm-frames 150    # benchmark mid-grasp instead of at rest
    uv run python 04_scaling_study.py --nconmax 32 --njmax 128   # tighter buffers: less memory, less work

Every run is added to results/stage4_scaling.json; report.py plots them together.
"""

import argparse
import subprocess
import time

import mujoco
import numpy as np
import warp as wp

from pick_place_common import ROBOTS, PickPlaceController, resolve_pick_place_scene
from viewing import load_result, save_result

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", default="so101", choices=sorted(ROBOTS))
parser.add_argument("--worlds", type=int, nargs="+", default=[1, 64, 1024, 2048, 8192])
parser.add_argument("--steps", type=int, default=200, help="timed physics steps per batch size")
parser.add_argument("--warm-frames", type=int, default=0,
                    help="advance the CPU task this many control frames before replicating its state")
parser.add_argument("--nconmax", type=int, help="contacts per world (default: from the robot profile)")
parser.add_argument("--njmax", type=int, help="constraints per world (default: from the robot profile)")
args = parser.parse_args()

wp.config.log_level = wp.LOG_WARNING
wp.init()
import mujoco_warp as mjw  # noqa: E402

device = wp.get_device()
if not device.is_cuda:
  raise SystemExit("CUDA graph capture needs a CUDA device")



def gpu_utilization() -> int | None:
  """How busy the GPU already is, in percent (None if nvidia-smi cannot say)."""
  try:
    query = ["nvidia-smi", f"--id={device.ordinal}", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
    return int(subprocess.run(query, capture_output=True, text=True, timeout=5).stdout.strip())
  except (OSError, ValueError, subprocess.TimeoutExpired):
    return None


# A benchmark on a shared GPU measures the sharing, not the simulator.
gpu_busy = gpu_utilization()
if gpu_busy and gpu_busy > 20:
  print(f"NOTE: the GPU is already {gpu_busy}% busy with other work; the numbers below will understate it")

spec = ROBOTS[args.robot]
mjm = mujoco.MjModel.from_xml_path(str(resolve_pick_place_scene(spec)))
mjd = mujoco.MjData(mjm)

fps = 50
sim_substeps = 10
frame_dt = 1.0 / fps
mjm.opt.timestep = frame_dt / sim_substeps  # same simulated time per step as Stages 2 and 3
mujoco.mj_forward(mjm, mjd)
nconmax = args.nconmax or spec.nconmax
njmax = args.njmax or spec.njmax

# Optionally move the host state somewhere more interesting than "arm at
# rest" before it is replicated, e.g. into the grasp where contacts peak.
controller = PickPlaceController(spec=spec)
for _ in range(args.warm_frames):
  ctrl = controller.step(mjm, mjd, frame_dt)
  for _ in range(sim_substeps):
    mjd.ctrl[: mjm.nu] = ctrl
    mujoco.mj_step(mjm, mjd)
print(f"replicated state: t={mjd.time:.2f} s, {mjd.ncon} contacts, {mjd.nefc} constraints per world")
print(f"buffers per world: nconmax={nconmax}, njmax={njmax}")

# --- CPU reference: one world, one thread, plain mj_step ------------------
cpu = mujoco.MjData(mjm)
cpu.qpos[:], cpu.qvel[:], cpu.ctrl[:] = mjd.qpos, mjd.qvel, mjd.ctrl
cpu_steps = 2000
t0 = time.perf_counter()
for _ in range(cpu_steps):
  mujoco.mj_step(mjm, cpu)
cpu_rate = cpu_steps / (time.perf_counter() - t0)

m = mjw.put_model(mjm)
# Buffer overflows invalidate a benchmark; the solver reaching its iteration
# cap does not, so mute that warning (see Stage 3) and check the rest below.
solver_notes = mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS
m.opt.warn_overflow = int(mjw.OverflowType.ALL & ~solver_notes)

print(f"\n{'worlds':>8} {'ms/step':>9} {'world-steps/s':>15} {'vs CPU':>8} {'x realtime':>11}  check")
print(f"{'CPU 1':>8} {1e3 / cpu_rate:>9.3f} {cpu_rate:>15,.0f} {1.0:>7.2f}x {cpu_rate * mjm.opt.timestep:>11,.0f}")

def benchmark(nworld: int) -> tuple[float, str]:
  """Return (seconds per batched step, verification result) for nworld worlds."""
  # --- Gate 4: allocate the batch and give every world the same start state.
  d = mjw.make_data(mjm, nworld=nworld, nconmax=nconmax, njmax=njmax)

  wp.copy(d.qpos, wp.array(np.tile(mjd.qpos, (nworld, 1)), dtype=wp.float32, device=device))
  wp.copy(d.qvel, wp.array(np.tile(mjd.qvel, (nworld, 1)), dtype=wp.float32, device=device))
  wp.copy(d.ctrl, wp.array(np.tile(mjd.ctrl, (nworld, 1)), dtype=wp.float32, device=device))
  mjw.forward(m, d)

  # mjw.step is many kernel launches. Record them once as a CUDA graph and
  # replay the graph; it is tied to the buffers of this `m` and `d`, so it is
  # captured again for every batch size.
  with wp.ScopedCapture() as capture:
    mjw.step(m, d)
  step_graph = capture.graph

  # --- Gate 5: warm up, then synchronize on both sides of the timed region.
  for _ in range(10):  # warm-up: compilation, allocation, caches
    wp.capture_launch(step_graph)
  wp.synchronize()

  t0 = time.perf_counter()
  for _ in range(args.steps):
    wp.capture_launch(step_graph)
  wp.synchronize()  # without this you time the queue, not the work
  elapsed = time.perf_counter() - t0

  # Verify before trusting the number: finite state and no buffer overflow.
  overflow = mjw.OverflowType(int(np.bitwise_or.reduce(d.overflow.numpy()))) & ~solver_notes
  finite = bool(np.isfinite(d.qpos.numpy()).all())
  check = "ok" if finite and not overflow else f"INVALID ({overflow!r}, finite={finite})"
  return elapsed / args.steps, check


rows = []
out_of_memory_at = None
for nworld in args.worlds:
  try:
    seconds, check = benchmark(nworld)
  except RuntimeError as error:  # Warp raises this when a device allocation fails
    if "allocate" not in str(error):
      raise
    print(f"{nworld:>8,}  out of GPU memory: buffers grow with nworld x nconmax / njmax, so size them tighter")
    out_of_memory_at = nworld
    break
  rate = nworld / seconds
  rows.append({"nworld": nworld, "ms_per_step": 1e3 * seconds, "rate": rate, "valid": check == "ok"})
  print(
    f"{nworld:>8,} {1e3 * seconds:>9.3f} {rate:>15,.0f} {rate / cpu_rate:>7.2f}x "
    f"{rate * mjm.opt.timestep:>11,.0f}  {check}"
  )

# Keep one entry per configuration so that reruns replace their earlier result.
state = "at rest" if args.warm_frames == 0 else f"mid-task, t={mjd.time:.1f} s"
label = f"{state}, nconmax {nconmax}, njmax {njmax}"
study = load_result("stage4_scaling") or {"runs": {}}
study["runs"][label] = {
  "state": state, "nconmax": nconmax, "njmax": njmax, "contacts": int(mjd.ncon), "constraints": int(mjd.nefc),
  "cpu_rate": cpu_rate, "timestep": mjm.opt.timestep, "device": device.name, "rows": rows,
  "out_of_memory_at": out_of_memory_at, "gpu_busy_percent": gpu_busy if gpu_busy and gpu_busy > 20 else None,
}
save_result("stage4_scaling", study)

print(
  "\nms/step is latency: wall-clock time for one batched step."
  "\nworld-steps/s is aggregate throughput: worlds x steps finished per wall-clock second."
  "\nx realtime is simulated seconds per wall-clock second, summed over all worlds."
  "\n\nrecorded to results/stage4_scaling.json (see report.py)"
)
