"""Stage 1 - one useful Warp kernel.

Warp lets you write a statically typed kernel in Python and compiles it to
native code for the CPU or to CUDA for the GPU. The kernel body describes the
work of ONE logical thread; ``wp.launch(dim=n)`` runs n of them in parallel.

Part A is the example from the article: two points falling under gravity.
Part B launches the very same kernel on four million points and times it on
both devices, which is the whole idea MJWarp is built on.

Usage:
    uv run python 01_warp_kernel.py
"""

import time

import numpy as np
import warp as wp


@wp.kernel
def integrate(
  positions: wp.array[wp.vec3],
  velocities: wp.array[wp.vec3],
  dt: float,
):
  i = wp.tid()  # which point this thread owns
  velocities[i] += wp.vec3(0.0, 0.0, -9.81) * dt
  positions[i] += velocities[i] * dt


wp.init()
device = "cuda:0" if wp.is_cuda_available() else "cpu"

# --- Part A: two points, one step of 0.01 s -------------------------------
start = np.array([[0.0, 0.0, 0.5], [0.2, 0.0, 0.5]], dtype=np.float32)
positions = wp.array(start, dtype=wp.vec3, device=device)  # lives on `device`
velocities = wp.zeros_like(positions)

# The first launch compiles the kernel (watch for "Module __main__ ... load"),
# later launches and later runs reuse the cached binary.
wp.launch(
  integrate,
  dim=len(start),
  inputs=[positions, velocities, 0.01],
  device=device,
)
wp.synchronize_device(device)
# .numpy() on a CUDA array synchronizes and COPIES to host memory.
print(f"\nPart A - positions after one step on {device}:")
print(positions.numpy())


# --- Part B: same kernel, millions of points ------------------------------
def time_launches(n: int, device: str, steps: int = 100) -> float:
  """Seconds per launch of `integrate` over n points."""
  positions = wp.zeros(n, dtype=wp.vec3, device=device)
  velocities = wp.zeros(n, dtype=wp.vec3, device=device)
  inputs = [positions, velocities, 0.01]

  wp.launch(integrate, dim=n, inputs=inputs, device=device)  # warm-up (compile / load)
  wp.synchronize_device(device)

  t0 = time.perf_counter()
  for _ in range(steps):
    wp.launch(integrate, dim=n, inputs=inputs, device=device)
  # GPU launches are asynchronous: without this you time how fast Python
  # queued the work, not how fast the GPU finished it.
  wp.synchronize_device(device)
  return (time.perf_counter() - t0) / steps


print("\nPart B - the same kernel at scale:")
print(f"{'points':>10} {'device':>8} {'ms/launch':>10} {'points/second':>16}")
for n in (2, 4_000, 4_000_000):
  for dev in dict.fromkeys(("cpu", device)):
    seconds = time_launches(n, dev)
    print(f"{n:>10,} {dev:>8} {1e3 * seconds:>10.4f} {n / seconds:>16,.0f}")

print(
  "\nA launch has a fixed overhead, so tiny problems are no faster on the GPU."
  "\nThe GPU wins when one launch carries a lot of parallel work. MJWarp gets"
  "\nthat work by stepping thousands of independent worlds per launch (Stage 4)."
)
