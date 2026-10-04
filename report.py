"""Build an interactive report from what the stage scripts recorded in results/.

    uv run python 02_so101_pick_place.py --headless-steps 600    # records the CPU rollout
    uv run python 03_so101_mjwarp.py --headless-steps 600        # records the MJWarp rollout
    uv run python 04_scaling_study.py                            # records a scaling run (repeat with other flags)
    uv run python 05_view_batch.py --headless-steps 600          # records the per-world outcomes
    uv run python report.py                                      # writes results/report.html and opens it

Sections whose data has not been recorded yet say which command to run.
Hover a chart (or focus it and use the arrow keys) to read values; every
chart also has a data table underneath.
"""

import argparse
import datetime
import json
import math
import webbrowser

from pick_place_common import DZ_MAX, DZ_MIN, ROOT, XY_TOL
from viewing import RESULTS, load_result

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--no-open", action="store_true", help="write the report without opening a browser")
args = parser.parse_args()

MAX_RUNS = 4  # scaling runs drawn together: one colour each, no more than the palette supports

cpu = load_result("stage2_cpu")
gpu = load_result("stage3_mjwarp")
scaling = load_result("stage4_scaling")
batch = load_result("stage5_batch")

tiles = []
sections = []


def mm(value: float) -> float:
  return round(1e3 * value, 2)


def compact(value: float) -> str:
  for limit, suffix in ((1e6, "M"), (1e3, "K")):
    if value >= limit:
      return f"{value / limit:.1f}{suffix}"
  return f"{value:.0f}"


def stack_tile(result: dict, stage: str) -> dict:
  summary = result["summary"]
  return {
    "label": f"Stack offset, {summary['backend']}",
    "value": f"{mm(summary['xy_err']):.1f} mm",
    "note": f"{stage} · limit {mm(XY_TOL):.0f} mm · {'stacked' if summary['ok'] else 'NOT stacked'}",
  }


# --- Parity: the same task on both backends ---------------------------------
parity = {
  "title": "Parity: one world on CPU and on GPU (Stages 2 and 3)",
  "intro": "The same controller drives the same scene on both backends. If MJWarp simulates the scene "
  "correctly, the red cube follows the same path and ends in the same place.",
}
if cpu and gpu:
  tiles += [stack_tile(cpu, "Stage 2"), stack_tile(gpu, "Stage 3")]
  a, b = cpu["series"], gpu["series"]
  n = min(len(a["t"]), len(b["t"]))
  marks = [{"x": a["t"][i], "label": a["phase"][i]} for i in range(n) if i == 0 or a["phase"][i] != a["phase"][i - 1]]
  gap = [[a["t"][i], mm(math.dist(a["red"][i], b["red"][i]))] for i in range(n)]
  parity["blocks"] = [
    {
      "type": "line",
      "title": "Red cube height",
      "subtitle": "Lifted, carried and set down on the blue cube. The two lines overlap almost exactly.",
      "xLabel": "time [s]", "yLabel": "height [mm]", "xFmt": "f1", "xTipFmt": "f2", "yFmt": "int", "tipFmt": "f1", "unit": "mm",
      "marks": marks,
      "series": [
        {"name": cpu["summary"]["backend"], "pts": [[a["t"][i], mm(a["red"][i][2])] for i in range(n)]},
        {"name": gpu["summary"]["backend"], "pts": [[b["t"][i], mm(b["red"][i][2])] for i in range(n)]},
      ],
    },
    {
      "type": "line",
      "title": "Distance between the two simulations' red cubes",
      "subtitle": "Zero until the gripper touches the cube; contact amplifies float32 and ordering differences.",
      "xLabel": "time [s]", "yLabel": "distance [mm]", "xFmt": "f1", "xTipFmt": "f2", "yFmt": "f1", "tipFmt": "f2", "unit": "mm",
      "marks": marks,
      "series": [{"name": "distance", "annotateMax": "max", "pts": gap}],
      "foot": f"Final gap {gap[-1][1]:.2f} mm, against a success tolerance of {mm(XY_TOL):.0f} mm.",
    },
  ]
else:
  parity["missing"] = "Not recorded yet. Run 02_so101_pick_place.py and 03_so101_mjwarp.py with --headless-steps 600."
sections.append(parity)

# --- Capacity: how full the MJWarp buffers get -------------------------------
capacity = {
  "title": "Capacity: contact and constraint buffers (Stage 3)",
  "intro": "MJWarp allocates these buffers before stepping. They have to cover the busiest moment of the "
  "task, which is the grasp and the placement, not the arm moving through free space.",
}
if gpu:
  g, series = gpu["summary"], gpu["series"]
  marks = [
    {"x": series["t"][i], "label": series["phase"][i]}
    for i in range(len(series["t"]))
    if i == 0 or series["phase"][i] != series["phase"][i - 1]
  ]
  common = {"type": "line", "xLabel": "time [s]", "xFmt": "f1", "xTipFmt": "f2", "yFmt": "int", "tipFmt": "int", "marks": marks}
  capacity["blocks"] = [
    {
      "type": "meters", "wide": True,
      "title": "Peak use against the allocated budget",
      "subtitle": "Overflow: " + ("YES, this rollout is invalid" if g["buffer_overflow"] else "none"),
      "meters": [
        {"label": "Contacts (nconmax)", "value": g["peak_contacts"], "max": g["nconmax"]},
        {"label": "Constraints (njmax)", "value": g["peak_constraints"], "max": g["njmax"]},
      ],
      "foot": "Headroom is memory. Stage 4 shows what tightening these limits buys.",
    },
    {
      **common, "title": "Constraint rows in use", "subtitle": "Peak per control frame, one world.", "yLabel": "constraints",
      "series": [{"name": "constraints", "slot": 1, "annotateMax": "peak", "pts": list(map(list, zip(series["t"], series["constraints"])))}],
    },
    {
      **common, "title": "Contacts in use", "subtitle": "Peak per control frame, one world.", "yLabel": "contacts",
      "series": [{"name": "contacts", "slot": 0, "annotateMax": "peak", "pts": list(map(list, zip(series["t"], series["contacts"])))}],
    },
  ]
else:
  capacity["missing"] = "Not recorded yet. Run: uv run python 03_so101_mjwarp.py --headless-steps 600"
sections.append(capacity)

# --- Scaling: throughput and latency against batch size ----------------------
scale = {
  "title": "Scaling: many worlds per step (Stage 4)",
  "intro": "Latency is the time for one batched step. Throughput is worlds times steps per second. "
  "A GPU step is slow for one world and nearly as slow for thousands, which is the whole point.",
}
if scaling and scaling["runs"]:
  runs = list(scaling["runs"].items())[-MAX_RUNS:]
  best_label, best_row = max(
    ((label, row) for label, run in runs for row in run["rows"]), key=lambda item: item[1]["rate"]
  )
  tiles.append({
    "label": "Peak throughput, world-steps per second",
    "value": compact(best_row["rate"]),
    "note": f"Stage 4 · {best_row['nworld']:,} worlds · {best_label.split(',')[0]}",
  })
  references = {}
  for _, run in runs:
    references.setdefault(run["state"], run["cpu_rate"])
  notes = []
  busy = [label for label, run in runs if run.get("gpu_busy_percent")]
  if busy:
    which = "All runs were" if len(busy) == len(runs) else "These runs were: " + "; ".join(busy) + ". They were"
    notes.append(f"{which} recorded while the GPU was busy with other work, so they understate the hardware.")
  for label, run in runs:
    if run["out_of_memory_at"]:
      notes.append(f"“{label}” ran out of GPU memory at {run['out_of_memory_at']:,} worlds.")
  common = {"type": "line", "xLabel": "worlds", "xLog": True, "yLog": True, "xFmt": "si", "xTipFmt": "int", "points": True}
  scale["blocks"] = [
    {
      **common, "title": "Aggregate throughput", "subtitle": "World-steps per second. Both axes are logarithmic.",
      "yLabel": "world-steps / s", "yFmt": "si", "tipFmt": "int",
      "hlines": [{"y": rate, "label": f"one CPU world, {state}"} for state, rate in references.items()],
      "series": [
        {"name": label, "pts": [[row["nworld"], row["rate"], f"{row['rate'] / run['cpu_rate']:.1f}x one CPU world"] for row in run["rows"]]}
        for label, run in runs
      ],
      "foot": " ".join(notes) or None,
    },
    {
      **common, "title": "Latency of one batched step", "subtitle": "Milliseconds per step, for the whole batch. Both axes are logarithmic.",
      "yLabel": "ms / step", "yFmt": "g", "tipFmt": "f3", "unit": "ms",
      "series": [{"name": label, "pts": [[row["nworld"], row["ms_per_step"]] for row in run["rows"]]} for label, run in runs],
    },
    {
      "type": "table", "wide": True, "title": "Recorded runs",
      "subtitle": f"{runs[0][1]['device']} · physics step {1e3 * runs[0][1]['timestep']:.0f} ms",
      "columns": ["Run", "Contacts / world", "CPU, 1 world [steps/s]", "Best GPU [world-steps/s]", "at worlds", "Speedup"],
      "rows": [
        [
          label, str(run["contacts"]), f"{run['cpu_rate']:,.0f}",
          f"{max(r['rate'] for r in run['rows']):,.0f}",
          f"{max(run['rows'], key=lambda r: r['rate'])['nworld']:,}",
          f"{max(r['rate'] for r in run['rows']) / run['cpu_rate']:.1f}x",
        ]
        for label, run in runs if run["rows"]
      ],
      "foot": "Speedup is against one world on one CPU core.",
    },
  ]
else:
  scale["missing"] = "Not recorded yet. Run: uv run python 04_scaling_study.py"
sections.append(scale)

# --- Batch: per-world outcomes with randomized cube layouts ------------------
outcomes = {
  "title": "Batch outcomes: every world has a different layout (Stage 5)",
  "intro": "Each world starts with its cubes shifted by a random offset and runs the whole task on the GPU. "
  "A point is where that world's red cube ended up relative to its blue cube.",
}
if batch:
  summary = batch["summary"]
  tiles.append({
    "label": "Worlds that stacked the cube",
    "value": f"{summary['stacked']} / {summary['nworld']}",
    "note": f"Stage 5 · cube start jitter ±{mm(summary['jitter']):.0f} mm",
  })
  outcomes["blocks"] = [
    {
      "type": "scatter", "wide": True,
      "title": "Final cube placement per world",
      "subtitle": f"{summary['nworld']} worlds. Inside the shaded region counts as stacked.",
      "xLabel": "horizontal offset between cube centres [mm]", "yLabel": "vertical separation [mm]",
      "region": {"x0": 0, "x1": mm(XY_TOL), "y0": mm(DZ_MIN), "y1": mm(DZ_MAX)},
      "points": [
        {
          "label": f"world {w['world']}", "x": mm(w["xy_err"]), "y": mm(w["dz"]), "ok": w["ok"],
          "start": f"({mm(w['red_offset'][0]):+.0f}, {mm(w['red_offset'][1]):+.0f}) mm",
        }
        for w in batch["worlds"]
      ],
    }
  ]
else:
  outcomes["missing"] = "Not recorded yet. Run: uv run python 05_view_batch.py --headless-steps 600"
sections.append(outcomes)

report = {
  "title": "MJWarp walkthrough: results",
  "lede": f"Built {datetime.datetime.now():%Y-%m-%d %H:%M} from the recordings in results/. "
  "Hover a chart to read values, or focus it and use the arrow keys.",
  "tiles": tiles,
  "sections": sections,
}

template = (ROOT / "report_template.html").read_text()
payload = json.dumps(report).replace("</", "<\\/")
RESULTS.mkdir(exist_ok=True)
path = RESULTS / "report.html"
path.write_text(template.replace("/*DATA*/null", payload))
print(f"wrote {path.relative_to(ROOT)}")
if not args.no_open:
  webbrowser.open(path.as_uri())
