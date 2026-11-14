"""Export real 3D soma coordinates without rebuilding or overwriting the RPS network."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--annotations", type=Path, default=Path("data/raw/mcns_annotations.feather"))
parser.add_argument("--subgraph", type=Path, default=Path("data/subgraph.json"))
parser.add_argument("--top-down-points", type=Path, default=Path("data/brain_points.bin"))
parser.add_argument("--output-dir", type=Path, required=True,
                    help="new or empty directory; existing outputs are never overwritten")
args = parser.parse_args()

outputs = {
    "all_points": args.output_dir / "brain_points_3d.bin",
    "sim_points": args.output_dir / "sim_neuron_positions_3d.bin",
    "metadata": args.output_dir / "brain_3d_metadata.json",
}
existing = [str(path) for path in outputs.values() if path.exists()]
if existing:
    raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))

annotation = pd.read_feather(args.annotations, columns=[
    "bodyId", "type", "superclass", "somaSide", "somaLocation", "statusLabel",
]).drop_duplicates("bodyId").set_index("bodyId")
placed = annotation[
    annotation.somaLocation.notna()
    & annotation.superclass.notna()
    & (annotation.statusLabel != "Glia")
]
source_xyz = np.stack(placed.somaLocation.to_numpy()).astype(np.float64)
x0, x1 = source_xyz[:, 0].min(), source_xyz[:, 0].max()
y0, y1 = source_xyz[:, 1].min(), source_xyz[:, 1].max()
z0, z1 = source_xyz[:, 2].min(), source_xyz[:, 2].max()
if min(x1 - x0, y1 - y0, z1 - z0) <= 0:
    raise ValueError("Source soma coordinates have a degenerate axis")
xspan = x1 - x0

# The old drawing uses mirrored source-x horizontally and source-z vertically.
view_x = (x1 - source_xyz[:, 0]) / xspan
view_y = (source_xyz[:, 2] - z0) / (z1 - z0)
view_z = (source_xyz[:, 1] - y0) / (y1 - y0)
projection = np.round(np.column_stack((view_x, view_y)) * 65535).astype("<u2")
original_projection = np.fromfile(args.top_down_points, dtype="<u2").reshape(-1, 2)
if not np.array_equal(projection, original_projection):
    raise ValueError("Source annotations do not exactly reproduce brain_points.bin")

# Use the exact Python float-rounding and type normalization used by extract_subgraph.py.
source = pd.DataFrame({
    "type": [value if isinstance(value, str) else "" for value in placed.type],
    "x": [round(float(value), 4) for value in view_x],
    "y": [round(float(value), 4) for value in view_y],
    "bodyId": placed.index.to_numpy(),
    "view_x": (view_x - .5) * 2.5,
    "view_y": (.5 - view_y) * 2.5 * ((z1 - z0) / xspan),
    "view_z": (view_z - .5) * 2.5 * ((y1 - y0) / xspan),
})
source["view_x"] = source["view_x"].astype("<f4")
source["view_y"] = source["view_y"].astype("<f4")
source["view_z"] = source["view_z"].astype("<f4")

subgraph = json.loads(args.subgraph.read_text(encoding="utf-8"))
neurons = subgraph["neurons"]
simulated = pd.DataFrame({
    "type": neurons["type"], "x": neurons["x"], "y": neurons["y"],
    "index": np.arange(len(neurons["type"])),
})
source_keys = source.groupby(["type", "x", "y"], dropna=False).size()
needed_keys = set(zip(simulated["type"], simulated["x"], simulated["y"]))
ambiguous = [key for key in needed_keys if source_keys.get(key, 0) > 1]
if ambiguous:
    raise ValueError(f"Ambiguous soma coordinates for {len(ambiguous)} simulated-neuron keys")
lookup = source.set_index(["type", "x", "y"])
positions = np.full((len(simulated), 3), np.nan, dtype="<f4")
matched = np.zeros(len(simulated), dtype=bool)
coordinate_lookup = {
    key: value for key, value in zip(
        lookup.index.tolist(), lookup[["view_x", "view_y", "view_z"]].to_numpy(dtype="<f4"),
    )
}
for index, (role, x, y) in enumerate(zip(simulated["type"], simulated["x"], simulated["y"])):
    if x < 0:
        continue
    key = (role, x, y)
    if key not in coordinate_lookup:
        raise ValueError(f"No source soma matched simulated neuron index {index}: {key}")
    positions[index] = coordinate_lookup[key]
    matched[index] = True

args.output_dir.mkdir(parents=True, exist_ok=True)
np.asarray(source[["view_x", "view_y", "view_z"]], dtype="<f4").tofile(outputs["all_points"])
positions.tofile(outputs["sim_points"])
metadata = {
    "source": "Janelia/Google male CNS v1.0 somaLocation, derived from raw annotations",
    "coordinateOrder": ["viewer x (mirrored source x)", "viewer y (source z)", "viewer z (source y depth)"],
    "units": "source coordinate axes normalized with one common x-axis scale; no neural positions are synthesized",
    "backgroundPoints": len(source),
    "simulatedNeurons": len(simulated),
    "placedSimulatedNeurons": int(matched.sum()),
    "unplacedSimulatedNeurons": int((~matched).sum()),
    "topDownProjectionExact": True,
    "depthRange": [float(source_xyz[:, 1].min()), float(source_xyz[:, 1].max())],
}
outputs["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(json.dumps(metadata, indent=2))
