import json
from pathlib import Path
import numpy as np

from .objects import Hist1D


def derive_hist1d_weights(
    data: Hist1D,
    mc: Hist1D,
    *,
    normalize=True,
    clip=(0.0, 5.0),
    empty_mc_weight=1.0,
):
    """
    Build binned Data/MC weights.

    If normalize=True, both histograms are first normalized to unit integral,
    so the result corrects the *shape* of the MC distribution.
    """
    if not np.allclose(data.edges, mc.edges):
        raise ValueError("Data and MC histogram binning differs.")

    d = data.normalized() if normalize else data
    m = mc.normalized() if normalize else mc

    weights = np.full_like(d.values, float(empty_mc_weight), dtype=float)

    valid = np.isfinite(m.values) & (m.values > 0)
    weights[valid] = d.values[valid] / m.values[valid]

    weights[~np.isfinite(weights)] = float(empty_mc_weight)

    if clip is not None:
        lo, hi = clip
        weights = np.clip(weights, lo, hi)

    return {
        "edges": data.edges.tolist(),
        "weights": weights.tolist(),
        "normalize": bool(normalize),
        "clip": None if clip is None else list(clip),
        "empty_mc_weight": float(empty_mc_weight),
    }


def save_weights_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return path


def load_weights_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    edges = np.asarray(payload["edges"], dtype=float)
    weights = np.asarray(payload["weights"], dtype=float)

    if len(edges) != len(weights) + 1:
        raise ValueError(
            "Invalid weight JSON: len(edges) must equal len(weights)+1"
        )

    return payload
