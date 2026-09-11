#!/usr/bin/env python3
'''
Create a CMS-style correctionlib pileup-weight JSON from pileupCalc data
histograms and an MC pileup histogram.

Optionally produce diagnostic plots using the plotting package shipped with
SkimRDFAnalysisBase.

Example:

python3 make_pu_weights_json_with_plots.py \
    --data-nominal pileup_69200.root \
    --data-up pileup_up.root \
    --data-down pileup_down.root \
    --data-hist pileup \
    --mc mc_pileup.root \
    --mc-hist pileup_nTrueInt \
    --name Collisions25_myData \
    -o puWeights_myData.json \
    --make-plots \
    --plot-dir plots
'''

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import uproot

try:
    from correctionlib import schemav2
    from rich import print as rich_print
    HAVE_CORRECTIONLIB = True
except ImportError as e:
    HAVE_CORRECTIONLIB = False
    CORRECTIONLIB_IMPORT_ERROR = str(e)


def validate_correction(correction):
    corr = correction["corrections"][0]

    if correction["schema_version"] != 2:
        raise RuntimeError("Expected correctionlib schema version 2")

    entries = corr["data"]["content"]
    if {x["key"] for x in entries} != {"nominal", "up", "down"}:
        raise RuntimeError("Expected nominal, up and down corrections")

    for entry in entries:
        node = entry["value"]
        if len(node["edges"]) != len(node["content"]) + 1:
            raise RuntimeError(
                f"Invalid binning for {entry['key']}: N(edges) must equal N(weights) + 1"
            )


def read_histogram(filename, histname):
    with uproot.open(filename) as f:
        hist = f[histname]
        values, edges = hist.to_numpy()
        try:
            errors = hist.errors()
        except Exception:
            errors = None

        if errors is None:
            errors = np.sqrt(np.clip(values, 0.0, None))

    return values.astype(float), edges.astype(float), np.asarray(errors, dtype=float)


def normalize(hist):
    integral = np.sum(hist)
    if integral <= 0:
        raise RuntimeError("Histogram has zero or negative integral")
    return hist / integral


def make_weights(data_hist, mc_hist):
    data_hist = normalize(data_hist)
    mc_hist = normalize(mc_hist)

    weights = np.ones_like(data_hist)
    mask = mc_hist > 0
    weights[mask] = data_hist[mask] / mc_hist[mask]
    return weights


def make_binning(edges, weights):
    return {
        "nodetype": "binning",
        "input": "NumTrueInteractions",
        "flow": "clamp",
        "edges": edges.tolist(),
        "content": weights.tolist(),
    }


def find_analysis_base_root():
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]

    for candidate in candidates:
        if (candidate / "plotting" / "__init__.py").is_file():
            return candidate

    raise ImportError(
        "Could not locate the SkimRDFAnalysisBase plotting package. "
        "Run the script from inside SkimRDFAnalysisBase or place it under the repository."
    )


def get_plotting_helpers():
    repo_root = find_analysis_base_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from plotting import Hist1D, auto_range, save_figure
        from plotting.plotters import draw_cms_label, use_hep_style
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Plotting was requested, but the SkimRDFAnalysisBase plotting dependencies "
            f"could not be imported: {e}"
        ) from e

    return Hist1D, auto_range, save_figure, draw_cms_label, use_hep_style, plt


def make_diagnostic_plots(
    edges,
    data_nominal,
    data_nominal_err,
    data_up,
    data_up_err,
    data_down,
    data_down_err,
    mc,
    mc_err,
    weights_nominal,
    weights_up,
    weights_down,
    output_stem,
    plot_dir,
    plot_format,
    cms_label,
    lumi,
    com,
):
    Hist1D, auto_range, save_figure, draw_cms_label, use_hep_style, plt = get_plotting_helpers()
    use_hep_style()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    data_nominal_hist = Hist1D(data_nominal, edges, data_nominal_err, name="data_nominal").normalized()
    data_up_hist = Hist1D(data_up, edges, data_up_err, name="data_up").normalized()
    data_down_hist = Hist1D(data_down, edges, data_down_err, name="data_down").normalized()
    mc_hist = Hist1D(mc, edges, mc_err, name="mc").normalized()

    mc_weighted_nominal = Hist1D(
        mc_hist.values * weights_nominal,
        edges,
        None if mc_hist.errors is None else mc_hist.errors * np.abs(weights_nominal),
        name="mc_weighted_nominal",
    )
    mc_weighted_up = Hist1D(
        mc_hist.values * weights_up,
        edges,
        None if mc_hist.errors is None else mc_hist.errors * np.abs(weights_up),
        name="mc_weighted_up",
    )
    mc_weighted_down = Hist1D(
        mc_hist.values * weights_down,
        edges,
        None if mc_hist.errors is None else mc_hist.errors * np.abs(weights_down),
        name="mc_weighted_down",
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    nominal_color = colors[0] if len(colors) > 0 else None
    up_color = colors[1] if len(colors) > 1 else None
    down_color = colors[2] if len(colors) > 2 else None
    default_mc_color = colors[3] if len(colors) > 3 else None

    # ------------------------------------------------------------------
    # Pileup distribution overlay
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10.8, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.42], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    side_ax = fig.add_subplot(gs[0, 1])
    side_ax.axis("off")

    for hist, label, color in [
        (data_nominal_hist, "Data nominal", nominal_color),
        (data_up_hist, "Data up", up_color),
        (data_down_hist, "Data down", down_color),
    ]:
        ax.errorbar(
            hist.centers,
            hist.values,
            yerr=hist.errors,
            fmt="o",
            linestyle="none",
            markersize=7.0,
            markeredgewidth=1.0,
            elinewidth=1.5,
            capsize=2.5,
            color=color,
            label=label,
            zorder=5,
        )

    for hist, label, color, linewidth in [
        (mc_hist, "MC default", default_mc_color, 2.3),
        (mc_weighted_nominal, "MC weighted nominal", nominal_color, 2.0),
        (mc_weighted_up, "MC weighted up", up_color, 2.0),
        (mc_weighted_down, "MC weighted down", down_color, 2.0),
    ]:
        ax.stairs(hist.values, hist.edges, color=color, linewidth=linewidth, label=label)

    y_range = auto_range([
        (data_nominal_hist.values, data_nominal_hist.errors, None),
        (data_up_hist.values, data_up_hist.errors, None),
        (data_down_hist.values, data_down_hist.errors, None),
        (mc_hist.values, None, None),
        (mc_weighted_nominal.values, None, None),
        (mc_weighted_up.values, None, None),
        (mc_weighted_down.values, None, None),
    ], padding=0.12)

    ax.set_xlim(edges[0], edges[-1])
    if y_range is not None:
        ax.set_ylim(max(0.0, y_range[0]), y_range[1])

    ax.set_xlabel(r"Number of true interactions $\mu$")
    ax.set_ylabel("Normalized entries")

    handles, labels = ax.get_legend_handles_labels()
    side_ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
        frameon=False,
        fontsize=15,
    )

    draw_cms_label(ax, cms_label=cms_label, lumi=lumi, com=com, data=True)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.94)

    overlay_output = plot_dir / f"{output_stem}_pileup_overlay.{plot_format}"
    save_figure(fig, overlay_output, dpi=200)
    print(f"Wrote: {overlay_output}")

    # ------------------------------------------------------------------
    # Pileup correction factors
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 7.2))

    ax.stairs(weights_nominal, edges, color=nominal_color, linewidth=2.0, label="Nominal")
    ax.stairs(weights_up, edges, color=up_color, linewidth=2.0, label="Up")
    ax.stairs(weights_down, edges, color=down_color, linewidth=2.0, label="Down")
    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="Unity")

    useful = np.isfinite(mc_hist.values) & (mc_hist.values > 1e-6 * np.nanmax(mc_hist.values))
    weight_range = auto_range([
        (weights_nominal, None, useful),
        (weights_up, None, useful),
        (weights_down, None, useful),
        (np.ones_like(weights_nominal), None, useful),
    ], include_errors=False, padding=0.15)

    ax.set_xlim(edges[0], edges[-1])
    if weight_range is not None:
        ax.set_ylim(*weight_range)

    ax.set_xlabel(r"Number of true interactions $\mu$")
    ax.set_ylabel("Pileup correction factor")
    ax.legend(frameon=False, fontsize="small")

    draw_cms_label(ax, cms_label=cms_label, lumi=lumi, com=com, data=True)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.94)

    correction_output = plot_dir / f"{output_stem}_correction_factors.{plot_format}"
    save_figure(fig, correction_output, dpi=200)
    print(f"Wrote: {correction_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Create CMS-style correctionlib pileup weights JSON"
    )

    parser.add_argument("--data-nominal", required=True, help="pileupCalc ROOT file for nominal data pileup")
    parser.add_argument("--data-up", required=True, help="pileupCalc ROOT file for pileup up variation")
    parser.add_argument("--data-down", required=True, help="pileupCalc ROOT file for pileup down variation")
    parser.add_argument("--data-hist", default="pileup", help="Histogram name in the data ROOT files")
    parser.add_argument("--mc", required=True, help="ROOT file containing the MC pileup histogram")
    parser.add_argument("--mc-hist", default="pileup", help="Histogram name in the MC ROOT file")
    parser.add_argument("--name", default="pileup", help="Correction name stored in the JSON")
    parser.add_argument("-o", "--output", default="puWeights.json", help="Output correctionlib JSON")

    parser.add_argument("--make-plots", action="store_true", help="Produce pileup overlay and correction-factor plots")
    parser.add_argument("--plot-dir", default="plots", help="Directory for diagnostic plots")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf"], help="Diagnostic plot format")
    parser.add_argument("--cms-label", default="Preliminary", help="CMS label used on diagnostic plots")
    parser.add_argument("--lumi", type=float, default=None, help="Integrated luminosity in fb^-1 shown on plots")
    parser.add_argument("--com", type=float, default=13.6, help="Center-of-mass energy in TeV shown on plots")

    args = parser.parse_args()

    print("Reading nominal data pileup...")
    data_nominal, edges, data_nominal_err = read_histogram(args.data_nominal, args.data_hist)

    print("Reading up data pileup...")
    data_up, edges_up, data_up_err = read_histogram(args.data_up, args.data_hist)

    print("Reading down data pileup...")
    data_down, edges_down, data_down_err = read_histogram(args.data_down, args.data_hist)

    print("Reading MC pileup...")
    mc, mc_edges, mc_err = read_histogram(args.mc, args.mc_hist)

    if not np.array_equal(edges, edges_up):
        raise RuntimeError("Nominal/up data binning does not match")
    if not np.array_equal(edges, edges_down):
        raise RuntimeError("Nominal/down data binning does not match")
    if not np.array_equal(edges, mc_edges):
        raise RuntimeError("Data/MC pileup binning does not match")

    weights_nominal = make_weights(data_nominal, mc)
    weights_up = make_weights(data_up, mc)
    weights_down = make_weights(data_down, mc)

    if args.make_plots:
        print()
        print("Producing pileup diagnostic plots...")
        make_diagnostic_plots(
            edges,
            data_nominal,
            data_nominal_err,
            data_up,
            data_up_err,
            data_down,
            data_down_err,
            mc,
            mc_err,
            weights_nominal,
            weights_up,
            weights_down,
            Path(args.output).stem,
            args.plot_dir,
            args.plot_format,
            args.cms_label,
            args.lumi,
            args.com,
        )

    correction = {
        "schema_version": 2,
        "corrections": [
            {
                "name": args.name,
                "version": 0,
                "inputs": [
                    {
                        "name": "NumTrueInteractions",
                        "type": "real",
                        "description": "Number of true interactions",
                    },
                    {
                        "name": "weights",
                        "type": "string",
                        "description": "nominal, up, or down",
                    },
                ],
                "output": {
                    "name": "weight",
                    "type": "real",
                    "description": "Event weight for pileup reweighting",
                },
                "data": {
                    "nodetype": "category",
                    "input": "weights",
                    "content": [
                        {"key": "nominal", "value": make_binning(edges, weights_nominal)},
                        {"key": "up", "value": make_binning(edges, weights_up)},
                        {"key": "down", "value": make_binning(edges, weights_down)},
                    ],
                },
            }
        ],
    }

    validate_correction(correction)

    if HAVE_CORRECTIONLIB:
        print()
        print("Validating with correctionlib...")
        cset = schemav2.CorrectionSet.model_validate(correction)
    else:
        print()
        print("WARNING: correctionlib Python validation skipped.")
        print(f"Reason: {CORRECTIONLIB_IMPORT_ERROR}")
        print("The JSON will still be written.")

    with open(args.output, "w") as f:
        json.dump(correction, f, indent=2)

    print()
    print("=" * 60)
    print("Correction summary")
    print("=" * 60)

    if HAVE_CORRECTIONLIB:
        rich_print(cset)
    else:
        corr = correction["corrections"][0]
        summary_edges = corr["data"]["content"][0]["value"]["edges"]

        print("CorrectionSet (schema v2)")
        print("No description")
        print("📂")
        print(f"└── 📈 {corr['name']} (v{corr['version']})")
        print("    No description")
        print("    Node counts: Category: 1, Binning: 3")
        print()
        print("    NumTrueInteractions (real)")
        print("    Number of true interactions")
        print(f"    Range: [{summary_edges[0]}, {summary_edges[-1]}), overflow ok")
        print()
        print("    weights (string)")
        print("    nominal, up, or down")
        print("    Values: down, nominal, up")
        print()
        print("    weight (real)")
        print("    Event weight for pileup reweighting")

    print()
    print("=" * 60)
    print(f"JSON contents: {args.output}")
    print("=" * 60)
    print()
    print(json.dumps(correction, indent=2))

    print()
    print("=" * 60)
    print(f"Output written to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
