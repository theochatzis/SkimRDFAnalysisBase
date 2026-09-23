#!/usr/bin/env python3

"""
Scale-factor variation checks for the Z+jet example.

Reads the histograms written by

    run_analysis.py ... --weights-defs <weights.yaml> \
                        --histogram-weight-output all

which for every definition <name> produces

    <name>                     every weight, including all scale factors
    <name>_unweighted          baseline weights only (the generator weight)
    <name>_<source>_up/_down   one scale factor varied, the others nominal

and, in the `weights` directory, the weight distributions themselves:

    <source>, <source>_up, <source>_down

For each source in SOURCES and each distribution in DISTRIBUTIONS this writes

    <output-dir>/<source>/<distribution>.png
        nominal, up and down overlaid, with a ratio to nominal underneath

    <output-dir>/<source>/weight.png
        the weight itself, nominal / up / down in one plot

Both lists are defined below and can be overridden on the command line.

Example
-------

  python3 example_analyses/plot_zjet_variations.py \
      --input zjet_example_output/DYTo2L.root \
      --region zjet \
      --output-dir zjet_variation_plots

Restricting either list:

      --sources puWeight \
      --distributions Z_mass,Z_pt,MET_pt
"""

import argparse
import os
import sys

from collections import OrderedDict
from pathlib import Path

import numpy as np


REPO_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if REPO_DIR not in sys.path:
    sys.path.insert(
        0,
        REPO_DIR,
    )


import matplotlib.pyplot as plt
import mplhep as hep

from plotting import RootFileReader
from plotting.plotters import (
    apply_axis_style,
    apply_figure_margins,
    draw_cms_label,
    ratio_axes,
    save_figure,
    style_legend_patches,
    use_hep_style,
)


# ------------------------------------------------------------------
# What to check. Both lists mirror the weights YAML and the histogram
# YAML; anything absent from the input file is skipped with a warning.
# ------------------------------------------------------------------

# Weight sources, as named in the weights YAML. The key is the source name
# used in the histogram suffixes, the value is the axis label for its weight.
SOURCES = OrderedDict([
    ("puWeight", "Pileup weight"),
])

# Distributions to check, as name -> (xlabel, logy).
DISTRIBUTIONS = OrderedDict([
    ("Z_mass", (r"$m_{\mu\mu}$ [GeV]", False)),
    ("Z_pt", (r"$p_{T}^{Z}$ [GeV]", True)),
    ("Probe_pt", (r"$p_{T}^{jet}$ [GeV]", True)),
    ("Probe_eta", (r"$\eta^{jet}$", False)),
    ("MET_pt", (r"$p_{T}^{miss}$ [GeV]", True)),
    ("U_par", (r"$u_{\parallel}$ [GeV]", True)),
    ("U_perp", (r"$u_{\perp}$ [GeV]", True)),
    ("npvsGood", (r"$N_{PV}^{good}$", False)),
    ("rho", (r"$\rho$ [GeV]", False)),
])

# Directory the runner writes the weight distributions into.
WEIGHTS_DIRECTORY = "weights"

# Colours are fixed per role so that every plot reads the same way.
VARIATION_STYLES = OrderedDict([
    ("nominal", ("black", "-", "Nominal (all SFs)")),
    ("up", ("tab:red", "-", "{} up")),
    ("down", ("tab:blue", "-", "{} down")),
    ("unweighted", ("tab:gray", "--", "No SFs (genWeight only)")),
])


def parse_list(raw):
    return [
        item.strip()
        for item in (raw or "").split(",")
        if item.strip()
    ]


def object_path(directory, name):
    directory = (directory or "").strip("/")

    if directory:
        return "{}/{}".format(directory, name)

    return name


def try_hist(reader, directory, name, *, quiet=False):
    """Return a histogram, or None with a warning when it is absent."""
    path = object_path(directory, name)

    try:
        return reader.get(path)
    except (KeyError, TypeError) as error:
        if not quiet:
            print(
                "WARNING: skipping '{}': {}".format(path, error)
            )
        return None


def ratio_to_nominal(variation, nominal):
    """
    Bin-by-bin variation / nominal.

    No uncertainty is attached. The variation and the nominal are the same
    events with different weights, so their statistical fluctuations are
    almost perfectly correlated and an independent-error band here would be
    meaningless - and far larger than the systematic effect being shown.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            nominal.values != 0.0,
            variation.values / np.where(
                nominal.values != 0.0,
                nominal.values,
                1.0,
            ),
            np.nan,
        )


def ratio_limits(ratios, nominal, *, fallback=(0.9, 1.1), widest=(0.5, 1.5)):
    """
    Choose the ratio-panel range from the up/down variations.

    Only bins carrying a reasonable share of the nominal yield are used, so
    that a single almost-empty tail bin - where the ratio is dominated by
    one event - does not set the scale for the whole panel. The result is
    padded, kept at least as wide as `fallback`, and clipped to `widest`.
    """
    usable = []

    threshold = 0.0

    if nominal.values.size:
        positive = nominal.values[nominal.values > 0.0]

        if positive.size:
            threshold = 0.01 * np.max(positive)

    for ratio in ratios:
        mask = np.isfinite(ratio) & (nominal.values > threshold)

        if np.any(mask):
            usable.append(ratio[mask])

    if not usable:
        return fallback

    values = np.concatenate(usable)

    low = float(np.min(values))
    high = float(np.max(values))

    padding = 0.25 * max(high - low, 1e-3)

    low = min(low - padding, fallback[0])
    high = max(high + padding, fallback[1])

    return (
        max(low, widest[0]),
        min(high, widest[1]),
    )


def plot_variation(
    histograms,
    output,
    *,
    xlabel,
    source_label,
    logy=False,
    ylabel="Events",
    ratio_label="Var. / nom.",
    title=None,
    lumi=None,
    com=13.6,
    with_ratio=True,
):
    """
    Overlay nominal, up, down (and optionally the no-SF reference), with the
    ratio to nominal underneath.

    with_ratio=False draws the overlay alone. That is what the weight
    distributions use: a ratio of two weight distributions compares different
    weight values falling in the same bin, which is not a meaningful
    comparison.

    `histograms` is a mapping of role -> Hist1D, where role is one of the
    keys of VARIATION_STYLES. The nominal is required.
    """
    nominal = histograms.get("nominal")

    if nominal is None:
        return False

    use_hep_style()

    if with_ratio:
        fig, ax, rax, _ = ratio_axes(
            figsize=(8.2, 7.6),
        )
    else:
        fig, ax = plt.subplots(
            figsize=(8.2, 7.2),
        )
        rax = None

    variation_ratios = []

    for role, (color, linestyle, label_template) in VARIATION_STYLES.items():
        hist = histograms.get(role)

        if hist is None:
            continue

        label = (
            label_template.format(source_label)
            if "{}" in label_template
            else label_template
        )

        hep.histplot(
            hist.values,
            hist.edges,
            ax=ax,
            histtype="step",
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
            label=label,
        )

        if role == "nominal" or rax is None:
            continue

        ratio = ratio_to_nominal(hist, nominal)

        hep.histplot(
            ratio,
            hist.edges,
            ax=rax,
            histtype="step",
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
        )

        if role in ("up", "down"):
            variation_ratios.append(ratio)

    if logy:
        ax.set_yscale("log")

    ax.set_ylabel(ylabel)

    if rax is None:
        ax.set_xlabel(xlabel)
    else:
        rax.axhline(
            1.0,
            linewidth=1,
            linestyle="--",
            color="black",
        )

        rax.set_xlabel(xlabel)
        rax.set_ylabel(ratio_label)
        rax.set_ylim(
            *ratio_limits(variation_ratios, nominal)
        )

    if title:
        ax.text(
            0.04,
            0.92,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
        )

    style_legend_patches(
        ax.legend(
            frameon=False,
            fontsize="x-small",
        )
    )

    apply_axis_style(
        ax,
        log_minor_labels=(rax is None),
    )

    if rax is not None:
        apply_axis_style(
            rax,
            sci_digits=None,
        )

    draw_cms_label(
        ax,
        # data=False makes mplhep prepend "Simulation" itself, so the label
        # here must not repeat it.
        cms_label="Preliminary",
        lumi=lumi,
        com=com,
        data=False,
    )

    apply_figure_margins(fig)

    save_figure(fig, output)

    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "ROOT file written with --histogram-weight-output all "
            "(normally the MC file)."
        ),
    )

    parser.add_argument(
        "--region",
        default="zjet",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--sources",
        default="",
        help=(
            "Comma-separated weight sources to check. "
            "Default: " + ",".join(SOURCES)
        ),
    )

    parser.add_argument(
        "--distributions",
        default="",
        help=(
            "Comma-separated distributions to check. "
            "Default: " + ",".join(DISTRIBUTIONS)
        ),
    )

    parser.add_argument(
        "--no-unweighted",
        action="store_true",
        help=(
            "Do not overlay the no-scale-factor reference "
            "(<name>_unweighted)."
        ),
    )

    parser.add_argument(
        "--lumi",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--com",
        type=float,
        default=13.6,
    )

    args = parser.parse_args()

    sources = parse_list(args.sources) or list(SOURCES)
    distributions = parse_list(args.distributions) or list(DISTRIBUTIONS)

    output_dir = Path(args.output_dir)

    written = 0
    missing_sources = []

    with RootFileReader(args.input) as reader:
        for source in sources:
            source_label = SOURCES.get(source, source)

            target = output_dir / source
            target.mkdir(parents=True, exist_ok=True)

            # ----------------------------------------------------------
            # The weight itself: nominal, up and down in one plot.
            # ----------------------------------------------------------
            weight_histograms = OrderedDict()

            for role, suffix in (
                ("nominal", ""),
                ("up", "_up"),
                ("down", "_down"),
            ):
                hist = try_hist(
                    reader,
                    WEIGHTS_DIRECTORY,
                    "{}{}".format(source, suffix),
                    quiet=True,
                )

                if hist is not None:
                    weight_histograms[role] = hist

            if "nominal" not in weight_histograms:
                print(
                    "WARNING: no weight histogram '{}' in '{}'; is this "
                    "source configured, and was the file written with "
                    "--weights-defs?".format(
                        source,
                        WEIGHTS_DIRECTORY,
                    )
                )
                missing_sources.append(source)
            else:
                if plot_variation(
                    weight_histograms,
                    target / "weight.png",
                    xlabel=source_label,
                    source_label=source_label,
                    ylabel="Events",
                    logy=True,
                    with_ratio=False,
                    lumi=args.lumi,
                    com=args.com,
                ):
                    written += 1

            # ----------------------------------------------------------
            # The distributions, one plot each.
            # ----------------------------------------------------------
            for name in distributions:
                xlabel, logy = DISTRIBUTIONS.get(name, (name, False))

                roles = OrderedDict([
                    ("nominal", name),
                    ("up", "{}_{}_up".format(name, source)),
                    ("down", "{}_{}_down".format(name, source)),
                ])

                if not args.no_unweighted:
                    roles["unweighted"] = "{}_unweighted".format(name)

                histograms = OrderedDict()

                for role, hist_name in roles.items():
                    hist = try_hist(
                        reader,
                        args.region,
                        hist_name,
                        quiet=(role == "unweighted"),
                    )

                    if hist is not None:
                        histograms[role] = hist

                if "nominal" not in histograms:
                    continue

                if "up" not in histograms and "down" not in histograms:
                    print(
                        "WARNING: no '{}' variations for '{}'; skipping."
                        .format(source, name)
                    )
                    continue

                if plot_variation(
                    histograms,
                    target / "{}.png".format(name),
                    xlabel=xlabel,
                    source_label=source_label,
                    logy=logy,
                    title=name,
                    lumi=args.lumi,
                    com=args.com,
                ):
                    written += 1

    print(
        "\n{} plot(s) written under: {}".format(
            written,
            output_dir,
        )
    )

    if missing_sources:
        print(
            "Sources with no weight histogram: "
            + ", ".join(missing_sources)
        )


if __name__ == "__main__":
    main()
