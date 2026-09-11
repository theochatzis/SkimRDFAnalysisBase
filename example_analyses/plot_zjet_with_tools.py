#!/usr/bin/env python3

"""
Example Z+jet plotting using the generic SkimRDFAnalysisBase plotting package.

Examples
--------

Data/MC stack plus MC efficiency/purity:

  python3 example_analyses/plot_zjet_with_tools.py \
      --data zjet_outputs/Muon2025G.root \
      --mc DY=zjet_outputs/DYto2L.root \
      --mc TT=zjet_outputs/TT.root \
      --mc WJets=zjet_outputs/WJets.root \
      --region zjet \
      --output-dir zjet_plots \
      --efficiency-process DY

Optional per-process normalization:

      --mc-scale DY=1.23 \
      --mc-scale TT=0.42

The scale factors are applied before stacking. If the analysis histograms
already contain luminosity-normalized weights, leave all scales at 1.
"""

#!/usr/bin/env python3

import argparse
import os
import sys

from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path


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


from plotting import (
    Hist1D,
    RootFileReader,
    efficiency_graph,
    hist_to_graph,
    plot_graphs,
    plot_hist1d_data_mc_stack,
)


CORE_DISTRIBUTIONS = OrderedDict([
    ("Z_mass", (r"$m_{\mu\mu}$ [GeV]", "Events")),
    ("Z_pt", (r"$p_{T}^{Z}$ [GeV]", "Events")),
    ("Probe_pt", (r"$p_{T}^{jet}$ [GeV]", "Events")),
    ("Probe_eta", (r"$\eta^{jet}$", "Events")),
    ("Probe_chHEF", (r"chHEF", "Events")),
    ("MET_pt", (r"$p_{T}^{miss}$ [GeV]", "Events")),
    ("alpha", (r"$\alpha$", "Events")),
    ("nGoodJets", (r"$N_{jets}$", "Events")),
    ("Jet2_pt", (r"$p_{T}^{jet2}$ [GeV]", "Events")),
    ("dPhi_ZProbe", (r"|$\Delta\phi(Z, jet)$|", "Events")),
])

ETA_CATEGORIES = OrderedDict([
    ("Incl", "Inclusive"),
    ("HB", "HB"),
    ("HE1", "HE1"),
    ("HE2", "HE2"),
    ("HF", "HF"),
])

PT_CATEGORIES = OrderedDict([
    ("pt15to30", "15 < pT < 30 GeV"),
    ("pt30to60", "30 < pT < 60 GeV"),
    ("pt60to120", "60 < pT < 120 GeV"),
    ("pt120to300", "120 < pT < 300 GeV"),
    ("pt300to1000", "300 < pT < 1000 GeV"),
    ("pt1000plus", "pT > 1000 GeV"),
])


def parse_key_value(items, value_type=str):
    result = OrderedDict()

    for item in items or []:
        if "=" not in item:
            raise ValueError(
                "Expected KEY=VALUE, got '{}'".format(
                    item
                )
            )

        key, value = item.split(
            "=",
            1,
        )

        result[key] = value_type(
            value
        )

    return result


def object_path(region, name):
    region = region.strip("/")

    if region:
        return "{}/{}".format(
            region,
            name,
        )

    return name


def get_hist(reader, region, name):
    obj = reader.get(
        object_path(
            region,
            name,
        )
    )

    if not isinstance(
        obj,
        Hist1D,
    ):
        raise TypeError(
            "{} is not a Hist1D/TProfile-like object."
            .format(
                object_path(
                    region,
                    name,
                )
            )
        )

    return obj


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--mc",
        action="append",
        default=[],
        help=(
            "MC process in PROCESS=FILE form. "
            "Repeat for multiple processes."
        ),
    )

    parser.add_argument(
        "--mc-scale",
        action="append",
        default=[],
        help=(
            "Optional PROCESS=SCALE. "
            "Repeat for multiple processes."
        ),
    )

    parser.add_argument(
        "--efficiency-process",
        default="",
        help=(
            "Which MC process to use for jet efficiency/purity. "
            "Default: first --mc process."
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
        "--normalize-mc-to-data",
        action="store_true",
        help=(
            "Shape-normalize the total MC stack to data. "
            "Do not use this for yield comparisons."
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

    mc_files = parse_key_value(
        args.mc,
        str,
    )

    if not mc_files:
        raise ValueError(
            "At least one --mc PROCESS=FILE is required."
        )

    mc_scales = parse_key_value(
        args.mc_scale,
        float,
    )

    efficiency_process = (
        args.efficiency_process
        or next(
            iter(mc_files)
        )
    )

    if efficiency_process not in mc_files:
        raise ValueError(
            "Efficiency process '{}' was not supplied with --mc."
            .format(
                efficiency_process
            )
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ExitStack() as stack:
        data_reader = stack.enter_context(
            RootFileReader(
                args.data
            )
        )

        mc_readers = OrderedDict([
            (
                process,
                stack.enter_context(
                    RootFileReader(
                        file_path
                    )
                ),
            )
            for process, file_path
            in mc_files.items()
        ])

        # ====================================================
        # Data / MC event-yield stacks
        # ====================================================
        for name, (
            xlabel,
            ylabel,
        ) in CORE_DISTRIBUTIONS.items():

            data_hist = get_hist(
                data_reader,
                args.region,
                name,
            )

            mc_hists = OrderedDict([
                (
                    process,
                    get_hist(
                        reader,
                        args.region,
                        name,
                    ),
                )
                for process, reader
                in mc_readers.items()
            ])

            plot_hist1d_data_mc_stack(
                data_hist,
                mc_hists,
                output_dir
                / "{}_stack.png".format(
                    name
                ),
                labels={
                    process: process
                    for process in mc_hists
                },
                scales=mc_scales,
                xlabel=xlabel,
                ylabel=ylabel,
                normalize_mc_to_data=(
                    args.normalize_mc_to_data
                ),
                legend_outside=(
                    len(mc_hists) >= 4
                ),
                lumi=args.lumi,
                com=args.com,
            )

        # ====================================================
        # MC jet reconstruction efficiency / purity
        # ====================================================
        efficiency_reader = (
            mc_readers[
                efficiency_process
            ]
        )

        efficiency_pt = OrderedDict()
        purity_pt = OrderedDict()

        for category, label in ETA_CATEGORIES.items():
            efficiency_pt[category] = efficiency_graph(
                get_hist(
                    efficiency_reader,
                    args.region,
                    "eff_num_pt_{}".format(
                        category
                    ),
                ),
                get_hist(
                    efficiency_reader,
                    args.region,
                    "eff_den_pt_{}".format(
                        category
                    ),
                ),
                name="eff_{}".format(
                    category
                ),
            )

            purity_pt[category] = efficiency_graph(
                get_hist(
                    efficiency_reader,
                    args.region,
                    "purity_num_pt_{}".format(
                        category
                    ),
                ),
                get_hist(
                    efficiency_reader,
                    args.region,
                    "purity_den_pt_{}".format(
                        category
                    ),
                ),
                name="purity_{}".format(
                    category
                ),
            )

        plot_graphs(
            efficiency_pt,
            output_dir
            / "jet_reco_efficiency_vs_pt.png",
            labels=ETA_CATEGORIES,
            xlabel=r"$p_{T}^{gen jet}$ [GeV]",
            ylabel="Reconstruction efficiency",
            ylim=(0.0, 1.05),
            logx=True,
            legend_outside=True,
            cms_label="Simulation",
            com=args.com,
            title=efficiency_process,
        )

        plot_graphs(
            purity_pt,
            output_dir
            / "jet_reco_purity_vs_pt.png",
            labels=ETA_CATEGORIES,
            xlabel=r"$_{T}^{reco jet}$ [GeV]",
            ylabel="Reconstruction purity",
            ylim=(0.0, 1.05),
            logx=True,
            legend_outside=True,
            cms_label="Simulation",
            com=args.com,
            title=efficiency_process,
        )

        efficiency_eta = OrderedDict()
        purity_eta = OrderedDict()

        for category, label in PT_CATEGORIES.items():
            efficiency_eta[category] = efficiency_graph(
                get_hist(
                    efficiency_reader,
                    args.region,
                    "eff_num_eta_{}".format(
                        category
                    ),
                ),
                get_hist(
                    efficiency_reader,
                    args.region,
                    "eff_den_eta_{}".format(
                        category
                    ),
                ),
                name="eff_eta_{}".format(
                    category
                ),
            )

            purity_eta[category] = efficiency_graph(
                get_hist(
                    efficiency_reader,
                    args.region,
                    "purity_num_eta_{}".format(
                        category
                    ),
                ),
                get_hist(
                    efficiency_reader,
                    args.region,
                    "purity_den_eta_{}".format(
                        category
                    ),
                ),
                name="purity_eta_{}".format(
                    category
                ),
            )

        plot_graphs(
            efficiency_eta,
            output_dir
            / "jet_reco_efficiency_vs_eta.png",
            labels=PT_CATEGORIES,
            xlabel="#eta^{gen jet}",
            ylabel="Reconstruction efficiency",
            ylim=(0.0, 1.05),
            legend_outside=True,
            cms_label="Simulation",
            com=args.com,
            title=efficiency_process,
        )

        plot_graphs(
            purity_eta,
            output_dir
            / "jet_reco_purity_vs_eta.png",
            labels=PT_CATEGORIES,
            xlabel="#eta^{reco jet}",
            ylabel="Reconstruction purity",
            ylim=(0.0, 1.05),
            legend_outside=True,
            cms_label="Simulation",
            com=args.com,
            title=efficiency_process,
        )

        # ====================================================
        # chHEF profiles through the same graph plotter
        # ====================================================
        chhef_pt = OrderedDict()

        for category, label in ETA_CATEGORIES.items():
            chhef_pt[category] = hist_to_graph(
                get_hist(
                    efficiency_reader,
                    args.region,
                    "chHEF_vs_pt_{}".format(
                        category
                    ),
                ),
                drop_empty=True,
            )

        plot_graphs(
            chhef_pt,
            output_dir
            / "chHEF_vs_pt.png",
            labels=ETA_CATEGORIES,
            xlabel=r"$p_{T}^{reco jet}$ [GeV]",
            ylabel="<chHEF>",
            ylim=(0.0, 1.0),
            logx=True,
            legend_outside=True,
            cms_label="Simulation",
            com=args.com,
            title=efficiency_process,
        )

        # ====================================================
        # DB / MPF profiles
        #
        # Profiles are overlaid process-by-process. They are NOT stacked:
        # means from separate TProfiles are not additive.
        # ====================================================
        for category, category_label in ETA_CATEGORIES.items():
            for quantity in (
                "DB",
                "MPF",
            ):
                name = "{}_vs_Zpt_{}".format(
                    quantity,
                    category,
                )

                graphs = OrderedDict()

                graphs["Data"] = hist_to_graph(
                    get_hist(
                        data_reader,
                        args.region,
                        name,
                    ),
                    drop_empty=True,
                )

                for process, reader in mc_readers.items():
                    graphs[process] = hist_to_graph(
                        get_hist(
                            reader,
                            args.region,
                            name,
                        ),
                        drop_empty=True,
                    )

                plot_graphs(
                    graphs,
                    output_dir
                    / "{}_profile_{}.png".format(
                        quantity,
                        category,
                    ),
                    labels={
                        key: key
                        for key in graphs
                    },
                    xlabel=r"$p_{T}^{Z}$ [GeV]",
                    ylabel=quantity,
                    ylim=(0.5, 1.5),
                    logx=True,
                    reference_line=1.0,
                    legend_outside=(
                        len(graphs) >= 4
                    ),
                    cms_label="Preliminary",
                    lumi=args.lumi,
                    com=args.com,
                    data=True,
                    title=category_label,
                )


if __name__ == "__main__":
    main()
