#!/usr/bin/env python3

"""
Z+jet plotting using the generic SkimRDFAnalysisBase plotting package.

The plots follow example_analyses/zjet_histograms.yaml one-to-one and are
written into subdirectories of --output-dir:

    01_inclusive/            rho, N_PV, Z mass, jet multiplicity, alpha
    02_kinematics/           Z, probe-jet, muon, MET and recoil distributions
    03_eta_in_Zpt/           eta distributions in regions of pT(Z)
    04_met_in_probe_eta/     MET and recoil in regions of probe-jet |eta|
    05_energy_fractions/     probe-jet PF fractions, in |eta| and pT(Z) regions
    06_response/             DB and MPF versus pT(Z), in |eta| regions
    07_recoil_components/    u_par / u_perp mean and resolution versus pT(Z),
                             raw and corrected for the recoil scale
    08_recoil_response/      hadronic-recoil response versus pT(Z)
    09_mc_jets/              MC efficiency, purity, response and resolution
    10_pu_dependence/        the same recoil quantities versus N_PV, split by
                             probe-jet |eta| and by pT(Z)

Examples
--------

Data/MC stacks plus the MC jet studies:

  python3 example_analyses/plot_zjet_with_tools.py \
      --data zjet_example_output/Muon2025G.root \
      --mc DY=zjet_example_output/DYTo2L.root \
      --region zjet \
      --output-dir zjet_plots \
      --efficiency-process DY

Optional per-process normalization:

      --mc-scale DY=1.23 \
      --mc-scale TT=0.42

The scale factors are applied before stacking. If the analysis histograms
already contain luminosity-normalized weights, leave all scales at 1.
"""

import argparse
import os
import sys

from collections import OrderedDict
from contextlib import ExitStack
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


from plotting import (
    Graph1D,
    Hist1D,
    RootFileReader,
    efficiency_graph,
    hist_to_graph,
    plot_graphs,
    plot_hist1d_data_mc_stack,
)


# ------------------------------------------------------------------
# Categories. These mirror zjet_rdf_definition.py.
# ------------------------------------------------------------------

ETA_CATEGORIES = OrderedDict([
    ("Incl", "Inclusive"),
    ("HB", "HB"),
    ("HE1", "HE1"),
    ("HE2", "HE2"),
    ("HF", "HF"),
])

ZPT_CATEGORIES = OrderedDict([
    ("Incl", "all $p_{T}^{Z}$"),
    ("zpt0to20", r"$p_{T}^{Z} < 20$ GeV"),
    ("zpt20to40", r"$20 < p_{T}^{Z} < 40$ GeV"),
    ("zpt40to80", r"$40 < p_{T}^{Z} < 80$ GeV"),
    ("zpt80to150", r"$80 < p_{T}^{Z} < 150$ GeV"),
    ("zpt150to300", r"$150 < p_{T}^{Z} < 300$ GeV"),
    ("zpt300plus", r"$p_{T}^{Z} > 300$ GeV"),
])

JETPT_CATEGORIES = OrderedDict([
    ("pt15to30", r"$15 < p_{T} < 30$ GeV"),
    ("pt30to60", r"$30 < p_{T} < 60$ GeV"),
    ("pt60to120", r"$60 < p_{T} < 120$ GeV"),
    ("pt120to300", r"$120 < p_{T} < 300$ GeV"),
    ("pt300to1000", r"$300 < p_{T} < 1000$ GeV"),
    ("pt1000plus", r"$p_{T} > 1000$ GeV"),
])

ENERGY_FRACTIONS = OrderedDict([
    ("chHEF", "Charged hadron energy fraction"),
    ("neHEF", "Neutral hadron energy fraction"),
    ("chEmEF", "Charged EM energy fraction"),
    ("neEmEF", "Neutral EM energy fraction"),
    ("muEF", "Muon energy fraction"),
])


# ------------------------------------------------------------------
# Distribution catalogues: name -> (xlabel, ylabel, logy)
# ------------------------------------------------------------------

INCLUSIVE_DISTRIBUTIONS = OrderedDict([
    ("rho", (r"$\rho$ [GeV]", "Events", False)),
    ("npvs", (r"$N_{PV}$", "Events", False)),
    ("npvsGood", (r"$N_{PV}^{good}$", "Events", False)),
    ("Z_mass", (r"$m_{\mu\mu}$ [GeV]", "Events", False)),
    ("nGoodJets", (r"$N_{jets}$", "Events", True)),
    ("alpha", (r"$\alpha = p_{T}^{jet2}/p_{T}^{Z}$", "Events", False)),
    ("Jet2_pt", (r"$p_{T}^{jet2}$ [GeV]", "Events", True)),
])

KINEMATIC_DISTRIBUTIONS = OrderedDict([
    ("Z_pt", (r"$p_{T}^{Z}$ [GeV]", "Events", True)),
    ("Z_eta", (r"$\eta^{Z}$", "Events", False)),
    ("Z_phi", (r"$\phi^{Z}$", "Events", False)),
    ("Probe_pt", (r"$p_{T}^{jet}$ [GeV]", "Events", True)),
    ("Probe_eta", (r"$\eta^{jet}$", "Events", False)),
    ("Probe_phi", (r"$\phi^{jet}$", "Events", False)),
    ("Mu1_pt", (r"$p_{T}^{\mu_{1}}$ [GeV]", "Events", True)),
    ("Mu1_eta", (r"$\eta^{\mu_{1}}$", "Events", False)),
    ("Mu1_phi", (r"$\phi^{\mu_{1}}$", "Events", False)),
    ("Mu2_pt", (r"$p_{T}^{\mu_{2}}$ [GeV]", "Events", True)),
    ("Mu2_eta", (r"$\eta^{\mu_{2}}$", "Events", False)),
    ("Mu2_phi", (r"$\phi^{\mu_{2}}$", "Events", False)),
    ("dPhi_mumu", (r"$|\Delta\phi(\mu_{1}, \mu_{2})|$", "Events", False)),
    ("dPhi_ZProbe", (r"$|\Delta\phi(Z, jet)|$", "Events", True)),
    ("MET_pt", (r"$p_{T}^{miss}$ [GeV]", "Events", True)),
    ("MET_phi", (r"$\phi^{miss}$", "Events", False)),
    ("U_pt", (r"$|u|$ [GeV]", "Events", True)),
    ("U_par", (r"$u_{\parallel}$ [GeV]", "Events", True)),
    ("U_perp", (r"$u_{\perp}$ [GeV]", "Events", True)),
])

# Eta distributions booked in regions of pT(Z).
ETA_IN_ZPT_VARIABLES = OrderedDict([
    ("Z_eta", r"$\eta^{Z}$"),
    ("Probe_eta", r"$\eta^{jet}$"),
    ("Mu1_eta", r"$\eta^{\mu_{1}}$"),
    ("Mu2_eta", r"$\eta^{\mu_{2}}$"),
])

# MET and hadronic-recoil distributions booked in regions of probe-jet |eta|.
MET_IN_ETA_VARIABLES = OrderedDict([
    ("MET_pt", (r"$p_{T}^{miss}$ [GeV]", True)),
    ("MET_phi", (r"$\phi^{miss}$", False)),
    ("U_par", (r"$u_{\parallel}$ [GeV]", True)),
    ("U_perp", (r"$u_{\perp}$ [GeV]", True)),
])

# Hadronic-recoil mean/mean-of-squares profile pairs. Each entry is
#   (mean pattern, mean-of-squares pattern, mean label, sigma label, sigma/R label)
# and the patterns take one format argument: the axis-and-category suffix, for
# instance "Zpt_HB" or "Npv_zpt_zpt40to80".
RECOIL_PROFILE_PAIRS = OrderedDict([
    (
        "u_par",
        (
            "Upar_vs_{}",
            "UparSq_vs_{}",
            r"$\langle u_{\parallel} \rangle$ [GeV]",
            r"$\sigma(u_{\parallel})$ [GeV]",
            r"$\sigma(u_{\parallel})\,/\,R$ [GeV]",
        ),
    ),
    (
        "u_perp",
        (
            "Uperp_vs_{}",
            "UperpSq_vs_{}",
            r"$\langle u_{\perp} \rangle$ [GeV]",
            r"$\sigma(u_{\perp})$ [GeV]",
            r"$\sigma(u_{\perp})\,/\,R$ [GeV]",
        ),
    ),
])

# The response is not booked. It is formed bin by bin from the <u_par> and
# <pT(Z)> profiles over the same binning:
#
#     R = -<u_par> / <pT(Z)>
#
# rather than as a profile of the per-event ratio, which diverges as
# pT(Z) -> 0 and so is unusable now that the region does not cut on pT(Z).
RECOIL_UPAR_PATTERN = "Upar_vs_{}"
RECOIL_ZPT_PATTERN = "Zpt_vs_{}"

RECOIL_RESPONSE_LABEL = (
    r"$R = -\langle u_{\parallel} \rangle\,/\,\langle p_{T}^{Z} \rangle$"
)


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


def try_hist(reader, region, name):
    """Return a histogram, or None with a warning when it is absent."""
    try:
        return get_hist(
            reader,
            region,
            name,
        )
    except (KeyError, TypeError) as error:
        print(
            "WARNING: skipping '{}': {}".format(
                name,
                error,
            )
        )
        return None


def response_arrays(upar_hist, zpt_hist):
    """
    Recoil response per bin, R = -<u_par> / <pT(Z)>, with its uncertainty.

    Both inputs are TProfiles over the same binning and carrying the same
    weight, so the ratio is taken bin by bin. Bins where <pT(Z)> is not
    positive carry no scale information and come back as NaN, which every
    consumer masks away.

    The two profiles are strongly correlated - u_par is essentially -pT(Z) -
    so adding their relative errors in quadrature overestimates the error on
    R. That is the conservative direction, and the alternative needs a
    covariance the profiles do not store.
    """
    if upar_hist is None or zpt_hist is None:
        return None, None

    upar = upar_hist.values
    zpt = zpt_hist.values

    if len(upar) != len(zpt):
        raise ValueError(
            "Response profile pair '{}' / '{}' has incompatible binning."
            .format(
                upar_hist.name,
                zpt_hist.name,
            )
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        response = np.where(zpt > 0.0, -upar / zpt, np.nan)

        relative = np.zeros_like(response)

        for hist, values in (
            (upar_hist, upar),
            (zpt_hist, zpt),
        ):
            if hist.errors is None:
                continue

            term = np.where(
                np.isfinite(hist.errors) & (values != 0.0),
                hist.errors / np.abs(np.where(values != 0.0, values, 1.0)),
                0.0,
            )

            relative = relative + term ** 2

        response_error = np.abs(response) * np.sqrt(relative)

    return response, response_error


def resolution_graph(
    mean_hist,
    mean_square_hist,
    *,
    name="resolution",
    scale=None,
):
    """
    Spread of a quantity from its mean and mean-of-squares TProfiles.

        sigma = sqrt(<x^2> - <x>^2)

    A default TProfile stores the error on the mean, sigma / sqrt(N_eff), so
    the uncertainty on the spread itself, sigma / sqrt(2 N_eff), is simply
    that error divided by sqrt(2). Bins with a vanishing error hold at most
    one effective entry and carry no spread information, so they are dropped.

    With scale, a (values, errors) pair over the same binning, the spread is
    divided by it bin by bin and the two relative uncertainties are added in
    quadrature. For the hadronic recoil that pair is the response R from
    response_arrays: dividing by it removes the part of the apparent
    resolution that is really a mismeasured recoil scale, so that different
    responses can be compared like for like.
    """
    if mean_hist is None or mean_square_hist is None:
        return None

    mean = mean_hist.values
    mean_square = mean_square_hist.values

    if mean_hist.errors is None:
        return None

    mean_error = mean_hist.errors

    if len(mean) != len(mean_square):
        raise ValueError(
            "Profile pair '{}' / '{}' has incompatible binning."
            .format(
                mean_hist.name,
                mean_square_hist.name,
            )
        )

    variance = mean_square - mean ** 2

    mask = (
        np.isfinite(variance)
        & np.isfinite(mean_error)
        & (mean_error > 0.0)
        & (variance > 0.0)
    )

    if scale is not None:
        scale_values, scale_errors = scale

        if scale_values is None:
            return None

        if len(scale_values) != len(mean):
            raise ValueError(
                "Scale array does not share the binning of '{}'."
                .format(
                    mean_hist.name,
                )
            )

        mask = (
            mask
            & np.isfinite(scale_values)
            & (scale_values > 0.0)
        )

    sigma = np.sqrt(variance[mask])
    sigma_error = mean_error[mask] / np.sqrt(2.0)

    if scale is not None:
        scale_values, scale_errors = scale

        values = scale_values[mask]

        relative_error = sigma_error / sigma

        if scale_errors is not None:
            relative_error = np.sqrt(
                relative_error ** 2
                + np.where(
                    np.isfinite(scale_errors[mask]),
                    (scale_errors[mask] / values) ** 2,
                    0.0,
                )
            )

        sigma = sigma / values
        sigma_error = relative_error * sigma

    half_width = 0.5 * mean_hist.widths

    return Graph1D(
        x=mean_hist.centers[mask],
        y=sigma,
        xerr_low=half_width[mask],
        xerr_high=half_width[mask],
        yerr_low=sigma_error,
        yerr_high=sigma_error,
        name=name,
        title=mean_hist.title,
        xlabel=mean_hist.xlabel,
    )


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
            "Which MC process to use for the jet efficiency, purity, "
            "response and resolution. Default: first --mc process."
        ),
    )

    parser.add_argument(
        "--efficiency-tolerance",
        type=float,
        default=1.0,
        help=(
            "Maximum propagated numerator-minus-denominator pull, in sigma, "
            "accepted before capping an efficiency or purity at 1.0."
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

    def directory(*parts):
        path = output_dir.joinpath(*parts)
        path.mkdir(
            parents=True,
            exist_ok=True,
        )
        return path

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
            for process, file_path in mc_files.items()
        ])

        # ------------------------------------------------------------
        # Shared helpers
        # ------------------------------------------------------------

        def stack_plot(
            name,
            output,
            xlabel,
            ylabel="Events",
            logy=False,
            title=None,
        ):
            """Data versus the MC stack for one histogram name."""
            data_hist = try_hist(
                data_reader,
                args.region,
                name,
            )

            if data_hist is None:
                return

            mc_hists = OrderedDict()

            for process, reader in mc_readers.items():
                mc_hist = try_hist(
                    reader,
                    args.region,
                    name,
                )

                if mc_hist is None:
                    return

                mc_hists[process] = mc_hist

            plot_hist1d_data_mc_stack(
                data_hist,
                mc_hists,
                output,
                labels={
                    process: process
                    for process in mc_hists
                },
                scales=mc_scales,
                xlabel=xlabel,
                ylabel=ylabel,
                logy=logy,
                normalize_mc_to_data=args.normalize_mc_to_data,
                legend_outside=(
                    len(mc_hists) >= 4
                ),
                lumi=args.lumi,
                com=args.com,
                title=title,
            )

        def profile_graphs(name):
            """Data and every MC process for one profile, as Graph1D objects."""
            graphs = OrderedDict()

            data_hist = try_hist(
                data_reader,
                args.region,
                name,
            )

            if data_hist is None:
                return None

            graphs["Data"] = hist_to_graph(
                data_hist,
                drop_empty=True,
            )

            for process, reader in mc_readers.items():
                mc_hist = try_hist(
                    reader,
                    args.region,
                    name,
                )

                if mc_hist is None:
                    return None

                graphs[process] = hist_to_graph(
                    mc_hist,
                    drop_empty=True,
                )

            return graphs

        def all_readers():
            readers = [("Data", data_reader)]
            readers.extend(mc_readers.items())
            return readers

        def process_response(reader, suffix):
            """R = -<u_par>/<pT(Z)> for one process, as (values, errors)."""
            return response_arrays(
                try_hist(
                    reader,
                    args.region,
                    RECOIL_UPAR_PATTERN.format(suffix),
                ),
                try_hist(
                    reader,
                    args.region,
                    RECOIL_ZPT_PATTERN.format(suffix),
                ),
            )

        def response_graphs(suffix, xlabel):
            """Recoil response versus one axis, for data and every MC process."""
            graphs = OrderedDict()

            for key, reader in all_readers():
                values, errors = process_response(reader, suffix)

                if values is None:
                    return None

                reference = try_hist(
                    reader,
                    args.region,
                    RECOIL_UPAR_PATTERN.format(suffix),
                )

                mask = np.isfinite(values)

                if not mask.any():
                    return None

                half_width = 0.5 * reference.widths

                graphs[key] = Graph1D(
                    x=reference.centers[mask],
                    y=values[mask],
                    xerr_low=half_width[mask],
                    xerr_high=half_width[mask],
                    yerr_low=errors[mask],
                    yerr_high=errors[mask],
                    name="response_{}_{}".format(suffix, key),
                    title=reference.title,
                    xlabel=xlabel,
                )

            return graphs

        def resolution_graphs(mean_name, square_name, scale_suffix=None):
            """
            Data and every MC process for one mean/mean-of-squares pair.

            With scale_suffix, each spread is divided by the recoil response
            of the same process over the same binning, which is how the
            scale-corrected resolutions are built.
            """
            graphs = OrderedDict()

            for key, reader in all_readers():
                mean_hist = try_hist(
                    reader,
                    args.region,
                    mean_name,
                )
                square_hist = try_hist(
                    reader,
                    args.region,
                    square_name,
                )

                scale = None

                if scale_suffix is not None:
                    scale = process_response(reader, scale_suffix)

                    if scale[0] is None:
                        return None

                graph = resolution_graph(
                    mean_hist,
                    square_hist,
                    name="{}_{}".format(
                        mean_name,
                        key,
                    ),
                    scale=scale,
                )

                if graph is None:
                    return None

                graphs[key] = graph

            return graphs

        def graph_plot(
            graphs,
            output,
            xlabel,
            ylabel,
            ylim=None,
            logx=True,
            reference_line=None,
            title=None,
        ):
            if not graphs:
                return

            plot_graphs(
                graphs,
                output,
                labels={
                    key: key
                    for key in graphs
                },
                xlabel=xlabel,
                ylabel=ylabel,
                ylim=ylim,
                logx=logx,
                reference_line=reference_line,
                legend_outside=(
                    len(graphs) >= 4
                ),
                cms_label="Preliminary",
                lumi=args.lumi,
                com=args.com,
                data=True,
                title=title,
            )

        # ============================================================
        # 01 Inclusive event-level distributions
        # ============================================================
        target = directory("01_inclusive")

        for name, (
            xlabel,
            ylabel,
            logy,
        ) in INCLUSIVE_DISTRIBUTIONS.items():
            stack_plot(
                name,
                target / "{}.png".format(name),
                xlabel,
                ylabel=ylabel,
                logy=logy,
            )

        # ============================================================
        # 02 Inclusive kinematics
        # ============================================================
        target = directory("02_kinematics")

        for name, (
            xlabel,
            ylabel,
            logy,
        ) in KINEMATIC_DISTRIBUTIONS.items():
            stack_plot(
                name,
                target / "{}.png".format(name),
                xlabel,
                ylabel=ylabel,
                logy=logy,
            )

        # ============================================================
        # 03 Eta distributions in regions of pT(Z)
        # ============================================================
        for variable, xlabel in ETA_IN_ZPT_VARIABLES.items():
            target = directory("03_eta_in_Zpt", variable)

            for category, label in ZPT_CATEGORIES.items():
                stack_plot(
                    "{}_zpt_{}".format(variable, category),
                    target / "{}_{}.png".format(variable, category),
                    xlabel,
                    title=label,
                )

        # ============================================================
        # 04 MET distributions in regions of probe-jet |eta|
        # ============================================================
        for variable, (xlabel, logy) in MET_IN_ETA_VARIABLES.items():
            target = directory("04_met_in_probe_eta", variable)

            for category, label in ETA_CATEGORIES.items():
                stack_plot(
                    "{}_eta_{}".format(variable, category),
                    target / "{}_{}.png".format(variable, category),
                    xlabel,
                    logy=logy,
                    title=label,
                )

        # ============================================================
        # 05 Probe-jet energy fractions
        # ============================================================
        for fraction, xlabel in ENERGY_FRACTIONS.items():
            target = directory("05_energy_fractions", "eta", fraction)

            for category, label in ETA_CATEGORIES.items():
                stack_plot(
                    "Probe_{}_eta_{}".format(fraction, category),
                    target / "{}_{}.png".format(fraction, category),
                    xlabel,
                    title=label,
                )

            target = directory("05_energy_fractions", "Zpt", fraction)

            for category, label in ZPT_CATEGORIES.items():
                stack_plot(
                    "Probe_{}_zpt_{}".format(fraction, category),
                    target / "{}_{}.png".format(fraction, category),
                    xlabel,
                    title=label,
                )

        # ============================================================
        # 06 DB / MPF response versus pT(Z)
        #
        # Profiles are overlaid process-by-process. They are NOT stacked:
        # means from separate TProfiles are not additive.
        # ============================================================
        target = directory("06_response")

        for category, label in ETA_CATEGORIES.items():
            for quantity in ("DB", "MPF"):
                graph_plot(
                    profile_graphs(
                        "{}_vs_Zpt_{}".format(quantity, category)
                    ),
                    target / "{}_{}.png".format(quantity, category),
                    r"$p_{T}^{Z}$ [GeV]",
                    quantity,
                    ylim=(0.5, 1.5),
                    reference_line=1.0,
                    title=label,
                )

        # ============================================================
        # 07 Parallel / transverse hadronic recoil versus pT(Z)
        #
        # The mean of each quantity is its bias and sits at zero; the spread
        # is the resolution, shown both raw and divided by the recoil
        # response of the same process.
        # ============================================================
        def recoil_block(
            target_parts,
            suffix_pattern,
            categories,
            xlabel,
            logx,
            mean_reference=None,
        ):
            """
            One full recoil block against a chosen axis.

            Per category: <u_par> and <u_perp>, their spreads, and the same
            spreads divided by that process's response.
            """
            for variable, (
                mean_pattern,
                square_pattern,
                mean_label,
                resolution_label,
                corrected_label,
            ) in RECOIL_PROFILE_PAIRS.items():
                target = directory(*target_parts, variable)

                for category, label in categories.items():
                    suffix = suffix_pattern.format(category)

                    graph_plot(
                        profile_graphs(mean_pattern.format(suffix)),
                        target / "{}_mean_{}.png".format(variable, category),
                        xlabel,
                        mean_label,
                        logx=logx,
                        reference_line=mean_reference,
                        title=label,
                    )

                    graph_plot(
                        resolution_graphs(
                            mean_pattern.format(suffix),
                            square_pattern.format(suffix),
                        ),
                        target / "{}_resolution_{}.png".format(
                            variable, category
                        ),
                        xlabel,
                        resolution_label,
                        logx=logx,
                        title=label,
                    )

                    graph_plot(
                        resolution_graphs(
                            mean_pattern.format(suffix),
                            square_pattern.format(suffix),
                            suffix,
                        ),
                        target / "{}_resolution_corrected_{}.png".format(
                            variable, category
                        ),
                        xlabel,
                        corrected_label,
                        logx=logx,
                        title=label,
                    )

        recoil_block(
            ("07_recoil_components",),
            "Zpt_{}",
            ETA_CATEGORIES,
            r"$p_{T}^{Z}$ [GeV]",
            True,
        )

        # ============================================================
        # 08 Hadronic-recoil response versus pT(Z)
        # ============================================================
        target = directory("08_recoil_response")

        for category, label in ETA_CATEGORIES.items():
            graph_plot(
                response_graphs(
                    "Zpt_{}".format(category),
                    r"$p_{T}^{Z}$ [GeV]",
                ),
                target / "recoil_response_{}.png".format(category),
                r"$p_{T}^{Z}$ [GeV]",
                RECOIL_RESPONSE_LABEL,
                ylim=(0.0, 1.5),
                reference_line=1.0,
                title=label,
            )

        # ============================================================
        # 09 MC jet efficiency / purity / response / resolution
        # ============================================================
        efficiency_reader = mc_readers[efficiency_process]

        def mc_graphs(numerator_pattern, denominator_pattern, categories):
            graphs = OrderedDict()

            for category in categories:
                numerator = try_hist(
                    efficiency_reader,
                    args.region,
                    numerator_pattern.format(category),
                )
                denominator = try_hist(
                    efficiency_reader,
                    args.region,
                    denominator_pattern.format(category),
                )

                if numerator is None or denominator is None:
                    continue

                graphs[category] = efficiency_graph(
                    numerator,
                    denominator,
                    name="{}_{}".format(
                        numerator_pattern.format(category),
                        category,
                    ),
                    tolerance=args.efficiency_tolerance,
                )

            return graphs

        def mc_plot(graphs, output, labels, xlabel, ylabel, **kwargs):
            if not graphs:
                return

            plot_graphs(
                graphs,
                output,
                labels=labels,
                xlabel=xlabel,
                ylabel=ylabel,
                legend_outside=True,
                # plot_graphs draws the simulation tag from data=False, so the
                # label itself must not repeat the word "Simulation".
                cms_label="Preliminary",
                com=args.com,
                title=efficiency_process,
                **kwargs
            )

        # --- efficiency -------------------------------------------
        target = directory("09_mc_jets", "efficiency")

        mc_plot(
            mc_graphs("eff_num_pt_{}", "eff_den_pt_{}", ETA_CATEGORIES),
            target / "efficiency_vs_pt.png",
            ETA_CATEGORIES,
            r"$p_{T}^{gen\ jet}$ [GeV]",
            "Reconstruction efficiency",
            ylim=(0.0, 1.05),
            logx=True,
        )

        mc_plot(
            mc_graphs("eff_num_eta_{}", "eff_den_eta_{}", JETPT_CATEGORIES),
            target / "efficiency_vs_eta.png",
            JETPT_CATEGORIES,
            r"$\eta^{gen\ jet}$",
            "Reconstruction efficiency",
            ylim=(0.0, 1.05),
        )

        # --- purity -----------------------------------------------
        target = directory("09_mc_jets", "purity")

        mc_plot(
            mc_graphs("purity_num_pt_{}", "purity_den_pt_{}", ETA_CATEGORIES),
            target / "purity_vs_pt.png",
            ETA_CATEGORIES,
            r"$p_{T}^{reco\ jet}$ [GeV]",
            "Reconstruction purity",
            ylim=(0.0, 1.05),
            logx=True,
        )

        mc_plot(
            mc_graphs("purity_num_eta_{}", "purity_den_eta_{}", JETPT_CATEGORIES),
            target / "purity_vs_eta.png",
            JETPT_CATEGORIES,
            r"$\eta^{reco\ jet}$",
            "Reconstruction purity",
            ylim=(0.0, 1.05),
        )

        # --- response ---------------------------------------------
        target = directory("09_mc_jets", "response")

        response_pt = OrderedDict()
        resolution_pt = OrderedDict()

        for category in ETA_CATEGORIES:
            mean_hist = try_hist(
                efficiency_reader,
                args.region,
                "resp_vs_pt_{}".format(category),
            )
            square_hist = try_hist(
                efficiency_reader,
                args.region,
                "respSq_vs_pt_{}".format(category),
            )

            if mean_hist is not None:
                response_pt[category] = hist_to_graph(
                    mean_hist,
                    drop_empty=True,
                )

            graph = resolution_graph(
                mean_hist,
                square_hist,
                name="resolution_pt_{}".format(category),
            )

            if graph is not None:
                resolution_pt[category] = graph

        response_eta = OrderedDict()
        resolution_eta = OrderedDict()

        for category in JETPT_CATEGORIES:
            mean_hist = try_hist(
                efficiency_reader,
                args.region,
                "resp_vs_eta_{}".format(category),
            )
            square_hist = try_hist(
                efficiency_reader,
                args.region,
                "respSq_vs_eta_{}".format(category),
            )

            if mean_hist is not None:
                response_eta[category] = hist_to_graph(
                    mean_hist,
                    drop_empty=True,
                )

            graph = resolution_graph(
                mean_hist,
                square_hist,
                name="resolution_eta_{}".format(category),
            )

            if graph is not None:
                resolution_eta[category] = graph

        mc_plot(
            response_pt,
            target / "response_vs_pt.png",
            ETA_CATEGORIES,
            r"$p_{T}^{gen\ jet}$ [GeV]",
            r"$\langle p_{T}^{reco}/p_{T}^{gen} \rangle$",
            ylim=(0.5, 1.5),
            logx=True,
            reference_line=1.0,
        )

        mc_plot(
            response_eta,
            target / "response_vs_eta.png",
            JETPT_CATEGORIES,
            r"$\eta^{gen\ jet}$",
            r"$\langle p_{T}^{reco}/p_{T}^{gen} \rangle$",
            ylim=(0.5, 1.5),
            reference_line=1.0,
        )

        # --- resolution -------------------------------------------
        target = directory("09_mc_jets", "resolution")

        mc_plot(
            resolution_pt,
            target / "resolution_vs_pt.png",
            ETA_CATEGORIES,
            r"$p_{T}^{gen\ jet}$ [GeV]",
            r"$\sigma(p_{T}^{reco}/p_{T}^{gen})$",
            logx=True,
        )

        mc_plot(
            resolution_eta,
            target / "resolution_vs_eta.png",
            JETPT_CATEGORIES,
            r"$\eta^{gen\ jet}$",
            r"$\sigma(p_{T}^{reco}/p_{T}^{gen})$",
        )

        # ============================================================
        # 10 Pileup dependence: the same recoil quantities versus N_PV
        #
        # The pT(Z) split is the one to read for the pileup dependence,
        # because it holds the recoil scale roughly fixed while N_PV varies,
        # so the slope of sigma against N_PV is the quantity to compare
        # between data and MC. A linear axis is used throughout.
        # ============================================================
        npv_label = r"$N_{PV}^{good}$"

        for split, suffix_pattern, categories in (
            ("probe_eta", "Npv_eta_{}", ETA_CATEGORIES),
            ("Zpt", "Npv_zpt_{}", ZPT_CATEGORIES),
        ):
            recoil_block(
                ("10_pu_dependence", split),
                suffix_pattern,
                categories,
                npv_label,
                False,
            )

            target = directory("10_pu_dependence", split, "response")

            for category, label in categories.items():
                graph_plot(
                    response_graphs(
                        suffix_pattern.format(category),
                        npv_label,
                    ),
                    target / "recoil_response_{}.png".format(category),
                    npv_label,
                    RECOIL_RESPONSE_LABEL,
                    ylim=(0.0, 1.5),
                    logx=False,
                    reference_line=1.0,
                    title=label,
                )

    print(
        "\nPlots written under: {}".format(
            output_dir
        )
    )


if __name__ == "__main__":
    main()
