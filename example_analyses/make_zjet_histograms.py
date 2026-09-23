#!/usr/bin/env python3

"""
Generate zjet_histograms.yaml.

The YAML file is long and highly repetitive: most of it is the same handful
of distributions repeated over the |eta|, pT(Z) and jet-pT categories. Hand
editing it is how the categories drift apart from the analysis definition,
so the file is generated from this script instead.

The categories themselves are imported from zjet_rdf_definition.py, which is
the single place they are declared. Only the binning and the titles live
here.

Usage
-----

    python3 example_analyses/make_zjet_histograms.py            # regenerate
    python3 example_analyses/make_zjet_histograms.py --check    # CI-friendly

`--check` exits nonzero when the checked-in YAML no longer matches this
script, without touching it.
"""

import argparse
import os
import sys

from collections import OrderedDict


HERE = os.path.dirname(
    os.path.abspath(__file__)
)

if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Importing the analysis definition pulls in ROOT, which is a little heavy for
# a text generator, but it is what keeps the categories from drifting.
from zjet_rdf_definition import (  # noqa: E402
    ENERGY_FRACTIONS,
    ETA_CATEGORIES,
    PT_CATEGORIES,
    ZPT_CATEGORIES,
)


DEFAULT_OUTPUT = os.path.join(
    HERE,
    "zjet_histograms.yaml",
)

# The event weight used by every data/MC comparison histogram. The MC-only
# blocks pass weight=None instead, which omits the key and lets the runner
# apply its own default (unweighted unless --weights-defs is given).
EVENT_WEIGHT = "eventWeight"


# ----------------------------------------------------------------------
# Binning
# ----------------------------------------------------------------------

# pT axis for the MC studies versus jet pT. It starts at the 15 GeV matching
# threshold, below which the gen/reco jet collections are empty by construction.
PT_EDGES = [
    20, 28, 40, 44, 49, 56, 64, 74, 84, 97, 114, 133, 153, 174,
    220, 300, 430, 638, 1032, 2000,
]

# Axis for every profile versus pT(Z). The zjet region selects the Z by muon
# kinematics rather than by pT(Z), so the spectrum is not cut at 20 GeV and the
# axis has to reach down to zero; the jet-pT studies above keep their own axis,
# which starts at the 15 GeV matching threshold.
ZPT_EDGES = [0, 5, 10, 15] + PT_EDGES

# Vertex axis for the pileup-dependence profiles. Bins of 5 keep every bin
# populated in both data (<N_PV^good> = 37) and MC (45).
NPV_EDGES = [
    0, 5, 10, 15, 20, 25, 30, 35,
    40, 45, 50, 55, 60, 65, 70, 80,
]

NPV_VARIABLE = "analysis_npvsGood"

# Calorimeter-tower eta boundaries.
ETA_EDGES = [
    -5.191, -3.839, -3.489, -3.139, -2.964, -2.853, -2.65, -2.5,
    -2.322, -2.172, -1.93, -1.653, -1.479, -1.305, -1.044, -0.783,
    -0.522, -0.261, 0.0, 0.261, 0.522, 0.783, 1.044, 1.305, 1.479,
    1.653, 1.93, 2.172, 2.322, 2.5, 2.65, 2.853, 2.964, 3.139,
    3.489, 3.839, 5.191,
]

# Fixed-width axes, as (nbins, low, high).
PILEUP_BINS = (100, 0, 100)
ZPT_BINS = (100, 0, 1000)
JETPT_BINS = (100, 0, 1000)
MUPT_BINS = (100, 0, 500)
METPT_BINS = (100, 0, 500)
# The parallel recoil sits at -pT(Z), so its axis is one-sided; the transverse
# component is centred on zero.
UPAR_BINS = (120, -600, 60)
UPERP_BINS = (100, -250, 250)
WIDE_ETA_BINS = (60, -5, 5)
JET_ETA_BINS = (60, -5.2, 5.2)
MUON_ETA_BINS = (48, -2.4, 2.4)
PHI_BINS = (64, -3.2, 3.2)
DPHI_BINS = (64, 0, 3.2)
FRACTION_BINS = (50, 0, 1)


# ----------------------------------------------------------------------
# Titles
# ----------------------------------------------------------------------

FRACTION_TITLES = OrderedDict([
    ("chHEF", "charged hadron energy fraction"),
    ("neHEF", "neutral hadron energy fraction"),
    ("chEmEF", "charged EM energy fraction"),
    ("neEmEF", "neutral EM energy fraction"),
    ("muEF", "muon energy fraction"),
])

if set(FRACTION_TITLES) != set(ENERGY_FRACTIONS):
    raise RuntimeError(
        "FRACTION_TITLES and zjet_rdf_definition.ENERGY_FRACTIONS disagree: "
        "{} vs {}".format(
            sorted(FRACTION_TITLES),
            sorted(ENERGY_FRACTIONS),
        )
    )


def range_label(low, high, symbol, unit="GeV"):
    """Human-readable label for one of the category ranges."""
    open_below = low <= 0.0
    open_above = high >= 1.0e9

    if open_below and open_above:
        return "all {}".format(symbol)

    if open_above:
        return "{} > {:g} {}".format(symbol, low, unit)

    if open_below:
        return "{} < {:g} {}".format(symbol, high, unit)

    return "{:g} < {} < {:g} {}".format(low, symbol, high, unit)


ZPT_LABELS = OrderedDict(
    (category, range_label(low, high, "p_{T}^{Z}"))
    for category, (low, high) in ZPT_CATEGORIES.items()
)

JETPT_LABELS = OrderedDict(
    (category, range_label(low, high, "p_{T}"))
    for category, (low, high) in PT_CATEGORIES.items()
)


# ----------------------------------------------------------------------
# YAML emitter
# ----------------------------------------------------------------------

class HistogramFile:
    """Small append-only writer for the histogram YAML."""

    def __init__(self):
        self._lines = []
        self._names = set()

    # -- structure -----------------------------------------------------

    def preamble(self, *paragraphs):
        for index, paragraph in enumerate(paragraphs):
            if index:
                self._lines.append("#")
            for line in paragraph.splitlines():
                self._lines.append(
                    "# {}".format(line).rstrip()
                )
        self._lines.append("")

    def section(self, heading, *notes):
        rule = "# " + "=" * 74
        self._lines.extend([rule, "# " + heading, rule])

        for note in notes:
            for line in note.splitlines():
                self._lines.append(
                    "# {}".format(line).rstrip()
                )

        self._lines.append("")

    # -- histograms ----------------------------------------------------

    def th1(self, name, title, variable, bins=None, edges=None,
            weight=EVENT_WEIGHT):
        self._open(name, "TH1D", title, bins, edges)
        self._lines.append("  variable: {}".format(variable))
        self._close(weight)

    def profile(self, name, title, variable_x, variable_y, edges,
                weight=EVENT_WEIGHT):
        self._open(name, "TProfile", title, None, edges)
        self._lines.append("  variable_x: {}".format(variable_x))
        self._lines.append("  variable_y: {}".format(variable_y))
        self._close(weight)

    # -- internals -----------------------------------------------------

    def _open(self, name, hist_type, title, bins, edges):
        if name in self._names:
            raise ValueError(
                "Duplicate histogram name: {}".format(name)
            )

        self._names.add(name)

        self._lines.append("{}:".format(name))
        self._lines.append("  type: {}".format(hist_type))
        self._lines.append('  title: "{}"'.format(title))

        if edges is not None:
            self._lines.append("  edges: {}".format(list(edges)))
        elif bins is not None:
            self._lines.append(
                "  bins: [{}, {}, {}]".format(*bins)
            )
        else:
            raise ValueError(
                "Histogram '{}' needs either bins or edges.".format(name)
            )

    def _close(self, weight):
        if weight is not None:
            self._lines.append("  weight: {}".format(weight))
        self._lines.append("")

    def text(self):
        return "\n".join(self._lines).rstrip() + "\n"

    def __len__(self):
        return len(self._names)


# ----------------------------------------------------------------------
# Content
# ----------------------------------------------------------------------

def build():
    out = HistogramFile()

    out.preamble(
        "Z+jet histogram definitions for run_analysis.py.",
        "GENERATED FILE - do not edit by hand.\n"
        "Regenerate with:\n"
        "    python3 example_analyses/make_zjet_histograms.py",
        "The |eta|, pT(Z) and jet-pT categories come from\n"
        "example_analyses/zjet_rdf_definition.py, which declares them once.",
        "Histograms without a 'weight' key are booked with the runner's\n"
        "default weight, which is unweighted unless --weights-defs is used.\n"
        "The MC-only blocks are filled from vector columns that are empty on\n"
        "data, so the same file can be used for both.",
    )

    inclusive_distributions(out)
    inclusive_kinematics(out)
    eta_in_zpt_regions(out)
    met_and_recoil_in_eta_regions(out)
    energy_fractions(out)
    balance_response(out)
    recoil_response_and_resolution(out)
    recoil_versus_npv(out)
    mc_versus_pt(out)
    mc_versus_eta(out)

    return out


def inclusive_distributions(out):
    out.section("Inclusive event-level distributions")

    out.th1("rho", "Pileup energy density;#rho [GeV];Events",
            "analysis_rho", bins=PILEUP_BINS)
    out.th1("npvs", "Reconstructed vertices;N_{PV};Events",
            "analysis_npvs", bins=PILEUP_BINS)
    out.th1("npvsGood", "Good reconstructed vertices;N_{PV}^{good};Events",
            "analysis_npvsGood", bins=PILEUP_BINS)
    out.th1("Z_mass", "Z mass;m_{#mu#mu} [GeV];Events",
            "Z_mass", bins=(60, 80, 102))
    out.th1("nGoodJets", "Selected jet multiplicity;N_{jets};Events",
            "nGoodJets", bins=(15, 0, 15))
    out.th1("alpha", "#alpha;p_{T}^{jet2}/p_{T}^{Z};Events",
            "alpha", bins=(50, 0, 1))
    out.th1("Jet2_pt", "Second selected jet p_{T};p_{T}^{jet2} [GeV];Events",
            "Jet2_pt", bins=(80, 0, 400))


def inclusive_kinematics(out):
    out.section("Inclusive kinematics")

    out.th1("Z_pt", "Z p_{T};p_{T}^{Z} [GeV];Events", "Z_pt", bins=ZPT_BINS)
    out.th1("Z_eta", "Z #eta;#eta^{Z};Events", "Z_eta", bins=WIDE_ETA_BINS)
    out.th1("Z_phi", "Z #phi;#phi^{Z};Events", "Z_phi", bins=PHI_BINS)

    out.th1("Probe_pt", "Probe jet p_{T};p_{T}^{jet} [GeV];Events",
            "Probe_pt", bins=JETPT_BINS)
    out.th1("Probe_eta", "Probe jet #eta;#eta^{jet};Events",
            "Probe_eta", bins=JET_ETA_BINS)
    out.th1("Probe_phi", "Probe jet #phi;#phi^{jet};Events",
            "Probe_phi", bins=PHI_BINS)

    for index, prefix in ((1, "Leading"), (2, "Subleading")):
        out.th1(
            "Mu{}_pt".format(index),
            "{} muon p_{{T}};p_{{T}}^{{#mu{}}} [GeV];Events".format(prefix, index),
            "Mu{}_pt".format(index), bins=MUPT_BINS,
        )
        out.th1(
            "Mu{}_eta".format(index),
            "{} muon #eta;#eta^{{#mu{}}};Events".format(prefix, index),
            "Mu{}_eta".format(index), bins=MUON_ETA_BINS,
        )
        out.th1(
            "Mu{}_phi".format(index),
            "{} muon #phi;#phi^{{#mu{}}};Events".format(prefix, index),
            "Mu{}_phi".format(index), bins=PHI_BINS,
        )

    out.th1("dPhi_mumu",
            "#Delta#phi(#mu1, #mu2);|#Delta#phi(#mu1, #mu2)|;Events",
            "dPhi_mumu", bins=DPHI_BINS)
    out.th1("dPhi_ZProbe",
            "#Delta#phi(Z, probe jet);|#Delta#phi(Z, jet)|;Events",
            "dPhi_ZProbe", bins=DPHI_BINS)

    out.th1("MET_pt", "MET p_{T};p_{T}^{miss} [GeV];Events",
            "MET_pt", bins=METPT_BINS)
    out.th1("MET_phi", "MET #phi;#phi^{miss};Events",
            "MET_phi", bins=PHI_BINS)
    out.th1("U_pt", "Hadronic recoil magnitude;|u| [GeV];Events",
            "U_pt", bins=METPT_BINS)
    out.th1("U_par",
            "Parallel hadronic recoil;u_{#parallel} [GeV];Events",
            "U_par", bins=UPAR_BINS)
    out.th1("U_perp",
            "Transverse hadronic recoil;u_{#perp} [GeV];Events",
            "U_perp", bins=UPERP_BINS)


def eta_in_zpt_regions(out):
    out.section("Eta distributions in regions of p_{T}(Z)")

    variables = (
        ("Z_eta", "Z", "#eta^{Z}", WIDE_ETA_BINS),
        ("Probe_eta", "Probe jet", "#eta^{jet}", JET_ETA_BINS),
        ("Mu1_eta", "Leading muon", "#eta^{#mu1}", MUON_ETA_BINS),
        ("Mu2_eta", "Subleading muon", "#eta^{#mu2}", MUON_ETA_BINS),
    )

    for category, label in ZPT_LABELS.items():
        for variable, subject, axis, bins in variables:
            out.th1(
                "{}_zpt_{}".format(variable, category),
                "{} #eta ({});{};Events".format(subject, label, axis),
                variable,
                bins=bins,
                weight="zptWeight_{}".format(category),
            )


def met_and_recoil_in_eta_regions(out):
    out.section(
        "MET and hadronic-recoil distributions in regions of probe-jet |eta|"
    )

    variables = (
        ("MET_pt", "MET p_{T}", "p_{T}^{miss} [GeV]", METPT_BINS),
        ("MET_phi", "MET #phi", "#phi^{miss}", PHI_BINS),
        ("U_par", "Parallel hadronic recoil",
         "u_{#parallel} [GeV]", UPAR_BINS),
        ("U_perp", "Transverse hadronic recoil",
         "u_{#perp} [GeV]", UPERP_BINS),
    )

    for category in ETA_CATEGORIES:
        for variable, subject, axis, bins in variables:
            out.th1(
                "{}_eta_{}".format(variable, category),
                "{} ({});{};Events".format(subject, category, axis),
                variable,
                bins=bins,
                weight="etaWeight_{}".format(category),
            )


def energy_fractions(out):
    out.section("Probe-jet energy fractions in regions of probe-jet |eta|")

    for fraction, description in FRACTION_TITLES.items():
        for category in ETA_CATEGORIES:
            out.th1(
                "Probe_{}_eta_{}".format(fraction, category),
                "Probe jet {} ({});{};Events".format(
                    description, category, fraction
                ),
                "Probe_{}".format(fraction),
                bins=FRACTION_BINS,
                weight="etaWeight_{}".format(category),
            )

    out.section("Probe-jet energy fractions in regions of p_{T}(Z)")

    for fraction, description in FRACTION_TITLES.items():
        for category, label in ZPT_LABELS.items():
            out.th1(
                "Probe_{}_zpt_{}".format(fraction, category),
                "Probe jet {} ({});{};Events".format(
                    description, label, fraction
                ),
                "Probe_{}".format(fraction),
                bins=FRACTION_BINS,
                weight="zptWeight_{}".format(category),
            )


def balance_response(out):
    out.section(
        "DB / MPF response vs p_{T}(Z), in regions of probe-jet |eta|"
    )

    for category in ETA_CATEGORIES:
        for quantity in ("DB", "MPF"):
            out.profile(
                "{}_vs_Zpt_{}".format(quantity, category),
                "{} vs Z p_{{T}} ({});p_{{T}}^{{Z}} [GeV];{}".format(
                    quantity, category, quantity
                ),
                "Z_pt",
                quantity,
                ZPT_EDGES,
                weight="etaWeight_{}".format(category),
            )


# The two recoil components whose spread is a resolution, as
# (name prefix, column, axis symbol, unit).
RECOIL_QUANTITIES = (
    ("Upar", "U_par", "u_{#parallel}", "GeV"),
    ("Uperp", "U_perp", "u_{#perp}", "GeV"),
)


def _recoil_profiles(out, suffix, variable_x, edges, weight, axis_title):
    """
    Book one recoil block against an arbitrary x axis.

    The block is <u_par>, <u_par^2>, <u_perp>, <u_perp^2> and <pT(Z)>. The
    first pair gives both the response numerator and sigma(u_par), the second
    gives sigma(u_perp), and the pT(Z) profile is the response denominator.
    Every member carries the same weight, which is what makes them
    combinable bin by bin.
    """
    for prefix, variable, symbol, unit in RECOIL_QUANTITIES:
        out.profile(
            "{}_vs_{}".format(prefix, suffix),
            "<{}> vs {};<{}> [{}]".format(
                symbol, axis_title, symbol, unit
            ),
            variable_x,
            variable,
            edges,
            weight=weight,
        )

        out.profile(
            "{}Sq_vs_{}".format(prefix, suffix),
            "<({})^{{2}}> vs {};<({})^{{2}}> [{}^{{2}}]".format(
                symbol, axis_title, symbol, unit
            ),
            variable_x,
            "{}_sq".format(variable),
            edges,
            weight=weight,
        )

    out.profile(
        "Zpt_vs_{}".format(suffix),
        "<p_{{T}}^{{Z}}> vs {};<p_{{T}}^{{Z}}> [GeV]".format(axis_title),
        variable_x,
        "Z_pt",
        edges,
        weight=weight,
    )


def recoil_response_and_resolution(out):
    out.section(
        "Hadronic-recoil response and resolution vs p_{T}(Z)",
        "The response is formed in the plotting step as",
        "    R = -<u_par> / <pT(Z)>",
        "from the Upar and Zpt profiles over the same bins, NOT as a profile",
        "of the per-event ratio -u_par/pT(Z), which diverges as pT(Z) -> 0.",
        "R is also what the resolutions are divided by to correct them for",
        "the recoil scale.",
        "The resolutions are reconstructed in the plotting step as",
        "    sigma = sqrt(<x^2> - <x>^2)",
        "from each mean/mean-of-squares profile pair below. Both members of a",
        "pair must always carry the same weight column.",
    )

    for category in ETA_CATEGORIES:
        _recoil_profiles(
            out,
            "Zpt_{}".format(category),
            "Z_pt",
            ZPT_EDGES,
            "etaWeight_{}".format(category),
            "Z p_{{T}} ({});p_{{T}}^{{Z}} [GeV]".format(category),
        )


def recoil_versus_npv(out):
    out.section(
        "Hadronic-recoil response and resolution vs N_PV",
        "Same quantities as the block above, profiled against the number of",
        "good vertices instead of p_T(Z), which is what exposes the pileup",
        "dependence: sigma grows with N_PV and the slope is the number to",
        "compare between data and MC.",
        "Booked twice, split by probe-jet |eta| and by p_T(Z). The p_T(Z)",
        "split is the more direct one, because it holds the recoil scale",
        "roughly fixed while N_PV varies.",
    )

    for category in ETA_CATEGORIES:
        _recoil_profiles(
            out,
            "Npv_eta_{}".format(category),
            NPV_VARIABLE,
            NPV_EDGES,
            "etaWeight_{}".format(category),
            "N_{{PV}}^{{good}} ({});N_{{PV}}^{{good}}".format(category),
        )

    for category, label in ZPT_LABELS.items():
        _recoil_profiles(
            out,
            "Npv_zpt_{}".format(category),
            NPV_VARIABLE,
            NPV_EDGES,
            "zptWeight_{}".format(category),
            "N_{{PV}}^{{good}} ({});N_{{PV}}^{{good}}".format(label),
        )


def mc_versus_pt(out):
    out.section(
        "MC only: jet efficiency / purity / response vs p_{T}, "
        "in |eta| regions",
        "Gen jets above 15 GeV are matched to reco jets above 10 GeV within",
        "dR < 0.2 for the efficiency; the roles are swapped for the purity.",
        "Both collections are cleaned against the muons of the Z candidate.",
    )

    for category in ETA_CATEGORIES:
        out.th1(
            "eff_den_pt_{}".format(category),
            "Gen jets ({});p_{{T}}^{{gen}} [GeV];Gen jets".format(category),
            "GenPt_{}".format(category), edges=PT_EDGES, weight=None,
        )
        out.th1(
            "eff_num_pt_{}".format(category),
            "Matched gen jets ({});p_{{T}}^{{gen}} [GeV];"
            "Matched gen jets".format(category),
            "GenMatchedPt_{}".format(category), edges=PT_EDGES, weight=None,
        )
        out.th1(
            "purity_den_pt_{}".format(category),
            "Reco jets ({});p_{{T}}^{{reco}} [GeV];Reco jets".format(category),
            "RecoPt_{}".format(category), edges=PT_EDGES, weight=None,
        )
        out.th1(
            "purity_num_pt_{}".format(category),
            "Matched reco jets ({});p_{{T}}^{{reco}} [GeV];"
            "Matched reco jets".format(category),
            "RecoMatchedPt_{}".format(category), edges=PT_EDGES, weight=None,
        )
        out.profile(
            "resp_vs_pt_{}".format(category),
            "<p_{{T}}^{{reco}}/p_{{T}}^{{gen}}> vs p_{{T}}^{{gen}} ({});"
            "p_{{T}}^{{gen}} [GeV];<p_{{T}}^{{reco}}/p_{{T}}^{{gen}}>"
            .format(category),
            "GenMatchedPt_{}".format(category),
            "Resp_pt_{}".format(category),
            PT_EDGES, weight=None,
        )
        out.profile(
            "respSq_vs_pt_{}".format(category),
            "<(p_{{T}}^{{reco}}/p_{{T}}^{{gen}})^{{2}}> vs p_{{T}}^{{gen}} "
            "({});p_{{T}}^{{gen}} [GeV];"
            "<(p_{{T}}^{{reco}}/p_{{T}}^{{gen}})^{{2}}>".format(category),
            "GenMatchedPt_{}".format(category),
            "RespSq_pt_{}".format(category),
            PT_EDGES, weight=None,
        )


def mc_versus_eta(out):
    out.section(
        "MC only: jet efficiency / purity / response vs eta, "
        "in p_{T} regions"
    )

    for category, label in JETPT_LABELS.items():
        out.th1(
            "eff_den_eta_{}".format(category),
            "Gen jets ({});#eta^{{gen}};Gen jets".format(label),
            "GenEta_{}".format(category), edges=ETA_EDGES, weight=None,
        )
        out.th1(
            "eff_num_eta_{}".format(category),
            "Matched gen jets ({});#eta^{{gen}};Matched gen jets".format(label),
            "GenMatchedEta_{}".format(category), edges=ETA_EDGES, weight=None,
        )
        out.th1(
            "purity_den_eta_{}".format(category),
            "Reco jets ({});#eta^{{reco}};Reco jets".format(label),
            "RecoEta_{}".format(category), edges=ETA_EDGES, weight=None,
        )
        out.th1(
            "purity_num_eta_{}".format(category),
            "Matched reco jets ({});#eta^{{reco}};Matched reco jets".format(label),
            "RecoMatchedEta_{}".format(category), edges=ETA_EDGES, weight=None,
        )
        out.profile(
            "resp_vs_eta_{}".format(category),
            "<p_{{T}}^{{reco}}/p_{{T}}^{{gen}}> vs #eta^{{gen}} ({});"
            "#eta^{{gen}};<p_{{T}}^{{reco}}/p_{{T}}^{{gen}}>".format(label),
            "GenMatchedEta_{}".format(category),
            "Resp_eta_{}".format(category),
            ETA_EDGES, weight=None,
        )
        out.profile(
            "respSq_vs_eta_{}".format(category),
            "<(p_{{T}}^{{reco}}/p_{{T}}^{{gen}})^{{2}}> vs #eta^{{gen}} ({});"
            "#eta^{{gen}};<(p_{{T}}^{{reco}}/p_{{T}}^{{gen}})^{{2}}>"
            .format(label),
            "GenMatchedEta_{}".format(category),
            "RespSq_eta_{}".format(category),
            ETA_EDGES, weight=None,
        )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Destination YAML file (default: {})".format(
            os.path.relpath(DEFAULT_OUTPUT)
        ),
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write anything; exit nonzero if --output differs "
            "from what this script would produce."
        ),
    )

    args = parser.parse_args()

    out = build()
    text = out.text()

    if args.check:
        if not os.path.isfile(args.output):
            print("MISSING: {}".format(args.output))
            return 1

        with open(args.output, "r", encoding="utf-8") as handle:
            current = handle.read()

        if current != text:
            print(
                "OUT OF DATE: {}\n"
                "Regenerate with: python3 {}".format(
                    args.output,
                    os.path.relpath(__file__),
                )
            )
            return 1

        print(
            "up to date: {} ({} histograms)".format(
                args.output,
                len(out),
            )
        )
        return 0

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(text)

    print(
        "wrote {} ({} histograms)".format(
            args.output,
            len(out),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
