"""
Z+jet RDF definition for the generic SkimRDFAnalysisBase/run_analysis.py driver.

Required interface used by run_analysis.py:
    setup(args, config)
    define_columns(df, sample, args, config)
    define_weighted_columns(df, sample, args, config, event_weight)
    get_regions(sample, args, config)

The analysis-specific file only defines physics objects/columns and regions.
Input discovery, event-loop execution, histogram booking and output writing stay
inside the generic run_analysis.py.

Booked content (see zjet_histograms.yaml):

    data and MC
        pileup observables (rho, N_PV)
        Z / probe-jet / muon / MET kinematics
        eta distributions in Z-pT regions
        MET and hadronic-recoil components in probe-|eta| regions
        probe-jet energy fractions, inclusively and in |eta| and Z-pT regions
        hadronic-recoil resolutions sigma(u_par) and sigma(u_perp) vs pT(Z),
            raw and corrected for the recoil scale
        hadronic-recoil response -<u_par>/<pT(Z)> vs pT(Z)
        the same recoil response and resolutions vs N_PV, in |eta| and
            Z-pT regions, for the pileup dependence
        DB / MPF response vs pT(Z), inclusively and in |eta| regions

    MC only
        jet reconstruction efficiency, purity, response and resolution with
        respect to gen jets, matched within dR < 0.2
"""

import os
import ROOT


# Probe-jet |eta| regions.
ETA_CATEGORIES = {
    "Incl": (0.0, 5.0),
    "HB":   (0.0, 1.3),
    "HE1":  (1.5, 2.5),
    "HE2":  (2.5, 3.0),
    "HF":   (3.0, 5.0),
}

# Jet-pT regions, used for the MC efficiency/purity/response versus eta.
PT_CATEGORIES = {
    "pt15to30":    (15.0, 30.0),
    "pt30to60":    (30.0, 60.0),
    "pt60to120":   (60.0, 120.0),
    "pt120to300":  (120.0, 300.0),
    "pt300to1000": (300.0, 1000.0),
    "pt1000plus":  (1000.0, 1.0e9),
}

# Z-pT regions, used for the eta and energy-fraction distributions.
ZPT_CATEGORIES = {
    "Incl":        (0.0, 1.0e9),
    "zpt0to20":    (0.0, 20.0),
    "zpt20to40":   (20.0, 40.0),
    "zpt40to80":   (40.0, 80.0),
    "zpt80to150":  (80.0, 150.0),
    "zpt150to300": (150.0, 300.0),
    "zpt300plus":  (300.0, 1.0e9),
}

# PF energy fractions of the probe jet.
ENERGY_FRACTIONS = {
    "chHEF":  "chHEF()",
    "neHEF":  "neHEF()",
    "chEmEF": "chEmEF()",
    "neEmEF": "neEmEF()",
    "muEF":   "muEF()",
}

# Reco and gen jets enter the matching with different thresholds: the object
# whose efficiency (or purity) is measured must be above 15 GeV, the object it
# is matched against only above 10 GeV.
MATCH_REFERENCE_PT = 15.0
MATCH_CANDIDATE_PT = 10.0

# Analysis muon working point. Both muons of the Z candidate satisfy it: tight
# ID plus a dBeta-corrected pfRelIso04 below 0.15, the Muon POG tight
# isolation point. The same TightMuons collection is counted by the region,
# cleans the jets, and provides the Z candidate, so the working point is
# written here once and cannot drift between those three uses.
MUON_PT_MIN = 10.0
MUON_ETA_MAX = 2.4
MUON_REL_ISO_MAX = 0.15

# Veto muon working point: the standard loose veto-lepton point. A muon that
# passes this but fails the analysis working point above is an extra muon, and
# the region requires there to be none, so the event has exactly the two
# muons of the Z and nothing else.
VETO_MUON_PT_MIN = 10.0
VETO_MUON_REL_ISO_MAX = 0.25

# Trigger paths accepted by the zjet region, OR-ed together. The default is
# matched to the `Mu2_pt > 27` region cut, which puts both muons on the
# IsoMu24 efficiency plateau.
#
# The same requirement is applied to data and to MC. Data come from a
# trigger-selected primary dataset, so leaving MC untriggered makes MC
# over-predict the data yield by the trigger inefficiency.
#
# Override with `triggers: [...]` in the optional analysis config.
DEFAULT_TRIGGERS = ("HLT_IsoMu24",)

# MET trigger measured in the `zjet_<MET_TRIGGER>` region, which is the zjet
# region plus this path. It is not part of the zjet selection itself.
MET_TRIGGER = "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight"

_JEC_ENABLED = False


def _repo_dir():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )


def _resolve_optional_path(path):
    """Resolve absolute paths, cwd-relative paths, then repository-relative paths."""
    if not path:
        return ""

    path = os.path.expanduser(path)

    if os.path.isabs(path):
        return path

    if os.path.exists(path):
        return os.path.abspath(path)

    candidate = os.path.join(_repo_dir(), path)
    return os.path.abspath(candidate)


def _column_names(df):
    return {str(name) for name in df.GetColumnNames()}


def _ensure_vector(df, columns, name, ctype, size_column, default_value):
    """Define an optional NanoAOD vector branch when it is absent."""
    if name in columns:
        return df

    return df.Define(
        name,
        f"ROOT::VecOps::RVec<{ctype}>({size_column}, {default_value})"
    )


def _ensure_scalar(df, columns, target, candidates, default_expression):
    """Alias the first available candidate to target, otherwise define a default."""
    if target in columns:
        return df

    for candidate in candidates:
        if candidate in columns:
            return df.Define(target, candidate)

    return df.Define(target, default_expression)


def setup(args=None, config=None):
    """
    One-time setup called by run_analysis.py.

    Loads the generic C++ physics-object/helper headers. If JEC re-application
    is enabled in the optional analysis config, also load libAnalysisCommon.so
    and initialize the correctionlib JEC payload.
    """
    global _JEC_ENABLED

    config = config or {}

    repo_dir = _repo_dir()
    common_dir = os.path.join(repo_dir, "Common")
    interface_dir = os.path.join(common_dir, "interface")

    ROOT.gInterpreter.AddIncludePath(interface_dir)

    for header in (
        "Kinematics.h",
        "PhysicsObjects.h",
        "ObjectBuilders.h",
        "Balance.h",
        "JetMatching.h",
        "Corrections.h",
    ):
        ROOT.gInterpreter.Declare(
            f'#include "{header}"'
        )

    jec_config = config.get("jec", {}) or {}
    _JEC_ENABLED = bool(jec_config.get("enabled", False))

    if not _JEC_ENABLED:
        return

    library = os.path.join(common_dir, "libAnalysisCommon.so")

    if not os.path.isfile(library):
        raise RuntimeError(
            f"JEC is enabled but {library} does not exist.\n"
            "Compile Common first with:\n"
            "  cd Common && source setup_Common_cpp.sh"
        )

    json_path = _resolve_optional_path(
        jec_config.get("json", "")
    )
    correction_name = jec_config.get("name", "")

    if not json_path or not correction_name:
        raise RuntimeError(
            "jec.enabled is true, but jec.json and/or jec.name is missing."
        )

    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"JEC JSON does not exist: {json_path}"
        )

    ROOT.gSystem.Load(library)
    ROOT.skimrdf.initJEC(
        json_path,
        correction_name
    )


def define_columns(df, sample=None, args=None, config=None):
    """
    Build the Z+jet analysis columns.

    This function intentionally performs Define/Redefine operations only.
    Event selection is returned separately by get_regions().
    """
    config = config or {}
    columns = _column_names(df)

    # ------------------------------------------------------------------
    # Input-schema capabilities
    # ------------------------------------------------------------------
    # Detect these BEFORE defining fallback vectors. A missing branch should
    # change the selection logic; it must not silently become an all-zero
    # vector that makes every object fail.
    has_muon_loose_id = "Muon_looseId" in columns
    has_muon_medium_id = "Muon_mediumId" in columns
    has_muon_tight_id = "Muon_tightId" in columns
    has_muon_rel_iso = "Muon_pfRelIso04_all" in columns
    has_jet_id = "Jet_jetId" in columns

    sample_name = (sample or {}).get("name", "<sample>")

    # ID level actually requested of the two collections, given what the input
    # schema provides. A missing branch weakens the requirement; it must never
    # become an all-zero vector that makes every muon fail.
    muon_id = "Tight" if has_muon_tight_id else "Any"

    if has_muon_loose_id:
        veto_muon_id = "Loose"
    elif has_muon_medium_id:
        veto_muon_id = "Medium"
    else:
        veto_muon_id = "Any"

    if not has_muon_tight_id:
        print(
            f"[zjet:{sample_name}] Muon_tightId not found: "
            "tight muon-ID requirement disabled."
        )

    if not has_muon_rel_iso:
        print(
            f"[zjet:{sample_name}] Muon_pfRelIso04_all not found: "
            "muon isolation requirement disabled."
        )

    if not has_muon_loose_id:
        if has_muon_medium_id:
            print(
                f"[zjet:{sample_name}] Muon_looseId not found: "
                "veto muons defined with mediumId instead."
            )
        else:
            print(
                f"[zjet:{sample_name}] No loose or medium muon ID found: "
                "veto muons defined by kinematics and isolation only."
            )

    if not has_jet_id:
        print(
            f"[zjet:{sample_name}] Jet_jetId not found: "
            "tight jet-ID requirement disabled."
        )

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------
    requested_triggers = list(
        config.get("triggers", DEFAULT_TRIGGERS)
    )

    available_triggers = [
        path for path in requested_triggers
        if path in columns
    ]

    missing_triggers = [
        path for path in requested_triggers
        if path not in columns
    ]

    if missing_triggers:
        print(
            f"[zjet:{sample_name}] Trigger path(s) not found: "
            + ", ".join(missing_triggers)
        )

    if available_triggers:
        print(
            f"[zjet:{sample_name}] Trigger requirement: "
            + " || ".join(available_triggers)
        )

        df = df.Define(
            "passTrigger",
            " || ".join(available_triggers)
        )
    else:
        # Falling back to "always true" keeps the region definition valid on
        # inputs without trigger branches, but it silently removes the
        # requirement, so say so loudly rather than in passing.
        print(
            f"[zjet:{sample_name}] WARNING: none of the requested trigger "
            "paths exist in this input; the trigger requirement is DISABLED. "
            "Data and MC are no longer selected consistently."
        )

        df = df.Define("passTrigger", "true")

    if MET_TRIGGER in columns:
        df = df.Define("passMETTrigger", MET_TRIGGER)
    else:
        # Fail closed: an empty region is visible, whereas an always-true cut
        # would silently report a MET-trigger efficiency of 1.
        print(
            f"[zjet:{sample_name}] WARNING: {MET_TRIGGER} not found; "
            "the MET-trigger region will be empty."
        )
        df = df.Define("passMETTrigger", "false")

    # ------------------------------------------------------------------
    # NanoAOD compatibility / optional branches
    # ------------------------------------------------------------------
    df = _ensure_vector(
        df, columns,
        "Muon_mass", "float", "nMuon", "0.105658f"
    )
    df = _ensure_vector(
        df, columns,
        "Muon_looseId", "unsigned char", "nMuon", "0"
    )
    df = _ensure_vector(
        df, columns,
        "Muon_mediumId", "unsigned char", "nMuon", "0"
    )
    df = _ensure_vector(
        df, columns,
        "Muon_tightId", "unsigned char", "nMuon", "0"
    )
    df = _ensure_vector(
        df, columns,
        "Muon_pfIsoId", "unsigned char", "nMuon", "0"
    )
    df = _ensure_vector(
        df, columns,
        "Muon_pfRelIso04_all", "float", "nMuon", "-1.f"
    )

    df = _ensure_vector(
        df, columns,
        "Jet_rawFactor", "float", "nJet", "0.f"
    )
    df = _ensure_vector(
        df, columns,
        "Jet_area", "float", "nJet", "0.f"
    )
    df = _ensure_vector(
        df, columns,
        "Jet_jetId", "int", "nJet", "0"
    )
    df = _ensure_vector(
        df, columns,
        "Jet_chHEF", "float", "nJet", "-1.f"
    )
    df = _ensure_vector(
        df, columns,
        "Jet_neHEF", "float", "nJet", "-1.f"
    )
    df = _ensure_vector(
        df, columns,
        "Jet_chEmEF", "float", "nJet", "-1.f"
    )
    df = _ensure_vector(
        df, columns,
        "Jet_neEmEF", "float", "nJet", "-1.f"
    )
    df = _ensure_vector(
        df, columns,
        "Jet_muEF", "float", "nJet", "-1.f"
    )
    df = _ensure_vector(
        df, columns,
        "Jet_genJetIdx", "int", "nJet", "-1"
    )

    # ------------------------------------------------------------------
    # Pileup observables
    # ------------------------------------------------------------------
    df = _ensure_scalar(
        df,
        columns,
        "analysis_rho",
        (
            "fixedGridRhoFastjetAll",
            "Rho_fixedGridRhoFastjetAll",
        ),
        "0.f",
    )

    df = _ensure_scalar(
        df,
        columns,
        "analysis_npvs",
        (
            "PV_npvs",
        ),
        "0",
    )

    df = _ensure_scalar(
        df,
        columns,
        "analysis_npvsGood",
        (
            "PV_npvsGood",
            "PV_npvs",
        ),
        "0",
    )

    # ------------------------------------------------------------------
    # Muons and Z candidate
    # ------------------------------------------------------------------
    df = df.Define(
        "Muons",
        """
        skimrdf::buildMuons(
            Muon_pt,
            Muon_eta,
            Muon_phi,
            Muon_mass,
            Muon_charge,
            Muon_looseId,
            Muon_mediumId,
            Muon_tightId,
            Muon_pfIsoId,
            Muon_pfRelIso04_all
        )
        """
    )

    # A non-positive relIso bound disables the isolation cut, which is the
    # documented way to run on a schema without an isolation branch.
    muon_rel_iso_max = MUON_REL_ISO_MAX if has_muon_rel_iso else -1.0
    veto_muon_rel_iso_max = VETO_MUON_REL_ISO_MAX if has_muon_rel_iso else -1.0

    # The analysis muons. This one collection is counted by the region, cleans
    # the reco and gen jets, and supplies the Z candidate.
    df = df.Define(
        "TightMuons",
        f"""
        skimrdf::selectMuons(
            Muons,
            skimrdf::MuonId::{muon_id},
            {muon_rel_iso_max}f,
            {MUON_PT_MIN}f,
            {MUON_ETA_MAX}f
        )
        """
    )

    # Extra muons: loose enough to matter, but not part of the Z. The region
    # requires this collection to be empty.
    df = df.Define(
        "VetoMuons",
        f"""
        skimrdf::selectVetoMuons(
            Muons,
            skimrdf::MuonId::{veto_muon_id},
            {veto_muon_rel_iso_max}f,
            {VETO_MUON_PT_MIN}f,
            skimrdf::MuonId::{muon_id},
            {muon_rel_iso_max}f,
            {MUON_PT_MIN}f,
            {MUON_ETA_MAX}f
        )
        """
    )

    df = (
        df
        .Define("nTightMuons", "static_cast<int>(TightMuons.size())")
        .Define("nVetoMuons", "static_cast<int>(VetoMuons.size())")
    )

    # Opposite-sign pair closest to the Z mass. TightMuons is already at the
    # analysis working point, so no ID or isolation requirement is repeated.
    df = df.Define(
        "Z",
        "skimrdf::selectBestDimuon(TightMuons, 91.1876)"
    )

    # Leading / subleading muon of the Z candidate, ordered in pT.
    df = (
        df
        .Define(
            "ZMuLead",
            """
            Z.valid()
            ? (Z.muon1().pt() >= Z.muon2().pt() ? Z.muon1() : Z.muon2())
            : skimrdf::Muon()
            """
        )
        .Define(
            "ZMuSub",
            """
            Z.valid()
            ? (Z.muon1().pt() >= Z.muon2().pt() ? Z.muon2() : Z.muon1())
            : skimrdf::Muon()
            """
        )
    )

    # ------------------------------------------------------------------
    # Jets
    # ------------------------------------------------------------------
    df = df.Define(
        "JetsNano",
        """
        skimrdf::buildJets(
            Jet_pt,
            Jet_eta,
            Jet_phi,
            Jet_mass,
            Jet_rawFactor,
            Jet_area,
            Jet_jetId,
            Jet_chHEF,
            Jet_neHEF,
            Jet_chEmEF,
            Jet_neEmEF,
            Jet_muEF,
            Jet_genJetIdx
        )
        """
    )

    # ------------------------------------------------------------------
    # MET collection
    # ------------------------------------------------------------------
    requested_met = config.get(
        "met_prefix",
        "PuppiMET"
    )
    
    # Fallback to MET in case it doesn't find PuppiMET
    candidate_prefixes = [
        requested_met,
        "PuppiMET",
        "MET",
    ]

    met_prefix = None
    for prefix in candidate_prefixes:
        if (
            f"{prefix}_pt" in columns
            and f"{prefix}_phi" in columns
        ):
            met_prefix = prefix
            break

    if met_prefix is None:
        raise RuntimeError(
            "Could not find a MET collection. Tried: "
            + ", ".join(candidate_prefixes)
        )

    df = df.Define(
        "InputMET",
        f"skimrdf::MET({met_prefix}_pt, {met_prefix}_phi)"
    )

    # ------------------------------------------------------------------
    # Optional JEC re-application and MET propagation
    # ------------------------------------------------------------------
    if _JEC_ENABLED:
        df = df.Define(
            "Jets",
            "skimrdf::applyJEC(JetsNano, analysis_rho)"
        )

        df = df.Define(
            "AnalysisMET",
            """
            skimrdf::propagateType1(
                InputMET,
                JetsNano,
                Jets,
                15.f,
                5.2f
            )
            """
        )
    else:
        df = df.Define(
            "Jets",
            "JetsNano"
        )

        df = df.Define(
            "AnalysisMET",
            "InputMET"
        )

    # Collection the jets are cleaned against: every tight muon, not only the
    # two that form the Z. The region requires nVetoMuons == 0, so any muon
    # beyond the Z legs is itself tight, and a jet overlapping it would still
    # be a muon jet. Any collection of objects with eta()/phi() accessors works
    # here, so swapping in electrons, photons or a merged lepton collection
    # only means changing this Define. The same collection also cleans the gen
    # jets below.
    df = df.Define(
        "JetCleaningObjects",
        "TightMuons"
    )

    # Clean reco jets from the tight muons and order them in pT.
    df = df.Define(
        "GoodJets",
        f"""
        skimrdf::selectJets(
            Jets,
            JetCleaningObjects,
            {MATCH_REFERENCE_PT}f,
            5.f,
            {str(has_jet_id).lower()},
            0.2f
        )
        """
    )

    # Looser copy of the same selection, used only as the match candidates for
    # the gen jets so that the matched object may be as soft as 10 GeV.
    df = df.Define(
        "GoodJetsLoose",
        f"""
        skimrdf::selectJets(
            Jets,
            JetCleaningObjects,
            {MATCH_CANDIDATE_PT}f,
            5.f,
            {str(has_jet_id).lower()},
            0.2f
        )
        """
    )

    df = df.Define(
        "Probe",
        "skimrdf::leadingJet(GoodJets)"
    )

    # ------------------------------------------------------------------
    # Scalar analysis quantities
    # ------------------------------------------------------------------
    df = (
        df
        .Define("Z_pt", "Z.pt()")
        .Define("Z_eta", "Z.valid() ? Z.eta() : -999.f")
        .Define("Z_phi", "Z.valid() ? Z.phi() : -999.f")
        .Define("Z_mass", "Z.mass()")
        .Define("Mu1_pt", "ZMuLead.pt()")
        .Define("Mu1_eta", "ZMuLead.valid() ? ZMuLead.eta() : -999.f")
        .Define("Mu1_phi", "ZMuLead.valid() ? ZMuLead.phi() : -999.f")
        .Define("Mu2_pt", "ZMuSub.pt()")
        .Define("Mu2_eta", "ZMuSub.valid() ? ZMuSub.eta() : -999.f")
        .Define("Mu2_phi", "ZMuSub.valid() ? ZMuSub.phi() : -999.f")
        .Define(
            "dPhi_mumu",
            """
            Z.valid()
            ? skimrdf::absDeltaPhi(ZMuLead.phi(), ZMuSub.phi())
            : -1.0
            """
        )
        .Define("Probe_pt", "Probe.pt()")
        .Define("Probe_eta", "Probe.valid() ? Probe.eta() : -999.f")
        .Define("Probe_phi", "Probe.valid() ? Probe.phi() : -999.f")
        .Define("MET_pt", "AnalysisMET.pt()")
        .Define("MET_phi", "AnalysisMET.valid() ? AnalysisMET.phi() : -999.f")
        .Define("nGoodJets", "static_cast<int>(GoodJets.size())")
        .Define(
            "Jet2_pt",
            "GoodJets.size() > 1 ? GoodJets[1].pt() : 0.f"
        )
        .Define(
            "dPhi_ZProbe",
            """
            Z.valid() && Probe.valid()
            ? skimrdf::absDeltaPhi(Z.phi(), Probe.phi())
            : -1.0
            """
        )
        .Define(
            "DB",
            "skimrdf::directBalance(Z, Probe)"
        )
        .Define(
            "MPF",
            "skimrdf::mpfResponse(Z, AnalysisMET)"
        )
        .Define(
            "alpha",
            "skimrdf::alpha(Z, GoodJets)"
        )
    )

    # Probe-jet PF energy fractions.
    for fraction, accessor in ENERGY_FRACTIONS.items():
        df = df.Define(
            f"Probe_{fraction}",
            f"Probe.{accessor}"
        )

    # ------------------------------------------------------------------
    # Hadronic recoil
    # ------------------------------------------------------------------
    #
    # The hadronic recoil is everything in the event that is not the Z. With
    # the MET defined as minus the vector sum of all reconstructed objects,
    #
    #     u_vec = -(MET_vec + qT_vec),      qT_vec = the Z transverse momentum
    #
    # and it is decomposed along the Z direction qhat and along the direction
    # transverse to it:
    #
    #     u_par  = u_vec . qhat            ~ -pT(Z)
    #     u_perp = u_vec x qhat            ~ 0
    #
    # The MET performance observables built from this are the recoil response
    #
    #     R = -<u_par> / <pT(Z)>           ~ 1
    #
    # and the two resolutions sigma(u_par) and sigma(u_perp).
    #
    # Nothing here is a per-event ratio. The response is formed in the
    # plotting step from a profile of u_par and a profile of pT(Z) over the
    # same bins, because the per-event ratio -u_par/pT(Z) diverges as
    # pT(Z) -> 0 and the region no longer cuts on pT(Z).
    #
    # The squared columns exist so that each resolution can be reconstructed
    # from a pair of TProfiles: sigma = sqrt(<x^2> - <x>^2).
    df = (
        df
        .Define(
            "U_x",
            "-(AnalysisMET.px() + Z.pt() * std::cos(Z.phi()))"
        )
        .Define(
            "U_y",
            "-(AnalysisMET.py() + Z.pt() * std::sin(Z.phi()))"
        )
        .Define(
            "U_par",
            """
            Z.valid() && AnalysisMET.valid()
            ? U_x * std::cos(Z.phi()) + U_y * std::sin(Z.phi())
            : -999.0
            """
        )
        .Define(
            "U_perp",
            """
            Z.valid() && AnalysisMET.valid()
            ? -U_x * std::sin(Z.phi()) + U_y * std::cos(Z.phi())
            : -999.0
            """
        )
        .Define(
            "U_pt",
            """
            Z.valid() && AnalysisMET.valid()
            ? std::hypot(U_x, U_y)
            : -999.0
            """
        )
        .Define("U_par_sq", "U_par * U_par")
        .Define("U_perp_sq", "U_perp * U_perp")
    )

    # ------------------------------------------------------------------
    # MC-only reco <-> gen matching
    # ------------------------------------------------------------------
    has_genjets = all(
        name in columns
        for name in (
            "nGenJet",
            "GenJet_pt",
            "GenJet_eta",
            "GenJet_phi",
        )
    )

    match_dr = float(
        config.get("match_dr", 0.2)
    )

    if has_genjets:
        if "GenJet_mass" not in columns:
            df = df.Define(
                "GenJet_mass",
                "ROOT::VecOps::RVec<float>(nGenJet, 0.f)"
            )

        df = (
            df
            .Define(
                "GenJetsAll",
                """
                skimrdf::buildGenJets(
                    GenJet_pt,
                    GenJet_eta,
                    GenJet_phi,
                    GenJet_mass
                )
                """
            )
            # Gen jets are cleaned against the same tight muons as the reco
            # jets, so muon jets do not enter the efficiency denominator. Using
            # one collection for both sides keeps the numerator and denominator
            # of the matching efficiency defined on the same footing.
            .Define(
                "GenJetsClean",
                """
                skimrdf::cleanByDeltaR(
                    GenJetsAll,
                    JetCleaningObjects,
                    0.2f
                )
                """
            )
            .Define(
                "GenJets",
                f"skimrdf::selectGenJets(GenJetsClean, {MATCH_REFERENCE_PT}f, 5.f)"
            )
            .Define(
                "GenJetsLoose",
                f"skimrdf::selectGenJets(GenJetsClean, {MATCH_CANDIDATE_PT}f, 5.f)"
            )
            # Gen jets above 15 GeV matched to reco jets above 10 GeV, and the
            # mirrored assignment for the purity.
            .Define(
                "GenMatchIdx",
                f"skimrdf::matchIndices(GenJets, GoodJetsLoose, {match_dr}f)"
            )
            .Define(
                "RecoMatchIdx",
                f"skimrdf::matchIndices(GoodJets, GenJetsLoose, {match_dr}f)"
            )
            .Define(
                "GenMatchFlags",
                "skimrdf::matchFlags(GenMatchIdx)"
            )
            .Define(
                "RecoMatchFlags",
                "skimrdf::matchFlags(RecoMatchIdx)"
            )
        )

        # Efficiency / purity / response versus pT in |eta| categories.
        for category, (eta_min, eta_max) in ETA_CATEGORIES.items():
            df = (
                df
                .Define(
                    f"GenPt_{category}",
                    f"skimrdf::objectPt(GenJets, {eta_min}f, {eta_max}f)"
                )
                .Define(
                    f"GenMatchedPt_{category}",
                    f"""
                    skimrdf::matchedObjectPt(
                        GenJets,
                        GenMatchFlags,
                        {eta_min}f,
                        {eta_max}f
                    )
                    """
                )
                .Define(
                    f"RecoPt_{category}",
                    f"skimrdf::objectPt(GoodJets, {eta_min}f, {eta_max}f)"
                )
                .Define(
                    f"RecoMatchedPt_{category}",
                    f"""
                    skimrdf::matchedObjectPt(
                        GoodJets,
                        RecoMatchFlags,
                        {eta_min}f,
                        {eta_max}f
                    )
                    """
                )
                .Define(
                    f"Resp_pt_{category}",
                    f"""
                    skimrdf::matchedResponse(
                        GenJets,
                        GoodJetsLoose,
                        GenMatchIdx,
                        {eta_min}f,
                        {eta_max}f
                    )
                    """
                )
                .Define(
                    f"RespSq_pt_{category}",
                    f"Resp_pt_{category} * Resp_pt_{category}"
                )
            )

        # Efficiency / purity / response versus eta in pT categories.
        for category, (pt_min, pt_max) in PT_CATEGORIES.items():
            df = (
                df
                .Define(
                    f"GenEta_{category}",
                    f"skimrdf::objectEta(GenJets, {pt_min}f, {pt_max}f)"
                )
                .Define(
                    f"GenMatchedEta_{category}",
                    f"""
                    skimrdf::matchedObjectEta(
                        GenJets,
                        GenMatchFlags,
                        {pt_min}f,
                        {pt_max}f
                    )
                    """
                )
                .Define(
                    f"RecoEta_{category}",
                    f"skimrdf::objectEta(GoodJets, {pt_min}f, {pt_max}f)"
                )
                .Define(
                    f"RecoMatchedEta_{category}",
                    f"""
                    skimrdf::matchedObjectEta(
                        GoodJets,
                        RecoMatchFlags,
                        {pt_min}f,
                        {pt_max}f
                    )
                    """
                )
                .Define(
                    f"Resp_eta_{category}",
                    f"""
                    skimrdf::matchedResponse(
                        GenJets,
                        GoodJetsLoose,
                        GenMatchIdx,
                        0.f,
                        999.f,
                        {pt_min}f,
                        {pt_max}f
                    )
                    """
                )
                .Define(
                    f"RespSq_eta_{category}",
                    f"Resp_eta_{category} * Resp_eta_{category}"
                )
            )

    else:
        # Keep the MC-only columns truly empty on data, so the same histogram
        # YAML can be used for both without filling misleading denominators.
        for category in ETA_CATEGORIES:
            for column in (
                f"GenPt_{category}",
                f"GenMatchedPt_{category}",
                f"RecoPt_{category}",
                f"RecoMatchedPt_{category}",
                f"Resp_pt_{category}",
                f"RespSq_pt_{category}",
            ):
                df = df.Define(
                    column,
                    "ROOT::VecOps::RVec<float>{}"
                )

        for category in PT_CATEGORIES:
            for column in (
                f"GenEta_{category}",
                f"GenMatchedEta_{category}",
                f"RecoEta_{category}",
                f"RecoMatchedEta_{category}",
                f"Resp_eta_{category}",
                f"RespSq_eta_{category}",
            ):
                df = df.Define(
                    column,
                    "ROOT::VecOps::RVec<float>{}"
                )

    return df


def define_weighted_columns(df, sample=None, args=None, config=None, event_weight=None):
    """Called after the runner combines weights, so profiles use the final weight."""
    if event_weight is None:
        expression = (config or {}).get("event_weight_expression", "")
        expression = expression or ("genWeight" if "genWeight" in _column_names(df) else "1.0")
        df = df.Define("eventWeight", expression)
    elif event_weight != "eventWeight":
        df = df.Define("eventWeight", event_weight)

    # A Z+jet event only enters a category plot once both the Z and the probe
    # jet exist. This matters for the histograms booked by --add-no-selection.
    base = "Z.valid() && Probe.valid()"

    for category, (eta_min, eta_max) in ETA_CATEGORIES.items():
        selection = (
            f"{base}"
            f" && fabs(Probe.eta()) >= {eta_min}f"
            f" && fabs(Probe.eta()) < {eta_max}f"
        )
        df = df.Define(
            f"etaWeight_{category}",
            f"({selection}) ? eventWeight : 0.0"
        )

    for category, (pt_min, pt_max) in ZPT_CATEGORIES.items():
        selection = (
            f"{base}"
            f" && Z.pt() >= {pt_min}f"
            f" && Z.pt() < {pt_max}f"
        )
        df = df.Define(
            f"zptWeight_{category}",
            f"({selection}) ? eventWeight : 0.0"
        )

    return df


def get_regions(sample=None, args=None, config=None):
    """
    Regions are deliberately minimal.

    The |eta| and Z-pT categories are encoded in weight columns, so the generic
    runner only needs one physical Z+jet selection.

    passTrigger is defined by define_columns from DEFAULT_TRIGGERS, or from
    `triggers` in the analysis config, and is applied to data and MC alike.

    The muon requirements are two-sided. nTightMuons >= 2 asks for the two
    legs of the Z at the analysis working point, and nVetoMuons == 0 rejects
    any additional muon that is merely loose, so the event is a clean dimuon
    event rather than one with a third muon hiding under the threshold.
    Everything the Z muons must satisfy is already in TightMuons, so the
    region itself only adds kinematics.

    The Z is selected by muon kinematics rather than by pT(Z): requiring the
    subleading muon above 27 GeV (which implies the leading one) puts both legs
    on the IsoMu24 plateau, and is the only thing that does so now that the
    dimuon builder is symmetric. It also leaves the pT(Z) spectrum unsculpted,
    so pT(Z) stays available as an observable instead of a selection variable.
    """
    return {
        "zjet": {
            "cuts": [
                "passTrigger",
                "nTightMuons >= 2",
                "nVetoMuons == 0",
                "Z.valid()",
                "fabs(Z.mass() - 91.1876f) < 3.743f",
                "Mu2_pt > 27.f",
                "GoodJets.size() > 0",
                "dPhi_ZProbe > 2.7"
            ]
        },
        f"zjet_{MET_TRIGGER}": {
            "cuts": [
                "passTrigger",
                "nTightMuons >= 2",
                "nVetoMuons == 0",
                "Z.valid()",
                "fabs(Z.mass() - 91.1876f) < 3.743f",
                "Mu2_pt > 27.f",
                "GoodJets.size() > 0",
                "dPhi_ZProbe > 2.7",
                "passMETTrigger"
            ]
        }
    }
