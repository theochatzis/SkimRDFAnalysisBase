"""
Z+jet RDF definition for the generic SkimRDFAnalysisBase/run_analysis.py driver.

Required interface used by run_analysis.py:
    setup(args, config)
    define_columns(df, sample, args, config)
    get_regions(sample, args, config)

The analysis-specific file only defines physics objects/columns and regions.
Input discovery, event-loop execution, histogram booking and output writing stay
inside the generic run_analysis.py.
"""

import os
import ROOT


ETA_CATEGORIES = {
    "Incl": (0.0, 5.0),
    "HB":   (0.0, 1.3),
    "HE1":  (1.5, 2.5),
    "HE2":  (2.5, 3.0),
    "HF":   (3.0, 5.0),
}

PT_CATEGORIES = {
    "pt15to30":    (15.0, 30.0),
    "pt30to60":    (30.0, 60.0),
    "pt60to120":   (60.0, 120.0),
    "pt120to300":  (120.0, 300.0),
    "pt300to1000": (300.0, 1000.0),
    "pt1000plus":  (1000.0, 1.0e9),
}

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
    has_muon_medium_id = "Muon_mediumId" in columns
    has_muon_tight_id = "Muon_tightId" in columns
    has_muon_pf_iso_id = "Muon_pfIsoId" in columns
    has_muon_rel_iso = "Muon_pfRelIso04_all" in columns
    has_jet_id = "Jet_jetId" in columns

    sample_name = (sample or {}).get("name", "<sample>")

    if not has_muon_medium_id:
        print(
            f"[zjet:{sample_name}] Muon_mediumId not found: "
            "baseline medium-ID requirement disabled."
        )

    if not has_muon_tight_id:
        print(
            f"[zjet:{sample_name}] Muon_tightId not found: "
            "tight tag-ID requirement disabled."
        )

    if not has_muon_pf_iso_id:
        if has_muon_rel_iso:
            print(
                f"[zjet:{sample_name}] Muon_pfIsoId not found: "
                "using pfRelIso04_all (<0.25 probe, <0.15 tag)."
            )
        else:
            print(
                f"[zjet:{sample_name}] No muon isolation branch found: "
                "muon isolation requirement disabled."
            )

    if not has_jet_id:
        print(
            f"[zjet:{sample_name}] Jet_jetId not found: "
            "tight jet-ID requirement disabled."
        )

    # ------------------------------------------------------------------
    # NanoAOD compatibility / optional branches
    # ------------------------------------------------------------------
    df = _ensure_vector(
        df, columns,
        "Muon_mass", "float", "nMuon", "0.105658f"
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

    # ------------------------------------------------------------------
    # Data / MC event weight
    # ------------------------------------------------------------------
    weight_expression = config.get(
        "event_weight_expression",
        ""
    )

    if not weight_expression:
        if "genWeight" in columns:
            weight_expression = (
                "(genWeight >= 0.f ? 1.0 : -1.0)"
            )
        else:
            weight_expression = "1.0"

    df = df.Define(
        "eventWeight",
        weight_expression
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
            Muon_mediumId,
            Muon_tightId,
            Muon_pfIsoId,
            Muon_pfRelIso04_all
        )
        """
    )

    use_rel_iso = (
        (not has_muon_pf_iso_id)
        and has_muon_rel_iso
    )

    df = df.Define(
        "Z",
        f"""
        skimrdf::selectBestDimuon(
            Muons,
            {str(has_muon_medium_id).lower()},
            {str(has_muon_tight_id).lower()},
            {str(has_muon_pf_iso_id).lower()},
            {str(use_rel_iso).lower()},
            0.25f,
            0.15f,
            91.1876
        )
        """
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

    # Clean reco jets from the Z muons and order them in pT.
    df = df.Define(
        "GoodJets",
        f"""
        skimrdf::selectJets(
            Jets,
            Z,
            15.f,
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
        .Define("Z_eta", "Z.eta()")
        .Define("Z_phi", "Z.phi()")
        .Define("Z_mass", "Z.mass()")
        .Define("Probe_pt", "Probe.pt()")
        .Define("Probe_eta", "Probe.eta()")
        .Define("Probe_phi", "Probe.phi()")
        .Define("Probe_chHEF", "Probe.chHEF()")
        .Define("MET_pt", "AnalysisMET.pt()")
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
        .Define(
            "zjetValid",
            """
            Z.valid()
            && fabs(Z.mass() - 91.1876f) < 3.743f
            && Z.pt() > 20.f
            && GoodJets.size() > 0
            """
        )
        .Define(
            "signalWindowWeight",
            """
            zjetValid
            ? eventWeight * (
                skimrdf::inOppositeWindow(
                    Z.phi(),
                    Probe.phi()
                ) ? 1.0 : 0.0
              )
            : 0.0
            """
        )
        .Define(
            "windowWeight",
            """
            zjetValid
            ? eventWeight * skimrdf::windowedBalanceWeight(
                Z.phi(),
                Probe.phi()
              )
            : 0.0
            """
        )
    )

    # ------------------------------------------------------------------
    # Eta-category weights for DB / MPF profiles
    # ------------------------------------------------------------------
    #
    # Keeping eta categories as weight columns means the same dataframe and
    # one generic zjet region can produce all eta-dependent response plots.
    for category, (eta_min, eta_max) in ETA_CATEGORIES.items():
        eta_selection = (
            f"(fabs(Probe.eta()) >= {eta_min}f "
            f"&& fabs(Probe.eta()) < {eta_max}f)"
        )

        df = (
            df
            .Define(
                f"signalWeight_{category}",
                f"signalWindowWeight * ({eta_selection} ? 1.0 : 0.0)"
            )
            .Define(
                f"windowWeight_{category}",
                f"windowWeight * ({eta_selection} ? 1.0 : 0.0)"
            )
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
            .Define(
                "GenJets",
                "skimrdf::selectGenJets(GenJetsAll, 15.f, 5.f)"
            )
            .Define(
                "RecoMatchFlags",
                f"""
                skimrdf::matchRecoToGenFlags(
                    GoodJets,
                    GenJets,
                    {match_dr}f
                )
                """
            )
            .Define(
                "GenMatchFlags",
                f"""
                skimrdf::matchGenToRecoFlags(
                    GenJets,
                    GoodJets,
                    {match_dr}f
                )
                """
            )
        )
    else:
        # Define empty MC collections on data so the same histogram YAML can
        # be used for data and simulation. MC-only histograms are simply empty.
        df = (
            df
            .Define(
                "GenJets",
                "ROOT::VecOps::RVec<skimrdf::GenJet>{}"
            )
            .Define(
                "RecoMatchFlags",
                "ROOT::VecOps::RVec<int>(GoodJets.size(), 0)"
            )
            .Define(
                "GenMatchFlags",
                "ROOT::VecOps::RVec<int>{}"
            )
        )

    # ------------------------------------------------------------------
    # MC study columns
    # ------------------------------------------------------------------
    #
    # Keep these truly empty on data so the same histogram YAML can be used
    # for data and MC without producing misleading purity denominators.
    if has_genjets:
        # Efficiency / purity versus pT in eta categories.
        for category, (eta_min, eta_max) in ETA_CATEGORIES.items():
            df = (
                df
                .Define(
                    f"GenPt_{category}",
                    f"""
                    skimrdf::objectPt(
                        GenJets,
                        {eta_min}f,
                        {eta_max}f
                    )
                    """
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
                    f"""
                    skimrdf::objectPt(
                        GoodJets,
                        {eta_min}f,
                        {eta_max}f
                    )
                    """
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
                    f"RecoCompPt_{category}",
                    f"""
                    skimrdf::recoJetPtForComposition(
                        GoodJets,
                        {eta_min}f,
                        {eta_max}f
                    )
                    """
                )
                .Define(
                    f"RecoChHEF_{category}",
                    f"""
                    skimrdf::recoJetChHEF(
                        GoodJets,
                        {eta_min}f,
                        {eta_max}f
                    )
                    """
                )
            )

        # Efficiency / purity versus eta in pT categories.
        for category, (pt_min, pt_max) in PT_CATEGORIES.items():
            df = (
                df
                .Define(
                    f"GenEta_{category}",
                    f"""
                    skimrdf::objectEta(
                        GenJets,
                        {pt_min}f,
                        {pt_max}f
                    )
                    """
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
                    f"""
                    skimrdf::objectEta(
                        GoodJets,
                        {pt_min}f,
                        {pt_max}f
                    )
                    """
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
                    f"RecoCompEta_{category}",
                    f"""
                    skimrdf::recoJetEtaForComposition(
                        GoodJets,
                        {pt_min}f,
                        {pt_max}f
                    )
                    """
                )
                .Define(
                    f"RecoChHEF_eta_{category}",
                    f"""
                    skimrdf::recoJetChHEFForEta(
                        GoodJets,
                        {pt_min}f,
                        {pt_max}f
                    )
                    """
                )
            )

    else:
        # Empty float vectors for all MC-only histogram variables.
        for category in ETA_CATEGORIES:
            for column in (
                f"GenPt_{category}",
                f"GenMatchedPt_{category}",
                f"RecoPt_{category}",
                f"RecoMatchedPt_{category}",
                f"RecoCompPt_{category}",
                f"RecoChHEF_{category}",
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
                f"RecoCompEta_{category}",
                f"RecoChHEF_eta_{category}",
            ):
                df = df.Define(
                    column,
                    "ROOT::VecOps::RVec<float>{}"
                )

    return df


def get_regions(sample=None, args=None, config=None):
    """
    Regions are deliberately minimal.

    Eta categories are encoded in profile weights / vector columns, so the
    generic runner only needs one physical Z+jet selection.
    """
    return {
        "zjet": {
            "cuts": [
                "Z.valid()",
                "fabs(Z.mass() - 91.1876f) < 3.743f",
                "Z.pt() > 20.f",
                "GoodJets.size() > 0",
            ]
        }
    }
