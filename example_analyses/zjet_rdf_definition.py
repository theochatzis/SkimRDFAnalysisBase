"""
Z+jet example.
"""

import ROOT


CPP_HELPERS = r"""
#include <ROOT/RVec.hxx>
#include <Math/Vector4D.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace zjet {

using ROOT::VecOps::RVec;
using P4 = ROOT::Math::PtEtaPhiMVector;

constexpr double PI = 3.14159265358979323846;
constexpr double MZ = 91.1880;


// Return {positive-muon index, negative-muon index} for the nominal Z pair.
//   both: mediumId, pfIsoId >= 2, pT > 10, |eta| < 2.4
//   leading muon pT > 27
//   at least one tightId, pfIsoId >= 4, pT > 27 tag
//   choose OS pair closest to mZ
template <typename TMedium, typename TTight, typename TIso>
RVec<int> selectZPair(
    const RVec<float>& pt,
    const RVec<float>& eta,
    const RVec<float>& phi,
    const RVec<float>& mass,
    const RVec<int>& charge,
    const RVec<TMedium>& mediumId,
    const RVec<TTight>& tightId,
    const RVec<TIso>& pfIsoId
) {
    int bestPlus = -1;
    int bestMinus = -1;
    double bestDistance = std::numeric_limits<double>::max();

    auto passProbe = [&](std::size_t i) {
        return (
            static_cast<bool>(mediumId[i]) &&
            static_cast<int>(pfIsoId[i]) >= 2 &&
            pt[i] > 10.f &&
            std::abs(eta[i]) < 2.4f
        );
    };

    auto isTag = [&](std::size_t i) {
        return (
            pt[i] > 27.f &&
            static_cast<bool>(tightId[i]) &&
            static_cast<int>(pfIsoId[i]) >= 4
        );
    };

    for (std::size_t ip = 0; ip < pt.size(); ++ip) {
        if (charge[ip] <= 0 || !passProbe(ip))
            continue;

        const P4 plus(pt[ip], eta[ip], phi[ip], mass[ip]);

        for (std::size_t im = 0; im < pt.size(); ++im) {
            if (charge[im] >= 0 || !passProbe(im))
                continue;

            const P4 minus(pt[im], eta[im], phi[im], mass[im]);

            if (std::max(plus.Pt(), minus.Pt()) <= 27.)
                continue;

            if (!isTag(ip) && !isTag(im))
                continue;

            const double distance = std::abs((plus + minus).M() - MZ);

            if (distance < bestDistance) {
                bestDistance = distance;
                bestPlus = static_cast<int>(ip);
                bestMinus = static_cast<int>(im);
            }
        }
    }

    return {bestPlus, bestMinus};
}


inline double deltaPhi(double phi1, double phi2) {
    return std::remainder(phi1 - phi2, 2.0 * PI);
}

inline double deltaR(
    double eta1, double phi1,
    double eta2, double phi2
) {
    const double deta = eta1 - eta2;
    const double dphi = deltaPhi(phi1, phi2);
    return std::sqrt(deta*deta + dphi*dphi);
}


// Explicit Run-3 Tight PF Jet ID.
template <typename TChMult, typename TNeMult, typename TNConst>
bool passTightJetId(
    std::size_t i,
    const RVec<float>& eta,
    const RVec<float>& neHEF,
    const RVec<float>& neEmEF,
    const RVec<float>& chHEF,
    const RVec<TChMult>& chMultiplicity,
    const RVec<TNeMult>& neMultiplicity,
    const RVec<TNConst>& nConstituents
) {
    const double abseta = std::abs(eta[i]);

    if (abseta <= 2.6)
        return (
            neHEF[i] < 0.99 &&
            neEmEF[i] < 0.90 &&
            nConstituents[i] > 1 &&
            chHEF[i] > 0.01 &&
            chMultiplicity[i] > 0
        );

    if (abseta <= 2.7)
        return (
            neHEF[i] < 0.90 &&
            neEmEF[i] < 0.99
        );

    if (abseta <= 3.0)
        return neHEF[i] < 0.99;

    if (abseta < 5.0)
        return (
            neEmEF[i] < 0.40 &&
            neMultiplicity[i] >= 2
        );

    return false;
}


// Return the highest-pT jet passing the simple Z+jet recoil selection.
template <typename TChMult, typename TNeMult, typename TNConst>
int leadingRecoilJet(
    const RVec<float>& jetPt,
    const RVec<float>& jetEta,
    const RVec<float>& jetPhi,
    const RVec<float>& jetNeHEF,
    const RVec<float>& jetNeEmEF,
    const RVec<float>& jetChHEF,
    const RVec<TChMult>& jetChMultiplicity,
    const RVec<TNeMult>& jetNeMultiplicity,
    const RVec<TNConst>& jetNConstituents,
    double zPt,
    double zPhi,
    double muPlusEta,
    double muPlusPhi,
    double muMinusEta,
    double muMinusPhi
) {
    int best = -1;
    double bestPt = -1.;

    const double recoilPhi = std::remainder(zPhi + PI, 2.0*PI);

    for (std::size_t i = 0; i < jetPt.size(); ++i) {

        if (!passTightJetId(
                i,
                jetEta,
                jetNeHEF,
                jetNeEmEF,
                jetChHEF,
                jetChMultiplicity,
                jetNeMultiplicity,
                jetNConstituents
            ))
            continue;

        if (deltaR(jetEta[i], jetPhi[i], muPlusEta, muPlusPhi) <= 0.2)
            continue;

        if (deltaR(jetEta[i], jetPhi[i], muMinusEta, muMinusPhi) <= 0.2)
            continue;

        // Narrow recoil direction
        if (std::abs(deltaPhi(jetPhi[i], recoilPhi)) >= PI/16.0)
            continue;

        // Individual jet/Z balance range.
        if (jetPt[i] <= 0.5*zPt || jetPt[i] >= 2.0*zPt)
            continue;

        if (jetPt[i] > bestPt) {
            bestPt = jetPt[i];
            best = static_cast<int>(i);
        }
    }

    return best;
}

} // namespace zjet
"""


def setup(args, config):
    ROOT.gInterpreter.Declare(CPP_HELPERS)


def define_columns(df, sample, args, config):

    return (
        df

        # ------------------------------------------------------
        # Z candidate
        # ------------------------------------------------------
        .Define(
            "Z_pairIdx",
            """
            zjet::selectZPair(
                Muon_pt,
                Muon_eta,
                Muon_phi,
                Muon_mass,
                Muon_charge,
                Muon_mediumId,
                Muon_tightId,
                Muon_pfIsoId
            )
            """
        )
        .Define("Z_valid", "Z_pairIdx[0] >= 0 && Z_pairIdx[1] >= 0")

        .Define(
            "Z_p4",
            """
            Z_valid ?
                ROOT::Math::PtEtaPhiMVector(
                    Muon_pt[Z_pairIdx[0]],
                    Muon_eta[Z_pairIdx[0]],
                    Muon_phi[Z_pairIdx[0]],
                    Muon_mass[Z_pairIdx[0]]
                )
                +
                ROOT::Math::PtEtaPhiMVector(
                    Muon_pt[Z_pairIdx[1]],
                    Muon_eta[Z_pairIdx[1]],
                    Muon_phi[Z_pairIdx[1]],
                    Muon_mass[Z_pairIdx[1]]
                )
                :
                ROOT::Math::PtEtaPhiMVector(0., 0., 0., 0.)
            """
        )

        .Define("Z_pt",   "Z_p4.Pt()")
        .Define("Z_eta",  "Z_p4.Eta()")
        .Define("Z_phi",  "Z_p4.Phi()")
        .Define("Z_mass", "Z_p4.M()")

        # ------------------------------------------------------
        # One leading recoil jet
        # ------------------------------------------------------
        .Define(
            "Probe_idx",
            """
            Z_valid ?
                zjet::leadingRecoilJet(
                    Jet_pt,
                    Jet_eta,
                    Jet_phi,
                    Jet_neHEF,
                    Jet_neEmEF,
                    Jet_chHEF,
                    Jet_chMultiplicity,
                    Jet_neMultiplicity,
                    Jet_nConstituents,
                    Z_pt,
                    Z_phi,
                    Muon_eta[Z_pairIdx[0]],
                    Muon_phi[Z_pairIdx[0]],
                    Muon_eta[Z_pairIdx[1]],
                    Muon_phi[Z_pairIdx[1]]
                )
                :
                -1
            """
        )

        .Define("Probe_valid", "Probe_idx >= 0")
        .Define("Probe_pt",  "Probe_valid ? Jet_pt[Probe_idx]  : -1.f")
        .Define("Probe_eta", "Probe_valid ? Jet_eta[Probe_idx] : -99.f")
        .Define("Probe_phi", "Probe_valid ? Jet_phi[Probe_idx] : 0.f")

        # ------------------------------------------------------
        # Scalar response observables
        # ------------------------------------------------------

        # Direct Balance
        .Define(
            "DB",
            "Probe_valid && Z_pt > 0.f ? Probe_pt / Z_pt : -1.f"
        )

        # Missing-pT Projection Fraction
        .Define(
            "MPF",
            """
            Z_pt > 0.f ?
                1.f
                + PuppiMET_pt / Z_pt
                  * cos(zjet::deltaPhi(PuppiMET_phi, Z_phi))
                :
                -999.f
            """
        )
    )


def get_regions(sample, args, config):

    return {
        "zjet": {
            "cuts": [
                (
                    "Flag_goodVertices && "
                    "Flag_globalSuperTightHalo2016Filter && "
                    "Flag_EcalDeadCellTriggerPrimitiveFilter && "
                    "Flag_BadPFMuonFilter && "
                    "Flag_BadPFMuonDzFilter && "
                    "Flag_hfNoisyHitsFilter && "
                    "Flag_eeBadScFilter && "
                    "Flag_ecalBadCalibFilter"
                ),

                "HLT_IsoMu24",

                "Z_valid",

                # |m_mumu - mZ| < 1.5 Gamma_Z
                "abs(Z_mass - 91.1880f) < 3.74325f",

                "Probe_valid",
            ]
        }
    }
