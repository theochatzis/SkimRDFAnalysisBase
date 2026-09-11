#ifndef SKIMRDF_OBJECTBUILDERS_H
#define SKIMRDF_OBJECTBUILDERS_H

#include "PhysicsObjects.h"
#include "Kinematics.h"
#include "ROOT/RVec.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace skimrdf {

using ROOT::VecOps::RVec;

template <typename TMedium, typename TTight, typename TIso>
RVec<Muon> buildMuons(
    const RVec<float>& pt,
    const RVec<float>& eta,
    const RVec<float>& phi,
    const RVec<float>& mass,
    const RVec<int>& charge,
    const RVec<TMedium>& mediumId,
    const RVec<TTight>& tightId,
    const RVec<TIso>& pfIsoId,
    const RVec<float>& pfRelIso04
) {
    RVec<Muon> out;
    out.reserve(pt.size());

    for (std::size_t i = 0; i < pt.size(); ++i) {
        out.emplace_back(
            pt[i], eta[i], phi[i], mass[i], charge[i],
            static_cast<bool>(mediumId[i]),
            static_cast<bool>(tightId[i]),
            static_cast<int>(pfIsoId[i]),
            pfRelIso04[i]
        );
    }
    return out;
}

template <typename TJetId>
RVec<Jet> buildJets(
    const RVec<float>& pt,
    const RVec<float>& eta,
    const RVec<float>& phi,
    const RVec<float>& mass,
    const RVec<float>& rawFactor,
    const RVec<float>& area,
    const RVec<TJetId>& jetId,
    const RVec<float>& chHEF,
    const RVec<float>& neHEF,
    const RVec<float>& chEmEF,
    const RVec<float>& neEmEF,
    const RVec<float>& muEF,
    const RVec<int>& genJetIdx
) {
    RVec<Jet> out;
    out.reserve(pt.size());

    for (std::size_t i = 0; i < pt.size(); ++i) {
        out.emplace_back(
            pt[i], eta[i], phi[i], mass[i],
            rawFactor[i], area[i], static_cast<int>(jetId[i]),
            chHEF[i], neHEF[i], chEmEF[i], neEmEF[i], muEF[i],
            genJetIdx[i]
        );
    }
    return out;
}

RVec<GenJet> buildGenJets(
    const RVec<float>& pt,
    const RVec<float>& eta,
    const RVec<float>& phi,
    const RVec<float>& mass
) {
    RVec<GenJet> out;
    out.reserve(pt.size());
    for (std::size_t i = 0; i < pt.size(); ++i) {
        out.emplace_back(pt[i], eta[i], phi[i], mass[i]);
    }
    return out;
}

inline Dimuon selectBestDimuon(
    const RVec<Muon>& muons,
    bool requireMediumId = true,
    bool requireTightTag = true,
    bool usePfIsoId = true,
    bool useRelIso = false,
    float probeRelIsoMax = 0.25f,
    float tagRelIsoMax = 0.15f,
    double targetMass = 91.1876
) {
    Dimuon best;
    double bestDistance = std::numeric_limits<double>::max();

    auto passBaseline = [&](const Muon& muon) {
        if (muon.pt() <= 10.f) return false;
        if (std::abs(muon.eta()) >= 2.4f) return false;

        if (requireMediumId && !muon.mediumId()) return false;

        if (usePfIsoId) {
            if (muon.pfIsoId() < 2) return false;
        } else if (useRelIso) {
            if (muon.pfRelIso04() < 0.f) return false;
            if (muon.pfRelIso04() >= probeRelIsoMax) return false;
        }

        return true;
    };

    auto passTag = [&](const Muon& muon) {
        if (muon.pt() <= 27.f) return false;
        if (std::abs(muon.eta()) >= 2.4f) return false;

        if (requireTightTag && !muon.tightId()) return false;

        if (usePfIsoId) {
            if (muon.pfIsoId() < 4) return false;
        } else if (useRelIso) {
            if (muon.pfRelIso04() < 0.f) return false;
            if (muon.pfRelIso04() >= tagRelIsoMax) return false;
        }

        return true;
    };

    for (std::size_t i = 0; i < muons.size(); ++i) {
        if (!passBaseline(muons[i])) continue;

        for (std::size_t j = i + 1; j < muons.size(); ++j) {
            if (!passBaseline(muons[j])) continue;
            if (muons[i].charge() * muons[j].charge() >= 0) continue;

            if (!passTag(muons[i]) && !passTag(muons[j])) continue;

            Dimuon candidate(muons[i], muons[j]);

            const double dm = std::abs(
                candidate.mass() - targetMass
            );

            if (dm < bestDistance) {
                bestDistance = dm;
                best = candidate;
            }
        }
    }

    return best;
}

inline RVec<Jet> selectJets(
    const RVec<Jet>& jets,
    const Dimuon& z,
    float minPt = 15.f,
    float maxAbsEta = 5.f,
    bool requireTightId = true,
    float minMuonDr = 0.2f
) {
    RVec<Jet> out;
    out.reserve(jets.size());

    for (const auto& jet : jets) {
        if (!jet.valid()) continue;
        if (jet.pt() <= minPt) continue;
        if (std::abs(jet.eta()) >= maxAbsEta) continue;
        if (requireTightId && !jet.passTightId()) continue;

        if (z.valid()) {
            if (deltaR(jet.eta(), jet.phi(), z.muon1().eta(), z.muon1().phi()) <= minMuonDr) continue;
            if (deltaR(jet.eta(), jet.phi(), z.muon2().eta(), z.muon2().phi()) <= minMuonDr) continue;
        }

        out.push_back(jet);
    }

    std::sort(out.begin(), out.end(),
              [](const Jet& a, const Jet& b) { return a.pt() > b.pt(); });
    return out;
}

inline RVec<GenJet> selectGenJets(
    const RVec<GenJet>& jets,
    float minPt = 15.f,
    float maxAbsEta = 5.f
) {
    RVec<GenJet> out;
    out.reserve(jets.size());
    for (const auto& jet : jets) {
        if (jet.pt() <= minPt) continue;
        if (std::abs(jet.eta()) >= maxAbsEta) continue;
        out.push_back(jet);
    }
    return out;
}

inline Jet leadingJet(const RVec<Jet>& jets) {
    if (jets.empty()) return Jet();
    return jets.front();
}

inline float secondJetPt(const RVec<Jet>& jets) {
    return jets.size() > 1 ? jets[1].pt() : 0.f;
}

}  // namespace skimrdf

#endif
