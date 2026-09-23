#ifndef SKIMRDF_OBJECTBUILDERS_H
#define SKIMRDF_OBJECTBUILDERS_H

#include "PhysicsObjects.h"
#include "Kinematics.h"
#include "ROOT/RVec.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace skimrdf {

using ROOT::VecOps::RVec;

// -----------------------------------------------------------------------------
// Generic dR cleaning
// -----------------------------------------------------------------------------
//
// These helpers are deliberately agnostic about what they are cleaning and what
// they are cleaning against: any object exposing eta() and phi() works on both
// sides. The same code therefore removes jets overlapping with muons, gen jets
// overlapping with leptons, jets overlapping with photons, and so on.
//
// Objects that also expose valid() are skipped on the veto side when invalid,
// so a default-constructed placeholder never vetoes anything.
//
namespace detail {

template <typename T, typename = void>
struct HasValid : std::false_type {};

template <typename T>
struct HasValid<T, std::void_t<decltype(std::declval<const T&>().valid())>>
    : std::true_type {};

template <typename T>
inline bool isUsable(const T& object) {
    if constexpr (HasValid<T>::value) {
        return object.valid();
    } else {
        return true;
    }
}

}  // namespace detail

// True if `object` is farther than `minDr` from every usable element of `vetoes`.
template <typename TObject, typename TVeto>
inline bool isSeparatedFrom(
    const TObject& object,
    const RVec<TVeto>& vetoes,
    float minDr
) {
    const double minDr2 = static_cast<double>(minDr) * static_cast<double>(minDr);

    for (const auto& veto : vetoes) {
        if (!detail::isUsable(veto)) continue;
        if (deltaR2(object.eta(), object.phi(), veto.eta(), veto.phi()) <= minDr2) {
            return false;
        }
    }
    return true;
}

// Keep the elements of `objects` that are farther than `minDr` from every
// element of `vetoes`. The order of `objects` is preserved.
template <typename TObject, typename TVeto>
inline RVec<TObject> cleanByDeltaR(
    const RVec<TObject>& objects,
    const RVec<TVeto>& vetoes,
    float minDr
) {
    RVec<TObject> out;
    out.reserve(objects.size());

    for (const auto& object : objects) {
        if (!isSeparatedFrom(object, vetoes, minDr)) continue;
        out.push_back(object);
    }
    return out;
}

// Per-object cleaning decision, for when the mask itself is of interest
// (control plots, cutflows, or cleaning several collections consistently).
template <typename TObject, typename TVeto>
inline RVec<int> cleaningMask(
    const RVec<TObject>& objects,
    const RVec<TVeto>& vetoes,
    float minDr
) {
    RVec<int> mask(objects.size(), 0);
    for (std::size_t i = 0; i < objects.size(); ++i) {
        mask[i] = isSeparatedFrom(objects[i], vetoes, minDr) ? 1 : 0;
    }
    return mask;
}


template <typename TLoose, typename TMedium, typename TTight, typename TIso>
RVec<Muon> buildMuons(
    const RVec<float>& pt,
    const RVec<float>& eta,
    const RVec<float>& phi,
    const RVec<float>& mass,
    const RVec<int>& charge,
    const RVec<TLoose>& looseId,
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
            static_cast<bool>(looseId[i]),
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

// -----------------------------------------------------------------------------
// Muon working points
// -----------------------------------------------------------------------------
//
// A working point is an ID level plus an isolation bound plus kinematics. Both
// the analysis muons and the veto muons are built from the same helpers, so a
// threshold is written once and the two collections cannot drift apart.
//
// `Any` imposes no ID requirement. It exists so that a caller can degrade
// gracefully when the input schema has no such ID branch, rather than silently
// failing every muon against an all-zero fallback vector. It is spelled `Any`
// rather than `None` because X11 headers, which ROOT may pull in, define
// `None` as a macro.
enum class MuonId { Any = 0, Loose = 1, Medium = 2, Tight = 3 };

inline bool passMuonId(const Muon& muon, MuonId id) {
    switch (id) {
        case MuonId::Tight:  return muon.tightId();
        case MuonId::Medium: return muon.mediumId();
        case MuonId::Loose:  return muon.looseId();
        case MuonId::Any:    return true;
    }
    return true;
}

// `relIsoMax` is the upper bound on the dBeta-corrected pfRelIso04. A
// non-positive value disables the isolation requirement, which is what the
// caller should pass when the input schema has no isolation branch.
inline bool passMuonWP(
    const Muon& muon,
    MuonId id,
    float relIsoMax,
    float ptMin,
    float etaMax = 2.4f
) {
    if (muon.pt() <= ptMin) return false;
    if (std::abs(muon.eta()) >= etaMax) return false;

    if (!passMuonId(muon, id)) return false;

    if (relIsoMax > 0.f) {
        // A negative isolation means the value is missing, not small.
        if (muon.pfRelIso04() < 0.f) return false;
        if (muon.pfRelIso04() >= relIsoMax) return false;
    }

    return true;
}

// Muons passing a working point, in the input order.
inline RVec<Muon> selectMuons(
    const RVec<Muon>& muons,
    MuonId id,
    float relIsoMax,
    float ptMin,
    float etaMax = 2.4f
) {
    RVec<Muon> out;
    out.reserve(muons.size());

    for (const auto& muon : muons) {
        if (!passMuonWP(muon, id, relIsoMax, ptMin, etaMax)) continue;
        out.push_back(muon);
    }
    return out;
}

// Additional muons in the event: those passing the loose veto working point
// but failing the analysis one. Requiring this collection to be empty rejects
// events with a third loose muon without also rejecting the two analysis
// muons, which pass the veto working point by construction.
inline RVec<Muon> selectVetoMuons(
    const RVec<Muon>& muons,
    MuonId vetoId,
    float vetoRelIsoMax,
    float vetoPtMin,
    MuonId analysisId,
    float analysisRelIsoMax,
    float analysisPtMin,
    float etaMax = 2.4f
) {
    RVec<Muon> out;
    out.reserve(muons.size());

    for (const auto& muon : muons) {
        if (!passMuonWP(muon, vetoId, vetoRelIsoMax, vetoPtMin, etaMax)) continue;
        if (passMuonWP(muon, analysisId, analysisRelIsoMax, analysisPtMin, etaMax)) continue;
        out.push_back(muon);
    }
    return out;
}

// Opposite-sign pair from `muons` whose invariant mass is closest to
// `targetMass`.
//
// The muons are expected to be pre-selected by the caller, typically with
// selectMuons, so that the same collection can also be counted and used to
// clean the jets. There is deliberately no tag/probe asymmetry and no working
// point here: for a Z+jet measurement the dimuon system is the pT reference,
// so both legs must satisfy the same requirement, and that requirement belongs
// in one place rather than being re-stated inside this function.
inline Dimuon selectBestDimuon(
    const RVec<Muon>& muons,
    double targetMass = 91.1876
) {
    Dimuon best;
    double bestDistance = std::numeric_limits<double>::max();

    for (std::size_t i = 0; i < muons.size(); ++i) {
        for (std::size_t j = i + 1; j < muons.size(); ++j) {
            if (muons[i].charge() * muons[j].charge() >= 0) continue;

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

// Expose the daughters of a composite object as a collection, so that a Z
// candidate (or any other composite) can be used as a cleaning collection.
inline RVec<Muon> dimuonMuons(const Dimuon& z) {
    RVec<Muon> out;
    if (!z.valid()) return out;

    out.reserve(2);
    out.push_back(z.muon1());
    out.push_back(z.muon2());
    return out;
}

// Kinematic + ID jet selection, cleaned against an arbitrary collection.
// `cleaningCollection` can hold any object with eta()/phi() accessors, e.g.
// the muons of the Z candidate (skimrdf::dimuonMuons(Z)), electrons, photons.
template <typename TVeto>
inline RVec<Jet> selectJets(
    const RVec<Jet>& jets,
    const RVec<TVeto>& cleaningCollection,
    float minPt = 15.f,
    float maxAbsEta = 5.f,
    bool requireTightId = true,
    float minCleaningDr = 0.2f,
    bool order = true
) {
    RVec<Jet> out;
    out.reserve(jets.size());

    for (const auto& jet : jets) {
        if (!jet.valid()) continue;
        if (jet.pt() <= minPt) continue;
        if (std::abs(jet.eta()) >= maxAbsEta) continue;
        if (requireTightId && !jet.passTightId()) continue;
        if (!isSeparatedFrom(jet, cleaningCollection, minCleaningDr)) continue;

        out.push_back(jet);
    }

    if (order) {
        std::sort(out.begin(), out.end(),
                  [](const Jet& a, const Jet& b) { return a.pt() > b.pt(); });
    }
    return out;
}

// Same selection without any cleaning.
inline RVec<Jet> selectJets(
    const RVec<Jet>& jets,
    float minPt = 15.f,
    float maxAbsEta = 5.f,
    bool requireTightId = true,
    bool order = true
) {
    return selectJets(jets, RVec<Jet>{}, minPt, maxAbsEta,
                      requireTightId, -1.f, order);
}

inline RVec<GenJet> selectGenJets(
    const RVec<GenJet>& jets,
    float minPt = 15.f,
    float maxAbsEta = 5.f,
    bool order = true
) {
    RVec<GenJet> out;
    out.reserve(jets.size());
    for (const auto& jet : jets) {
        if (jet.pt() <= minPt) continue;
        if (std::abs(jet.eta()) >= maxAbsEta) continue;
        out.push_back(jet);
    }

    if (order) {
        std::sort(out.begin(), out.end(),
                  [](const GenJet& a, const GenJet& b) {
                      return a.pt() > b.pt();
                  });
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
