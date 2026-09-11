#ifndef SKIMRDF_JETMATCHING_H
#define SKIMRDF_JETMATCHING_H

#include "PhysicsObjects.h"
#include "Kinematics.h"
#include "ROOT/RVec.hxx"

#include <cmath>
#include <cstddef>
#include <limits>

namespace skimrdf {

using ROOT::VecOps::RVec;

inline RVec<int> matchRecoToGenFlags(
    const RVec<Jet>& recoJets,
    const RVec<GenJet>& genJets,
    float maxDR = 0.2f
) {
    RVec<int> matched(recoJets.size(), 0);
    const double maxDR2 = static_cast<double>(maxDR) * maxDR;

    for (std::size_t i = 0; i < recoJets.size(); ++i) {
        double best = maxDR2;
        for (std::size_t j = 0; j < genJets.size(); ++j) {
            const double dr2 = deltaR2(
                recoJets[i].eta(), recoJets[i].phi(),
                genJets[j].eta(), genJets[j].phi()
            );
            if (dr2 < best) {
                best = dr2;
                matched[i] = 1;
            }
        }
    }
    return matched;
}

inline RVec<int> matchGenToRecoFlags(
    const RVec<GenJet>& genJets,
    const RVec<Jet>& recoJets,
    float maxDR = 0.2f
) {
    RVec<int> matched(genJets.size(), 0);
    const double maxDR2 = static_cast<double>(maxDR) * maxDR;

    for (std::size_t i = 0; i < genJets.size(); ++i) {
        double best = maxDR2;
        for (std::size_t j = 0; j < recoJets.size(); ++j) {
            const double dr2 = deltaR2(
                genJets[i].eta(), genJets[i].phi(),
                recoJets[j].eta(), recoJets[j].phi()
            );
            if (dr2 < best) {
                best = dr2;
                matched[i] = 1;
            }
        }
    }
    return matched;
}

template <typename TObject>
inline bool inEtaRange(const TObject& obj, float minAbsEta, float maxAbsEta) {
    const float aeta = std::abs(obj.eta());
    return aeta >= minAbsEta && aeta < maxAbsEta;
}

template <typename TObject>
RVec<float> objectPt(
    const RVec<TObject>& objs,
    float minAbsEta = 0.f,
    float maxAbsEta = 999.f,
    float minPt = 0.f,
    float maxPt = 1.e9f
) {
    RVec<float> out;
    for (const auto& obj : objs) {
        if (!inEtaRange(obj, minAbsEta, maxAbsEta)) continue;
        if (obj.pt() < minPt || obj.pt() >= maxPt) continue;
        out.push_back(obj.pt());
    }
    return out;
}

template <typename TObject>
RVec<float> objectEta(
    const RVec<TObject>& objs,
    float minPt = 0.f,
    float maxPt = 1.e9f
) {
    RVec<float> out;
    for (const auto& obj : objs) {
        if (obj.pt() < minPt || obj.pt() >= maxPt) continue;
        out.push_back(obj.eta());
    }
    return out;
}

template <typename TObject>
RVec<float> matchedObjectPt(
    const RVec<TObject>& objs,
    const RVec<int>& flags,
    float minAbsEta = 0.f,
    float maxAbsEta = 999.f,
    float minPt = 0.f,
    float maxPt = 1.e9f
) {
    RVec<float> out;
    const std::size_t n = std::min(objs.size(), flags.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (!flags[i]) continue;
        if (!inEtaRange(objs[i], minAbsEta, maxAbsEta)) continue;
        if (objs[i].pt() < minPt || objs[i].pt() >= maxPt) continue;
        out.push_back(objs[i].pt());
    }
    return out;
}

template <typename TObject>
RVec<float> matchedObjectEta(
    const RVec<TObject>& objs,
    const RVec<int>& flags,
    float minPt = 0.f,
    float maxPt = 1.e9f
) {
    RVec<float> out;
    const std::size_t n = std::min(objs.size(), flags.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (!flags[i]) continue;
        if (objs[i].pt() < minPt || objs[i].pt() >= maxPt) continue;
        out.push_back(objs[i].eta());
    }
    return out;
}

inline RVec<float> recoJetChHEF(
    const RVec<Jet>& jets,
    float minAbsEta = 0.f,
    float maxAbsEta = 999.f,
    float minPt = 0.f,
    float maxPt = 1.e9f
) {
    RVec<float> out;
    for (const auto& jet : jets) {
        if (!inEtaRange(jet, minAbsEta, maxAbsEta)) continue;
        if (jet.pt() < minPt || jet.pt() >= maxPt) continue;
        out.push_back(jet.chHEF());
    }
    return out;
}

inline RVec<float> recoJetPtForComposition(
    const RVec<Jet>& jets,
    float minAbsEta = 0.f,
    float maxAbsEta = 999.f,
    float minPt = 0.f,
    float maxPt = 1.e9f
) {
    return objectPt(jets, minAbsEta, maxAbsEta, minPt, maxPt);
}

inline RVec<float> recoJetEtaForComposition(
    const RVec<Jet>& jets,
    float minPt = 0.f,
    float maxPt = 1.e9f
) {
    return objectEta(jets, minPt, maxPt);
}

inline RVec<float> recoJetChHEFForEta(
    const RVec<Jet>& jets,
    float minPt = 0.f,
    float maxPt = 1.e9f
) {
    RVec<float> out;
    for (const auto& jet : jets) {
        if (jet.pt() < minPt || jet.pt() >= maxPt) continue;
        out.push_back(jet.chHEF());
    }
    return out;
}

}  // namespace skimrdf

#endif
