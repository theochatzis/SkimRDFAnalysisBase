#include "Corrections.h"
#include "CorrectionRegistry.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace skimrdf {

namespace {

std::unique_ptr<correction::CorrectionSet> gJecSet;
correction::Correction::Ref gJec = nullptr;
correction::CompoundCorrection::Ref gCompoundJec = nullptr;

struct RegisteredCorrection {
    std::unique_ptr<correction::CorrectionSet> cset;
    correction::Correction::Ref corr = nullptr;
    correction::CompoundCorrection::Ref compound = nullptr;
};

std::unordered_map<std::string, std::unique_ptr<RegisteredCorrection>> gRegistry;

std::vector<correction::Variable::Type>
jecArguments(const Jet& jet, float rho) {
    std::vector<correction::Variable::Type> args;

    const auto inputs = (gCompoundJec != nullptr)
        ? gCompoundJec->inputs()
        : gJec->inputs();

    for (const auto& input : inputs) {
        const std::string name = input.name();

        if (name == "JetEta" || name == "eta") {
            args.emplace_back(static_cast<double>(jet.eta()));
        } else if (name == "JetPt" || name == "pt") {
            // CMS JEC payloads are evaluated from raw jet pT.
            args.emplace_back(static_cast<double>(jet.rawPt()));
        } else if (name == "JetA" || name == "area") {
            args.emplace_back(static_cast<double>(jet.area()));
        } else if (name == "Rho" || name == "rho") {
            args.emplace_back(static_cast<double>(rho));
        } else if (name == "JetPhi" || name == "phi") {
            args.emplace_back(static_cast<double>(jet.phi()));
        } else {
            throw std::runtime_error(
                "Unknown JEC input parameter requested by payload: " + name
            );
        }
    }
    return args;
}

}  // namespace

void initJEC(const char* filepath, const char* correctionName) {
    gJecSet = correction::CorrectionSet::from_file(std::string(filepath));
    gJec = nullptr;
    gCompoundJec = nullptr;

    const std::string key(correctionName);

    try {
        gCompoundJec = gJecSet->compound().at(key);
        std::cout << "[Corrections] Loaded compound JEC: " << key << "\n";
        return;
    } catch (...) {
        // Fall through to a normal correction.
    }

    gJec = gJecSet->at(key);
    std::cout << "[Corrections] Loaded JEC: " << key << "\n";
}

float evaluateJEC(const Jet& jet, float rho) {
    if (gJec == nullptr && gCompoundJec == nullptr) return 1.f;

    const auto args = jecArguments(jet, rho);

    const double factor = (gCompoundJec != nullptr)
        ? gCompoundJec->evaluate(args)
        : gJec->evaluate(args);

    return static_cast<float>(factor);
}

Jet applyJEC(const Jet& jet, float rho) {
    if (!jet.valid()) return jet;

    const float factor = evaluateJEC(jet, rho);

    return jet.withPtAndMass(
        jet.rawPt() * factor,
        jet.rawMass() * factor
    );
}

ROOT::VecOps::RVec<Jet> applyJEC(
    const ROOT::VecOps::RVec<Jet>& jets,
    float rho
) {
    ROOT::VecOps::RVec<Jet> out;
    out.reserve(jets.size());

    for (const auto& jet : jets) {
        out.push_back(applyJEC(jet, rho));
    }
    return out;
}

MET propagateType1(
    const MET& met,
    const ROOT::VecOps::RVec<Jet>& oldJets,
    const ROOT::VecOps::RVec<Jet>& newJets,
    float minJetPt,
    float maxAbsEta
) {
    if (!met.valid()) return met;

    double px = met.px();
    double py = met.py();

    const std::size_t n = std::min(oldJets.size(), newJets.size());

    for (std::size_t i = 0; i < n; ++i) {
        const auto& oldJet = oldJets[i];
        const auto& newJet = newJets[i];

        if (std::abs(oldJet.eta()) >= maxAbsEta) continue;
        if (std::max(oldJet.pt(), newJet.pt()) < minJetPt) continue;

        const double dpt = static_cast<double>(newJet.pt()) - oldJet.pt();
        px -= dpt * std::cos(oldJet.phi());
        py -= dpt * std::sin(oldJet.phi());
    }

    return MET::fromPxPy(px, py);
}

void registerCorrection(
    const char* alias,
    const char* filepath,
    const char* correctionName,
    bool compound
) {
    auto entry = std::make_unique<RegisteredCorrection>();
    entry->cset = correction::CorrectionSet::from_file(std::string(filepath));

    const std::string key(correctionName);

    if (compound) {
        entry->compound = entry->cset->compound().at(key);
    } else {
        entry->corr = entry->cset->at(key);
    }

    gRegistry[std::string(alias)] = std::move(entry);
}

double evaluateCorrection(
    const char* alias,
    const std::vector<correction::Variable::Type>& args
) {
    const auto it = gRegistry.find(std::string(alias));
    if (it == gRegistry.end()) {
        throw std::runtime_error(
            "Correction alias is not registered: " + std::string(alias)
        );
    }

    const auto& entry = *it->second;
    if (entry.compound != nullptr) return entry.compound->evaluate(args);
    return entry.corr->evaluate(args);
}

}  // namespace skimrdf
