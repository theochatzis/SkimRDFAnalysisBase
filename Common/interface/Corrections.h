#ifndef SKIMRDF_CORRECTIONS_H
#define SKIMRDF_CORRECTIONS_H

#include "PhysicsObjects.h"
#include "ROOT/RVec.hxx"

namespace skimrdf {

void initJEC(const char* filepath, const char* correctionName);

float evaluateJEC(const Jet& jet, float rho);
Jet applyJEC(const Jet& jet, float rho);
ROOT::VecOps::RVec<Jet> applyJEC(
    const ROOT::VecOps::RVec<Jet>& jets,
    float rho
);

MET propagateType1(
    const MET& met,
    const ROOT::VecOps::RVec<Jet>& oldJets,
    const ROOT::VecOps::RVec<Jet>& newJets,
    float minJetPt = 15.f,
    float maxAbsEta = 5.2f
);

}  // namespace skimrdf

#endif
