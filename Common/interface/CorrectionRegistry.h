#ifndef SKIMRDF_CORRECTIONREGISTRY_H
#define SKIMRDF_CORRECTIONREGISTRY_H

#include "correction.h"
#include <vector>

namespace skimrdf {

// Low-level generic correctionlib registry.
//
// Keep payload-specific knowledge in small analysis wrappers rather than in
// the physics-object classes. For example, a future MuonSF helper can inspect
// the payload convention and construct the ordered heterogeneous argument list.
//
// Example:
//   registerCorrection("muon_id", json, key, false);
//   std::vector<correction::Variable::Type> args = {eta, pt, "nominal"};
//   const double sf = evaluateCorrection("muon_id", args);

void registerCorrection(
    const char* alias,
    const char* filepath,
    const char* correctionName,
    bool compound = false
);

double evaluateCorrection(
    const char* alias,
    const std::vector<correction::Variable::Type>& args
);

}  // namespace skimrdf

#endif
