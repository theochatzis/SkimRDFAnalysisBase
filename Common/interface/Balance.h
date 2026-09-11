#ifndef SKIMRDF_BALANCE_H
#define SKIMRDF_BALANCE_H

#include "PhysicsObjects.h"
#include "Kinematics.h"

#include <cmath>

namespace skimrdf {

inline double directBalance(const Dimuon& z, const Jet& probe) {
    if (!z.valid() || !probe.valid() || z.pt() <= 0.) return -999.;
    return probe.pt() / z.pt();
}

inline double mpfResponse(const Dimuon& z, const MET& met) {
    if (!z.valid() || !met.valid() || z.pt() <= 0.) return -999.;

    const double zpx = z.pt() * std::cos(z.phi());
    const double zpy = z.pt() * std::sin(z.phi());
    const double dot = met.px() * zpx + met.py() * zpy;
    const double zpt2 = static_cast<double>(z.pt()) * z.pt();

    return 1.0 + dot / zpt2;
}

inline double alpha(const Dimuon& z, const ROOT::VecOps::RVec<Jet>& jets) {
    if (!z.valid() || z.pt() <= 0.) return -999.;
    if (jets.size() < 2) return 0.;
    return jets[1].pt() / z.pt();
}

inline bool inOppositeWindow(
    double tagPhi,
    double probePhi,
    double halfWidth = M_PI / 16.0
) {
    return std::abs(std::abs(deltaPhi(probePhi, tagPhi)) - M_PI) < halfWidth;
}

inline double windowedBalanceWeight(
    double tagPhi,
    double probePhi,
    double halfWidth = M_PI / 16.0
) {
    const double dphi = deltaPhi(probePhi, tagPhi);

    // Opposite-side signal window around +/- pi.
    if (std::abs(std::abs(dphi) - M_PI) < halfWidth) return +1.0;

    // Two transverse control windows around +/- pi/2.
    if (std::abs(dphi - M_PI / 2.0) < halfWidth) return -0.5;
    if (std::abs(dphi + M_PI / 2.0) < halfWidth) return -0.5;

    return 0.0;
}

}  // namespace skimrdf

#endif
