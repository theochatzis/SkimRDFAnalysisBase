#ifndef SKIMRDF_KINEMATICS_H
#define SKIMRDF_KINEMATICS_H

#include <cmath>

namespace skimrdf {

inline double deltaPhi(double phi1, double phi2) {
    double dphi = std::fmod(phi1 - phi2, 2.0 * M_PI);
    if (dphi > M_PI) dphi -= 2.0 * M_PI;
    if (dphi <= -M_PI) dphi += 2.0 * M_PI;
    return dphi;
}

inline double absDeltaPhi(double phi1, double phi2) {
    return std::abs(deltaPhi(phi1, phi2));
}

inline double deltaR2(double eta1, double phi1, double eta2, double phi2) {
    const double deta = eta1 - eta2;
    const double dphi = deltaPhi(phi1, phi2);
    return deta * deta + dphi * dphi;
}

inline double deltaR(double eta1, double phi1, double eta2, double phi2) {
    return std::sqrt(deltaR2(eta1, phi1, eta2, phi2));
}

}  // namespace skimrdf

#endif
