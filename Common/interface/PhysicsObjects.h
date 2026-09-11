#ifndef SKIMRDF_PHYSICSOBJECTS_H
#define SKIMRDF_PHYSICSOBJECTS_H

#include "Math/Vector4D.h"
#include <cmath>

namespace skimrdf {

using P4 = ROOT::Math::PtEtaPhiMVector;

// -----------------------------------------------------------------------------
// Jet
// -----------------------------------------------------------------------------
//
// Intentionally follows a CMSSW-like user-facing style:
//   jet.pt(), jet.eta(), jet.phi(), jet.mass(), jet.p4()
//
// The stored state is private. Analysis code reads it through accessors;
// transformations such as JEC return a new Jet rather than modifying fields
// directly.
//
class Jet {
public:
    Jet() = default;

    Jet(float pt, float eta, float phi, float mass,
        float rawFactor, float area, int jetId,
        float chHEF, float neHEF, float chEmEF,
        float neEmEF, float muEF, int genJetIdx = -1)
        : pt_(pt),
          eta_(eta),
          phi_(phi),
          mass_(mass),
          rawFactor_(rawFactor),
          area_(area),
          chHEF_(chHEF),
          neHEF_(neHEF),
          chEmEF_(chEmEF),
          neEmEF_(neEmEF),
          muEF_(muEF),
          jetId_(jetId),
          genJetIdx_(genJetIdx) {}

    bool valid() const { return pt_ >= 0.f; }

    // Kinematics
    float pt()   const { return pt_; }
    float eta()  const { return eta_; }
    float phi()  const { return phi_; }
    float mass() const { return mass_; }

    P4 p4() const {
        return P4(pt_, eta_, phi_, mass_);
    }

    // NanoAOD / jet-specific information
    float rawFactor() const { return rawFactor_; }
    float area()      const { return area_; }

    float chHEF()  const { return chHEF_; }
    float neHEF()  const { return neHEF_; }
    float chEmEF() const { return chEmEF_; }
    float neEmEF() const { return neEmEF_; }
    float muEF()   const { return muEF_; }

    int jetId() const { return jetId_; }
    int genJetIdx() const { return genJetIdx_; }

    float rawPt() const {
        return pt_ * (1.f - rawFactor_);
    }

    float rawMass() const {
        return mass_ * (1.f - rawFactor_);
    }

    bool passTightId() const {
        // NanoAOD jetId bit 1 = tight jet ID.
        return (jetId_ & 2) != 0;
    }

    // Return a copy with new reconstructed kinematics.
    // This is used by correction providers and avoids exposing mutable fields.
    Jet withPtAndMass(float newPt, float newMass) const {
        Jet out(*this);
        out.pt_ = newPt;
        out.mass_ = newMass;
        return out;
    }

private:
    float pt_ = -1.f;
    float eta_ = 0.f;
    float phi_ = 0.f;
    float mass_ = 0.f;

    float rawFactor_ = 0.f;
    float area_ = 0.f;

    float chHEF_ = -1.f;
    float neHEF_ = -1.f;
    float chEmEF_ = -1.f;
    float neEmEF_ = -1.f;
    float muEF_ = -1.f;

    int jetId_ = 0;
    int genJetIdx_ = -1;
};


// -----------------------------------------------------------------------------
// GenJet
// -----------------------------------------------------------------------------
class GenJet {
public:
    GenJet() = default;

    GenJet(float pt, float eta, float phi, float mass)
        : pt_(pt), eta_(eta), phi_(phi), mass_(mass) {}

    bool valid() const { return pt_ >= 0.f; }

    float pt()   const { return pt_; }
    float eta()  const { return eta_; }
    float phi()  const { return phi_; }
    float mass() const { return mass_; }

    P4 p4() const {
        return P4(pt_, eta_, phi_, mass_);
    }

private:
    float pt_ = -1.f;
    float eta_ = 0.f;
    float phi_ = 0.f;
    float mass_ = 0.f;
};


// -----------------------------------------------------------------------------
// Muon
// -----------------------------------------------------------------------------
class Muon {
public:
    Muon() = default;

    Muon(float pt, float eta, float phi, float mass,
         int charge, bool mediumId, bool tightId,
         int pfIsoId, float pfRelIso04)
        : pt_(pt),
          eta_(eta),
          phi_(phi),
          mass_(mass),
          charge_(charge),
          mediumId_(mediumId),
          tightId_(tightId),
          pfIsoId_(pfIsoId),
          pfRelIso04_(pfRelIso04) {}

    bool valid() const { return pt_ >= 0.f; }

    // Kinematics
    float pt()   const { return pt_; }
    float eta()  const { return eta_; }
    float phi()  const { return phi_; }
    float mass() const { return mass_; }

    P4 p4() const {
        return P4(pt_, eta_, phi_, mass_);
    }

    // Identification / isolation information
    int charge() const { return charge_; }
    bool mediumId() const { return mediumId_; }
    bool tightId() const { return tightId_; }
    int pfIsoId() const { return pfIsoId_; }
    float pfRelIso04() const { return pfRelIso04_; }

    bool passZBaseline() const {
        return pt_ > 10.f &&
               std::abs(eta_) < 2.4f &&
               mediumId_ &&
               pfIsoId_ >= 2;
    }

    bool passZTag() const {
        return pt_ > 27.f &&
               std::abs(eta_) < 2.4f &&
               tightId_ &&
               pfIsoId_ >= 4;
    }

private:
    float pt_ = -1.f;
    float eta_ = 0.f;
    float phi_ = 0.f;
    float mass_ = 0.105658f;

    int charge_ = 0;
    bool mediumId_ = false;
    bool tightId_ = false;
    int pfIsoId_ = 0;
    float pfRelIso04_ = -1.f;
};


// -----------------------------------------------------------------------------
// MET
// -----------------------------------------------------------------------------
class MET {
public:
    MET() = default;
    MET(float pt, float phi) : pt_(pt), phi_(phi) {}

    bool valid() const { return pt_ >= 0.f; }

    float pt()  const { return pt_; }
    float phi() const { return phi_; }

    float px() const {
        return pt_ * std::cos(phi_);
    }

    float py() const {
        return pt_ * std::sin(phi_);
    }

    P4 p4() const {
        return P4(pt_, 0.f, phi_, 0.f);
    }

    static MET fromPxPy(double px, double py) {
        return MET(
            static_cast<float>(std::hypot(px, py)),
            static_cast<float>(std::atan2(py, px))
        );
    }

private:
    float pt_ = -1.f;
    float phi_ = 0.f;
};


// -----------------------------------------------------------------------------
// Dimuon
// -----------------------------------------------------------------------------
//
// Composite object: its kinematics are calculated from its daughters.
// The constituent muons are also exposed through accessor methods.
//
class Dimuon {
public:
    Dimuon() = default;

    Dimuon(const Muon& muon1, const Muon& muon2)
        : muon1_(muon1), muon2_(muon2), valid_(true) {}

    bool valid() const { return valid_; }

    const Muon& muon1() const { return muon1_; }
    const Muon& muon2() const { return muon2_; }

    P4 p4() const {
        if (!valid_) return P4(0., 0., 0., 0.);
        return muon1_.p4() + muon2_.p4();
    }

    float pt() const {
        return static_cast<float>(p4().Pt());
    }

    float eta() const {
        return static_cast<float>(p4().Eta());
    }

    float phi() const {
        return static_cast<float>(p4().Phi());
    }

    float mass() const {
        return static_cast<float>(p4().M());
    }

    int charge() const {
        if (!valid_) return 0;
        return muon1_.charge() + muon2_.charge();
    }

    bool oppositeSign() const {
        return valid_ && (muon1_.charge() * muon2_.charge() < 0);
    }

    bool hasTag() const {
        return valid_ && (muon1_.passZTag() || muon2_.passZTag());
    }

private:
    Muon muon1_;
    Muon muon2_;
    bool valid_ = false;
};

}  // namespace skimrdf

#endif
