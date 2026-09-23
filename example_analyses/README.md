# Z+jet example through the generic `run_analysis.py`

The Z+jet example no longer has its own event-loop driver.

The generic runner owns:

- input-file discovery,
- RDataFrame construction,
- event ranges / multithreading,
- histogram booking,
- regions,
- execution,
- ROOT output.

The analysis-specific files provide only:

```text
zjet_rdf_definition.py
    setup()
    define_columns()
    get_regions()

zjet_histograms.yaml
    histogram/profile definitions (generated)

make_zjet_histograms.py
    generator for zjet_histograms.yaml

zjet_config.yaml
    optional runtime physics configuration
```

Run:

```bash
python3 run_analysis.py \
  --input-files-dir /eos/user/t/tchatzis/reco2pico/myPicosDirectory/zjet_window_default/default/DATA/Muon2025G/ \
  --output-dir zjet_example_output \
  --file-pattern "*.root" \
  --histograms-defs example_analyses/zjet_histograms.yaml \
  --rdf-definition example_analyses/zjet_rdf_definition.py \
  --add-no-selection
```

The physical selection is returned by `get_regions()` as the `zjet` region:

```text
passTrigger
nTightMuons >= 2
nVetoMuons == 0
Z.valid()
|m_mumu - mZ| < 3.743 GeV
pT(mu2) > 27 GeV
at least one selected clean jet
|dPhi(Z, probe jet)| > 2.7
```

## Muons

Two collections are built from the same helpers in `ObjectBuilders.h`, so the
thresholds are written once:

| collection | ID | isolation | pT | \|eta\| |
|---|---|---|---|---|
| `TightMuons` | `tightId` | `pfRelIso04_all < 0.15` | > 10 GeV | < 2.4 |
| `VetoMuons` | `looseId`, failing the tight WP | `pfRelIso04_all < 0.25` | > 10 GeV | < 2.4 |

`nTightMuons >= 2` asks for the two legs of the Z, and `nVetoMuons == 0`
rejects events with any additional muon that is merely loose, so the event is
a clean dimuon event rather than one with a third muon under the threshold.
The two Z muons pass the veto working point by construction and are excluded
from `VetoMuons`, so the two cuts do not fight each other.

`TightMuons` is the single source of the working point: the region counts it,
the jets are cleaned against it, and the Z is built from it. `Z.valid()`
therefore implies that **both** legs are tight and isolated. There is no
tag/probe asymmetry — the dimuon system is the pT reference of the
measurement, so a loosely identified second leg would smear the quantity being
measured.

Both the reco jets and the gen jets are cleaned against `TightMuons` (via
`JetCleaningObjects`) within `dR < 0.2`, so muon jets enter neither the
matching numerator nor its denominator.

The Z is then selected by muon kinematics rather than by pT(Z). Requiring the
subleading muon above 27 GeV implies the leading one and puts both legs on the
IsoMu24 plateau. It also leaves the pT(Z) spectrum unsculpted, so pT(Z) remains
an observable instead of a selection variable — which is why the profiles versus
pT(Z) use `ZPT_EDGES`, reaching down to 0, while the MC jet studies keep
`PT_EDGES` starting at the 15 GeV matching threshold.

## Trigger

`passTrigger` is the OR of the paths in `DEFAULT_TRIGGERS`
(`zjet_rdf_definition.py`), or of the `triggers` list in the analysis config.
The default is `HLT_IsoMu24`, matched to the `pT(mu2) > 27 GeV` region cut.
Together with the tight ID and tight isolation that `selectBestDimuon` applies
to both legs, that cut is what puts the muons on the IsoMu24 efficiency
plateau; since the dimuon builder no longer carries a `pT > 27` tag
requirement of its own, the region cut is the only place this is enforced.

The requirement is applied to data **and** to MC. Data come from a
trigger-selected primary dataset, so an untriggered MC over-predicts the data
yield by the trigger inefficiency; requiring the path on both sides removes
that bias. What remains is the data/MC difference in trigger efficiency, which
needs a scale factor and is not applied here.

If none of the requested paths exist in an input, `passTrigger` is defined as
`true` and the runner prints a warning: the region stays valid, but data and
MC are then no longer selected consistently.

## Data / MC

The definition inspects the input columns.

If `genWeight` exists, the default event weight is `genWeight`, which is
what pairs with the runner's `1 / sum(genEventSumw)` normalization;
otherwise it is 1.

If `GenJet_*` exists, reco/gen matching is enabled automatically.
For data, all MC-study histogram vectors are defined empty, which means
the same histogram YAML can be used for both data and MC without filling
misleading efficiency/purity denominators.

## Booked results

For data and MC:

- pileup observables: `rho`, `npvs`, `npvsGood`;
- kinematics: Z, probe-jet and muon pT / eta / phi, `dPhi_mumu`,
  `dPhi_ZProbe`, MET pT and phi;
- the eta distributions repeated in regions of pT(Z);
- the MET and hadronic-recoil distributions repeated in regions of
  probe-jet |eta|;
- probe-jet PF energy fractions (`chHEF`, `neHEF`, `chEmEF`, `neEmEF`,
  `muEF`), in regions of probe-jet |eta| and of pT(Z);
- the hadronic-recoil means and resolutions versus pT(Z), raw and
  corrected for the recoil scale, inclusively and in probe-jet |eta|
  regions;
- the hadronic-recoil response versus pT(Z);
- the same recoil response and resolutions versus N_PV, split by probe-jet
  |eta| and by pT(Z), for the pileup dependence;
- DB and MPF response versus pT(Z), inclusively and in |eta| regions.

For MC only:

- jet reconstruction efficiency, purity, response and resolution with
  respect to gen jets.

## Regenerating the histogram definitions

`zjet_histograms.yaml` is generated, not hand written: most of it is the same
few distributions repeated over the |eta|, pT(Z) and jet-pT categories, and
hand editing is how those drift apart from the analysis definition. The
generator imports the categories from `zjet_rdf_definition.py`, so they are
declared in exactly one place.

```bash
python3 example_analyses/make_zjet_histograms.py           # regenerate
python3 example_analyses/make_zjet_histograms.py --check   # verify, no write
```

`--check` exits nonzero when the checked-in YAML no longer matches the
script. Binning and titles live in the generator; categories do not.

## Event weights and their variations

`--weights-defs example_analyses/weights_pu_example.yaml` turns on the weight
system; the example configures the generator weight and a correctionlib
pileup weight with up/down variations.

A weight marked `baseline: true` is one that is *always* applied. Here that is
`generatorWeight`, and it is what defines the meaning of "unweighted": no
scale factors, but still the generator weight, because for a sample with
negative weights a genuinely unweighted distribution is not meaningful.

Adding `--histogram-weight-output all` writes, for every histogram `<name>`:

```text
<name>                      every weight, including all scale factors
<name>_unweighted           baseline weights only (here: genWeight)
<name>_puWeight_up          pileup weight varied up, all others nominal
<name>_puWeight_down        pileup weight varied down, all others nominal
```

The fully corrected histogram keeps the plain name, so `plot_zjet_with_tools.py`
works against either output mode without changes.

The variations preserve the histogram's own weight column. A category weight
such as `etaWeight_HB` is `passes_category ? eventWeight : 0`, so a variation
cannot simply overwrite it; the runner rescales by
`varied_event_weight / eventWeight`, which swaps the weight and keeps the
category selection.

`--histogram-weight-output all` does nothing useful on its own: without
`--weights-defs` there are no scale factors to vary, only the nominal
histograms are written, and the runner prints a warning saying so.

The weight distributions themselves are written to the `weights` directory as
`puWeight`, `puWeight_up` and `puWeight_down`.

### Checking the variations

```bash
python3 example_analyses/plot_zjet_variations.py \
  --input zjet_example_output/DYTo2L.root \
  --region zjet \
  --output-dir zjet_variation_plots
```

For each source it writes `<source>/weight.png`, the weight itself with its
up and down variations overlaid, and one plot per distribution showing
nominal, up, down and the no-scale-factor reference with a ratio to nominal.
The sources and the distributions are the `SOURCES` and `DISTRIBUTIONS` lists
at the top of that script, and both can be overridden with `--sources` and
`--distributions`. Pass `--no-unweighted` to drop the no-SF reference.

## Category weights

The |eta| and pT(Z) categories are encoded as weight columns rather than as
regions, so every categorized histogram is booked in the single `zjet`
region:

```text
etaWeight_Incl
etaWeight_HB
etaWeight_HE1
...
zptWeight_Incl
zptWeight_zpt20to40
...
```

A category weight is the event weight when the Z and the probe jet exist and
the probe falls in the category, and zero otherwise.

## Hadronic recoil

The MET performance observables are built from the hadronic recoil, that is
from everything in the event that is not the Z:

```text
u_vec  = -(MET_vec + qT_vec)        qT_vec = the Z transverse momentum
u_par  = u_vec . qhat               ~ -pT(Z)
u_perp = u_vec x qhat               ~ 0
```

so that a perfectly balanced, perfectly measured event has
`u_par = -pT(Z)` and `u_perp = 0`. The three performance quantities are then

```text
response    R = -<u_par> / <pT(Z)>               ~ 1
resolution  sigma(u_par)                         parallel
            sigma(u_perp)                        transverse
```

Nothing is stored as a per-event ratio. The response is formed in the plotting
step from the `Upar_vs_*` and `Zpt_vs_*` profiles over the same bins, because
the per-event ratio `-u_par / pT(Z)` diverges as pT(Z) approaches zero, and the
region no longer cuts on pT(Z). The plotting script also divides each
resolution by the response of the same process, which removes the part of the
apparent resolution that is really a mismeasured recoil scale and makes
processes with different responses comparable.

One caveat on `sigma(u_par)`. Within a pT(Z) bin, `u_par` sits at `-pT(Z)`, so
its spread contains the spread of pT(Z) across the bin on top of the detector
resolution. The wider the bin, the larger that contribution — it is most
noticeable in the coarse high-pT bins. The usual way to remove it is to take
the spread of `u_par + pT(Z)` instead, which is bin-width independent.

## Pileup dependence

The same five profiles are booked a second time against `analysis_npvsGood`
rather than pT(Z), split both by probe-jet |eta| and by pT(Z), and plotted
under `10_pu_dependence/`. The pT(Z) split is the one to read: it holds the
recoil scale roughly fixed while the vertex count varies, so the slope of
sigma against N_PV is the quantity to compare between data and MC.

## Resolutions

Resolutions are not booked directly. Every mean profile has a companion
mean-of-squares profile carrying the same weight, for instance
`Upar_vs_Zpt_HB` and `UparSq_vs_Zpt_HB`, and the plotting script forms

```text
sigma = sqrt(<x^2> - <x>^2)
```

A default `TProfile` stores the error on the mean, `sigma / sqrt(N_eff)`, so
the uncertainty on the spread is that error divided by `sqrt(2)`. For the
scale-corrected resolutions the relative error on the response profile is
added to this in quadrature.

## Gen-jet matching

Gen jets are cleaned against the muons of the Z candidate with the same
`dR > 0.2` requirement as the reco jets, so muon jets do not enter the
efficiency denominator.

The matching is nearest-neighbour within `dR < 0.2`, with asymmetric
thresholds: the object whose efficiency (or purity) is measured must have
`pT > 15 GeV`, while the object it is matched against only needs
`pT > 10 GeV`. The efficiency therefore uses gen jets above 15 GeV matched
to reco jets above 10 GeV, and the purity swaps the two roles.

## Plots

`plot_zjet_with_tools.py` reads the same names and writes into subdirectories
of `--output-dir`:

```text
01_inclusive/            rho, N_PV, Z mass, jet multiplicity, alpha
02_kinematics/           Z, probe-jet, muon, MET and recoil distributions
03_eta_in_Zpt/           eta distributions in regions of pT(Z)
04_met_in_probe_eta/     MET and recoil in regions of probe-jet |eta|
05_energy_fractions/     probe-jet PF fractions, in |eta| and pT(Z) regions
06_response/             DB and MPF versus pT(Z), in |eta| regions
07_recoil_components/    u_par / u_perp mean and resolution versus pT(Z),
                         raw and corrected for the recoil scale
08_recoil_response/      hadronic-recoil response versus pT(Z)
09_mc_jets/              MC efficiency, purity, response and resolution
10_pu_dependence/        the same recoil quantities versus N_PV, split by
                         probe-jet |eta| and by pT(Z)
```

Data/MC comparisons are stacks; profiles are overlaid process by process,
because means from separate `TProfile`s are not additive.

## JEC

JEC re-application is disabled by default.

To enable it, edit `zjet_config.yaml`:

```yaml
jec:
  enabled: true
  json: /path/to/JEC.json
  name: Your_JEC_Key
```

Compile the common library once:

```bash
cd Common
source setup_Common_cpp.sh
cd ..
```

and add:

```bash
--analysis-config example_analyses/zjet_config.yaml
```

to the generic `run_analysis.py` command.

## Recommended framework pattern

For future analyses, use the same interface:

```python
def setup(args=None, config=None):
    # one-time library/header/correction initialization
    ...

def define_columns(df, sample=None, args=None, config=None):
    # physics objects and derived quantities only
    return df

def get_regions(sample=None, args=None, config=None):
    # selections only
    return {
        "signal": {
            "cuts": [...]
        }
    }
```

This keeps `run_analysis.py` analysis-independent.
