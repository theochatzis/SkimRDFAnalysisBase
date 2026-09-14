# SkimRDFAnalysisBase

Basic tools to analyze ntuple skims with ROOT RDataFrame.

The repository separates the generic analysis execution from reusable tools and
analysis-specific physics definitions:

```text
SkimRDFAnalysisBase/
├── run_analysis.py          # generic RDataFrame execution
├── plotting/                # reusable plotting tools
├── utils/                   # reusable analysis/helper tools
├── Common/                  # compiled C++ helpers and correctionlib registry
├── example_analyses/        # example RDF definitions and YAML configurations
└── tools/                   # standalone analysis utilities
```

## Setup

Compile the C++ helper library with:

```bash
bash setup.sh
```

---

# Analysis driver

## `run_analysis.py`

`run_analysis.py` is the generic RDataFrame analysis driver. It handles input
discovery, sample looping, event ranges, regions, histogram booking, event
weights, execution, and ROOT output writing.

The analysis-specific physics logic is kept outside the driver.

The main architecture is:

- `run_analysis.py` = generic execution
- `rdf_definition.py` = physics objects, derived columns, and regions
- histogram YAML = histogram/output definitions
- optional analysis YAML = numerical and analysis-specific configuration
- optional weights YAML = event weights and systematic variations

## RDF definition interface

An analysis module must implement:

```python
def define_columns(df, sample, args, config):
    return df


def get_regions(sample, args, config):
    return {
        "region_name": {
            "cuts": [
                "selection expression",
                "another selection",
            ]
        }
    }
```

It may also implement:

```python
def setup(args, config):
    ...
```

`setup()` is called once before any sample is processed. It is the appropriate
place to load shared libraries, declare C++ helpers, or initialize corrections.

`define_columns()` is called once per sample immediately after creating the base
RDataFrame. Put analysis-specific `Define()` and `Redefine()` operations there.

`get_regions()` is called once per sample, so regions may depend on the sample
name or analysis configuration.

Example:

```bash
python3 run_analysis.py \
  --input-files-dir /path/to/input/nano \
  --output-dir output \
  --file-pattern "*.root" \
  --histograms-defs example_analyses/zjet_histograms.yaml \
  --rdf-definition example_analyses/zjet_rdf_definition.py \
  --analysis-config example_analyses/zjet_config.yaml \
  --add-no-selection
```

---

# Plotting

`plotting/` is a reusable ROOT + matplotlib/mplhep plotting toolkit. ROOT
objects are converted into lightweight numpy-backed objects and can then be
plotted with a common CMS-style interface.

## Supported ROOT objects

- `TH1*`
- `TProfile`
- `TH2*`
- `TProfile2D`
- `TH3*`
- `TGraph`
- `TGraphErrors`
- `TGraphAsymmErrors`
- `TEfficiency`

## Core API

Typical imports are:

```python
from plotting import (
    RootFileReader,
    read_root_object,
    plot_hist1d_data_mc,
    plot_hist1d_methods_data_mc,
    plot_hist1d_data_mc_stack,
    plot_graphs,
    plot_efficiency,
)
```

## TH1 / TProfile Data vs MC

```python
from plotting import read_root_object, plot_hist1d_data_mc

data = read_root_object(
    "data.root",
    "zjet/Jet_eta_parallel",
)

mc = read_root_object(
    "mc.root",
    "zjet/Jet_eta_parallel",
)

plot_hist1d_data_mc(
    data,
    mc,
    "jet_eta.pdf",
    normalize_mc_to_data=True,
)
```

A Data/MC ratio panel is produced automatically.

## Multiple methods

Several Data/MC methods can be compared in the same figure:

```python
from collections import OrderedDict

from plotting import (
    read_root_object,
    plot_hist1d_methods_data_mc,
)

data_methods = OrderedDict([
    (
        "methodA",
        read_root_object("data.root", "region/methodA"),
    ),
    (
        "methodB",
        read_root_object("data.root", "region/methodB"),
    ),
])

mc_methods = OrderedDict([
    (
        "methodA",
        read_root_object("mc.root", "region/methodA"),
    ),
    (
        "methodB",
        read_root_object("mc.root", "region/methodB"),
    ),
])

plot_hist1d_methods_data_mc(
    data_methods,
    mc_methods,
    "methods.pdf",
)
```

## MC stack

```python
from collections import OrderedDict

from plotting import (
    read_root_object,
    plot_hist1d_data_mc_stack,
)

components = OrderedDict([
    (
        "processA",
        read_root_object("a.root", "region/h"),
    ),
    (
        "processB",
        read_root_object("b.root", "region/h"),
    ),
])

data = read_root_object(
    "data.root",
    "region/h",
)

plot_hist1d_data_mc_stack(
    data,
    components,
    "stack.pdf",
)
```

## TGraph / TProfile

Histogram-like objects can be converted to graphs when appropriate:

```python
from plotting import (
    hist_to_graph,
    plot_graphs,
    read_root_object,
)

profile = read_root_object(
    "data.root",
    "region/profile",
)

graph = hist_to_graph(profile)

plot_graphs(
    {"profile": graph},
    "profile.pdf",
    xlabel="pT",
    ylabel="Response",
)
```

## TEfficiency

```python
from plotting import (
    plot_efficiency,
    read_root_object,
)

numerator = read_root_object(
    "histograms.root",
    "trigger/numerator",
)

denominator = read_root_object(
    "histograms.root",
    "trigger/denominator",
)

plot_efficiency(
    numerator,
    denominator,
    "efficiency.pdf",
)
```

## Efficient repeated ROOT access

When reading many objects from the same ROOT file, keep the file open:

```python
from plotting import RootFileReader

with RootFileReader("data.root") as reader:
    h1 = reader.get("region/h1")
    h2 = reader.get("region/h2")
    graph = reader.get("region/graph")
```

`RootFileReader` caches converted objects, avoiding repeated ROOT-file opening.
This is particularly useful for EOS/XRootD files.

## Generic plotting CLI

For simple comparisons, the repository also provides `plot_root_objects.py`.

Example:

```bash
python3 plot_root_objects.py compare \
  --data data.root \
  --mc mc.root \
  --object zjet/Jet_eta \
  --output jet_eta.pdf \
  --normalize-mc-to-data
```

## Plotting requirements

PyROOT should come from CMSSW/ROOT.

Install the Python plotting dependencies if needed:

```bash
python3 -m pip install --user mplhep matplotlib numpy pyyaml
```

## lxplus matplotlib cache

Importing `plotting` automatically configures matplotlib to use local temporary
cache/config directories:

```text
/tmp/$USER/skimrdf_matplotlib/
```

This avoids slow matplotlib startup when the default cache lives on EOS/AFS.

Disable it with:

```bash
export SKIMRDF_MPL_LOCAL_CACHE=0
```

or redirect it with:

```bash
export SKIMRDF_MPLCONFIGDIR=/some/local/path
```

---

# Utils

`utils/` contains reusable analysis-side helper modules. Like `plotting/`, its
components are callable from other scripts, but its purpose is analysis
bookkeeping and generic execution support rather than visualization.

The event-weight machinery currently lives in:

```python
from utils.event_weights import (
    load_weights_config,
    setup_weight_corrections,
    annotate_sample_type,
    apply_event_weights,
    book_weight_histograms,
)
```

Normally these functions are called automatically by `run_analysis.py`, so an
analysis only needs to provide a weights YAML file.

## YAML-driven event weights

Weights are configured independently of the physics analysis:

```yaml
event_weight_name: eventWeight

weights:

  puWeight:
    enabled: true
    type: correctionlib
    apply_to: mc

    json: /path/to/puWeights.json
    correction: Collisions25_goldenJSON

    arguments:
      - column: Pileup_nTrueInt
        type: real

      - variation: true
        type: string

    variations:
      nominal: nominal
      up: up
      down: down

    histogram:
      title: "Pileup weight;Pileup weight;Events"
      bins: [100, 0.0, 5.0]
      include_variations: true
```

Run the analysis with:

```bash
python3 run_analysis.py \
  --input-files-dir /path/to/input/nano \
  --output-dir output \
  --file-pattern "*.root" \
  --histograms-defs example_analyses/zjet_histograms.yaml \
  --rdf-definition example_analyses/zjet_rdf_definition.py \
  --weights-defs example_analyses/weights_pu_example.yaml
```

For the configuration above, the framework defines:

```text
puWeight
puWeight_up
puWeight_down

eventWeight
eventWeight_puWeight_up
eventWeight_puWeight_down
```

With several configured weights:

```text
eventWeight = puWeight * muonWeight * btagWeight * ...
```

A systematic event-weight column varies only the requested factor:

```text
eventWeight_puWeight_up =
    puWeight_up * muonWeight * btagWeight * ...
```

This makes the naming directly usable later for systematic variations.

## Weight monitoring

Each weight may define its own monitoring-histogram binning:

```yaml
histogram:
  title: "Pileup weight;Pileup weight;Events"
  bins: [100, 0.0, 5.0]
  include_variations: true
```

The output ROOT file then contains:

```text
weights/
├── puWeight
├── puWeight_up
└── puWeight_down
```

These histograms are unweighted and show the event-by-event correction-factor
distribution itself.

## Histogram weighting

When a weights configuration is active, normal physics histograms use the
combined `eventWeight` automatically.

```yaml
Jet_pt:
  title: "Jet pT;Jet pT;Events"
  variable: Jet_pt
  bins: [100, 0, 500]
```

To explicitly keep a histogram unweighted:

```yaml
Jet_pt_unweighted:
  title: "Jet pT;Jet pT;Events"
  variable: Jet_pt
  bins: [100, 0, 500]
  weight: null
```

To book a specific systematic variation:

```yaml
Jet_pt_puUp:
  title: "Jet pT PU up;Jet pT;Events"
  variable: Jet_pt
  bins: [100, 0, 500]
  weight: eventWeight_puWeight_up
```

## Expression-based weights

Weights do not have to come from correctionlib. Existing RDataFrame columns can
also be registered:

```yaml
muonWeight:
  enabled: true
  type: expression
  apply_to: mc

  requires:
    - MuonSF
    - MuonSF_up
    - MuonSF_down

  expressions:
    nominal: MuonSF
    up: MuonSF_up
    down: MuonSF_down

  histogram:
    title: "Muon weight;Muon weight;Events"
    bins: [100, 0.5, 1.5]
```

The same nominal/up/down and combined-event-weight bookkeeping is then applied
automatically.

## Data and MC

For standard NanoAOD input, MC is detected through the presence of `genWeight`.

A weight configured with:

```yaml
apply_to: mc
```

is set to `1.0` for data, allowing the same histogram configuration to be used
for both data and simulation.

---

# Common C++ helpers

`Common/` contains reusable compiled C++ helpers used by RDataFrame analyses.

This includes the generic correctionlib registry:

```cpp
skimrdf::registerCorrection(...)
skimrdf::evaluateCorrection(...)
```

which is used by the YAML-driven weight utilities and can also be reused for
other correctionlib payloads.

Build the library with:

```bash
bash setup.sh
```

