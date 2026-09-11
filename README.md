# Skims Analysis Base
Basic tools to analyze ntuples skims for studies. 

# Setup
Compile the cpp libraries by using:
``` 
bash setup.sh
```

# `run_analysis.py` script
The `run_analysis.py` features YAML-defined region and histogram configurations in order to produce in the end histograms. Each region is a set of cuts.


In addition can so far natively integrates Python's correctionlib via a compiled C++ shared library to apply Jet Energy Corrections (JECs) and Type-1 MET corrections on the fly.

## RDF definition interface
The architecture is such that:

- `run_analysis.py` = generic execution
- `rdf_definition.py` = physics analysis
- histogram YAML = output description
- optional analysis YAML = numerical/configuration inputs

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

`setup()` is called exactly once before any sample is processed. It is the right
place to load shared libraries, declare C++ helpers, or initialize correctionlib
objects.

`define_columns()` is called once per sample, immediately after construction of
the base RDataFrame. Put `Define()` and `Redefine()` calls there.

`get_regions()` is called once per sample. Regions can therefore depend on the
sample name or user configuration.

Example:

```bash
python3 run_analysis.py \
  --input-files-dir /path/to/input/nano \
  --output-dir output \
  --file-pattern "*.root" \
  --histograms-defs analyses/zjet_histograms.yaml \
  --rdf-definition analyses/zjet_rdf_definition.py \
  --add-no-selection
```

# Plotting toolkit

Reusable ROOT + mplhep plotting helpers.

## Supported ROOT objects

- `TH1*`
- `TProfile`
- `TH2*`
- `TProfile2D`
- `TGraph`
- `TGraphErrors`
- `TGraphAsymmErrors`
- `TEfficiency`

ROOT objects are read with PyROOT and converted into small numpy dataclasses.
Plotting is then done with matplotlib/mplhep.

## Core API

```python
from plotting import (
    read_root_object,
    plot_hist1d_data_mc,
    plot_hist1d_stack,
    plot_graph_data_mc,
    plot_hist2d_data_mc,
)
```

### TH1 / TProfile Data vs MC

```python
data = read_root_object("data.root", "zjet/Jet_eta_parallel")
mc = read_root_object("mc.root", "zjet/Jet_eta_parallel")

plot_hist1d_data_mc(
    data,
    mc,
    "jet_eta.pdf",
    normalize_mc_to_data=True,
)
```

A Data/MC ratio panel is produced automatically.

### Generic stack

```python
from collections import OrderedDict

components = OrderedDict([
    ("processA", read_root_object("a.root", "region/h")),
    ("processB", read_root_object("b.root", "region/h")),
])

data = read_root_object("data.root", "region/h")

plot_hist1d_stack(
    components,
    "stack.pdf",
    data=data,
    scales={
        "processA": 1.2,
        "processB": 0.8,
    },
    normalize_stack_to_data=False,
)
```

The `scales` mapping provides generic scalar weighting.

### TGraph / TEfficiency

`TEfficiency` is converted internally through its asymmetric-error graph.

```python
data = read_root_object("data.root", "efficiency")
mc = read_root_object("mc.root", "efficiency")

plot_graph_data_mc(
    data,
    mc,
    "efficiency.pdf",
)
```

### TH2

```python
data = read_root_object("data.root", "region/h2")
mc = read_root_object("mc.root", "region/h2")

plot_hist2d_data_mc(
    data,
    mc,
    "h2.pdf",
)
```

This creates Data, MC, and Data/MC panels.

## Generic CLI

```bash
python3 plot_root_objects.py compare \
  --data data.root \
  --mc mc.root \
  --object zjet/Jet_eta_parallel \
  --output jet_eta.pdf \
  --normalize-mc-to-data
```

For TH2:

```bash
python3 plot_root_objects.py compare2d \
  --data data.root \
  --mc mc.root \
  --object region/h2 \
  --output h2.pdf
```

## Requirements

PyROOT should come from CMSSW/ROOT.

Install the plotting dependencies if needed:

```bash
python3 -m pip install --user mplhep matplotlib numpy pyyaml
```


## Efficient repeated ROOT access

For many objects from the same file, do not call `read_root_object()` in a
large loop. Keep the file open:

```python
from plotting import RootFileReader

with RootFileReader("data.root") as reader:
    h1 = reader.get("region/h1")
    h2 = reader.get("region/h2")
    graph = reader.get("region/graph")
```

`RootFileReader` caches converted objects, and the ROOT file is opened only
once. This is strongly recommended for EOS/XRootD files.


## Permanent lxplus matplotlib cache

Importing `SkimRDFAnalysisBase/plotting` now automatically configures
matplotlib to use local temporary cache/config directories:

```text
/tmp/$USER/skimrdf_matplotlib/
```

This prevents the very slow first matplotlib figure seen when `~/.cache` or
`~/.config` live on EOS/AFS.

The behavior can be disabled with:

```bash
export SKIMRDF_MPL_LOCAL_CACHE=0
```

or redirected explicitly with:

```bash
export SKIMRDF_MPLCONFIGDIR=/some/local/path
```

