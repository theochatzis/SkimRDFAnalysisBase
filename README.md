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


