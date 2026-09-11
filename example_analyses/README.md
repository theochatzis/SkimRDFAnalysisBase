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
    histogram/profile definitions

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
Z.valid()
|m_mumu - mZ| < 3.743 GeV
pT(Z) > 20 GeV
at least one selected clean jet
```

## Data / MC

The definition inspects the input columns.

If `genWeight` exists, the default event weight is `sign(genWeight)`;
otherwise it is 1.

If `GenJet_*` exists, reco/gen matching is enabled automatically.
For data, all MC-study histogram vectors are defined empty, which means
the same histogram YAML can be used for both data and MC without filling
misleading efficiency/purity denominators.

## DB / MPF eta categories

The eta categories are encoded as weight columns:

```text
signalWeight_HB
signalWeight_HE1
...
windowWeight_HB
windowWeight_HE1
...
```

This allows all response profiles to be booked in the same `zjet` region
without creating analysis-specific event-loop code.

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
