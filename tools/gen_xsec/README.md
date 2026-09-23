# Generator cross section

Derive the effective generator cross section from EDM ROOT files:

```bash
cmsRun tools/gen_xsec/gen_xsec_cfg.py \
  inputFiles=root://cms-xrd-global.cern.ch//store/mc/.../file.root
```

Use several comma-separated files for a statistically representative result.
The final `GenXSecAnalyzer` summary reports the cross section and, when
applicable, the filter efficiency. This requires a CMSSW release that supports
`GenXSecAnalyzer`.

Reference:
https://gitlab.cern.ch/cms-gen/genproductions_scripts/-/tree/master/Utilities/calculateXSectionAndFilterEfficiency?ref_type=heads
