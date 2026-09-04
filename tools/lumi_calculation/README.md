# Luminosity and $\mu$ calculations
Here the way to calculate the luminosity and creation of average pileup distribution from data NanoAOD ROOT files is described. The only thing needed is the input file to have the TTree `LuminosityBlocks` which contains the branches `run` and `luminosityBlock` to be able to calculate the Lumisections (LS) used.

## Making Data LS JSON
For each of those tasks the first step is to make a LS JSON file in the format CMS uses in the standard scripts of CMSSW.

This is done by using `make_lumi_json.py` as for example:
```bash
python3 make_lumi_json.py data_rootfiles/ \
    -o processed_lumis.json
```
it also supports to select only the certified lumisections by using a Certification JSON to clean for them. 
```bash
python3 make_lumi_json.py data_rootfiles/ \
    --cert-json Cert_Collisions2025_391658_398903_Golden.json \
    -o processed_lumis_golden.json
```
In this latter case you can also get from the output the number of the lumisections that was kept and the percentage - which is showing how much of the data is getting "lost". An example output could be :
```
=======================================================
Summary
=======================================================
ROOT files found             : 286
ROOT files with errors       : 0
Files without LumiBlocks     : 0
Raw LumiBlock entries read   : 42781
Unique run/LS in NanoAODs    : 32451
Certified run/LS retained    : 31992
Non-certified run/LS removed : 459
Fraction retained            : 98.59%
Runs in output               : 143
Output JSON                  : processed_lumis_golden.json
=======================================================
```

## Integrated Luminosity calculation (brilcal)
To calculate the integrated luminosity given the json produced in the previous step use `brilcalc`. Check [BrilcalcQuickStart](https://twiki.cern.ch/twiki/bin/view/CMS/BrilcalcQuickStart).

You can find documentation in [twiki](https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun3).
In this link you can find also the default values and the associated uncertainties per year.

First load the environment 
```bash
source /cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/brilws-env
```
Example:
```bash
brilcalc lumi -u /fb -b "STABLE BEAMS" --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json -i processed_lumis.json
```

Should produce such an output:
```
#Data tag : 24v2 , Norm tag: composite
+--------------+-------------------+------+------+----------------+---------------+
| run:fill     | time              | nls  | ncms | delivered(/fb) | recorded(/fb) |
+--------------+-------------------+------+------+----------------+---------------+
| 386478:10189 | 10/02/24 05:25:34 | 405  | 405  | 0.177307836    | 0.169060565   |
| 386505:10190 | 10/02/24 16:45:21 | 279  | 279  | 0.119552284    | 0.113142112   |
| 386509:10190 | 10/02/24 18:48:15 | 1029 | 1029 | 0.471178487    | 0.444940775   |
| 386553:10197 | 10/03/24 15:18:13 | 857  | 857  | 0.406073500    | 0.379080452   |
| 386554:10197 | 10/03/24 20:54:36 | 1004 | 1004 | 0.432720617    | 0.412413916   |
| 386592:10199 | 10/04/24 12:52:01 | 414  | 414  | 0.042547743    | 0.038168853   |
| 386593:10199 | 10/04/24 15:46:38 | 651  | 651  | 0.317787026    | 0.298663817   |
| 386594:10199 | 10/04/24 20:20:35 | 871  | 871  | 0.153090520    | 0.142146215   |
| 386604:10200 | 10/05/24 08:06:58 | 1992 | 1992 | 0.894621581    | 0.839470159   |
| 386605:10200 | 10/05/24 21:08:51 | 155  | 155  | 0.046244965    | 0.044711541   |
| 386614:10201 | 10/06/24 00:32:57 | 750  | 750  | 0.357070809    | 0.331437857   |
| 386615:10201 | 10/06/24 05:27:39 | 37   | 37   | 0.018022474    | 0.015984860   |
| 386616:10201 | 10/06/24 05:44:36 | 127  | 127  | 0.062064131    | 0.057227736   |
| 386617:10201 | 10/06/24 06:57:39 | 14   | 14   | 0.006849088    | 0.006137266   |
| 386618:10201 | 10/06/24 07:04:34 | 948  | 948  | 0.395405291    | 0.376681761   |
| 386629:10202 | 10/06/24 15:27:34 | 64   | 64   | 0.019895362    | 0.018559629   |
| 386630:10202 | 10/06/24 15:54:10 | 28   | 28   | 0.012779570    | 0.011608964   |
| 386640:10204 | 10/06/24 20:23:38 | 1142 | 1142 | 0.547800644    | 0.508521331   |
| 386642:10204 | 10/07/24 03:59:03 | 714  | 714  | 0.287822508    | 0.272393747   |
| 386661:10206 | 10/07/24 11:24:15 | 438  | 438  | 0.205469221    | 0.184320143   |
| 386668:10207 | 10/07/24 16:23:23 | 347  | 347  | 0.159575259    | 0.146234707   |
| 386672:10208 | 10/07/24 21:18:35 | 128  | 128  | 0.053657424    | 0.048535323   |
| 386673:10208 | 10/07/24 22:10:35 | 990  | 990  | 0.484040939    | 0.451952677   |
| 386679:10209 | 10/08/24 07:20:42 | 355  | 355  | 0.164939414    | 0.150953515   |
| 386693:10210 | 10/08/24 12:02:10 | 86   | 86   | 0.037544307    | 0.034554330   |
+--------------+-------------------+------+------+----------------+---------------+
#Summary: 
+-------+------+-------+-------+-------------------+------------------+
| nfill | nrun | nls   | ncms  | totdelivered(/fb) | totrecorded(/fb) |
+-------+------+-------+-------+-------------------+------------------+
| 13    | 25   | 13825 | 13825 | 5.874061002       | 5.496902252      |
+-------+------+-------+-------+-------------------+------------------+
```

The important part is the totrecorded(/fb) luminosity, which is the one you should scale the MC with.

## Getting the average pileup ($\mu$) distribution from data
This is done using the `pileupCalc.py` tool ([documentation](https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData#Using_pileupCalc)). 

This measures in data the expected average pileup $\mu$ from the recorded instantaneous luminosity $L_{inst}$ as follows:
$$\mu \sim \frac{L_{inst}\times \sigma_{inel}}{f_{crossings}} $$
where $\sigma_{inel}$ is the inelastic QCD cross-section and $f_{crossings}$ bunch crossing frequency (in Hz).
Note that the $\sigma_{inel}$ is important and this also contributes to the uncertainty of this measurement. Run2 can use 69.2 this is the one at 13TeV. The extrapolated cross-section at 13.6 TeV (Run3) is 80mb.

Example:
```bash
pileupCalc.py -i MyAnalysisJSON.txt --inputLumiJSON pileup_latest.txt --calcMode true --minBiasXsec 80000 --maxPileupBin 100 --numPileupBins 100 MyDataPileupHistogram.root
```

where:
- MyAnalysisJSON.txt is the JSON file defining the lumi sections that your analysis uses. This is generally the appropriate certification JSON file from PdmV or processedLumis.json from your CRAB job.
- pileup_latest.txt is the appropriate pileup file for your analysis. You can find those under `/eos/user/c/cmsdqm/www/CAF/certification/Collisions[YEAR]/PileUp/`.
- minBiasXsec defines the minimum bias cross section to use (in μb). The current run 2 recommended value is 69200.
- MyDataPileupHistogram.root is the name of the output file.

for 2024 an example:
```bash
pileupCalc.py -i processed_lumis.json --inputLumiJSON /eos/user/c/cmsdqm/www/CAF/certification/Collisions24/PileUp/pileup_JSON-2024I_Golden.txt --calcMode true --minBiasXsec 80000 --maxPileupBin 100 --numPileupBins 100 MyDataPileupHistogram.root
```