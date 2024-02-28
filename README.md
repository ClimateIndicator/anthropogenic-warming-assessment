# Anthropogenic Warming Assessment
![IGCC-2022 Published Headline Results Figure](https://essd.copernicus.org/articles/15/2295/2023/essd-15-2295-2023-f05-web.png)
![IGCC-2022 Published Headline Results Table](https://essd.copernicus.org/articles/15/2295/2023/essd-15-2295-2023-t06-web.png)
*Most recent IGCC-2022 headline results, published in Forster et al. (2023).*

## Introduction
This repository contains the results, and generating code, for the attributed contributions to global warming as a part of the Indicators of Global Climate Change (IGCC) [project](https://www.igcc.earth/), which provides annual updates to key IPCC assessments. The first IGCC report was [Forster et al. (2023)](https://essd.copernicus.org/articles/15/2295/2023); the section in the report corresponding to this repository is [section 7](https://essd.copernicus.org/articles/15/2295/2023/#section7).


## Versions
Each annual update of the IGCC will be available as a separate [release](https://github.com/ClimateIndicator/anthropogenic-warming-assessment/releases) in this repository. The release version is identified by the year of the assessed indictaor, and not the year of publication, e.g. results for the levels of attributed warming in 2022 (as publised in Forster et al. (2023)) are available in the release named IGCC-2022, not IGCC-2023.

By default, the version of the results and code displayed on the main page of the repository is the most recent version. Currently this is the 2022 update (as published in the [IGCC 2022 Report (Forster et al. (2023))](https://essd.copernicus.org/articles/15/2295/2023/)).

The code and results for a specific iteration of the IGCC can be accessed in a number of ways:
1. **Access the results directly on GitHub:**
    1. Navigate to the code for a specific version directly on GitHub at `https://github.com/ClimateIndicator/anthropogenic-warming-assessment/tree/<IGCC-version>`, where `<IGCC-version>` should be replaced by the required version, for example `IGCC-2022`.
    2. Equivalently, you can display a given release by selecting the requried version from the *tags* list in the *branch* drop down menu on this page (which will say `master` by default).
2. **Download the zip** containing the code and results:
    1. Navigate to the [releases](https://github.com/ClimateIndicator/anthropogenic-warming-assessment/releases) page
    2. Download the zip file associated wtih the required release version, which is listed in the 'Assets' section of a given release.
3. **Use the command line locally to clone the repository** and checkout the required version:
    1. `git clone <repo url>`: clone the repository
    2. `cd anthropogenic-warming-assessment`: move into the newly cloned local repository
    3. `git checkout <IGCC-version>`: checkout the required version, replacing `<IGCC-version>` with, for example, `IGCC-2022` - the local workspace will now contain the historical repo for the required version.

## Methods
### Multi-method assessment
The multi-method assessment of anthropogenic warming is based on the results from three individual attribution methods. The assessment approach is detailed in [section 7]((https://essd.copernicus.org/articles/15/2295/2023/#section7)) of the IGCC 2022 report (Forster et al. (2023)).

The code for the multi-method assessment is contained in the script `multi_method_assessment.py`. It takes as input the results from all three individual attribution methods, and produces the overall multi-method assessment.

Plots of the results can be created by running `python multi_method_assessemnt.py` inside the `anthropogenic-warming-assessment` directory. The plots will be saved in a newly created `plots/` directory. All the required data for this is contained in the `results/` directory.

### Individual Methods
The three attribution methods used for the multi-method assessment of anthropogenic warming in the [IGCC 2022 report (Forster et al. (2023))](https://essd.copernicus.org/articles/15/2295/2023/) are:
1. Global Warming Index (GWI), detailed in [supplement section S7.1](https://essd.copernicus.org/articles/15/2295/2023/essd-15-2295-2023-supplement.pdf), with results available in the files `results/Walsh_*.csv`.
2. Kriging for Climate Change (KCC), detailed in [supplement section S7.2](https://essd.copernicus.org/articles/15/2295/2023/essd-15-2295-2023-supplement.pdf), with results available in the files `results/Ribes_*.csv`.
3. Regularised Optimal Fingerprinting (ROF), detailed in [supplement section S7.3](https://essd.copernicus.org/articles/15/2295/2023/essd-15-2295-2023-supplement.pdf), with results available in the files `results/Gillett_*.csv`.

There is no pipeline that automatically pulls in the results for each method; the results files were simply manually copied into the `results/` directory, and pushed to GitHub.

#### Global Warming Index
The code for the GWI attribution in the IGCC 2022 report is available in this repository. The code is contained in the directory `attribution_methods/GlobalWarmingIndex`.

#### Regularised Optimal Fingerprinting
The esmvaltool code (which generates ROF attribution results based on CMIP6 outout) is available on GitHub at [ESMValGroup/ESMValTool at forster23 (github.com)](https://github.com/ESMValGroup/ESMValTool/tree/forster23). The actual python diagnostics code is available here: [ESMValTool/esmvaltool/diag_scripts/attribute at forster23 · ESMValGroup/ESMValTool (github.com)](https://github.com/ESMValGroup/ESMValTool/tree/forster23/esmvaltool/diag_scripts/attribute).

#### Kriging for Climate Change
The code for the KCC attribution in the IGCC 2022 report based on code contained in the following GitLab repo: [Global temperature constraint (gitlab.com)](https://gitlab.com/saidqasmi/global-temperature-constraint).



## Results
The results from all three individual attribution methods, and the overall multi-method assessment, are provided in the `results/` directory in this repository, which contains the following:
- `Assessment-6thIPCC_headlines.csv`: the previous assessment results from  AR6 and SR1.5
- `Assessment-UPdate-2022-GMST_headlines.csv`: the results from the multi-method assessment for the 2022 update of the IGCC
- `<Surname>_GMST_headlines.csv`: the results from the individual attribution methods that feed into the multi-method assessment. `<Surname>` is replaced by the surname of the lead author of the method, i.e. `Walsh`, `Ribes`, and `Gillett` for the GWI, KCC, and ROF methods, respectively.

The results from this `anthropogenic_warming_assessment` GitHub repository are also available on [GitHub](https://github.com/ClimateIndicator/data/) and [Zenodo](https://doi.org/10.5281/zenodo.8430424) - note the specific versioning for each release differs from this repository.