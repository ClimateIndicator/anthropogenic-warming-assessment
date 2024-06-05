# Anthropogenic Warming Assessment
The [Indicators of Global Climate Change (IGCC) project](https://www.igcc.earth/) provides annual updates to key IPCC assessments. This repository contains the results and generating code for the assessment of the level and rate of global warming, and the attributed contributions to them.


## IGCC Releases and Citations
### Summary Table
| Indicator Year | Code Release | Paper Reference | Paper DOI | Dataset Reference | Dataset DOI |
| --- | --- | --- | --- | --- | --- |
| **2022** | [`IGCC-2022`](https://github.com/ClimateIndicator/anthropogenic-warming-assessment/tree/IGCC-2022) | [Forster et al. (2023) - Section 7](https://doi.org/10.5194/essd-15-2295-2023) | [![DOI](https://zenodo.org/badge/DOI/10.5194/essd-15-2295-2023.svg)](https://doi.org/10.5194/essd-15-2295-2023) | [Smith et al. (2023)]() | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8430424.svg)](https://doi.org/10.5281/zenodo.8430424) |
| **2023** |  [`IGCC-2023`](https://github.com/ClimateIndicator/anthropogenic-warming-assessment/tree/IGCC-2023) | [Forster et al. (2024) - Section 7]() | [![DOI](https://zenodo.org/badge/DOI/10.5194/essd-16-2625-2024.svg)](https://doi.org/10.5194/essd-16-2625-2024) | [Smith et al. (2024)](https://doi.org/10.5281/zenodo.11388387) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11388387.svg)](https://doi.org/10.5281/zenodo.11388387) |

Note:
- The **GitHub Code Release** links to the specific release version of the code and results in this repository used to generate results and figures. The generating code here does not have a **DOI**.
- The **Paper Reference** links to the peer-reviewed publication that presents the results and figures of the IGCC assessment. This has a permanent **DOI**, and can be cited.
- The **Dataset Reference** is the data repository that contains the results of the IGCC assessment. This has a permanent **DOI**, and can be cited.

<!-- [![GITHUB](https://img.shields.io/badge/IGCC-2022-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ClimateIndicator/anthropogenic-warming-assessment/tree/IGCC-2022) -->


### Accessing historical release versions of this repository
Each annual update of the IGCC will be available as a separate [release](https://github.com/ClimateIndicator/anthropogenic-warming-assessment/releases) in this repository.

The release version is identified by the year of the assessed indicator, and not the year of publication, e.g. results for the levels of attributed warming in 2022 (as published in Forster et al. (2023)) are available in the release named IGCC-2022, not IGCC-2023.

By default, the version of the results and code displayed on the main page of the repository is the most recent version.

The code and results for a specific iteration of the IGCC can be accessed in a number of ways:
1. **Access the results directly on GitHub:**
    1. Navigate to the code for a specific version directly on GitHub at `https://github.com/ClimateIndicator/anthropogenic-warming-assessment/tree/<IGCC-version>`, where `<IGCC-version>` should be replaced by the required version, for example `IGCC-2022`.
    2. Equivalently, you can display a given release by selecting the required version from the *tags* list in the *branch* drop down menu on this page (which will say `master` by default).
2. **Download the zip** containing the code and results:
    1. Navigate to the [releases](https://github.com/ClimateIndicator/anthropogenic-warming-assessment/releases) page
    2. Download the zip file associated with the required release version, which is listed in the 'Assets' section of a given release.
3. **Use the command line locally to clone the repository** and checkout the required version:
    1. `git clone <repo url>`: clone the repository
    2. `cd anthropogenic-warming-assessment`: move into the newly cloned local repository
    3. `git checkout <IGCC-version>`: checkout the required version, replacing `<IGCC-version>` with, for example, `IGCC-2022` - the local workspace will now contain the historical repo for the required version.


## Methods
### Multi-method Assessment
The multi-method assessment of anthropogenic warming is based on the results from three individual attribution methods. The assessment approach is detailed in [section 7]((https://essd.copernicus.org/articles/15/2295/2023/#section7)) of the IGCC 2022 report (Forster et al. (2023)).

#### Running the code
The code for the multi-method assessment is contained in the script `multi_method_assessment.py`. It takes as input the results from all three individual attribution methods, and produces the overall multi-method assessment.

Plots of the results can be created by running `python multi_method_assessemnt.py` inside the `anthropogenic-warming-assessment` directory. The plots will be saved in a newly created `plots/` directory. All the required data for this is contained in the `results/` directory.

#### Conda Environment
The conda environment used to generate the results in this repository is available in the file `environment.yml` to aid reproducibility.

To keep the environment file as cross-platform as possible, only the directly installed packages and their version during the analysis are included. Note that the analysis was only carried out and tested on a Linux computing cluster.

The environment can be installed by running `conda env create --name <env-name> --file environment.yml` where you can specify your preferred environment name in place of `<env-name>`.

Note, you may need to rebuild the `matplotlib` font cache using `rm -rf ~/.cache/matplotlib/` to ensure that matplotlib can find the fonts that are installed through the `open-fonts` conda package included in the environment file.


### Individual Attribution Methods
Three attribution methods are used for the multi-method assessment of anthropogenic warming in the [IGCC papers](#igcc-releases). These methods are detailed in the *supplement* of the IGCC papers. They are:
1. **(GWI)** Global Warming Index, with results available in the files `results/Walsh_*.csv`.
2. **(KCC)** Kriging for Climate Change, with results available in the files `results/Ribes_*.csv`.
3. **(ROF)** Regularised Optimal Fingerprinting, with results available in the files `results/Gillett_*.csv`.


There is no pipeline that automatically pulls in the results for each method; the results files were simply manually copied into the `results/` directory, and pushed to this GitHub repository.

#### Global Warming Index
The code for the GWI attribution in the IGCC 2022 report is available in this repository. The code is contained in the directory `attribution_methods/GlobalWarmingIndex`.

#### Regularised Optimal Fingerprinting
The `esmvaltool` code (which generates ROF attribution results based on CMIP6 outout) is available on GitHub at [ESMValGroup/ESMValTool at forster23 (github.com)](https://github.com/ESMValGroup/ESMValTool/tree/forster23). The actual python diagnostics code is available here: [ESMValTool/esmvaltool/diag_scripts/attribute at forster23 · ESMValGroup/ESMValTool (github.com)](https://github.com/ESMValGroup/ESMValTool/tree/forster23/esmvaltool/diag_scripts/attribute).

#### Kriging for Climate Change
The code for the KCC attribution in the IGCC 2022 report based on code contained in the following GitLab repo: [Global temperature constraint (gitlab.com)](https://gitlab.com/saidqasmi/global-temperature-constraint).


## Results
The results from all three individual attribution methods, and the overall multi-method assessment, are provided in the `results/` directory in this repository, which contains the following:
- [Overall multi-method assessment](#multi-method-assessment)
    - `Assessment-6thIPCC_headlines.csv`: assessment results directly quoted from the IPCC's sixth assessment cycle (including both AR6 and SR1.5).
    - `Assessment-Update-<indicator-year>_GMST_headlines.csv`: the results from the multi-method assessment for the level of warming in the `<indicator-year>` update of the IGCC.
    - `Assessment-Update-<indicator-year>_GMST_rates.csv`: the results for the rates of warming in the `<indicator-year>` update of the IGCC.
- [Indivudual attribution methods](#individual-attribution-methods)
    - `<Surname>_GMST_headlines.csv`: the results from the individual attribution methods that feed into the multi-method assessment.
    - `<Surname>_GMST_timeseries.csv`: the single-year timeseries from the individual attribution methods. 
    - `<Surname>_GMST_rates.csv`: the decadal rates for the individual attribution methods.
    - **Note:** `<Surname>` is replaced in each case by the surname of the lead author of each method, i.e. `Walsh`, `Ribes`, and `Gillett` for the GWI, KCC, and ROF methods, respectively.

The results from this `anthropogenic_warming_assessment` GitHub repository are formally available with metadata in the [`ClimateIndicator/data` GitHub repository](https://github.com/ClimateIndicator/data/), and with a citable DOI in the [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.7883757) - note the specific versioning for each release differs from this feeder repository.


### Headline Results for Level of Warming
![IGCC-2023 Published Headline Results Figure](https://essd.copernicus.org/articles/16/2625/2024/essd-16-2625-2024-f07-web.png)
![IGCC-2023 Published Headline Results Table](https://essd.copernicus.org/articles/16/2625/2024/essd-16-2625-2024-t06-web.png)
*IGCC-2023 headline results; these and several other figures are produced by the code in this repo and published in [Forster et al. (2024)](https://essd.copernicus.org/articles/16/2625/2024/#section7)*

### Headline Results for Rate of Warming
![IGCC-2023 Published Rate Results Figure](https://essd.copernicus.org/articles/16/2625/2024/essd-16-2625-2024-f08-web.png)
![IGCC-2023 Published Rate Results Table](https://essd.copernicus.org/articles/16/2625/2024/essd-16-2625-2024-t07-web.png)
*IGCC-2023 headline results; these and several other figures are produced by the code in this repo and published in [Forster et al. (2024)](https://essd.copernicus.org/articles/16/2625/2024/#section7)*

