# 5G Localization

This is the main repository for my code implementation part of my masters thesis on enhanced localization in 5G networks
using machine learning.
This repository will be the foundation for experimenting with different positioning techniques and ML models. The data
im using is a dataset
collected in Oslo and Rome on both 5G and NB-IoT networks.

## Getting started

To set up the project, run:

```bash
./setup.sh
```

Run the Jupyter notebook:

```bash
jupyter notebook
``` 

It can also be run using PyCharm or Data Spell IDEA by pressing the play button on the notebook.

## About the project

## Project code

### Notebook

The code is run from a jupyter notebook found in `Main.ipynb`.
Here can you also find notes and in-line documentation.

### Scripts

Other code is found in the `scripts/` directory for things like data loading and formatting.
The data_loader is the module responsible for loading the matlab files and returning them
as pandas dataframes.

#### `weighted_coverage.py`

This is the main logic and implementation of the wKNN algorithm.

#### `data_loader.py`

The matlab data files are loaded and transformed into pandas dataframes
that are used throughout the project.

#### `utils.py`

Contains some utility functions to assist the main algorithm, e.g. haversine distance.

#### `plotting.py`

Methods to plot graphs.

## Dataset

### Pre-processed Matlab files

For now im using the NB-IoT dataset to get started, but eventually i will switch to the 5G dataset.

Im using the `Campaign_data_NBIoT_1_2_3_4_5_6_interpolated_smoothed.mat` file which contains three datasets.
This file is generated from the matlab code found [Here](https://github.com/lucadn/positioning-5G/tree/main), in the
NB-IoT part of the code. This reads the raw `.xlsx` files, interpolates and cleans them. Then the data files are
generated
as the `.mat` files found in the `data/` directory in this project.

**The file contains three datasets:**

- dataSet
- dataSet_interp
- dataSet_smooth

### Dataset content

| Column | Label               | Content                                                                                                                         |
|--------|---------------------|---------------------------------------------------------------------------------------------------------------------------------|
| 1      | lat                 | Latitude                                                                                                                        |
| 2      | lng                 | Longitude                                                                                                                       |
| 3      | measurements_matrix | A matrix that contains for each row the following info: NPCI; eNodeB ID; RSSI; NSINR; NRSRP; NRSRQ; ToA; operatorID; campaignID |
| 4      | num_npcis_rf_op1    | A scalar that reports the number of NPCIs with RF data for operator 1                                                           |
| 5      | logical_rf_op1      | A logical column vector that has 1s at positions of the matrix containing a NPCI with RF data for operator 1                    |
| 6      | num_npcis_toa_op1   | A scalar that reports the number of NPCIs with ToA data for operator 1                                                          |
| 7      | logical_toa_op1     | A logical column vector that has 1s at positions of the matrix containing a NPCI with ToA data for operator 1                   |
| 8      | num_npcis_rf_op2    | A scalar that reports the number of NPCIs with RF data for operator 2                                                           |
| 9      | logical_rf_op2      | A logical column vector that has 1s at positions of the matrix containing a NPCI with RF data for operator 2                    |
| 10     | num_npcis_toa_op2   | A scalar that reports the number of NPCIs with ToA data for operator 2                                                          |
| 11     | logical_toa_op2     | A logical column vector that has 1s at positions of the matrix containing a NPCI with ToA data for operator 2                   |
| 12     | num_npcis_rf_op3    | A scalar that reports the number of NPCIs with RF data for operator 3                                                           |
| 13     | logical_rf_op3      | A logical column vector that has 1s at positions of the matrix containing a NPCI with RF data for operator 3                    |
| 14     | num_npcis_toa_op3   | A scalar that reports the number of NPCIs with ToA data for operator 3                                                          |
| 15     | logical_toa_op3     | A logical column vector that has 1s at positions of the matrix containing a NPCI with ToA data for operator 3                   |
| 16     | campaign_ids        | A column vector that contains the list of campaign IDs that contributed to the data in the location                             |

## References

### Dataset

Luca De Nardis, Giuseppe Caso, Özgü Alay, Marco Neri, Anna Brunstrom, & Maria-Gabriella Di Benedetto. (2023). Outdoor
NB-IoT and 5G coverage and channel information data in urban environments [Data set].
Zenodo. https://doi.org/10.5281/zenodo.8161173