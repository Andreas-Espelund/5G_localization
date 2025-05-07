# 5G Localization

This is the main repository for my code implementation part of my masters thesis on enhanced localization in 5G networks
using machine learning.
This repository will be the foundation for experimenting with different positioning techniques and ML models. The data
im using is a dataset
collected in Oslo and Rome on both 5G and NB-IoT networks.

## Getting started

I used a venv to control packages and versions.
To set up the project, run the setup script:

```bash
# Allow execution privleges
chmod +x setup.sh

# Run the setup script
./setup.sh
```

This next part is a bit stupid, but I might fix it. Some parts of the code read a project root
path from a config file.

1. get the absolute path of your repo-root-dir by doing `pwd`. The path should end with `<...>/5G_localization`.
2. Replace the existing path in the file `config/config.json` with this value

It should look like this:

```json
{
  "project_root": "<your pwd>"
}
```

## About the project

### Experiments

The `experiments` directory has some python scripts with the setups for different experiments
I conducted in my thesis work. Due to many changes and little time, conventions changed a bit
here, making the code a bit repetitive and messy. I might clean this up later.

**Storing results**

I store the experiment results with the function `save_experiment_result` that stores the
results in a directory inside `data/results/experiments` with a json file with config values
as well as .csv files with the measurements.

### Scripts

Most code is found in the `scripts/` directory for algorithm implementation, utils,
data loading etc.

### Plotting

I used JupyterNotebooks to do most the plotting. These notebooks are in the `plotting/` directory
and usually there is a corresponding plotting notebook to an experiment.
I also have a file, `scripts/plotting.py`, that I initially used to make reusable plotting functions,
but I quickly realized that the plots alle are quite different. So here is some unused code that I might
clean up.

### Other notebooks

I have done a lot of experimentation and testing. Things like beam-matching and PCA. I thought
that this could be nice to have [(Kjekt å ha)](https://www.youtube.com/watch?v=ZyMtjM6aLgY), so
I put them inside `notebooks/`.

## Dataset

### Dataset (5G)

The 5G dataset I used was provided in a .mat file

### Dataset (NB-IoT)

At the start I used the NB-IoT dataset to get started, but eventually switched to the 5G dataset.

The `Campaign_data_NBIoT_1_2_3_4_5_6_interpolated_smoothed.mat` file contains three datasets.
This file is generated from the matlab code found [Here](https://github.com/lucadn/positioning-5G/tree/main), in the
NB-IoT part of the code. This reads the raw `.xlsx` files, interpolates and cleans them. Then the data files are
generated
as the `.mat` files found in the `data/` directory in this project.

**The file contains three datasets:**

- dataSet
- dataSet_interp
- dataSet_smooth

#### Dataset content

| Column   | Label                 | Content                                                                                                                           |
|----------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| 1        | lat                   | Latitude                                                                                                                          |
| 2        | lng                   | Longitude                                                                                                                         |
| 3        | measurements_matrix   | A matrix that contains for each row the following info: NPCI; eNodeB ID; RSSI; NSINR; NRSRP; NRSRQ; ToA; operatorID; campaignID   |
| 4        | num_npcis_rf_op1      | A scalar that reports the number of NPCIs with RF data for operator 1                                                             |
| 5        | logical_rf_op1        | A logical column vector that has 1s at positions of the matrix containing a NPCI with RF data for operator 1                      |
| 6        | num_npcis_toa_op1     | A scalar that reports the number of NPCIs with ToA data for operator 1                                                            |
| 7        | logical_toa_op1       | A logical column vector that has 1s at positions of the matrix containing a NPCI with ToA data for operator 1                     |
| 8        | num_npcis_rf_op2      | A scalar that reports the number of NPCIs with RF data for operator 2                                                             |
| 9        | logical_rf_op2        | A logical column vector that has 1s at positions of the matrix containing a NPCI with RF data for operator 2                      |
| 10       | num_npcis_toa_op2     | A scalar that reports the number of NPCIs with ToA data for operator 2                                                            |
| 11       | logical_toa_op2       | A logical column vector that has 1s at positions of the matrix containing a NPCI with ToA data for operator 2                     |
| 12       | num_npcis_rf_op3      | A scalar that reports the number of NPCIs with RF data for operator 3                                                             |
| 13       | logical_rf_op3        | A logical column vector that has 1s at positions of the matrix containing a NPCI with RF data for operator 3                      |
| 14       | num_npcis_toa_op3     | A scalar that reports the number of NPCIs with ToA data for operator 3                                                            |
| 15       | logical_toa_op3       | A logical column vector that has 1s at positions of the matrix containing a NPCI with ToA data for operator 3                     |
| 16       | campaign_ids          | A column vector that contains the list of campaign IDs that contributed to the data in the location                               |
| -------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |

---

## References

### Dataset

Luca De Nardis, Giuseppe Caso, Özgü Alay, Marco Neri, Anna Brunstrom, & Maria-Gabriella Di Benedetto. (2023). Outdoor
NB-IoT and 5G coverage and channel information data in urban environments [Data set].
Zenodo. https://doi.org/10.5281/zenodo.8161173