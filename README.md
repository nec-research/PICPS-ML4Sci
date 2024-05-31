# Physics-Informed Weakly Supervised Learning

(This repository is to check the reproducibility of the submitted paper). 

`PIWSL` is a codebase for training Machine Learning Interatomic Potentials (MLPIs) with newly proposed physics-informed 
weakly supervised learning method. This repo is based on the repo `mdsim` [[paper]](https://arxiv.org/abs/2210.07237) and 
OpenCatalyst Project repo [[OCP]](https://fair-chem.github.io/core/datasets/oc20.html). This implementation was tested under Ubuntu 20.04, Python 3.10, PyTorch 2.01, and CUDA 11.8. 

## Installation
We followed `mdsim` repo's description. 

### create a conda environment

```
conda create -n piwsl python=3.10 
```

### install other dependencies

After installing the conda environment, activate by running:

```
conda activate piwsl
```

Then install other dependicies using requirement.txt

## Download/preprocess data

A part of the data used in the submitted paper can be obtained from: 

[Ani-1x](https://github.com/aiqm/ANI1x_datasets)

The default path is `./DATAPATH`. Training should be immediately runnable if all preprocessed datasets are downloaded from Zenodo and the directory is properly renamed. Alternatively, the scripts for downloading and preprocessing each individual dataset are in `preprocessing/`. Specify `data_path` for the location of the saved source files, and specify `db_path` for the proprocessed files (Lmdb files for SchNet, DimeNet, etc. or npz files for NequIP). For example, preprocess Ani-1x dataset to `DATAPATH` and save processed files to `DBPATH` by running:

```
python preprocessing/ani-1x_new.py
```

All datasets have unit kcal/mol for energy and kcal/mol/$\AA$ for forces. The mean and standard deviation of energy/forces are stored in `metadata.npy`.


## Train ML force fields

We recommend logging with wandb and it is used by default. You need to have a wandb account and log in with `wandb init`. More details at https://wandb.ai/. You can use a different logger by changing `logger` in the `base.yml` config files.

Training is mostly through `main.py`. All training configs are stored in `configs/`. The training can be done using our script file. 
For example, train on MD17 can be done as:

```
sh perform_model_train_ani1x.sh
```

For baseline model results, use `configs/base_van.yml`. 

For PIWSL model results, use `configs/base_piwsl.yml`. 

For NoisyNode model results, use `configs/base_aug.yml`. 
