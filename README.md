# Physics-Informed Weakly Supervised Learning

![Schematic illustration of physics-informed weakly supervised losses used in this work.](/piwsl.png)

## Introduction

This repository provides the PyTorch implementation of _Physics-Informed Weakly Supervised Learning (PIWSL)_, 
a method for training Machine Learning Interatomic Potentials (MLPIs) with newly proposed physics-informed 
weakly supervised learning, accepted in ICML2025 [[Paper]](https://www.arxiv.org/abs/2408.05215). 

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

Training is mostly through `main.py`. All training configs are stored in `configs/`. The training can be done using our script file. 
For example, train on ANI-1x can be done as:

```
sh perform_model_train_ani1x.sh
```

For baseline model results, use `configs/base_van.yml`. 

For PIWSL model results, use `configs/base_piwsl.yml`. 

For NoisyNode model results, use `configs/base_aug.yml`. 

## Infrastructure

This implementation was tested under Ubuntu 20.04, Python 3.10, PyTorch 2.01, and CUDA 11.8. 

## Acknowledgements
This codebase is based on the repository `mdsim` [[paper]](https://arxiv.org/abs/2210.07237) and 
_OpenCatalyst Project_ repository [[OCP]](https://fair-chem.github.io/core/datasets/oc20.html).

## Citation 
```bibtex
@misc{takamoto2024physicsinformedweaklysupervisedlearning,
      title={Physics-Informed Weakly Supervised Learning for Interatomic Potentials}, 
      author={Makoto Takamoto and Viktor Zaverkin and Mathias Niepert},
      year={2024},
      eprint={2408.05215},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2408.05215}, 
}
```