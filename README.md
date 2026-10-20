# SF-FPA for Source-Free Time-Series Domain Adaptation

This repository contains the code for **SF-FPA**, a source-free domain adaptation method that learns sample-dependent amplitude and phase corrections at the input spectrum while keeping the source encoder and classifier weights frozen. The project also includes implementations of several SFDA baselines and an optional FAL adapter for SHOT, AaD, and NRC.

This codebase builds on the [MAPU framework](https://github.com/mohamedr002/MAPU_SFDA_TS).

## Methods

The main training entry points are `trainers/train.py` and `trainers/test.py`. The algorithms configured for the WISDM, FD, and Boiler experiments include SHOT, AaD, NRC, MAPU, TemSR, CE-SFDA (`CESFDA` in code), SAFA, and SF-FPA. The optional baseline+FAL path is implemented and runnable for SHOT, AaD, and NRC through `scripts/run_baseline_fal.sh`, retaining each method's target update objective.

TFDA is not reimplemented in this repository. TFDA baseline results were obtained using the [authors' TFDA repository](https://github.com/furqon3009/TFDA).

## Environment

The reference environment used for the experiments is Python 3.8 with PyTorch 2.4.1. Install the pinned dependencies with:

```bash
conda create -n sffpa python=3.8 -y
conda activate sffpa
pip install -r requirements.txt
```

For GPU use, select the PyTorch 2.4.1 wheel matching your CUDA platform from the official PyTorch installer. CUDA is optional; use `--device cpu` or `--device cuda` as appropriate.

## Datasets

The loaders expect the prepared domain-wise `.pt` files under `dataset/<DATASET>/`; exact file names, tensor shapes, transfer lists, and source dataset pages are documented in [DATASETS.md](DATASETS.md). The serialized files are dictionaries containing `samples` and `labels`. Validate a local dataset directory before running experiments:

```bash
python scripts/verify_dataset_layout.py --data-root dataset
```

The WISDM, FD/MFD, and Boiler tensors currently present on the development machine total about 2.6 GB. They are excluded from normal Git and configured for Git LFS. Before distributing them, preserve the source dataset attribution/readme and verify each dataset's redistribution terms. Do not use ordinary Git for these binaries.

## Reproduce SF-FPA

Run from the repository root. Each command evaluates all scenarios configured for that dataset with three seeds:

```bash
python trainers/train.py --dataset WISDM --da_method SF_FPA --num_runs 3 --device cuda --save_dir runs
python trainers/test.py  --dataset WISDM --da_method SF_FPA --num_runs 3 --device cuda --save_dir runs
```

Replace `WISDM` with `FD` or `Boiler` to run the other benchmarks. The selected protocol uses nine transfers for WISDM, nine for FD, and the three Boiler transfers `1->2`, `1->3`, and `2->3`. Dataset schedules and method hyperparameters are in `configs/hparams.py`; domain shapes and transfer scenarios are in `configs/data_model_configs.py`.

## Reproduce Baselines

For example, train and evaluate SHOT on WISDM:

```bash
python trainers/train.py --dataset WISDM --da_method SHOT --num_runs 3 --device cuda --save_dir runs --run_description SHOT
python trainers/test.py  --dataset WISDM --da_method SHOT --num_runs 3 --device cuda --save_dir runs --run_description SHOT
```

Use `AaD` or `NRC` in place of `SHOT` for those baselines. The source model is trained from the source split by `train.py`; target adaptation uses each baseline's implemented update loss.

## Add FAL to a Baseline

First train the baseline normally so that its per-scenario `checkpoint.pt` files contain the source model. Then initialize a separate run from those source weights and enable FAL. For SHOT on WISDM:

```bash
bash scripts/run_baseline_fal.sh WISDM SHOT ../runs/WISDM/SHOT runs_fal
```

Replace `SHOT` with `AaD` or `NRC`, and use the corresponding source checkpoint directory. The script accepts optional output directory, device, and run-count arguments. FAL is initialized with the dataset's SF-FPA spectral settings; encoder parameters are frozen, and the baseline's classifier update policy and target loss are retained. The source checkpoint directory must contain `<source>_to_<target>_run_<seed>/checkpoint.pt` for every configured scenario and seed.
