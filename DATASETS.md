# Dataset Preparation

The experiment code consumes preprocessed, domain-wise PyTorch files. Place them in `dataset/<DATASET>/` using the exact names below. Each file must deserialize to a dictionary with:

- `samples`: array/tensor shaped `(N, L, C)`, `(N, C, L)`, or `(N, L)` for univariate data.
- `labels`: integer labels with one value per sample, encoded from zero.

The dataloader normalizes each split independently and converts inputs to float tensors. Keep the train/test split and domain IDs intact; do not combine domains before adaptation.

| Dataset | Local directory | Domains | Channels x length | Classes | Files |
| --- | --- | --- | --- | --- | --- |
| WISDM | `dataset/WISDM/` | `0`--`35` | `3 x 128` | 6 | `train_<id>.pt`, `test_<id>.pt` |
| FD/MFD | `dataset/FD/` | `0`--`3` | `1 x 5120` | 3 | `train_<id>.pt`, `test_<id>.pt` |
| Boiler | `dataset/Boiler/` | `1`--`3` | `20 x 36` | 2 | `train_<id>.pt`, `test_<id>.pt` |

## Benchmark Transfers

The checked-in experiment configurations use these transfers:

- WISDM: `2->32`, `7->30`, `18->20`, `20->30`, `26->2`, `9->3`, `22->18`, `12->9`, `2->20`.
- FD/MFD: `0->1`, `0->2`, `0->3`, `1->0`, `1->3`, `2->0`, `2->1`, `3->0`, `3->2`.
- Boiler: `1->2`, `1->3`, `2->3`.

Training uses three runs with seeds `0`, `1`, and `2` by default. The configured training epochs and hyperparameters are in `configs/hparams.py`.

## Dataset Sources and Attribution

Obtain the original datasets from their respective providers and follow the provider's current terms for downloading, reuse, and redistribution:

- WISDM activity-recognition data: [Fordham WISDM Lab dataset page](https://www.cis.fordham.edu/wisdm/dataset.php). Follow the source page's attribution instructions and retain its README when redistributing source data.
- FD/MFD bearing dataset: [Data repository record](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/PU85XN). Cite the dataset and the AdaTime benchmark paper.
- Boiler simulated fault dataset: [IEEE DataPort record](https://ieee-dataport.org/open-access/simulated-boiler-data-fault-detection-and-classification), DOI `10.21227/awav-bn36`. Cite Shohet, Kandil, and McArthur (2019).

Prepared tensors remain local and are ignored by Git. The bundles are approximately 2.6 GB in total; FD and Boiler contain individual files larger than GitHub's ordinary 100 MB file limit. `.gitattributes` configures these directories for Git LFS if redistribution is approved. Confirm dataset terms and LFS quota before intentionally adding any files.

```bash
git lfs install
git lfs track 'dataset/WISDM/*.pt' 'dataset/FD/*.pt' 'dataset/Boiler/*.pt'
git add .gitattributes
# Add data files explicitly only after confirming redistribution and hosting terms.
```

Validate all expected files before training:

```bash
python scripts/verify_dataset_layout.py --data-root dataset
```

The validator loads local `.pt` files. Only run it on dataset files from trusted sources.
