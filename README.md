# EvoState

Code and processed data for **"EvoState learns state representations for predicting antimicrobial resistance evolution"**.

EvoState predicts which resistance-associated feature a bacterial lineage is likely to acquire next, given its current resistance state. A compact Transformer encoder is first pretrained on resistance states with masked state reconstruction and state-contrastive learning (Stage 1). It is then fine-tuned on observed acquisition events to rank the eligible candidate features (Stage 2).

## Repository layout

```
src/models/            EvoState model and training code
  two_stage_backbone.py          Transformer state encoder, Stage-1 objectives, Stage-2 event head
  train_state_pretrain.py        Stage 1 CLI: self-supervised state pretraining
  train_transition_two_stage.py  Stage 2 CLI: conditional event prediction
  train_dl.py, model_dl.py       data loading, feature metadata, shared utilities
  baselines.py                   Uniform and Prevalence baselines
src/inference.py       load a released checkpoint and rank next acquisitions (no training)
src/eval/
  run_two_stage_eval.py          grouped K-fold benchmark (EvoState, EvoState w/o pretraining, baselines, HyperTraPS)
  evaluator.py                   metrics (log-likelihood, Top-k, AUC) and grouped fold construction
  run_phylo_eval.py              shared evaluation helpers and the HyperTraPS wrapper
configs/<task>.conf    hyperparameters for each of the seven tasks (from the manuscript runs)
scripts/
  run_benchmark.sh               cross-validation for one task
  run_all_benchmarks.sh          all seven tasks plus a summary table
  train_full.sh                  fit Stage 1 and Stage 2 on all transitions of a task
  summarize_benchmark.py         Supplementary Table S1-style summary and comparison with the manuscript runs
  verify_datasets.py             checksum, Table S3 and fold-partition checks
  predict.py                     command-line predictions with a released checkpoint
  check_checkpoints.py           checksum and reference-output check of all released checkpoints
checkpoints/           20 released EvoState models fitted on all transitions (see checkpoints/MANIFEST.json)
preprocessing/         raw public data -> data/processed (TB, K. pneumoniae, E. coli)
data/processed/        model-ready transitions for the seven tasks and the K. pneumoniae subsets (see data/README.md)
data/manuscript/       inputs of the figure notebooks (evaluation results, source data, caches, map shapefile)
results/paper_runs/    archived evaluation summaries of the manuscript runs
notebooks/manuscript/  notebooks that draw Figures 2-5 and Supplementary Figures S1-S6
  lib/                           analysis modules imported by the Figure 3 and Figure 4 notebooks
results/rendered/      figures and source data written by the notebooks (git-ignored)
```

## Installation

```bash
conda env create -f environment.yml
conda activate evostate
```

Validated with Python 3.10, PyTorch 2.4.0, NumPy 2.2.6, pandas 2.3.3 and scikit-learn 1.7.2 on Linux. A GPU is optional.

HyperTraPS, used only as a comparator, is an external GPL-3.0 program. To include it, build it from https://github.com/sgreenbury/HyperTraPS into `refs/HyperTraPS/`, or set `HT_BIN=/path/to/RUN_MCMC_SAMPLER`. If it is not available, the HyperTraPS column is skipped.

## Tasks

| Config | Species | Task | Features | Transitions | HyperTraPS |
| --- | --- | --- | ---: | ---: | :---: |
| `tuberculosis_genes19` | *M. tuberculosis* | Genes19 | 19 | 264 | yes |
| `tuberculosis_antibiotics10` | *M. tuberculosis* | Antibiotics10 | 10 | 427 | yes |
| `tuberculosis_combined29` | *M. tuberculosis* | Combined29 | 29 | 485 | attempted (1,800 s limit) |
| `klebsiella_kleborate22` | *K. pneumoniae* | Kleborate22 | 22 | 9,026 | attempted (1,800 s limit) |
| `ecoli_nit` / `ecoli_mec` / `ecoli_tmp` | *E. coli* | NIT / MEC / TMP | 320 / 858 / 1,081 | 1,819 / 2,258 / 4,154 | no |

## Quick start

```bash
# 1. check the shipped datasets (a few seconds; no training)
python scripts/verify_datasets.py

# 2. print the manuscript results (Supplementary Table S1) from the archived runs
python scripts/summarize_benchmark.py --root results/paper_runs

# 3. check the released checkpoints and predict with one of them (seconds; no training)
python scripts/check_checkpoints.py
python scripts/predict.py --model tuberculosis_genes19 --present katG --top-k 5

# 4. smoke test of training (under a minute on CPU)
FOLDS=2 PRETRAIN_EPOCHS=3 FINETUNE_EPOCHS=3 WITH_HT=0 \
  bash scripts/run_benchmark.sh tuberculosis_genes19 outputs/smoke
```

## Pretrained models

`checkpoints/` contains the 20 EvoState models used for the descriptive analyses in the manuscript, so they can be used without retraining. Each directory holds `model.pt` (Stage 1 encoder + Stage 2 event head, feature names and training-data checksum) and `reference_first_step_ranking.csv` (expected predictions from the empty state). `checkpoints/MANIFEST.json` lists file checksums, architectures, training settings and training data.

| Checkpoint | Training data | Used in |
| --- | --- | --- |
| `tuberculosis_genes19`, `tuberculosis_antibiotics10`, `tuberculosis_combined29` | all transitions of the TB task | Figure 3 (Genes19) |
| `klebsiella_kleborate22` | all 9,026 Kleborate22 transitions | Figure 4 |
| `klebsiella_kleborate22_country_<Brazil, China, India, Italy, Thailand, UK, USA>` | per-country reconstructions (`data/processed/klebsiella_kleborate22/subsets/country_*`) | Figure 4, country analysis |
| `klebsiella_kleborate22_lineage_<ST11, ST15, ST147, ST258, ST307, ST512>` | transitions of one sequence type (`subsets/lineage_*`) | Figure 4, lineage analysis |
| `ecoli_nit`, `ecoli_mec`, `ecoli_tmp` | all transitions of the E. coli task | Figure 5 |

```python
import sys; sys.path.insert(0, "src")
from inference import list_models, load_model

m = load_model("klebsiella_kleborate22")            # CPU by default; load_model(..., device="cuda")
m.feature_names                                     # feature order of the state vector
m.rank(present=["AGly_acquired"], top_k=5)          # list of {rank, feature, feature_index, probability}
p = m.predict_proba(state="1000000000000000000000") # probability per feature (0 for ineligible ones)
s = m.encode(["AGly_acquired"])                     # 0/1 state vector; states may also be 0/1 strings or arrays
```

Predictions are a softmax over the features that are absent from the given state. For the E. coli models, genetic-background markers (`background__*`) are part of the state but are not scored, because they are never acquired; pass `include_non_candidates=True` (`--include-non-candidates` on the command line) for the unrestricted distribution used by the cross-validation metrics.

These models were fitted on every transition of their dataset. Use them for description and exploration, not to report held-out performance; the cross-validation fold models were not kept and are re-created by `scripts/run_benchmark.sh`. Checkpoints are loaded with `torch.load(weights_only=True)`. `checkpoints/MANIFEST.json` records their provenance.

## Reproducing the cross-validation benchmark

```bash
bash scripts/run_benchmark.sh tuberculosis_genes19            # -> outputs/benchmark/tuberculosis_genes19/
bash scripts/run_all_benchmarks.sh                            # all seven tasks
python scripts/summarize_benchmark.py --root outputs/benchmark --compare results/paper_runs
```

Each fold uses only that fold's training partition for Stage-1 pretraining, Stage-2 fine-tuning, the baselines and HyperTraPS. Folds are formed over lineage, sequence-type or genetic-background groups (`group_id`), so no group appears in both training and test data. Metrics are calculated on test transitions with exactly one acquisition and averaged over folds without weighting.

| Output method key | Manuscript name |
| --- | --- |
| `B0_uniform` | Uniform |
| `B1_prevalence` | Prevalence |
| `TS_random_init` | EvoState w/o pretraining |
| `TS_pretrained` | EvoState |
| `HT_second-order` | HyperTraPS |

Any configuration value can be overridden from the environment, for example `WITH_HT=0`, `FOLDS=5`, `DUMP_PREDICTIONS=1` (per-fold out-of-fold distributions) or `PYTHON_BIN=...`.

On a CPU, Genes19 takes about 8 minutes without HyperTraPS. With HyperTraPS, each fold is limited to 1,800 s.

**Expected variation.** Re-runs on the same machine are bit-identical. Across hardware and PyTorch builds, the neural results can differ by a few hundredths in Top-k accuracy on the small TB tasks, because early stopping can select a different epoch. The baselines reproduce the manuscript exactly.

## Training on a complete dataset

The descriptive analyses (forward simulation, dependency estimates, E. coli prioritisation and partial-state completion) use models fitted to all transitions of a task. The fitted models are released in `checkpoints/`; to refit one:

```bash
bash scripts/train_full.sh klebsiella_kleborate22    # -> outputs/full_fit/klebsiella_kleborate22/{stage1,stage2}/
```

These models have seen every transition, so do not use them to report held-out performance.

The lower-level CLIs can also be called directly; see `--help`:

```bash
python src/models/train_state_pretrain.py --help
python src/models/train_transition_two_stage.py --help
python src/eval/run_two_stage_eval.py --help
```

## Data

`data/processed/` contains the model inputs for all seven tasks. They are derived from published *M. tuberculosis* (HyperTraPS benchmark; Casali et al. 2014), *K. pneumoniae* (Lam et al. 2021; Aga et al. 2026) and *E. coli* (Petrungaro et al. 2026) data. [data/README.md](data/README.md) describes the file formats and original sources, and explains how to rebuild the data with `preprocessing/run_preprocessing.sh`. Running this script on the original raw files reproduces every shipped `transitions.csv` byte for byte.

## Publication figures

The notebooks in `notebooks/manuscript/` draw the manuscript figures. They keep their saved outputs, so the figures and printed numbers can be inspected without running anything. Figure 1 is a schematic and has no notebook.

| Notebook | Figures | Output files (`results/rendered/manuscript/<notebook>/`) |
| --- | --- | --- |
| `figure2.ipynb` | Figure 2; Supplementary Figure S1 | `figure2.pdf`, `supplementary_fig_compute_cost.pdf` |
| `figure3.ipynb` | Figure 3 | `figure3.pdf` |
| `figure4.ipynb` | Figure 4; Supplementary Figures S2–S5 | `figure4_main_A-F.pdf`, `supplement_S1_…` to `supplement_S4_…` (= S2–S5) |
| `figure5.ipynb` | Figure 5; Supplementary Figure S6 | `figure5.pdf`, `figure6_supplement_reveal_metrics.pdf` (= S6) |

Each notebook also writes its source-data tables next to the figures. All inputs are in the repository: `data/manuscript/` (evaluation results, source data, the Figure 4 forward-walk cache, the Figure 5 paired-rank cache and the Natural Earth country shapefile), `notebooks/manuscript/lib/` (plotting and analysis modules) and `checkpoints/` (Figures 4 and 5 check the stage-2 checkpoint hashes; the weights are loaded only if the Figure 4 walk cache has to be recomputed).

Figure PDFs, PNGs and source-data tables are saved under `results/rendered/manuscript/figure2/` through `figure5/`. There is no separate `figures/` directory. `results/rendered/` is currently git-ignored.

`requirements.txt` includes the notebook and plotting dependencies (Matplotlib, SciPy, Cartopy, IPython, ipykernel, nbformat, nbconvert and JupyterLab); `environment.yml` installs the same requirements. To update an existing environment and run a notebook:

```bash
python -m pip install -r requirements.txt
cd notebooks/manuscript
jupyter nbconvert --to notebook --execute figure3.ipynb --ExecutePreprocessor.kernel_name=python3 --output /tmp/figure3_run.ipynb
```

Use the kernel of the installed environment when opening the notebooks interactively; the saved kernel names may refer to the original author's environment. The dependency file targets Python 3.10 and Matplotlib 3.10.9, not the historical Python 3.12 rendering environment.

The notebooks find the repository root from the working directory, so start Jupyter anywhere inside the repository. Figures 2, 3 and 5 were rendered with Python 3.12 and matplotlib 3.9.2; Figure 4 with Python 3.10, matplotlib 3.10.9, scipy 1.15.3, cartopy 0.25.0 and PyTorch 2.4.0. Other matplotlib versions can shift text and line positions slightly. See [notebooks/manuscript/README.md](notebooks/manuscript/README.md) for details.

## Citation

```
@article{yang2026evostate,
  title={EvoState learns resistance-state representations for predicting antimicrobial resistance evolution},
  author={Ziyi Yang, Jie Ren, Wei Liu, Man Tat Alexander Ng},
  journal={},
  year={2026}
}
```

## License

Evostate were created and maintained by the Tencent AI for Life Sciences Lab.

© Tencent AI for Life Sciences Lab. All rights reserved.

Unauthorized reproduction, redistribution, or commercial use is prohibited without prior permission.
