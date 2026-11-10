# Data

## Processed datasets (`data/processed/`)

These are the model inputs for the seven prediction tasks in the manuscript. The files are
byte-identical to the inputs of the archived manuscript runs, and checksums are listed in
`processed/MANIFEST.json`. Run `python scripts/verify_datasets.py` to check them.

| Directory | Species | Task | Features | Groups (CV unit) | Transitions | Evidence |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `tuberculosis_genes19` | *M. tuberculosis* | Genes19 | 19 | 44 lineage clades | 264 | phylogenetic ancestral reconstruction |
| `tuberculosis_antibiotics10` | *M. tuberculosis* | Antibiotics10 | 10 | 58 lineage clades | 427 | phylogenetic ancestral reconstruction |
| `tuberculosis_combined29` | *M. tuberculosis* | Combined29 | 29 | 58 lineage clades | 485 | phylogenetic ancestral reconstruction |
| `klebsiella_kleborate22` | *K. pneumoniae* | Kleborate22 | 22 | 980 sequence types | 9,026 | ancestral reconstruction on Mash-NJ tree |
| `ecoli_nit` | *E. coli* | NIT | 320 | 36 genetic backgrounds | 1,819 | endpoint-derived single-gain sequences |
| `ecoli_mec` | *E. coli* | MEC | 858 | 33 genetic backgrounds | 2,258 | endpoint-derived single-gain sequences |
| `ecoli_tmp` | *E. coli* | TMP | 1,081 | 38 genetic backgrounds | 4,154 | endpoint-derived single-gain sequences |

"Groups" counts the groups that contain at least one acquisition event. Cross-validation folds
are formed over these groups, so no group appears in both training and test data.

### Files

| File | Content |
| --- | --- |
| `transitions/transitions.csv` | One row per transition, and the only file read by training and evaluation. Columns: `src_state` and `dest_state` (binary strings, one character per feature, in the row order of `feature_hierarchy.csv`, or the column order of `category_barcode.csv` for Kleborate22), `n_gained` and `gained_features` (`;`-separated feature names), `n_lost` and `lost_features`, `group_id` (CV group), and `transition_strategy`. |
| `transitions/transitions_summary.json` | Statistics written by `preprocessing/build_transitions.py`. |
| `feature_hierarchy.csv` | Feature names and annotations (layer, drug class, mechanism and related fields). E. coli files mark genetic-background features with `layer=background`. |
| `samples.csv` | Sample (isolate or E. coli population) identifiers with `group_id`. |
| `category_barcode.csv` | Binary sample-by-feature matrix (TB, Kp). |
| `tree_pruned.nwk` | Phylogeny pruned to the analysed isolates (TB). |
| `prep_report.json` / `manifest.json` | Preprocessing reports. Absolute local paths are replaced by `<evoamr>/` or `<project>/`. |
| `transition_timings.csv` | Population, step and gained feature for each E. coli transition, with phenotype time anchors (`src_time_h`, `dest_time_h`). |

### *K. pneumoniae* subsets (`klebsiella_kleborate22/subsets/`)

These are the training data of the country- and lineage-specific checkpoints in `checkpoints/` (Figure 4). They use the same
22 features and file format as `transitions/transitions.csv` and are not used by the cross-validation benchmark.

| Directory | Construction | Transitions |
| --- | --- | ---: |
| `country_<Brazil, China, India, Italy, Thailand, UK, USA>` | Ancestral states reconstructed independently on the tree pruned to the isolates of one country (`preprocessing/stratify_transitions_tree.py`, countries with at least 100 isolates and 30 transitions; `group_id` = ST). | 244 to 1,574 |
| `lineage_<ST11, ST15, ST147, ST258, ST307, ST512>` | Rows of the global `transitions.csv` whose `group_id` is that sequence type (`preprocessing/klebsiella/extract_lineage_subsets.py`). | 290 to 1,247 |

`bash preprocessing/run_preprocessing.sh klebsiella` rebuilds all 13 subsets, and they match the shipped files byte for byte.
Their checksums are listed in `processed/MANIFEST.json`.

## Original sources

The processed datasets are derived from previously published data. Cite the original studies
when you use them.

| Task(s) | Original data | Raw files used by `preprocessing/` |
| --- | --- | --- |
| Genes19, Antibiotics10, Combined29 | *M. tuberculosis* benchmark distributed with HyperTraPS (Greenbury, Barahona & Johnston, *Cell Systems* 2020; https://github.com/sgreenbury/HyperTraPS, `datasets/tuberculosis/`). Isolates and phylogeny come from Casali et al., *Nat. Genet.* 2014 (doi:10.1038/ng.2878). | `tuberculosis-v5-header-29-limit-19.csv`, `tuberculosis-v5-header-19-29.csv`, `tuberculosis-v5-header-29.csv`, `ng.2878-S2.txt` |
| Kleborate22 | Global *Klebsiella* genome collection genotyped with Kleborate (Lam et al., *Nat. Commun.* 2021), metadata table and tree from the accompanying Microreact project, as analysed by Aga et al., *PLoS Biol.* 2026. | `2021-Klebsiella.csv`, `2021-Klebsiella_tree.nwk` |
| NIT, MEC, TMP | High-throughput *E. coli* evolution experiment (Petrungaro et al., "Function-specific epistasis shapes evolutionary trajectories towards antibiotic resistance", *Nat. Commun.* 2026): source-data workbooks and plate data. | `SourceData_MainFigures.xlsx`, `SourceData_SupplementaryFigures.xlsx`, `fsepi_data/` |

Raw files are not redistributed here. To rebuild the processed datasets, place the raw files as follows and run
`bash preprocessing/run_preprocessing.sh {tuberculosis|klebsiella|ecoli|all}`:

```
data/raw/tuberculosis/   the four TB files listed above
data/raw/klebsiella/     2021-Klebsiella.csv, 2021-Klebsiella_tree.nwk
data/ecoli2026/          SourceData_*.xlsx and fsepi_data/   (fixed location used by preprocessing/ecoli/)
```

> Before release: confirm the redistribution terms of each upstream source for the derived files in
> `data/processed/`, and add the exact Microreact project URL and E. coli data-repository accession.
