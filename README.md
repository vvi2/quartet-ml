# quartet-ml

**Does a CNN beat logistic regression on site-pattern frequencies at inferring
four-taxon phylogenies, and does that answer change under model
misspecification or near-zero internal branch lengths?**

Short answer: TBD — this is a work in progress. See `report/` once written.

[Headline figure goes here once generated]

## What this is

Four inference strategies — CNN, site-pattern logistic regression, a hybrid
of the two, maximum likelihood (IQ-TREE), and neighbour joining — compared on
the smallest phylogenetic problem that exists: four taxa, three possible
unrooted topologies. Evaluated across three evolutionary regimes (easy,
Felsenstein zone, outbreak-like) and under substitution-model
misspecification (train JC69, test GTR+Γ).

## Package layout

```
quartet_ml/
  simulate/    tree generation, AliSim wrappers, regime definitions
  features/    site-pattern extraction (256-dim and 15-dim collapsed)
  train/       CNN, hybrid, logistic regression
  baselines/   IQ-TREE and ape::nj() wrappers (R)
  evaluate/    accuracy, confusion matrices, calibration, bootstrap CIs
report/        report.qmd (4-6 page writeup)
tests/         pytest suite
```

## Setup

Requires Python >= 3.10, R with the `ape` package, and IQ-TREE (bundles
AliSim; ships as the `iqtree3` binary).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

R dependency:

```r
install.packages("ape")
```

IQ-TREE, via conda/bioconda:

```bash
conda create -n quartet-ml -c bioconda -c conda-forge iqtree -y
conda activate quartet-ml
```

**Note:** the Homebrew `iqtree3` bottle (both Intel and native arm64 builds)
hangs indefinitely on AliSim on macOS 14.3.1/arm64 as of 2026-08 — the
process enters an unkillable uninterruptible-wait state. The bioconda build
(3.1.3) does not have this problem and is what's actually used here. Point
any `iqtree3` calls in `quartet_ml/simulate/` and `quartet_ml/baselines/` at
the conda env's binary, e.g. `~/miniconda3/envs/quartet-ml/bin/iqtree3`, or
`conda activate quartet-ml` before running.

## Usage

```bash
quartet-ml --help
```

(Subcommands land as each stage is implemented.)

## Tests

```bash
pytest
```
