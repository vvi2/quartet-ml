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

Full design and motivation: `docs/project2_quartet_phylogenetics.md` (project
spec).

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

Requires Python >= 3.10, R with the `ape` package, and IQ-TREE 2 (bundles
AliSim) on `PATH`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

R dependency:

```r
install.packages("ape")
```

IQ-TREE 2 (macOS, via Homebrew):

```bash
brew install brewsci/bio/iqtree
```

## Usage

```bash
quartet-ml --help
```

(Subcommands land as each stage is implemented — see the two-week plan in
the project spec.)

## Tests

```bash
pytest
```
