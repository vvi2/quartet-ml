"""Command-line entry point for quartet-ml.

Subcommands are added as the corresponding module is implemented
(simulate, featurize, train, evaluate).
"""

import click
import pandas as pd

from quartet_ml.simulate import create_tree_files
from quartet_ml.split import split_data

@click.group()
def main() -> None:
    """quartet-ml: CNN vs. site-pattern classifier on four-taxon phylogenetics."""

@main.command()
def simulate() -> None:
    create_tree_files("jc", "JC69", "jc69_simulated_data.csv", 17)
    create_tree_files("gtr", "GTR{1.25,4.30,0.65,0.80,4.90,1.0}+F{0.38/0.17/0.23/0.22}+G4{0.55}", "gtr_simulated_data.csv", 2)

@main.command()
def split() -> None:
    df = pd.read_csv("jc69_simulated_data.csv")
    train_rows, val_rows, test_rows = split_data(df)

    train_rows["split"] = "train"
    val_rows["split"] = "val"
    test_rows["split"] = "test"

    split_df = pd.concat([train_rows, val_rows, test_rows])
    split_df.to_csv("jc69_split_data.csv", index=False)
    
if __name__ == "__main__":
    main()
