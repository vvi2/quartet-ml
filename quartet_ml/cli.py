"""Command-line entry point for quartet-ml.

Subcommands are added as the corresponding module is implemented
(simulate, featurize, train, evaluate).
"""

import click


@click.group()
def main() -> None:
    """quartet-ml: CNN vs. site-pattern classifier on four-taxon phylogenetics."""


if __name__ == "__main__":
    main()
