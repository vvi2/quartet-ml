"""Tests for split_data(): stratified train/val/test partitioning of a manifest."""

import pandas as pd
import pytest

from quartet_ml.split import split_data

STRATA_COLUMNS = ["topology", "regime", "internal_branch_length"]


@pytest.fixture
def manifest_df():
    """A small synthetic manifest with enough replicates per stratum to survive
    two nested stratified splits (each of the 12 topology/regime/branch-length
    combinations needs at least ~2 rows in the smallest split, val/test)."""
    n_replicates = 14
    rows = []
    for topology in [1, 2, 3]:
        for regime in ["easy", "outbreak"]:
            for branch_length in [0.01, 0.05]:
                for replicate in range(n_replicates):
                    rows.append(
                        {
                            "filepath": f"sim_{topology}_{regime}_{branch_length}_{replicate}.phy",
                            "topology": topology,
                            "regime": regime,
                            "internal_branch_length": branch_length,
                            "replicate": replicate,
                        }
                    )
    return pd.DataFrame(rows)


def test_split_preserves_all_rows(manifest_df):
    train_rows, val_rows, test_rows = split_data(manifest_df.copy())
    combined = pd.concat([train_rows, val_rows, test_rows])

    assert len(combined) == len(manifest_df), "Row count changed after splitting"
    assert set(combined["filepath"]) == set(manifest_df["filepath"]), (
        "Some rows were dropped or duplicated"
    )
    assert combined["filepath"].is_unique, "A row appears in more than one split"


def test_split_proportions_are_roughly_70_15_15(manifest_df):
    train_rows, val_rows, test_rows = split_data(manifest_df.copy())
    total = len(manifest_df)

    train_frac = len(train_rows) / total
    val_frac = len(val_rows) / total
    test_frac = len(test_rows) / total

    assert train_frac == pytest.approx(0.70, abs=0.05)
    assert val_frac == pytest.approx(0.15, abs=0.05)
    assert test_frac == pytest.approx(0.15, abs=0.05)


def test_every_stratum_present_in_every_split(manifest_df):
    train_rows, val_rows, test_rows = split_data(manifest_df.copy())
    expected_strata = set(
        manifest_df[STRATA_COLUMNS].astype(str).agg("_".join, axis=1)
    )

    for name, split_df in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        actual_strata = set(
            split_df[STRATA_COLUMNS].astype(str).agg("_".join, axis=1)
        )
        missing = expected_strata - actual_strata
        assert not missing, f"{name} split is missing strata: {missing}"


def test_split_is_deterministic(manifest_df):
    train_1, val_1, test_1 = split_data(manifest_df.copy())
    train_2, val_2, test_2 = split_data(manifest_df.copy())

    assert list(train_1["filepath"]) == list(train_2["filepath"])
    assert list(val_1["filepath"]) == list(val_2["filepath"])
    assert list(test_1["filepath"]) == list(test_2["filepath"])
