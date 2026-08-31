"""Smoke test: confirms the package installs and imports cleanly."""

import quartet_ml
from quartet_ml.simulate import build_newick, create_trees, easy, felsenstein_zone, outbreak_like, possible_topologies


def test_import() -> None:
    assert quartet_ml is not None

def test_possible_topologies():
    result = possible_topologies()
    assert len(result) == 3, f"Expected 3 topologies, but got {len(result)}"

    actual_topologies = set()
    for topo in result.values():
        pair1, pair2 = topo
        norm_topo = frozenset([frozenset(pair1), frozenset(pair2)])
        actual_topologies.add(norm_topo)
    expected_topologies = {
        frozenset([frozenset({"A", "B"}), frozenset({"C", "D"})]),
        frozenset([frozenset({"A", "C"}), frozenset({"B", "D"})]),
        frozenset([frozenset({"A", "D"}), frozenset({"B", "C"})])
    }
    assert actual_topologies == expected_topologies, "The topologies do not cover all 3 unique pairings"

def test_build_newick():
    pair1 = ("A", "B")
    pair2 = ("C", "D")
    ext_lengths = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}
    internal_length = 0.5

    expected = "((A:0.1,B:0.2):0.5,(C:0.3,D:0.4));"
    result = build_newick(pair1, pair2, ext_lengths, internal_length)

    assert expected == result, "The Newick string is not correct"

def test_easy():
    pair1 = ("A", "B")
    pair2 = ("C", "D")
    int_length = 0.5

    expected = "((A:0.1,B:0.1):0.5,(C:0.1,D:0.1));"
    result = easy(pair1, pair2, int_length)

    assert expected == result, "easy() did not return expected Newick tree"

def test_felsenstein():
    pair1 = ("A", "B")
    pair2 = ("C", "D")
    int_length = 0.5

    expected = "((A:1.0,B:0.02):0.5,(C:1.0,D:0.02));"
    result = felsenstein_zone(pair1, pair2, int_length)

    assert expected == result, "felsenstein_zone() did not return expected Newick tree"

def test_outbreak():
    pair1 = ("A", "B")
    pair2 = ("C", "D")
    int_length = 0.5

    expected = "((A:0.01,B:0.01):0.5,(C:0.01,D:0.01));"
    result = outbreak_like(pair1, pair2, int_length)

    assert expected == result, "outbreak_like() did not return expected Newick tree"

def test_create_trees():
    trees = create_trees()
    assert len(trees)==300, f"Expected 300 entries, but got {len(trees)}"
    seen_topologies = set()
    expected_regimes = {"easy", "felsenstein", "outbreak"}

    for idx, item in enumerate(trees):
        topo = item.get("topology")
        assert topo in {"1", "2", "3"}, f"Invalid topology label {topo} at index {idx}"
        seen_topologies.add(topo)
        assert "length" in item, f"Missing 'length' key at index {idx}"
        regimes = item.get("regimes")
        assert isinstance(regimes, dict), f"'regimes' must be a dict at index {idx}"
        assert expected_regimes.issubset(regimes.keys()), (
            f"Missing regime keys at index {idx}. Found: {list(regimes.keys())}"
        )
    assert seen_topologies == {"1", "2", "3"}, f"Not all topologies {1, 2, 3} were present. Found: {seen_topologies}"
