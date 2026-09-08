"""Smoke test: confirms the package installs and imports cleanly."""

import numpy as np
import torch

import quartet_ml
from quartet_ml.baselines import nj_prediction, iq_tree
from quartet_ml.features import build_site_pattern_vector, cnn_matrix, jc69_collapsed_vector, read_phylip_data
from quartet_ml.simulate import build_newick, create_trees, easy, felsenstein_zone, outbreak_like, possible_topologies
from quartet_ml.train import CNN, HybridCNN, get_x_y_split, train_site_pattern_classifier_LR
from quartet_ml.evaluate import accuracy, confusion_matrix_3x3, bootstrap_ci, calibration


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

def test_build_site_pattern_vector():
    taxa_dict_ex = read_phylip_data("sim_outputs_jc/sim_1_0.081_easy_jc_0.phy")
    vector_256 = build_site_pattern_vector(taxa_dict_ex)
    assert vector_256.size == 256, f"Expected 256 entries in feature vector, but got {vector_256.size}"
    assert vector_256.sum() == 1000, f"Expected 1000 site patterns in the alignment, but got {vector_256.sum()}"

def test_jc69_collapsed_vector():
    taxa_dict_ex = read_phylip_data("sim_outputs_jc/sim_1_0.081_easy_jc_0.phy")
    vector_256 = build_site_pattern_vector(taxa_dict_ex)
    collapsed_vector = jc69_collapsed_vector(vector_256)
    assert sum(collapsed_vector) == vector_256.sum(), f"Expected 256-number vector sum to equal 15-number collapsed version sum (i.e. 1000), but collapsed vector sum is {sum(collapsed_vector)}"
    assert len(collapsed_vector) == 15, f"Expected collapsed vector to have 15 entries, but it has {len(collapsed_vector)}"

def test_cnn_matrix():
    taxa_dict_ex = read_phylip_data("sim_outputs_jc/sim_1_0.081_easy_jc_0.phy")
    matrix = cnn_matrix(taxa_dict_ex)
    assert matrix.shape == (4, 4, 1000), f"Expected shape of CNN matrix to be (4, 4, 1000) but instead got {matrix.shape}"
    assert (np.sum(matrix, axis=1) == 1).all(), f"Expected only one base present at each alignment position, but instead got {np.sum(matrix, axis=1)}"

def test_iq_tree():
    topology_1 = iq_tree("sim_outputs_jc/sim_1_0.081_easy_jc_0.phy", "sim_1_0.081_easy_jc_0")
    topology_2 = iq_tree("sim_outputs_jc/sim_2_0.081_easy_jc_0.phy", "sim_2_0.081_easy_jc_0")
    topology_3 = iq_tree("sim_outputs_jc/sim_3_0.081_easy_jc_0.phy", "sim_3_0.081_easy_jc_0")
    assert topology_1 == "1", f"Expected iqtree to predict 1 on this file, but instead got {topology_1}"
    assert topology_2 == "2", f"Expected iqtree to predict 2 on this file, but instead got {topology_2}"
    assert topology_3 == "3", f"Expected iqtree to predict 3 on this file, but instead got {topology_3}"

def test_neighbor_joining():
    topology_1 = nj_prediction("sim_outputs_jc/sim_1_0.081_easy_jc_0.phy")
    topology_2 = nj_prediction("sim_outputs_jc/sim_2_0.081_easy_jc_0.phy")
    topology_3 = nj_prediction("sim_outputs_jc/sim_3_0.081_easy_jc_0.phy")
    assert topology_1 == "1", f"Expected NJ to predict 1 on this file, but instead got {topology_1}"
    assert topology_2 == "2", f"Expected NJ to predict 2 on this file, but instead got {topology_2}"
    assert topology_3 == "3", f"Expected NJ to predict 3 on this file, but instead got {topology_3}"

def test_site_pattern_classifier_LR():
    models = train_site_pattern_classifier_LR("tests/fixtures/site_pattern_train_fixture.csv")
    x_train_256 = models["x_train_256"]
    x_train_15 = models["x_train_15"]
    y_train = models["y_train"]
    assert len(x_train_256) == len(y_train), f"x_train_256 does not match size of y_train"
    assert len(x_train_256[0]) == 256, f"x_train_256 vectors should have 256 values, but instead has {len(x_train_256[0])}"
    assert len(x_train_15[0]) == 15, f"x_train_15 vector should have 15 values, but instead has {len(x_train_15)}"
    assert all(topology in ["1", "2", "3"] for topology in y_train), f"Expected y_train to have only either 1, 2, 3, but that's not the case"
    assert "256" in models and "15" in models, f"Expected returned model dict to have keys 256 and 15."
    assert set(models["256"].named_steps["clf"].classes_) == {"1", "2", "3"}, f"Expected labels the 256 lr model learned to be 1, 2, 3 but got {set(models['256'].named_steps['clf'].classes_)}"
    assert set(models["15"].named_steps["clf"].classes_) == {"1", "2", "3"}, f"Expected labels the 15 lr model learned to be 1, 2, 3 but got {set(models['15'].named_steps['clf'].classes_)}"
    probs_256 = models["256"].predict_proba(x_train_256[:3])
    probs_15 = models["15"].predict_proba(x_train_15[:3])
    assert probs_256.shape == (3, 3), f"Shape of 256 model probabilities should be (3, 3), but is {probs_256.shape}"
    assert probs_15.shape == (3, 3), f"Shape of 15 model probabilities should be (3, 3), but is {probs_15.shape}"
    assert np.allclose(probs_256.sum(axis=1), 1), f"Probabilities of 256 model do not sum to 1"
    assert np.allclose(probs_15.sum(axis=1), 1), f"Probabilities of 15 model do not sum to 1"

def test_cnn_output_shape():
    model = CNN(in_channels=16, out_channels=8, kernel_size=3)
    fake_input = torch.randn(2, 16, 1000)
    logits = model(fake_input)
    probs = torch.softmax(logits, dim=1)
    assert probs.shape == (2, 3), f"Expected CNN output shape (2, 3), but got {tuple(probs.shape)}"
    assert torch.allclose(probs.sum(dim=1), torch.ones(2)), f"Expected CNN output probabilities to sum to 1 per row, but got {probs.sum(dim=1)}"

def test_hybrid_cnn_concatenation():
    model = HybridCNN(in_channels=16, out_channels=8, kernel_size=3)
    expected_in_features = model.conv2.out_channels + 15
    assert model.fc.in_features == expected_in_features, f"Expected hybrid fc input size to equal pooled-CNN-size + 15 ({expected_in_features}), but got {model.fc.in_features}"

def test_hybrid_cnn_output_shape():
    model = HybridCNN(in_channels=16, out_channels=8, kernel_size=3)
    fake_cnn_input = torch.randn(2, 16, 1000)
    fake_pattern_input = torch.randn(2, 15)
    logits = model(fake_cnn_input, fake_pattern_input)
    probs = torch.softmax(logits, dim=1)
    assert probs.shape == (2, 3), f"Expected hybrid CNN output shape (2, 3), but got {tuple(probs.shape)}"
    assert torch.allclose(probs.sum(dim=1), torch.ones(2)), f"Expected hybrid CNN output probabilities to sum to 1 per row, but got {probs.sum(dim=1)}"

def test_accuracy():
    y_true = ["1", "2", "3", "1", "2", "3"]
    y_pred = ["1", "2", "3", "1", "3", "3"]
    score = accuracy(y_true, y_pred)
    assert score == 5/6, f"Expected accuracy of 5/6, but got {score}"

def test_confusion_matrix_3x3():
    y_true = ["1", "2", "3", "1", "2", "3"]
    y_pred = ["1", "2", "3", "1", "3", "3"]
    expected = np.array([
        [2, 0, 0],
        [0, 1, 1],
        [0, 0, 2],
    ])
    matrix = confusion_matrix_3x3(y_true, y_pred)
    assert np.array_equal(matrix, expected), f"Expected confusion matrix {expected.tolist()}, but got {matrix.tolist()}"

def test_bootstrap_ci():
    y_true = ["1", "2", "3", "1", "2", "3"]
    y_pred = ["1", "2", "3", "1", "3", "3"]
    lo, hi = bootstrap_ci(y_true, y_pred, n_boot=2000, seed=42)
    assert 0 <= lo <= hi <= 1, f"Expected 0 <= lo <= hi <= 1, but got lo={lo}, hi={hi}"
    point_estimate = accuracy(y_true, y_pred)
    assert lo <= point_estimate <= hi, f"Expected point estimate {point_estimate} to fall within [{lo}, {hi}]"

def test_calibration():
    y_true = ["2", "1", "3", "1", "2", "3"]
    y_proba = np.array([
        [0.05, 0.90, 0.05],
        [0.60, 0.20, 0.20],
        [0.10, 0.10, 0.80],
        [0.34, 0.33, 0.33],
        [0.40, 0.55, 0.05],
        [0.20, 0.30, 0.50],
    ])
    result = calibration(y_true, y_proba, n_bins=2)
    assert np.isclose(result["brier_score"], 0.2855666666666667), f"Expected brier_score ~0.2856, but got {result['brier_score']}"
    reliability_table = result["reliability_table"]
    assert len(reliability_table) == 2, f"Expected 2 bins, but got {len(reliability_table)}"
    for mean_confidence, observed_accuracy, n_in_bin in reliability_table:
        assert 0 <= mean_confidence <= 1, f"Expected mean_confidence in [0, 1], but got {mean_confidence}"
        assert 0 <= observed_accuracy <= 1, f"Expected observed_accuracy in [0, 1], but got {observed_accuracy}"
        assert n_in_bin == 3, f"Expected 3 alignments per bin, but got {n_in_bin}"


