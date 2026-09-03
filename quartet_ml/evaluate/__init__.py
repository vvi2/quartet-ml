from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


def accuracy(y_true, y_pred):
  score = accuracy_score(y_true, y_pred)
  # need {"Accuracy": score, "Chance-Level Accuracy": 1/3} one layer up in the caller file
  return score

def confusion_matrix_3x3(y_true, y_pred, labels=["1", "2", "3"]):
  return confusion_matrix(y_true, y_pred, labels=labels)

def bootstrap_ci(y_true, y_pred, metric_fn=accuracy, n_boot=2000, ci=0.95, seed=42):
  boot_scores = []
  y_true_arr = np.asarray(y_true)
  y_pred_arr = np.asarray(y_pred)
  RNG = np.random.default_rng(seed)
  for _ in range(n_boot):
    idx = RNG.integers(0, len(y_true), len(y_true))
    boot_scores.append(metric_fn(y_true_arr[idx], y_pred_arr[idx]))

  lower_bound = (1.0-ci) / 2.0 * 100.0
  upper_bound = (1.0+ci) / 2.0 * 100.0
  lo, hi = np.percentile(boot_scores, [lower_bound, upper_bound])
  return lo, hi

def calibration(y_true, y_proba, n_bins=20):
  y_true = np.asarray(y_true)
  y_proba = np.asarray(y_proba)
  n_bins = min(n_bins, len(y_true))
  # confidence is the models own belief in its predicted topology, not whether it was right
  confidences = np.max(y_proba, axis=1)
  predicted_labels = np.argmax(y_proba, axis=1) + 1
  predicted_labels_str = predicted_labels.astype(str)
  correct = (predicted_labels_str == y_true).astype(int)
  # what the perfectly confident, correct model would have output for this alignment's true class
  ideal_vectors = (np.eye(3))[y_true.astype(int) - 1]
  # brier score
  brier_score = np.mean(np.sum((y_proba - ideal_vectors) ** 2, axis=1))
  #build reliability_table
  sort_order = np.argsort(confidences)
  bin_confidences = np.array_split(confidences[sort_order], n_bins)
  bin_correct = np.array_split(correct[sort_order], n_bins)
  reliability_table = []
  #well calilbrated bin has mean_conf close to observed accuracy; overconfident if mean_conf is consistently higher
  for conf, corr in zip(bin_confidences, bin_correct):
    mean_confidence = np.mean(conf)
    observed_accuracy = np.mean(corr)
    n_in_bin = len(conf)
    reliability_table.append((mean_confidence, observed_accuracy, n_in_bin))

  return {"brier_score": brier_score, "reliability_table": reliability_table}

