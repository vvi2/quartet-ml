from pathlib import Path
import time

import pandas as pd

from quartet_ml.baselines import nj_prediction
from quartet_ml.evaluate import accuracy, bootstrap_ci, confusion_matrix_3x3

def evaluate_nj():
  data_file_path = Path(__file__).resolve().parent / "jc69_split_data.csv"
  gtr_data_file_path = Path(__file__).resolve().parent / "gtr_simulated_data.csv"
  df = pd.read_csv(data_file_path)
  df_gtr = pd.read_csv(gtr_data_file_path)
  test_df = df[df["split"] == "test"]
  y_true = []
  y_pred = []
  regimes = []
  internal_branch_lengths = []
  failure_counter = 0
  start = time.time()
  for i, (_, row) in enumerate(test_df.iterrows()):
    true_label = str(row["topology"])
    pred_label = nj_prediction(row["filepath"])
    regime = row["regime"]
    branch_length = row["internal_branch_length"]
    if pred_label is None:
      failure_counter +=1
      continue
    else:
      y_true.append(true_label)
      y_pred.append(pred_label)
      regimes.append(regime)
      internal_branch_lengths.append(branch_length)
    if i % 100 == 0:
      print(i)
  print(f"JC69 Data: {time.time() - start:.1f}s for {len(y_true)} alignments")
  correct = [int(x==y) for x,y in zip(y_true, y_pred)]
  headline_file_path = Path("headline_data/")
  headline_file_path.mkdir(parents = True, exist_ok = True)
  pd.DataFrame({"method": "NJ", "regime": regimes, "internal_branch_length": internal_branch_lengths, "correct": correct}).to_csv(headline_file_path / "nj.csv", index=False)
  gtr_y_true = []
  gtr_y_pred = []
  gtr_failure_counter = 0
  gtr_start = time.time()
  for gtr_i, (_, gtr_row) in enumerate(df_gtr.iterrows()):
    gtr_true_label = str(gtr_row["topology"])
    gtr_pred_label = nj_prediction(gtr_row["filepath"])
    if gtr_pred_label is None:
      gtr_failure_counter +=1
      continue
    else:
      gtr_y_true.append(gtr_true_label)
      gtr_y_pred.append(gtr_pred_label)
    if gtr_i % 100 == 0:
      print(gtr_i)
  print(f"GTR Data: {time.time() - gtr_start:.1f}s for {len(gtr_y_true)} alignments")
  acc_score = {"Accuracy": accuracy(y_true, y_pred), "Chance-Level Accuracy": 1/3}
  acc_score_gtr = {"Accuracy": accuracy(gtr_y_true, gtr_y_pred), "Chance-Level Accuracy": 1/3}
  conf_mat = confusion_matrix_3x3(y_true, y_pred)
  conf_mat_gtr = confusion_matrix_3x3(gtr_y_true, gtr_y_pred)
  ci_low, ci_high = bootstrap_ci(y_true, y_pred)
  ci_low_gtr, ci_high_gtr = bootstrap_ci(gtr_y_true, gtr_y_pred)
  return {"accuracy": acc_score, "accuracy_gtr": acc_score_gtr, "accuracy_diff": acc_score["Accuracy"] - acc_score_gtr["Accuracy"], "confusion_matrix": conf_mat, "confusion_matrix_gtr": conf_mat_gtr, "confidence_interval": f"[{ci_low}, {ci_high}]", "confidence_interval_gtr": f"[{ci_low_gtr}, {ci_high_gtr}]", "failure_count": failure_counter, "failure_count_gtr": gtr_failure_counter}

if __name__ == "__main__":
  evaluation = evaluate_nj()
  print(evaluation)
  