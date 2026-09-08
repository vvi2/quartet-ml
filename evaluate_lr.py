from quartet_ml.evaluate import accuracy, bootstrap_ci, calibration, confusion_matrix_3x3
from quartet_ml.train import get_x_y_split, get_x_y_split_gtr, train_site_pattern_classifier_LR


def evaluate_lr():
  test_x_256, test_x_15, test_y = get_x_y_split(split="test")
  test_x_256_gtr, test_x_15_gtr, test_y_gtr = get_x_y_split_gtr()
  models = train_site_pattern_classifier_LR()
  y_pred_256 = models["256"].predict(test_x_256)
  y_proba_256 = models["256"].predict_proba(test_x_256)
  y_pred_15 = models["15"].predict(test_x_15)
  y_proba_15 = models["15"].predict_proba(test_x_15)
  y_pred_256_gtr = models["256"].predict(test_x_256_gtr)
  y_proba_256_gtr = models["256"].predict_proba(test_x_256_gtr)
  y_pred_15_gtr = models["15"].predict(test_x_15_gtr)
  y_proba_15_gtr = models["15"].predict_proba(test_x_15_gtr)
  acc_score_256 = {"Accuracy": accuracy(test_y, y_pred_256), "Chance-Level Accuracy": 1/3}
  acc_score_15 = {"Accuracy": accuracy(test_y, y_pred_15), "Chance-Level Accuracy": 1/3}
  acc_score_256_gtr = {"Accuracy": accuracy(test_y_gtr, y_pred_256_gtr), "Chance-Level Accuracy": 1/3}
  acc_score_15_gtr = {"Accuracy": accuracy(test_y_gtr, y_pred_15_gtr), "Chance-Level Accuracy": 1/3}
  conf_mat_256 = confusion_matrix_3x3(test_y, y_pred_256)
  conf_mat_15 = confusion_matrix_3x3(test_y, y_pred_15)
  conf_mat_256_gtr = confusion_matrix_3x3(test_y_gtr, y_pred_256_gtr)
  conf_mat_15_gtr = confusion_matrix_3x3(test_y_gtr, y_pred_15_gtr)
  ci_low_256, ci_high_256 = bootstrap_ci(test_y, y_pred_256)
  ci_low_15, ci_high_15 = bootstrap_ci(test_y, y_pred_15)
  ci_low_256_gtr, ci_high_256_gtr = bootstrap_ci(test_y_gtr, y_pred_256_gtr)
  ci_low_15_gtr, ci_high_15_gtr = bootstrap_ci(test_y_gtr, y_pred_15_gtr)
  calibration_dict_256 = calibration(test_y, y_proba_256)
  calibration_dict_15 = calibration(test_y, y_proba_15)
  calibration_dict_256_gtr = calibration(test_y_gtr, y_proba_256_gtr)
  calibration_dict_15_gtr = calibration(test_y_gtr, y_proba_15_gtr)
  return {"accuracy_256": acc_score_256, "accuracy_15": acc_score_15, "accuracy_256_gtr": acc_score_256_gtr, "accuracy_15_gtr": acc_score_15_gtr, "accuracy_256_diff": acc_score_256["Accuracy"] - acc_score_256_gtr["Accuracy"], "accuracy_15_diff": acc_score_15["Accuracy"] - acc_score_15_gtr["Accuracy"], "confusion_matrix_256": conf_mat_256, "confusion_matrix_15": conf_mat_15, "confusion_matrix_256_gtr": conf_mat_256_gtr, "confusion_matrix_15_gtr": conf_mat_15_gtr,"confidence_interval_256": f"[{ci_low_256}, {ci_high_256}]", "confidence_interval_15": f"[{ci_low_15}, {ci_high_15}]", "confidence_interval_256_gtr": f"[{ci_low_256_gtr}, {ci_high_256_gtr}]", "confidence_interval_15_gtr": f"[{ci_low_15_gtr}, {ci_high_15_gtr}]", "calibration_256": calibration_dict_256, "calibration_15": calibration_dict_15, "calibration_256_gtr": calibration_dict_256_gtr, "calibration_15_gtr": calibration_dict_15_gtr}
  
if __name__ == "__main__":
  evaluation = evaluate_lr()
  print(evaluation)