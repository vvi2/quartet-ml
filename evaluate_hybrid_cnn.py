import torch

from quartet_ml.evaluate import accuracy, bootstrap_ci, calibration, confusion_matrix_3x3
from quartet_ml.train import HybridCNN, get_hybrid_cnn_split_data, get_hybrid_cnn_split_data_gtr, prepare_hybrid_cnn_dataloader, train_hybrid_cnn_with_early_stopping

def evaluate_hybrid_cnn():
  torch.manual_seed(42)
  train_x, train_x_pattern, train_y = get_hybrid_cnn_split_data("train")
  val_x, val_x_pattern, val_y = get_hybrid_cnn_split_data("val")
  test_x, test_x_pattern, test_y = get_hybrid_cnn_split_data("test")
  test_x_gtr, test_x_pattern_gtr, test_y_gtr = get_hybrid_cnn_split_data_gtr()
  train_loader = prepare_hybrid_cnn_dataloader(train_x, train_x_pattern, train_y, batch_size=32, shuffle=True)
  val_loader = prepare_hybrid_cnn_dataloader(val_x, val_x_pattern, val_y, batch_size=32, shuffle=False)
  test_loader = prepare_hybrid_cnn_dataloader(test_x, test_x_pattern, test_y, batch_size=32, shuffle=False)
  test_loader_gtr = prepare_hybrid_cnn_dataloader(test_x_gtr, test_x_pattern_gtr, test_y_gtr, batch_size=32, shuffle=False)
  hybrid_cnn_model = HybridCNN(in_channels=16, out_channels=8, kernel_size=3)
  model, _, _ = train_hybrid_cnn_with_early_stopping(hybrid_cnn_model, train_loader, val_loader)
  model.eval()
  with torch.no_grad():
    y_probs = []
    y_preds = []
    y_true = []
    y_probs_gtr = []
    y_preds_gtr = []
    y_true_gtr = []
    for x_cnn_batch, x_pattern_batch, _ in test_loader:
      logits = model(x_cnn_batch, x_pattern_batch) #raw scores, 3 per row
      probs = torch.softmax(logits, dim=1) #probabilities that sum to 1 per row to feed into evaluate calibration, since CNN forward returns raw logits
      preds = torch.argmax(probs, dim=1) #what topology cnn predicted, 0/1/2
      y_probs.extend(probs.tolist())
      y_preds.extend(preds.tolist())
    for x_cnn_batch_gtr, x_pattern_batch_gtr, _ in test_loader_gtr:
      logits_gtr = model(x_cnn_batch_gtr, x_pattern_batch_gtr) #raw scores, 3 per row
      probs_gtr = torch.softmax(logits_gtr, dim=1) #probabilities that sum to 1 per row to feed into evaluate calibration, since CNN forward returns raw logits
      preds_gtr = torch.argmax(probs_gtr, dim=1) #what topology cnn predicted, 0/1/2
      y_probs_gtr.extend(probs_gtr.tolist())
      y_preds_gtr.extend(preds_gtr.tolist())
  y_preds = [str(value + 1) for value in y_preds]
  y_true = [str(value + 1) for value in test_y]
  y_preds_gtr = [str(value + 1) for value in y_preds_gtr]
  y_true_gtr = [str(value + 1) for value in test_y_gtr]
  acc_score = {"Accuracy": accuracy(y_true, y_preds), "Chance-Level Accuracy": 1/3}
  acc_score_gtr = {"Accuracy": accuracy(y_true_gtr, y_preds_gtr), "Chance-Level Accuracy": 1/3}
  conf_mat = confusion_matrix_3x3(y_true, y_preds)
  conf_mat_gtr = confusion_matrix_3x3(y_true_gtr, y_preds_gtr)
  ci_low, ci_high = bootstrap_ci(y_true, y_preds)
  ci_low_gtr, ci_high_gtr = bootstrap_ci(y_true_gtr, y_preds_gtr)
  calibration_dict = calibration(y_true, y_probs)
  calibration_dict_gtr = calibration(y_true_gtr, y_probs_gtr)
  return {"accuracy": acc_score, "accuracy_gtr": acc_score_gtr, "accuracy_diff": acc_score["Accuracy"] - acc_score_gtr["Accuracy"], "confusion_matrix": conf_mat, "confusion_matrix_gtr": conf_mat_gtr, "confidence_interval": f"[{ci_low}, {ci_high}]", "confidence_interval_gtr": f"[{ci_low_gtr}, {ci_high_gtr}]", "calibration": calibration_dict, "calibration_gtr": calibration_dict_gtr}

if __name__ == "__main__":
  evaluation = evaluate_hybrid_cnn()
  print(evaluation)
  