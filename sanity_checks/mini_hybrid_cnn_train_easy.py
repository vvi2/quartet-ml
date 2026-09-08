from pathlib import Path

from matplotlib import pyplot as plt

from quartet_ml.train import HybridCNN, get_hybrid_cnn_split_data, prepare_hybrid_cnn_dataloader, train_hybrid_cnn_with_early_stopping

def mini_hybrid_cnn_train_easy():
  train_x, train_x_pattern, train_y = get_hybrid_cnn_split_data("train", regime_type="easy")
  val_x, val_x_pattern, val_y = get_hybrid_cnn_split_data("val", regime_type="easy")
  train_loader = prepare_hybrid_cnn_dataloader(train_x, train_x_pattern, train_y, batch_size=32, shuffle=True)
  val_loader = prepare_hybrid_cnn_dataloader(val_x, val_x_pattern, val_y, batch_size=32, shuffle=False)
  cnn_model = HybridCNN(in_channels=16, out_channels=8, kernel_size=3)
  _, model_history, metrics = train_hybrid_cnn_with_early_stopping(cnn_model, train_loader, val_loader)
  fig, axis = plt.subplots(figsize=(9,5))
  axis.plot(model_history.epoch, model_history.train_loss, color="#4C72B0", linewidth=2.0, label="CNN Training Loss")
  axis.plot(model_history.epoch, model_history.val_loss, color="#C44E52", linewidth=2.0, label="CNN Validation Loss")
  axis.axvline(metrics["best_epoch"], color="#55A868", linewidth=2.4, label=f"Best epoch ({metrics['best_epoch']}) - weights restored here")
  axis.axvline(metrics["epochs_run"] - 1, color="black", linestyle="--", linewidth=1.8, label=f"Stopped (epoch {metrics['epochs_run'] - 1}, patience 25)")
  axis.axvspan(metrics["best_epoch"], metrics["epochs_run"] - 1, alpha=0.12, color="#8C8C8C")
  axis.set_xlabel("Epoch")
  axis.set_ylabel("3 Class Cross-Entropy Loss")
  axis.set_title("CNN w Early Stopping: train CNN past the minimum, then go back to this state")
  axis.legend(fontsize=10)
  fig.tight_layout()
  fig.savefig(Path(__file__).resolve().parent / "mini_hybrid_cnn_train_easy.png")
  plt.show()
  return metrics

if __name__ == "__main__":
  metrics = mini_hybrid_cnn_train_easy()
  print(metrics)
  

