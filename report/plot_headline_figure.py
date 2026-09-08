from matplotlib import pyplot as plt
import pandas as pd

def plot_headline_figure():
  csv_data_files = ["headline_data/cnn.csv", "headline_data/hybrid_cnn.csv", "headline_data/iqtree.csv", "headline_data/lr_15.csv", "headline_data/lr_256.csv", "headline_data/nj.csv"]
  df = pd.concat((pd.read_csv(file) for file in csv_data_files), ignore_index=True)
  df["bin"] = pd.cut(df["internal_branch_length"], bins=10)
  df["x_axis_values"] = [b.mid for b in df["bin"]]
  summary_df = df.groupby(["method", "regime", "x_axis_values"]).agg(
    accuracy=("correct", "mean"),
    bin_size=("correct", "size")
  ).reset_index()
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  regimes_ordered = ["easy", "felsenstein", "outbreak"]
  method_plot_mapping = {
    "CNN": ["#264653", "o"],
    "Hybrid CNN": ["#2a9d8f", "s"],
    "LR-256": ["#8ab17d", "^"],
    "LR-15": ["#e9c46a", "D"],
    "NJ": ["#f4a261", "H"],
    "IQ-TREE": ["#e76f51", "*"]
  }
  for ax, regime in zip(axes, regimes_ordered):
    regime_df = summary_df[summary_df["regime"] == regime]
    for method, (color, marker) in method_plot_mapping.items():
      method_df = regime_df[regime_df["method"] == method].sort_values("x_axis_values")
      ax.plot(method_df["x_axis_values"], method_df["accuracy"], label=method, color=color, marker=marker)
    ax.axhline(y=1/3, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Internal Branch Length")
    ax.set_ylabel("Accuracy")
    ax.set_title(regime.capitalize())
  axes[1].set_facecolor("#F0F0F0")
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.05))
  fig.tight_layout()
  fig.savefig("report/headline_figure.png", dpi=300, bbox_inches="tight")
  plt.show()
  return summary_df[["method", "regime", "x_axis_values", "bin_size"]]

if __name__ == "__main__":
  summary = plot_headline_figure()
  print(summary)
