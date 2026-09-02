from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quartet_ml.features import build_site_pattern_vector, jc69_collapsed_vector, read_phylip_data

def get_x_y_train(csv_file_path = None):
  """
  Parse x and y training data from the split labeled alignment data i.e.jc69_split_data.csv or another file path if specified. Featurize the data in a way that the site pattern classifier will accept for both the 256-num version and collapsed jc 15-num version.

  Parameters:
    csv_file_path (str) OPTIONAL: file path to csv file of split data
  """
  data_file_path = Path(__file__).resolve().parent.parent.parent / "jc69_split_data.csv" if csv_file_path is None else csv_file_path
  df = pd.read_csv(data_file_path)
  training_data = df[df["split"] == "train"]
  x_train_256 = []
  x_train_15 = []
  y_train = []
  for _,row in training_data.iterrows():
    taxa_dict = read_phylip_data(row["filepath"])
    site_pattern_vector = build_site_pattern_vector(taxa_dict)
    x_train_15.append(jc69_collapsed_vector(site_pattern_vector))
    x_train_256.append(site_pattern_vector)
    y_train.append(str(row["topology"]))
  return x_train_256, x_train_15, y_train

def site_pattern_classifier_LR(csv_file_path = None):
  """
  Run multinomial logistic regression on the 256 vector and collapsed 15 vector and return the split training data used and each model.

  Parameters:
      csv_file_path (str) OPTIONAL: file path to csv file of split data
  """
  training_data_lr = get_x_y_train(csv_file_path)
  model_256 = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, random_state=42))
  ])
  model_15 = Pipeline([
      ("scale", StandardScaler()),
      ("clf", LogisticRegression(max_iter=2000, random_state=42))
    ])
  model_256.fit(training_data_lr[0], training_data_lr[2])
  model_15.fit(training_data_lr[1], training_data_lr[2])
  return {"x_train_256": training_data_lr[0], "x_train_15": training_data_lr[1], "y_train": training_data_lr[2], "256": model_256, "15": model_15}
  