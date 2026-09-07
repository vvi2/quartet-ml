from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from quartet_ml.features import build_site_pattern_vector, jc69_collapsed_vector, read_phylip_data, cnn_matrix

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

def get_cnn_split_data(split, csv_file_path = None):
  data_file_path = Path(__file__).resolve().parent.parent.parent / "jc69_split_data.csv" if csv_file_path is None else csv_file_path
  df = pd.read_csv(data_file_path)
  split_data = df[df["split"] == split]
  x_split = []
  y_split = []
  pytorch_mapping = {1: 0, 2: 1, 3: 2}
  for _,row in split_data.iterrows():
    taxa_dict = read_phylip_data(row["filepath"])
    cnn_vector = cnn_matrix(taxa_dict)
    x_split.append(cnn_vector)
    y_split.append(pytorch_mapping[row["topology"]])
  return x_split, y_split

def prepare_cnn_dataloader(x_split, y_split, batch_size, shuffle):
  #take N loose alignments from x and n loose labels from y and turn them into one object to hand to training for cnn
  n = len(x_split)
  x_stack = np.stack(x_split, axis=0) #gets (N, 4, 4, 1000)
  x_reshaped_and_cast = torch.tensor(x_stack.reshape(n, 16, 1000), dtype=torch.float32) # cnn conv1d needs (batch, channels, length) shape matrix
  y_reshaped_and_cast = torch.tensor(y_split.reshape(n, 16, 1000), dtype=torch.long)
  tensor_data = TensorDataset(x_reshaped_and_cast, y_reshaped_and_cast) # links alignment features and labels
  data_loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle)
  return data_loader # groups examples into batches and scrambles order everytime you loop in training

class CNN(nn.Module):
  # in channels = 16, out channels = how many different filters i want cnn to learn at the layer, kernel size = how many neighboring site do i want each filter to look at per slide. slides out_channels n separate filters across 1000 size axis and prodcuts output shaped (batch, out_channels, new_length) - 1. number per filter per position saying how strongly did filter fire here
  # 1 conv layer, 1 nonlinear activation, 1 conv layer, 1 output layer
  def __init__(self, in_channels, out_channels, kernel_size):
    super().__init__()
    self.conv1 = nn.Conv1d(in_channels, out_channels=out_channels, kernel_size=kernel_size)
    self.conv2 = nn.Conv1d(self.conv1.out_channels, out_channels=out_channels, kernel_size=kernel_size)
    self.relu = nn.ReLU()
    self.pool = nn.AdaptiveMaxPool1d(1) # did the ilter fire strongly anywhere in the alignment
    self.fc = nn.Linear(in_features=self.conv2.out_channels, out_features=3)
  def forward(self, x):
    conv1_out = self.conv1(x)
    relu1_out = self.relu(conv1_out)
    conv2_out = self.conv2(relu1_out)
    relu2_out = self.relu(conv2_out)
    pool_out = self.pool(relu2_out).squeeze(-1)
    fc_out = self.fc(pool_out)
    return fc_out



  