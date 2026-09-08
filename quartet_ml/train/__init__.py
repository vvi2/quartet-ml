import copy
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

def get_x_y_split(csv_file_path = None, split="train"):
  """
  Parse x and y data from the split labeled alignment data i.e.jc69_split_data.csv or another file path if specified. Featurize the data in a way that the site pattern classifier will accept for both the 256-num version and collapsed jc 15-num version.

  Parameters:
    csv_file_path (str) OPTIONAL: file path to csv file of split data
    split (str): whatever type of data we want - train, val, test
  """
  data_file_path = Path(__file__).resolve().parent.parent.parent / "jc69_split_data.csv" if csv_file_path is None else csv_file_path
  df = pd.read_csv(data_file_path)
  split_data = df[df["split"] == split]
  x_256 = []
  x_15 = []
  y = []
  for _,row in split_data.iterrows():
    taxa_dict = read_phylip_data(row["filepath"])
    site_pattern_vector = build_site_pattern_vector(taxa_dict)
    x_15.append(jc69_collapsed_vector(site_pattern_vector))
    x_256.append(site_pattern_vector)
    y.append(str(row["topology"]))
  return x_256, x_15, y

def get_x_y_split_gtr(csv_file_path = None):
  """
  Parse x and y data from the GTR alignment data i.e.gtr_simulated_data.csv or another file path if specified. Featurize the data in a way that the site pattern classifier will accept for both the 256-num version and collapsed jc 15-num version.

  Parameters:
    csv_file_path (str) OPTIONAL: file path to csv file of split data
  """
  data_file_path = Path(__file__).resolve().parent.parent.parent / "gtr_simulated_data.csv" if csv_file_path is None else csv_file_path
  df = pd.read_csv(data_file_path)
  x_256 = []
  x_15 = []
  y = []
  for _,row in df.iterrows():
    taxa_dict = read_phylip_data(row["filepath"])
    site_pattern_vector = build_site_pattern_vector(taxa_dict)
    x_15.append(jc69_collapsed_vector(site_pattern_vector))
    x_256.append(site_pattern_vector)
    y.append(str(row["topology"]))
  return x_256, x_15, y

def train_site_pattern_classifier_LR(csv_file_path = None):
  """
  Run multinomial logistic regression on the 256 vector and collapsed 15 vector and return the split training data used and each model.

  Parameters:
      csv_file_path (str) OPTIONAL: file path to csv file of split data
  """
  training_data_lr = get_x_y_split(csv_file_path)
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

def get_cnn_split_data(split, regime_type = None, csv_file_path = None):
  """
  Parse x and y data from the split labeled alignment data i.e. jc69_split_data.csv or another file path if specified. Featurize the data in a way that the CNN will accept: a (4, 4, 1000) one-hot matrix per alignment, with the topology label mapped to 0/1/2 for PyTorch.

  Parameters:
    split (str): whatever type of data we want - train, val, test
    regime_type (str) OPTIONAL: filter to one regime - easy, felsenstein, outbreak. Defaults to all regimes combined.
    csv_file_path (str) OPTIONAL: file path to csv file of split data
  """
  data_file_path = Path(__file__).resolve().parent.parent.parent / "jc69_split_data.csv" if csv_file_path is None else csv_file_path
  df = pd.read_csv(data_file_path)
  split_data = df[(df["split"] == split) & (True if regime_type is None else (df["regime"] == regime_type))]
  x_split = []
  y_split = []
  pytorch_mapping = {1: 0, 2: 1, 3: 2}
  for _,row in split_data.iterrows():
    taxa_dict = read_phylip_data(row["filepath"])
    cnn_vector = cnn_matrix(taxa_dict)
    x_split.append(cnn_vector)
    y_split.append(pytorch_mapping[row["topology"]])
  return x_split, y_split

def get_cnn_split_data_gtr(regime_type = None, csv_file_path = None):
  """
  Parse x and y data from the GTR alignment data i.e. gtr_simulated_data.csv or another file path if specified. Featurize the data in a way that the CNN will accept: a (4, 4, 1000) one-hot matrix per alignment, with the topology label mapped to 0/1/2 for PyTorch. This data has no train/val/test split - it's meant to be used only as a misspecification test set for an already-trained model.

  Parameters:
    regime_type (str) OPTIONAL: filter to one regime - easy, felsenstein, outbreak. Defaults to all regimes combined.
    csv_file_path (str) OPTIONAL: file path to csv file of GTR data
  """
  data_file_path = Path(__file__).resolve().parent.parent.parent / "gtr_simulated_data.csv" if csv_file_path is None else csv_file_path
  df = pd.read_csv(data_file_path)
  split_data = df if regime_type is None else df[(df["regime"] == regime_type)]
  x_split = []
  y_split = []
  pytorch_mapping = {1: 0, 2: 1, 3: 2}
  for _,row in split_data.iterrows():
    taxa_dict = read_phylip_data(row["filepath"])
    cnn_vector = cnn_matrix(taxa_dict)
    x_split.append(cnn_vector)
    y_split.append(pytorch_mapping[row["topology"]])
  return x_split, y_split

def get_hybrid_cnn_split_data(split, regime_type = None, csv_file_path = None):
  """
  Parse x, pattern, and y data from the split labeled alignment data i.e. jc69_split_data.csv or another file path if specified. Featurize the data in a way that the hybrid CNN will accept: a (4, 4, 1000) one-hot matrix per alignment, the JC69-collapsed 15-num pattern vector (scaled to proportions of the alignment length), and the topology label mapped to 0/1/2 for PyTorch.

  Parameters:
    split (str): whatever type of data we want - train, val, test
    regime_type (str) OPTIONAL: filter to one regime - easy, felsenstein, outbreak. Defaults to all regimes combined.
    csv_file_path (str) OPTIONAL: file path to csv file of split data
  """
  data_file_path = Path(__file__).resolve().parent.parent.parent / "jc69_split_data.csv" if csv_file_path is None else csv_file_path
  df = pd.read_csv(data_file_path)
  split_data = df[(df["split"] == split) & (True if regime_type is None else (df["regime"] == regime_type))]
  x_cnn_split = []
  x_pattern_split = []
  y_split = []
  pytorch_mapping = {1: 0, 2: 1, 3: 2}
  for _,row in split_data.iterrows():
    taxa_dict = read_phylip_data(row["filepath"])
    cnn_vector = cnn_matrix(taxa_dict)
    jc69_vector = jc69_collapsed_vector(build_site_pattern_vector(taxa_dict))
    scaled_jc69_vector = jc69_vector / jc69_vector.sum()
    x_cnn_split.append(cnn_vector)
    x_pattern_split.append(scaled_jc69_vector)
    y_split.append(pytorch_mapping[row["topology"]])
  return x_cnn_split, x_pattern_split, y_split

def get_hybrid_cnn_split_data_gtr(regime_type = None, csv_file_path = None):
  """
  Parse x, pattern, and y data from the GTR alignment data i.e. gtr_simulated_data.csv or another file path if specified. Featurize the data in a way that the hybrid CNN will accept: a (4, 4, 1000) one-hot matrix per alignment, the JC69-collapsed 15-num pattern vector (scaled to proportions of the alignment length), and the topology label mapped to 0/1/2 for PyTorch. This data has no train/val/test split - it's meant to be used only as a misspecification test set for an already-trained model.

  Parameters:
    regime_type (str) OPTIONAL: filter to one regime - easy, felsenstein, outbreak. Defaults to all regimes combined.
    csv_file_path (str) OPTIONAL: file path to csv file of GTR data
  """
  data_file_path = Path(__file__).resolve().parent.parent.parent / "gtr_simulated_data.csv" if csv_file_path is None else csv_file_path
  df = pd.read_csv(data_file_path)
  split_data = df if regime_type is None else df[(df["regime"] == regime_type)]
  x_cnn_split = []
  x_pattern_split = []
  y_split = []
  pytorch_mapping = {1: 0, 2: 1, 3: 2}
  for _,row in split_data.iterrows():
    taxa_dict = read_phylip_data(row["filepath"])
    cnn_vector = cnn_matrix(taxa_dict)
    jc69_vector = jc69_collapsed_vector(build_site_pattern_vector(taxa_dict))
    scaled_jc69_vector = jc69_vector / jc69_vector.sum()
    x_cnn_split.append(cnn_vector)
    x_pattern_split.append(scaled_jc69_vector)
    y_split.append(pytorch_mapping[row["topology"]])
  return x_cnn_split, x_pattern_split, y_split

def prepare_cnn_dataloader(x_split, y_split, batch_size, shuffle):
  """
  Take the loose per-alignment CNN features and labels and turn them into one PyTorch DataLoader, ready to hand to the training loop.

  Parameters:
    x_split (list): list of (4, 4, 1000) one-hot alignment matrices
    y_split (list): list of topology labels, mapped to 0/1/2
    batch_size (int): how many alignments per batch
    shuffle (bool): whether to scramble the order every epoch - True for train, False for val/test
  """
  #take N loose alignments from x and n loose labels from y and turn them into one object to hand to training for cnn
  n = len(x_split)
  x_stack = np.stack(x_split, axis=0) #gets (N, 4, 4, 1000)
  x_reshaped_and_cast = torch.tensor(x_stack.reshape(n, 16, 1000), dtype=torch.float32) # cnn conv1d needs (batch, channels, length) shape matrix
  y_reshaped_and_cast = torch.tensor(y_split, dtype=torch.long)
  tensor_data = TensorDataset(x_reshaped_and_cast, y_reshaped_and_cast) # links alignment features and labels
  data_loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle)
  return data_loader # groups examples into batches and scrambles order everytime you loop in training

def prepare_hybrid_cnn_dataloader(x_cnn_split, x_pattern_split,y_split, batch_size, shuffle):
  """
  Take the loose per-alignment CNN features, pattern vectors, and labels and turn them into one PyTorch DataLoader, ready to hand to the hybrid training loop.

  Parameters:
    x_cnn_split (list): list of (4, 4, 1000) one-hot alignment matrices
    x_pattern_split (list): list of 15-num JC69-collapsed pattern vectors, scaled to proportions
    y_split (list): list of topology labels, mapped to 0/1/2
    batch_size (int): how many alignments per batch
    shuffle (bool): whether to scramble the order every epoch - True for train, False for val/test
  """
  #take N loose alignments from x and n loose labels from y and turn them into one object to hand to training for cnn
  n = len(x_cnn_split)
  x_cnn_stack = np.stack(x_cnn_split, axis=0) #gets (N, 4, 4, 1000)
  x_pattern_stack = np.stack(x_pattern_split, axis=0)
  x_cnn_reshaped_and_cast = torch.tensor(x_cnn_stack.reshape(n, 16, 1000), dtype=torch.float32) # cnn conv1d needs (batch, channels, length) shape matrix
  x_pattern_reshaped_and_cast = torch.tensor(x_pattern_stack, dtype=torch.float32)
  y_reshaped_and_cast = torch.tensor(y_split, dtype=torch.long)
  tensor_data = TensorDataset(x_cnn_reshaped_and_cast, x_pattern_reshaped_and_cast, y_reshaped_and_cast) # links alignment features and labels
  data_loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle)
  return data_loader # groups examples into batches and scrambles order everytime you loop in training

class CNN(nn.Module):
  """
  1 conv layer, 1 nonlinear activation, 1 conv layer, 1 linear output layer. Takes the raw one-hot alignment and outputs 3 raw scores, one per topology.

  Parameters:
    in_channels (int): number of channels per alignment position - 16, since the 4 taxa x 4 bases one-hot encoding gets flattened together
    out_channels (int): how many different filters we want the CNN to learn at each conv layer. Each filter slides across the 1000-site axis and produces one number per position, saying how strongly that filter fired there
    kernel_size (int): how many neighboring sites each filter looks at per slide
  """
  def __init__(self, in_channels, out_channels, kernel_size):
    super().__init__()
    self.conv1 = nn.Conv1d(in_channels, out_channels=out_channels, kernel_size=kernel_size)
    self.conv2 = nn.Conv1d(self.conv1.out_channels, out_channels=out_channels, kernel_size=kernel_size)
    self.relu = nn.ReLU()
    self.pool = nn.AdaptiveMaxPool1d(1) # did the ilter fire strongly anywhere in the alignment
    self.fc = nn.Linear(in_features=self.conv2.out_channels, out_features=3)
  def forward(self, x):
    """
    Run one batch of one-hot alignments through the conv layers, pool, and output layer.

    Parameters:
      x (tensor): batch of one-hot alignments, shape (batch, 16, 1000)
    """
    conv1_out = self.conv1(x)
    relu1_out = self.relu(conv1_out)
    conv2_out = self.conv2(relu1_out)
    relu2_out = self.relu(conv2_out)
    pool_out = self.pool(relu2_out).squeeze(-1)
    fc_out = self.fc(pool_out)
    return fc_out

class HybridCNN(nn.Module):
  """
  Same architecture as CNN, but concatenates the 15-num JC69-collapsed pattern vector onto the pooled conv features right before the final linear layer, so the classifier sees both what it detected in the raw sequence and the model-derived summary.

  Parameters:
    in_channels (int): number of channels per alignment position - 16, since the 4 taxa x 4 bases one-hot encoding gets flattened together
    out_channels (int): how many different filters we want the CNN to learn at each conv layer
    kernel_size (int): how many neighboring sites each filter looks at per slide
  """
  def __init__(self, in_channels, out_channels, kernel_size):
    super().__init__()
    self.conv1 = nn.Conv1d(in_channels, out_channels=out_channels, kernel_size=kernel_size)
    self.conv2 = nn.Conv1d(self.conv1.out_channels, out_channels=out_channels, kernel_size=kernel_size)
    self.relu = nn.ReLU()
    self.pool = nn.AdaptiveMaxPool1d(1) # did the ilter fire strongly anywhere in the alignment
    self.fc = nn.Linear(in_features=self.conv2.out_channels + 15, out_features=3)
  def forward(self, x, pattern_vector):
    """
    Run one batch of one-hot alignments through the conv layers and pool, concatenate the pattern vector onto the pooled features, then pass through the output layer.

    Parameters:
      x (tensor): batch of one-hot alignments, shape (batch, 16, 1000)
      pattern_vector (tensor): batch of 15-num JC69-collapsed pattern vectors, shape (batch, 15)
    """
    conv1_out = self.conv1(x)
    relu1_out = self.relu(conv1_out)
    conv2_out = self.conv2(relu1_out)
    relu2_out = self.relu(conv2_out)
    pool_out = self.pool(relu2_out).squeeze(-1)
    add_pattern_vector_to_pool = torch.cat([pool_out, pattern_vector], dim=1)
    fc_out = self.fc(add_pattern_vector_to_pool)
    return fc_out

def train_cnn_with_early_stopping(model, train_data_loader, val_data_loader, max_epochs=200, lr=0.01, patience=25, min_delta=1e-4):
  """
  Train the CNN, watching validation loss each epoch, and stop once it hasn't improved in `patience` epochs. Restores the best-epoch weights before returning, not whatever the model ended on.

  Parameters:
    model (CNN): the model to train
    train_data_loader (DataLoader): batches of (alignment, label) for training
    val_data_loader (DataLoader): batches of (alignment, label) for validation, used only to decide when to stop
    max_epochs (int): upper bound on how many epochs to run if early stopping never fires
    lr (float): learning rate for the Adam optimizer
    patience (int): how many epochs to wait without improvement before stopping
    min_delta (float): minimum decrease in validation loss that counts as an improvement, to guard against stopping on noise
  """
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  model_history = []
  best_score = float('inf')
  best_epoch = 0
  best_weights = None
  epochs_without_improvement = 0
  stopped_early = False
  for epoch in range(max_epochs):
    model.train()
    running_train_loss = 0.0
    running_val_loss = 0.0
    for t_x_batch, t_y_batch in train_data_loader:
      optimizer.zero_grad()
      t_loss = criterion(model(t_x_batch), t_y_batch)
      t_loss.backward()
      optimizer.step()
      running_train_loss += t_loss.item() * len(t_x_batch)
    model.eval()
    with torch.no_grad():
      for v_x_batch, v_y_batch in val_data_loader:
        v_loss = criterion(model(v_x_batch), v_y_batch)
        running_val_loss += v_loss.item() * len(v_x_batch)
    running_train_loss /= len(train_data_loader.dataset)
    running_val_loss /= len(val_data_loader.dataset)
    model_history.append({"epoch": epoch, "train_loss": running_train_loss, "val_loss": running_val_loss})
    val_improvement = running_val_loss < best_score - min_delta
    if val_improvement:
      best_score = running_val_loss
      best_epoch = epoch
      best_weights = copy.deepcopy(model.state_dict())
      epochs_without_improvement = 0
    else:
      epochs_without_improvement += 1
      if epochs_without_improvement >= patience:
        stopped_early = True
        break
  if best_weights is not None:
    model.load_state_dict(best_weights)
  return model, pd.DataFrame(model_history), {"best_epoch": best_epoch, "best_loss": best_score, "stopped_early": stopped_early, "epochs_run": len(model_history)}

def train_hybrid_cnn_with_early_stopping(model, train_data_loader, val_data_loader, max_epochs=200, lr=0.01, patience=25, min_delta=1e-4):
  """
  Train the HybridCNN, watching validation loss each epoch, and stop once it hasn't improved in `patience` epochs. Restores the best-epoch weights before returning, not whatever the model ended on. Same logic as train_cnn_with_early_stopping, just passing both the alignment and the pattern vector into the model at each step.

  Parameters:
    model (HybridCNN): the model to train
    train_data_loader (DataLoader): batches of (alignment, pattern vector, label) for training
    val_data_loader (DataLoader): batches of (alignment, pattern vector, label) for validation, used only to decide when to stop
    max_epochs (int): upper bound on how many epochs to run if early stopping never fires
    lr (float): learning rate for the Adam optimizer
    patience (int): how many epochs to wait without improvement before stopping
    min_delta (float): minimum decrease in validation loss that counts as an improvement, to guard against stopping on noise
  """
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  model_history = []
  best_score = float('inf')
  best_epoch = 0
  best_weights = None
  epochs_without_improvement = 0
  stopped_early = False
  for epoch in range(max_epochs):
    model.train()
    running_train_loss = 0.0
    running_val_loss = 0.0
    for t_x_cnn_batch, t_x_pattern_batch, t_y_batch in train_data_loader:
      optimizer.zero_grad()
      t_loss = criterion(model(t_x_cnn_batch, t_x_pattern_batch), t_y_batch)
      t_loss.backward()
      optimizer.step()
      running_train_loss += t_loss.item() * len(t_x_cnn_batch)
    model.eval()
    with torch.no_grad():
      for v_x_cnn_batch, v_x_pattern_batch, v_y_batch in val_data_loader:
        v_loss = criterion(model(v_x_cnn_batch, v_x_pattern_batch), v_y_batch)
        running_val_loss += v_loss.item() * len(v_x_cnn_batch)
    running_train_loss /= len(train_data_loader.dataset)
    running_val_loss /= len(val_data_loader.dataset)
    model_history.append({"epoch": epoch, "train_loss": running_train_loss, "val_loss": running_val_loss})
    val_improvement = running_val_loss < best_score - min_delta
    if val_improvement:
      best_score = running_val_loss
      best_epoch = epoch
      best_weights = copy.deepcopy(model.state_dict())
      epochs_without_improvement = 0
    else:
      epochs_without_improvement += 1
      if epochs_without_improvement >= patience:
        stopped_early = True
        break
  if best_weights is not None:
    model.load_state_dict(best_weights)
  return model, pd.DataFrame(model_history), {"best_epoch": best_epoch, "best_loss": best_score, "stopped_early": stopped_early, "epochs_run": len(model_history)}



  