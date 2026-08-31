import pandas as pd
from sklearn.model_selection import train_test_split
def split_data(df):
  """
  Splitting simulated alignment data into training set, validation set, and testing set to feed into models
  """
  df = df.copy()
  df["strata"] = df[["topology", "regime", "internal_branch_length"]].astype(str).agg("_".join, axis=1)
  train_rows, val_test_rows = train_test_split(df, test_size=0.3, random_state=42, stratify=df["strata"])
  val_rows, test_rows = train_test_split(val_test_rows, test_size=0.5, random_state=42, stratify=val_test_rows["strata"])
  return train_rows, val_rows, test_rows