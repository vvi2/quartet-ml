from pathlib import Path
import subprocess
import tempfile
from .iqtree import iq_tree

def run_nj(file_path_to_nj, alignment_path, temp_file_path):
  """
  Run nj package on an alignment file and output the prediction to a new temp file. This is a bridge from the R to Python.
  Parameters:
    file_path_to_nj (str): file path to nj function written in r
    alignment_path (str): file path to alignment data
    temp_file_path (str): file path to the temp file created to paste the result from nj
  """
  command = [
    "Rscript",
    file_path_to_nj,
    alignment_path,
    temp_file_path
  ]
  try:
    result = subprocess.run(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      check=True
    )
    return result.stdout
  except subprocess.CalledProcessError as e:
    print(f"NJ failed with exit code {e.returncode}")
    print(f"Error output:\n{e.stderr}")
    return None


def nj_prediction(alignment_path):
  """
  Retrieve prediction from nj method.
  Parameters:
    alignment_path (str): file path to alignment data
  """
  nj_file_path = Path(__file__).parent / "nj.R"
  with tempfile.NamedTemporaryFile(prefix="nj_prediction_file", suffix=".txt", mode="w+", delete=True) as temp_file:
    if(run_nj(nj_file_path, alignment_path, temp_file.name) is not None):
      prediction = temp_file.read().rstrip()
    else:
      return None
  return prediction



