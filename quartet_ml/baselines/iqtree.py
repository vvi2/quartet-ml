from pathlib import Path
import subprocess
from quartet_ml.simulate import possible_topologies

from Bio import Phylo

def run_iqtree_command(alignment_path, output_prefix):
  """
  Run iqtree package on an alignment file and output the tree to a new file.
  Parameters:
    alignment_path (str): file path to alignment data
    output_prefix (str): prefix of output file name
  """
  command = [
    "iqtree3",
    "-s", alignment_path,
    "-m", "JC",
    "-pre", output_prefix,
    "-quiet"
  ]
  try:
    #Cleanup files created from iqtree3 command from last run so it can run again
    for ext in {"bionj", "ckp.gz", "iqtree", "log", "mldist", "treefile"}:
      Path(f"{output_prefix}.{ext}").unlink(missing_ok=True)
    result = subprocess.run(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      check=True
    )
    return result.stdout
  except subprocess.CalledProcessError as e:
    print(f"IqTree failed with exit code {e.returncode}")
    print(f"Error output:\n{e.stderr}")
    return None

def iq_tree_prediction(output_prefix):
  """
  Determine which topology the iq tree predicted from the alignment data in the output file.
  Parameters:
    output_prefix (str): prefix of output file name
  """
  newick_tree = Phylo.read(f"{output_prefix}.treefile", "newick")
  taxon = {taxa.name: taxa for taxa in newick_tree.get_terminals()}
  for topology_label, newick in possible_topologies().items():
    if(newick_tree.is_monophyletic([taxon[newick[0][0]], taxon[newick[0][1]]]) or newick_tree.is_monophyletic([taxon[newick[1][0]], taxon[newick[1][1]]]) ):
      return topology_label

def iq_tree(alignment_path, output_prefix):
  """
  Run iqtree command and get its prediction if not None
  Parameters:
    alignment_path (str): file path to alignment data
    output_prefix (str): prefix of output file name
  """
  if(run_iqtree_command(alignment_path, output_prefix) is not None):
    return iq_tree_prediction(output_prefix)
  