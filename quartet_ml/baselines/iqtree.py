import subprocess
from quartet_ml.simulate import possible_topologies

from Bio import Phylo

def run_iqtree_command(alignment, output_prefix):
  command = [
    "iqtree3",
    "-s", alignment,
    "-m", "JC",
    "-pre", output_prefix,
    "-quiet"
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
    print(f"IqTree failed with exit code {e.returncode}")
    print(f"Error output:\n{e.stderr}")
    return None

def iq_tree_prediction(output_prefix):
  newick_tree = Phylo.read(f"{output_prefix}.treefile", "newick")
  taxon = {taxa.name: taxa for taxa in newick_tree.get_terminals()}
  for topology_label, newick in possible_topologies().items():
    if(newick_tree.is_monophyletic([taxon[newick[0][0]], taxon[newick[0][1]]]) or newick_tree.is_monophyletic([taxon[newick[1][0]], taxon[newick[1][1]]]) ):
      return topology_label

def iq_tree(alignment, output_prefix):
  if(run_iqtree_command(alignment, output_prefix) is not None):
    return iq_tree_prediction(output_prefix)