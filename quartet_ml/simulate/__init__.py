import subprocess
import tempfile
import os
import csv

"""
Simulated Data. Building 4-taxon trees in Newick format with various branch lengths for each of the 3 possible topologies. And then generate DNA alignment for each tree using AliSim.
"""

def build_newick(pair1, pair2, ext_lengths: dict, internal_length: float):
  """Return a Newick formatted string modelling a phylogenetic tree of 4 taxa ((pair1[0]: ext_length1, pair1[1]: ext_length2): internal_length, (pair2[0]: ext_length3, pair2[1]: ext_length4));
  """
  return f"(({pair1[0]}:{ext_lengths[pair1[0]]},{pair1[1]}:{ext_lengths[pair1[1]]}):{internal_length},({pair2[0]}:{ext_lengths[pair2[0]]},{pair2[1]}:{ext_lengths[pair2[1]]}));"

def easy(pair1, pair2, int_length):
  """
  The Control: each method should predict accurately here. Normal, balanced amount of evolutionary change.
  """
  ext_lengths = {pair1[0]: 0.1, pair1[1]: 0.1, pair2[0]: 0.1, pair2[1]: 0.1}
  return build_newick(pair1, pair2, ext_lengths, int_length)

def felsenstein_zone(pair1, pair2, int_length):
  """
  A trap: "similar-looking" sequences are actually not closely related - simple methods should get fooled. 2 of the 4 branches (non sisters) evolve fast for a long time.
  """
  ext_lengths = {pair1[0]: 1.0, pair1[1]: 0.02, pair2[0]: 1.0, pair2[1]: 0.02}
  return build_newick(pair1, pair2, ext_lengths, int_length)

def outbreak_like(pair1, pair2, int_length):
  """
  Realistic scenario: very little evidence to work with, since internal branches will be close to 0 length since mutations occur in very little time in an outbreak. Almost no time passed - sequences are nearly identical.
  """
  ext_lengths = {pair1[0]: 0.01, pair1[1]: 0.01, pair2[0]: 0.01, pair2[1]: 0.01}
  return build_newick(pair1, pair2, ext_lengths, int_length)

def possible_topologies():
  """
  Return the 3 possible ways to form a topology tree with 4 taxa
  """
  return {"1": (("A","B"), ("C","D")), "2": (("A","C"), ("B","D")), "3": (("A","D"), ("B","C"))}

def create_trees():
  """
  Return the created tree structures for each possible topology, with varying internal branch lengths, and across 3 different regimes of data
  """
  topologies = possible_topologies()
  trees = []
  for t in topologies:
    pair1 = topologies[t][0]
    pair2 = topologies[t][1]
    for i in range(1, 101):
      int_length = i/1000.0
      e = easy(pair1, pair2, int_length)
      fz = felsenstein_zone(pair1, pair2, int_length)
      ol = outbreak_like(pair1, pair2, int_length)
      trees.append({"topology": t, "length": str(int_length), "regimes": {"easy": e, "felsenstein": fz, "outbreak": ol}})
  return trees

def create_tree_files(model_type, alisim_model_parameters, data_file_name, replicate_range):
  """
  Create the trees and build the data csv files so the models can refer to them during training/validating/testing

  Parameters:
    model_type (str): simple jc or gtr to name the temp files and file paths for the alisim outputs
    alisim_model_parameters (str): exact string to pass into the alisim command for the model -m
    data_file_name (str): name of the csv file to build upon
    replicate_range (int): seed/how many replications to create a sufficient volume of data
  """
  headers = ["filepath", "topology", "regime", "internal_branch_length", "replicate"]
  trees = create_trees()
  with open(data_file_name, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()
    for r in range(0, replicate_range):
      for item in trees:
        topology = item["topology"]
        length = item["length"]
        regimes = item["regimes"]
        for regime_type, treetext in regimes.items():
          filename_prefix = f"Topology_{topology}_InternalLength_{length}_{regime_type}_Replicate_{r}"
          with tempfile.NamedTemporaryFile(prefix=filename_prefix, suffix=".nwk", mode="w+", delete=True) as temp_file:
            temp_file.write(treetext)
            temp_file.flush()

            output_dir = f"./sim_outputs_{model_type}"
            os.makedirs(output_dir, exist_ok=True)
            out_prefix = os.path.join(output_dir, f"sim_{topology}_{length}_{regime_type}_{model_type}_{r}")

            simulation_run = run_alisim(treefile=temp_file.name, model=alisim_model_parameters, output_prefix=out_prefix, seed=r)

            if(simulation_run is not None):
              row_data = {"filepath": f"{out_prefix}.phy", "topology": topology, "regime": regime_type, "internal_branch_length": length, "replicate": r}
              writer.writerow(row_data)

def run_alisim(treefile, model, output_prefix, seed, length=1000, seq_type="DNA", threads=1):
  """
  Calls AliSim via IQTREE3 to generate simulated DNA sequence alignments

  Parameters:
    treefile (str): path to Newick tree string file
    model (str): substitution model
    output_prefix (str): prefix for path of output alignment files
    length (int): always 1000, length of the DNA sequence
    seq_type (str): always DNA, sequence type generated from ali sim
    threads (int): number of threads for parallel computing
    seed (int): seed, 42 
  """
  command = [
    "iqtree3",
    "--alisim", output_prefix,
    "-t", treefile,
    "-m", model,
    "--length", str(length),
    "--seqtype", seq_type,
    "--threads", str(threads),
    "-seed", str(seed)
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
    print(f"AliSim failed with exit code {e.returncode}")
    print(f"Error output:\n{e.stderr}")
    return None