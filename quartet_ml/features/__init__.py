import numpy as np

def read_phylip_data(file_path):
  """
  Read the simulated data and reformat it so we can build the features to feed into models. Returns a dict where each key is a taxon and each value is that taxon's DNA sequence, both are strings.

  Parameters:
    file_path( str): file path to the simulated data file that we are reading
  """
  taxa_dict = {}
  with open(file_path, mode="r") as file:
    next(file) #skip header
    for line in file:
      line = line.strip("\n")
      if not line:
        continue
      taxa = line[0]
      dna_sequence = line[1:].lstrip()
      taxa_dict[taxa] = dna_sequence
  return taxa_dict

def build_site_pattern_vector(taxa_dict):
  """
  Turn raw alignment into 256-number site-pattern count vector as feature for LR model and hybrid model. Returns 256-number array where each number is a count of how many sites in the alignment had that pattern.

  Parameters:
    taxa_dict (dict): raw alignment in a dict form where keys are taxa and values are corresponding DNA sequence. Expects it to have exactly the keys A, B, C, D.
  """
  site_pattern_frequencies = np.zeros(256)
  base_vector_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
  sequence_length = len(next(iter(taxa_dict.values())))
  taxa_a_seq = taxa_dict['A']
  taxa_b_seq = taxa_dict['B']
  taxa_c_seq = taxa_dict['C']
  taxa_d_seq = taxa_dict['D']
  for i in range(sequence_length):
    site_pattern = taxa_a_seq[i] + taxa_b_seq[i] + taxa_c_seq[i] + taxa_d_seq[i]
    pattern_index = 0
    for char in site_pattern:
      pattern_index = (pattern_index << 2) | base_vector_map.get(char)
    site_pattern_frequencies[pattern_index] += 1
  return site_pattern_frequencies