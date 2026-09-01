from collections import defaultdict

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

base_vector_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def build_site_pattern_vector(taxa_dict):
  """
  Turn raw alignment into 256-number site-pattern count vector as feature for LR model and hybrid model. Returns 256-number array where each number is a count of how many sites in the alignment had that pattern.

  Parameters:
    taxa_dict (dict): raw alignment in a dict form where keys are taxa and values are corresponding DNA sequence. Expects it to have exactly the keys A, B, C, D.
  """
  site_pattern_frequencies = np.zeros(256)
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

def index_to_pattern(index):
  """
  index of pattern in 256 number site_pattern_frequency vector = (A * 64) + (B * 16) + (C * 4) + (D * 1). Need to reverse this to get letter version of the pattern.
  Parameters:
    index (int): index of pattern in site_pattern_frequency vector
  """
  taxa_a_base, remainder_a = divmod(index, 64)
  taxa_b_base, remainder_b = divmod(remainder_a, 16)
  taxa_c_base, remainder_c = divmod(remainder_b, 4)
  taxa_d_base = remainder_c
  reversed_base_vector_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
  return (reversed_base_vector_map[taxa_a_base] + reversed_base_vector_map[taxa_b_base] + reversed_base_vector_map[taxa_c_base] + reversed_base_vector_map[taxa_d_base])

def convert_pattern(pattern):
  """
  Convert base letter pattern to tuple format that identifies it as 1 of the 15 ways to group patterns in the way that JC69 assumes them as the same pattern, to map to my lookup table.

  Parameters:
    pattern (str): 4 letter string of the bases at the same alignment position in each of the 4 taxa
  """
  seen_bases = {}
  for i, base in enumerate(pattern):
    if base not in seen_bases:
      seen_bases[base] = []
    seen_bases[base].append(i)
  return (tuple(sorted(tuple(group) for group in seen_bases.values())))

# JC69 says it's which taxa match each other that matters for counting the patterns. For 4 taxa, there are 15 distinct ways you group them by "who shares a base with whom". (1) All 4 match - 1 way, (2) 3 match, 1 doesn't - 4 ways, (3) Two pairs, each pair matches internally but differs from other pair - 3 ways, (4) One pair matches, the other two are each different from everyone - 6 ways, (5) All 4 different - 1 way
categories_mapping ={
  ((0, 1, 2, 3),): "All_4_Match",
  ((0,), (1,), (2,), (3,)): "All_4_Different",
  ((0, 1, 2), (3,)): "3_Match_1_Not_1",
  ((0,), (1, 2, 3)): "3_Match_1_Not_2",
  ((0, 2, 3), (1,)): "3_Match_1_Not_3",
  ((0, 1, 3), (2,)): "3_Match_1_Not_4",
  ((0, 1), (2, 3)): "2_Pairs_1",
  ((0, 2), (1, 3)): "2_Pairs_2",
  ((0, 3), (1, 2)): "2_Pairs_3",
  ((0, 1), (2,), (3,)): "1_Pair_1",
  ((0, 2), (1,), (3,)): "1_Pair_2",
  ((0, 3), (1,), (2,)): "1_Pair_3",
  ((0,), (1, 2), (3,)): "1_Pair_4",
  ((0,), (1, 3), (2,)): "1_Pair_5",
  ((0,), (1,), (2, 3)): "1_Pair_6",
}

def jc69_collapsed_vector(site_pattern_vector):
  """
  Convert 256-number site pattern frequency to the 15-number vector to mirror JC69 assumption of only 15 distinct patterns

  Parameters:
    site_pattern_vector (np array): 256-num site pattern frequencies vector
  """
  collapsed_version_dict = defaultdict(int)
  for index, pattern_count in enumerate(site_pattern_vector):
    pattern = index_to_pattern(index)
    map_tuple = convert_pattern(pattern)
    category = categories_mapping[map_tuple]
    collapsed_version_dict[category] += pattern_count
  collapsed_vector = np.zeros(15)
  for index, category in enumerate(categories_mapping.values()):
    collapsed_vector[index] = int(collapsed_version_dict[category])
  return collapsed_vector

def cnn_matrix(taxa_dict):
  """
  Convert raw alignment data into a form that nn.Conv1d will accept, all numbers no letters

  Parameters:
    taxa_dict (dict): raw alignment in a dict form where keys are taxa and values are corresponding DNA sequence.
  """
  sequence_length = len(next(iter(taxa_dict.values())))
  output_matrix = np.zeros((4, 4, sequence_length))
  ordered_taxa_dict = [taxa_dict['A'], taxa_dict['B'], taxa_dict['C'], taxa_dict['D']]
  for taxon_index, sequence in enumerate(ordered_taxa_dict):
    for sequence_index, base in enumerate(sequence):
      base_index = base_vector_map[base]
      output_matrix[taxon_index, base_index, sequence_index] = 1
  return output_matrix




    



