# Neighbour-joining baseline arm: ape::nj() on JC69-corrected distances.
#
# Usage (once implemented):
#   Rscript nj.R <alignment.fasta> <output.nwk>

library(ape)

neighbor_joining_prediction_method <- function(file_path){
  dna <- read.dna(file_path, format = "sequential")
  distance_matrix <- dist.dna(dna, model = "JC69")
  nj_tree <- nj(distance_matrix)
  predicted_topology <- ""
  if(is.monophyletic(nj_tree, c("A", "B"))){
    predicted_topology <- "1"
  } else if(is.monophyletic(nj_tree, c("A", "C"))){
    predicted_topology <- "2"
  } else {
    predicted_topology <- "3"
  }
  return(predicted_topology)
}

args <- commandArgs(trailingOnly = TRUE)
writeLines(neighbor_joining_prediction_method(args[1]), args[2])
