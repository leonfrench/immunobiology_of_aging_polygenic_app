#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
output_path <- if (length(args) >= 1L) args[[1L]] else "data/homologene_to_human.csv.gz"

library(homologene)
data(homologeneData)

human <- homologeneData[
  homologeneData$Taxonomy == 9606,
  c("HID", "Gene.Symbol")
]
names(human)[2L] <- "human_symbol"

sources <- homologeneData[
  homologeneData$Taxonomy %in% c(10090, 9544),
  c("HID", "Gene.Symbol", "Taxonomy")
]
names(sources)[2L] <- "source_symbol"

mapping <- merge(sources, human, by = "HID")
mapping$species <- ifelse(
  mapping$Taxonomy == 10090,
  "Mouse",
  "Rhesus Macaque"
)
mapping <- unique(mapping[, c("species", "source_symbol", "human_symbol")])
mapping <- mapping[order(mapping$species, mapping$source_symbol, mapping$human_symbol), ]

connection <- gzfile(output_path, open = "wt")
write.csv(mapping, connection, row.names = FALSE)
close(connection)
