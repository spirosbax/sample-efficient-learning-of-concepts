#!/bin/bash

# Bash script that exports the required figures from the spline experiments into the
# LaTeX document. It consists of 2 parts, a curated list for the main text, and 
# a random selection

# Assign arguments to variables
DEST_DIR=~/Documents/Research/projects/active/crl-for-concept-explanations/crl-writing/icml/tables

TABLES=(
    "sample_table.tex" 
    "appendix_table_perm_error.tex"
    "appendix_table_r2.tex"
    "appendix_table_times.tex"
)

DATA_DIR=./tables
for ((i=0; i<${#TABLES[@]}; i++)); do
    echo "Processing table ${TABLES[i]}"

    full_path="$DATA_DIR/${TABLES[i]}"
    cp "$full_path" "$DEST_DIR/${TABLES[i]}"
done

# FIG_DIR=./cluster_checkpoints/figs
# cp "$FIG_DIR/times.pdf" "$DEST_DIR/dms-vae/times.pdf"

