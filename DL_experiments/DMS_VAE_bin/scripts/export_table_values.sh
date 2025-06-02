#!/bin/bash

# Bash script that exports the required figures from the spline experiments into the
# LaTeX document. It consists of 2 parts, a curated list for the main text, and 
# a random selection

# Assign arguments to variables
DEST_DIR=~/Documents/Research/projects/active/crl-for-concept-explanations/crl-writing/icml/tables

TABLES=(
    "aggregate_mse.tex"
    "aggregate_perm_errors.tex"
    "aggregate_r2.tex"
    "aggregate_times.tex"
)

DATA_DIR=./cluster_checkpoints/tables
for ((i=0; i<${#TABLES[@]}; i++)); do
    echo "Processing table ${TABLES[i]}"

    full_path="$DATA_DIR/${TABLES[i]}"
    cp "$full_path" "$DEST_DIR/dms-vae/${TABLES[i]}"
done

FIG_DIR=./cluster_checkpoints/figs
cp "$FIG_DIR/times.pdf" "$DEST_DIR/dms-vae/times.pdf"
cp "$FIG_DIR/legend.pdf" "$DEST_DIR/dms-vae/legend.pdf"

cp "$FIG_DIR/times_no_two_stage.pdf" "$DEST_DIR/dms-vae/times_no_two_stage.pdf"
cp "$FIG_DIR/legend_no_two_stage.pdf" "$DEST_DIR/dms-vae/legend_no_two_stage.pdf"


