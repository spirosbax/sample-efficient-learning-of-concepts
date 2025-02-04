#!/bin/bash

# Bash script that exports the required figures from the spline experiments into the
# LaTeX document. 

# Assign arguments to variables
DEST_DIR=./../../../crl-writing/icml/tables

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
    cp "$full_path" "$DEST_DIR/citris/${TABLES[i]}"
done


FIG_DIR=./cluster_checkpoints/figs
cp "$FIG_DIR/times.pdf" "$DEST_DIR/citris/times.pdf"
cp "$FIG_DIR/legend.pdf" "$DEST_DIR/citris/legend.pdf"
