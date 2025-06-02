#!/bin/bash

# Bash script that exports the required figures from the spline experiments into the
# LaTeX document. It consists of 2 parts, a curated list for the main text, and 
# a random selection

# Assign arguments to variables
DEST_DIR=~/Documents/Research/projects/active/crl-for-concept-explanations/crl-writing/neurips/tables

TABLES=(
    "sample_table.tex" 
    "appendix_table_perm_error.tex"
    "appendix_table_r2.tex"
    "appendix_table_times.tex"
    "table_concept_acc_roc.tex"
    "table_label_acc_roc.tex"
    "table_nis_ois.tex"
    "tabel_bin_main.tex"
    "table_bin_0.tex"
    "table_bin_1.tex"
    "table_bin_2.tex"
    "table_bin_main.tex"
)

DATA_DIR=./tables
for ((i=0; i<${#TABLES[@]}; i++)); do
    echo "Processing table ${TABLES[i]}"

    full_path="$DATA_DIR/${TABLES[i]}"
    cp "$full_path" "$DEST_DIR/${TABLES[i]}"
done

FIG_DIR=./figs
FIG_DEST_DIR=~/Documents/Research/projects/active/crl-for-concept-explanations/crl-writing/neurips/figs
cp "$FIG_DIR/citris_times.pdf" "$FIG_DEST_DIR/citris/times_bin.pdf"
cp "$FIG_DIR/dms_times.pdf" "$FIG_DEST_DIR/dms-vae/times_bin.pdf"
cp "$FIG_DIR/times_legend_bin.pdf" "$FIG_DEST_DIR/times_legend_bin.pdf"

