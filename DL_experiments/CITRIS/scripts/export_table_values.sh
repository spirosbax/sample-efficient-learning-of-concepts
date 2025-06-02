#!/bin/bash

# Bash script that exports the required figures from the spline experiments into the
# LaTeX document. It consists of 2 parts, a curated list for the main text, and 
# a random selection

# Assign arguments to variables
DEST_DIR=~/Documents/Research/projects/active/crl-for-concept-explanations/crl-writing/icml/tables
# TABLES=(
#     "table_perm_error_lasso_linear.tex"
#     "table_perm_error_lasso_spline.tex"
#     "table_perm_error_group_linear.tex"
#     "table_perm_error_group_spline.tex"

#     "table_perm_spear_lasso_linear.tex"
#     "table_perm_spear_lasso_spline.tex"
#     "table_perm_spear_group_linear.tex"
#     "table_perm_spear_group_spline.tex"

#     "table_mse_lasso_linear.tex"
#     "table_mse_lasso_spline.tex"
#     "table_mse_group_linear.tex"
#     "table_mse_group_spline.tex"

#     "table_r2_lasso_linear.tex"
#     "table_r2_lasso_spline.tex"
#     "table_r2_group_linear.tex"
#     "table_r2_group_spline.tex"
# )

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


# echo "Copying CITRISVAE tables"
# DATA_DIR=./cluster_checkpoints/CITRISVAE/CITRISVAE_32l_7b_32hid_causal3d/version_4
# for ((i=0; i<${#TABLES[@]}; i++)); do
#     echo "Processing table ${TABLES[i]}"

#     full_path="$DATA_DIR/tables/${TABLES[i]}"
#     cp "$full_path" "$DEST_DIR/citris_vae/${TABLES[i]}"
# done


# echo "Copying iVAE tables"
# DATA_DIR=./cluster_checkpoints/iVAE/iVAE_32l_7b_32hid_causal3d/version_3
# for ((i=0; i<${#TABLES[@]}; i++)); do
#     echo "Processing table ${TABLES[i]}"

#     full_path="$DATA_DIR/tables/${TABLES[i]}"
#     cp "$full_path" "$DEST_DIR/ivae/${TABLES[i]}"
# done