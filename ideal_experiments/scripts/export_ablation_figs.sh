#!/bin/bash

# Bash script that exports the required figures from the spline experiments into the
# LaTeX document. It consists of 2 parts, a curated list for the main text, and 
# a random selection

# Assign arguments to variables
SOURCE_DIR=./figs/binary/ablation
DEST_DIR=./../../crl-writing/neurips/figs/ablation
FIGURE_LIST_DIM=(
    "alpha=0.001_n_total=2000_wide.pdf"
    "alpha=0.01_n_total=2000_wide.pdf"
    "alpha=0.1_n_total=2000_wide.pdf"
)
FIGURE_LIST_N_TOTAL=(
    "alpha=0.001_d_variables=20_wide.pdf"
    "alpha=0.01_d_variables=20_wide.pdf"
    "alpha=0.1_d_variables=20_wide.pdf"
)

echo "Copying dim figures..."
for ((i=0; i<${#FIGURE_LIST_DIM[@]}; i++)); do
    echo "Processing figure ${FIGURE_LIST_DIM[i]}"

    cp "$SOURCE_DIR/d_variables/acc_label/${FIGURE_LIST_DIM[i]}" "$DEST_DIR/acc_label/dim_$i.pdf"
    cp "$SOURCE_DIR/d_variables/acc_concept/${FIGURE_LIST_DIM[i]}" "$DEST_DIR/acc_concept/dim_$i.pdf"
    cp "$SOURCE_DIR/d_variables/ois_concept/${FIGURE_LIST_DIM[i]}" "$DEST_DIR/ois_score/dim_$i.pdf"
    cp "$SOURCE_DIR/d_variables/nis_concept/${FIGURE_LIST_DIM[i]}" "$DEST_DIR/nis_score/dim_$i.pdf"
    cp "$SOURCE_DIR/d_variables/time/${FIGURE_LIST_DIM[i]}" "$DEST_DIR/time/dim_$i.pdf"
done

echo "Copying n_total figures..."
for ((i=0; i<${#FIGURE_LIST_N_TOTAL[@]}; i++)); do
    echo "Processing figure ${FIGURE_LIST_N_TOTAL[i]}"

    cp "$SOURCE_DIR/N/acc_label/${FIGURE_LIST_N_TOTAL[i]}" "$DEST_DIR/acc_label/n_total_$i.pdf"
    cp "$SOURCE_DIR/N/acc_concept/${FIGURE_LIST_N_TOTAL[i]}" "$DEST_DIR/acc_concept/n_total_$i.pdf"
    cp "$SOURCE_DIR/N/ois_concept/${FIGURE_LIST_N_TOTAL[i]}" "$DEST_DIR/ois_score/n_total_$i.pdf"
    cp "$SOURCE_DIR/N/nis_concept/${FIGURE_LIST_N_TOTAL[i]}" "$DEST_DIR/nis_score/n_total_$i.pdf"
    cp "$SOURCE_DIR/N/time/${FIGURE_LIST_N_TOTAL[i]}" "$DEST_DIR/time/n_total_$i.pdf"
done

# Copy legend
full_path="$SOURCE_DIR/legend.pdf"
cp "$full_path" "$DEST_DIR/legend.pdf"



echo "Finished copying figures to $DEST_DIR"