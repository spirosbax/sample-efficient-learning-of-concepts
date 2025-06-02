#!/bin/bash

# Bash script that exports the required figures from the spline experiments into the
# LaTeX document. It consists of 2 parts, a curated list for the main text, and 
# a random selection

# Assign arguments to variables
SOURCE_DIR=./figs/binary/spline
DEST_DIR=./../../crl-writing/neurips/figs/binary/spline
FIGURE_LIST_ALPHA=(
    "d_variables=10_entanglement=0_n_total=1250.pdf"
    "d_variables=20_entanglement=0_n_total=1250.pdf"
    "d_variables=30_entanglement=0_n_total=1250.pdf"
)
FIGURE_LIST_DIM=(
    "alpha=0.001_entanglement=0_n_total=1250.pdf"
    "alpha=0.01_entanglement=0_n_total=1250.pdf"
    "alpha=0.1_entanglement=0_n_total=1250.pdf"
)
FIGURE_LIST_ENTANGLEMENT=(
    "d_variables=10_alpha=0.1_n_total=1250.pdf"
    "d_variables=10_alpha=0.01_n_total=1250.pdf"
    "d_variables=10_alpha=0.001_n_total=1250.pdf"
)
FIGURE_LIST_N_TOTAL=(
    "d_variables=10_alpha=0.1_entanglement=0.pdf"
    "d_variables=10_alpha=0.01_entanglement=0.pdf"
    "d_variables=10_alpha=0.001_entanglement=0.pdf"
)


# Copy curated figures
echo "Copying regularization figures..."
for ((i=0; i<${#FIGURE_LIST_ALPHA[@]}; i++)); do
    echo "Processing figure ${FIGURE_LIST_ALPHA[i]}"

    cp "$SOURCE_DIR/alpha/acc_label/${FIGURE_LIST_ALPHA[i]}" "$DEST_DIR/acc_label/alpha_$i.pdf"
    cp "$SOURCE_DIR/alpha/acc_concept/${FIGURE_LIST_ALPHA[i]}" "$DEST_DIR/acc_concept/alpha_$i.pdf"
    cp "$SOURCE_DIR/alpha/ois_concept/${FIGURE_LIST_ALPHA[i]}" "$DEST_DIR/ois_score/alpha_$i.pdf"
done

echo "Copying dim figures..."
for ((i=0; i<${#FIGURE_LIST_DIM[@]}; i++)); do
    echo "Processing figure ${FIGURE_LIST_DIM[i]}"

    cp "$SOURCE_DIR/d_variables/acc_label/${FIGURE_LIST_DIM[i]}" "$DEST_DIR/acc_label/dim_$i.pdf"
    cp "$SOURCE_DIR/d_variables/acc_concept/${FIGURE_LIST_DIM[i]}" "$DEST_DIR/acc_concept/dim_$i.pdf"
    cp "$SOURCE_DIR/d_variables/ois_concept/${FIGURE_LIST_DIM[i]}" "$DEST_DIR/ois_score/dim_$i.pdf"
done

echo "Copying entanglement figures..."
for ((i=0; i<${#FIGURE_LIST_ENTANGLEMENT[@]}; i++)); do
    echo "Processing figure ${FIGURE_LIST_ENTANGLEMENT[i]}"

    cp "$SOURCE_DIR/entanglement/acc_label/${FIGURE_LIST_ENTANGLEMENT[i]}" "$DEST_DIR/acc_label/entanglement_$i.pdf"
    cp "$SOURCE_DIR/entanglement/acc_concept/${FIGURE_LIST_ENTANGLEMENT[i]}" "$DEST_DIR/acc_concept/entanglement_$i.pdf"
    cp "$SOURCE_DIR/entanglement/ois_concept/${FIGURE_LIST_ENTANGLEMENT[i]}" "$DEST_DIR/ois_score/entanglement_$i.pdf"
done

echo "Copying n_total figures..."
for ((i=0; i<${#FIGURE_LIST_N_TOTAL[@]}; i++)); do
    echo "Processing figure ${FIGURE_LIST_N_TOTAL[i]}"

    cp "$SOURCE_DIR/n_total/acc_label/${FIGURE_LIST_N_TOTAL[i]}" "$DEST_DIR/acc_label/n_total_$i.pdf"
    cp "$SOURCE_DIR/n_total/acc_concept/${FIGURE_LIST_N_TOTAL[i]}" "$DEST_DIR/acc_concept/n_total_$i.pdf"
    cp "$SOURCE_DIR/n_total/ois_concept/${FIGURE_LIST_N_TOTAL[i]}" "$DEST_DIR/ois_score/n_total_$i.pdf"
done

# Copy legend
full_path="$SOURCE_DIR/legend.pdf"
cp "$full_path" "$DEST_DIR/legend.pdf"



echo "Finished copying figures to $DEST_DIR"