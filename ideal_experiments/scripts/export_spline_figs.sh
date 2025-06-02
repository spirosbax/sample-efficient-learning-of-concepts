#!/bin/bash

# Bash script that exports the required figures from the spline experiments into the
# LaTeX document. It consists of 2 parts, a curated list for the main text, and 
# a random selection

# Assign arguments to variables
SOURCE_DIR=./figs/spline
DEST_DIR=./../../crl-writing/neurips/figs/spline
CURATED_LIST=(
    "alpha/perm_error_match/d_variables=100_entanglement=0_miss_well=true_n_total=1250_wide.pdf"
    "alpha/perm_error_match/d_variables=100_entanglement=0_miss_well=false_n_total=1250_wide.pdf"
    "d_variables/perm_error_match/alpha=0.01_entanglement=0_miss_well=true_n_total=1250_wide.pdf"
    "d_variables/perm_error_match/alpha=0.01_entanglement=0_miss_well=false_n_total=1250_wide.pdf"
    "entanglement/perm_error_match/d_variables=60_alpha=0.01_miss_well=true_n_total=1250_wide.pdf"
    "entanglement/perm_error_match/d_variables=60_alpha=0.01_miss_well=false_n_total=1250_wide.pdf"
    "n_total/perm_error_match/d_variables=60_alpha=0.01_entanglement=0_miss_well=true_wide.pdf"
    "n_total/perm_error_match/d_variables=60_alpha=0.01_entanglement=0_miss_well=false_wide.pdf"
)

# Copy curated figures
echo "Copying curated figures..."
for ((i=0; i<${#CURATED_LIST[@]}; i++)); do
    echo "Processing figure ${CURATED_LIST[i]}"

    full_path="$SOURCE_DIR/${CURATED_LIST[i]}"
    cp "$full_path" "$DEST_DIR/figure_$i.pdf"
done


# Copy legend
full_path="$SOURCE_DIR/legend.pdf"
cp "$full_path" "$DEST_DIR/legend.pdf"

FIGURE_LIST_ALPHA=(
    "d_variables=20_entanglement=0_miss_well=true_n_total=1250.pdf"
    "d_variables=20_entanglement=0_miss_well=false_n_total=1250.pdf"
    "d_variables=60_entanglement=0_miss_well=true_n_total=1250.pdf"
    "d_variables=60_entanglement=0_miss_well=false_n_total=1250.pdf"
    "d_variables=100_entanglement=0_miss_well=true_n_total=1250.pdf"
    "d_variables=100_entanglement=0_miss_well=false_n_total=1250.pdf"
)
FIGURE_LIST_DIM=(
    "alpha=0.001_entanglement=0_miss_well=true_n_total=1250.pdf"
    "alpha=0.001_entanglement=0_miss_well=false_n_total=1250.pdf"
    "alpha=0.01_entanglement=0_miss_well=true_n_total=1250.pdf"
    "alpha=0.01_entanglement=0_miss_well=false_n_total=1250.pdf"
    "alpha=0.1_entanglement=0_miss_well=true_n_total=1250.pdf"
    "alpha=0.1_entanglement=0_miss_well=false_n_total=1250.pdf"
)
FIGURE_LIST_ENTANGLEMENT=(
    "d_variables=60_alpha=0.001_miss_well=true_n_total=1250.pdf"
    "d_variables=60_alpha=0.001_miss_well=false_n_total=1250.pdf"
    "d_variables=60_alpha=0.01_miss_well=true_n_total=1250.pdf"
    "d_variables=60_alpha=0.01_miss_well=false_n_total=1250.pdf"
    "d_variables=60_alpha=0.1_miss_well=true_n_total=1250.pdf"
    "d_variables=60_alpha=0.1_miss_well=false_n_total=1250.pdf"
)
FIGURE_LIST_N_TOTAL=(
    "d_variables=60_alpha=0.001_entanglement=0_miss_well=true.pdf"
    "d_variables=60_alpha=0.001_entanglement=0_miss_well=false.pdf"
    "d_variables=60_alpha=0.01_entanglement=0_miss_well=true.pdf"
    "d_variables=60_alpha=0.01_entanglement=0_miss_well=false.pdf"
    "d_variables=60_alpha=0.1_entanglement=0_miss_well=true.pdf"
    "d_variables=60_alpha=0.1_entanglement=0_miss_well=false.pdf"
)


echo "Finished copying figures to $DEST_DIR"

# Copy curated figures
echo "Copying regularization figures..."
for ((i=0; i<${#FIGURE_LIST_ALPHA[@]}; i++)); do
    echo "Processing figure ${FIGURE_LIST_ALPHA[i]}"

    cp "$SOURCE_DIR/alpha/perm_error_match/${FIGURE_LIST_ALPHA[i]}" "$DEST_DIR/perm_error/alpha_$i.pdf"
    cp "$SOURCE_DIR/alpha/r2_match/${FIGURE_LIST_ALPHA[i]}" "$DEST_DIR/r2/alpha_$i.pdf"
    cp "$SOURCE_DIR/alpha/time_match/${FIGURE_LIST_ALPHA[i]}" "$DEST_DIR/time/alpha_$i.pdf"
done

echo "Copying dim figures..."
for ((i=0; i<${#FIGURE_LIST_DIM[@]}; i++)); do
    echo "Processing figure ${FIGURE_LIST_DIM[i]}"

    cp "$SOURCE_DIR/d_variables/perm_error_match/${FIGURE_LIST_DIM[i]}" "$DEST_DIR/perm_error/dim_$i.pdf"
    cp "$SOURCE_DIR/d_variables/r2_match/${FIGURE_LIST_DIM[i]}" "$DEST_DIR/r2/dim_$i.pdf"
    cp "$SOURCE_DIR/d_variables/time_match/${FIGURE_LIST_DIM[i]}" "$DEST_DIR/time/dim_$i.pdf"
done

echo "Copying entanglement figures..."
for ((i=0; i<${#FIGURE_LIST_ENTANGLEMENT[@]}; i++)); do
    echo "Processing figure ${FIGURE_LIST_ENTANGLEMENT[i]}"

    cp "$SOURCE_DIR/entanglement/perm_error_match/${FIGURE_LIST_ENTANGLEMENT[i]}" "$DEST_DIR/perm_error/entanglement_$i.pdf"
    cp "$SOURCE_DIR/entanglement/r2_match/${FIGURE_LIST_ENTANGLEMENT[i]}" "$DEST_DIR/r2/entanglement_$i.pdf"
    cp "$SOURCE_DIR/entanglement/time_match/${FIGURE_LIST_ENTANGLEMENT[i]}" "$DEST_DIR/time/entanglement_$i.pdf"
done

echo "Copying n_total figures..."
for ((i=0; i<${#FIGURE_LIST_N_TOTAL[@]}; i++)); do
    echo "Processing figure ${FIGURE_LIST_N_TOTAL[i]}"

    cp "$SOURCE_DIR/n_total/perm_error_match/${FIGURE_LIST_N_TOTAL[i]}" "$DEST_DIR/perm_error/n_total_$i.pdf"
    cp "$SOURCE_DIR/n_total/r2_match/${FIGURE_LIST_N_TOTAL[i]}" "$DEST_DIR/r2/n_total_$i.pdf"
    cp "$SOURCE_DIR/n_total/time_match/${FIGURE_LIST_N_TOTAL[i]}" "$DEST_DIR/time/n_total_$i.pdf"
done

echo "Finished copying figures to $DEST_DIR"