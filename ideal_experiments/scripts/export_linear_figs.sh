#!/bin/bash

# Bash script that exports the required figures from the linear experiments into the
# LaTeX document. 

# Assign arguments to variables
SOURCE_DIR=./figs/linear
DEST_DIR=./../../crl-writing/icml/figs/linear
CURATED_LIST=(
    "alpha/perm_error_match/d_variables=100_entanglement=0_miss_well=true_n_total=1250.pdf"
    "alpha/perm_error_match/d_variables=100_entanglement=0_miss_well=false_n_total=1250.pdf"
    "d_variables/perm_error_match/alpha=0.05_entanglement=0_miss_well=true_n_total=1250.pdf"
    "d_variables/perm_error_match/alpha=0.05_entanglement=0_miss_well=false_n_total=1250.pdf"
    "entanglement/perm_error_match/d_variables=100_alpha=0.05_n_total=1250_miss_well=true.pdf"
    "entanglement/perm_error_match/d_variables=100_alpha=0.05_n_total=1250_miss_well=false.pdf"
    "n_total/perm_error_match/d_variables=100_alpha=0.05_entanglement=0_miss_well=true.pdf"
    "n_total/perm_error_match/d_variables=100_alpha=0.05_entanglement=0_miss_well=false.pdf"
)
RANDOM_COUNT=5

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

echo "Finished copying figures to $DEST_DIR"

# TODO: write this
# Copy random figures
echo "Selecting and copying random figures..."
# Find all image files in the source directory (adjust extensions as needed)
find "$SOURCE_DIR/d_variables/perm_error_match" -type f -name "*.pdf" > /tmp/all_figures_dim.txt
find "$SOURCE_DIR/alpha/perm_error_match" -type f -name "*.pdf" > /tmp/all_figures_alpha.txt
find "$SOURCE_DIR/entanglement/perm_error_match" -type f -name "*.pdf" > /tmp/all_figures_entanglement.txt
find "$SOURCE_DIR/n_total/perm_error_match" -type f -name "*.pdf" > /tmp/all_figures_n_total.txt

# # Randomly select figures
idx=0
# sort -R |tail -$N |while read file; do
sort -R /tmp/all_figures_dim.txt | tail -"$RANDOM_COUNT"| while read random_figure; do
    if [ -f "$random_figure" ]; then
        cp "$random_figure" "$DEST_DIR/random_fig_dim_$idx.pdf"
        echo "Copied random figure: $(basename "$random_figure")"
        ((idx=idx+1))
    fi
done
sort -R /tmp/all_figures_alpha.txt | tail -"$RANDOM_COUNT"| while read random_figure; do
    if [ -f "$random_figure" ]; then
        cp "$random_figure" "$DEST_DIR/random_fig_alpha_$idx.pdf"
        echo "Copied random figure: $(basename "$random_figure")"
        ((idx=idx+1))
    fi
done
sort -R /tmp/all_figures_entanglement.txt | tail -"$RANDOM_COUNT"| while read random_figure; do
    if [ -f "$random_figure" ]; then
        cp "$random_figure" "$DEST_DIR/random_fig_entanglement_$idx.pdf"
        echo "Copied random figure: $(basename "$random_figure")"
        ((idx=idx+1))
    fi
done
sort -R /tmp/all_figures_n_total.txt | tail -"$RANDOM_COUNT"| while read random_figure; do
    if [ -f "$random_figure" ]; then
        cp "$random_figure" "$DEST_DIR/random_fig_n_total_$idx.pdf"
        echo "Copied random figure: $(basename "$random_figure")"
        ((idx=idx+1))
    fi
done

# # Clean up temporary files
rm /tmp/all_figures_dim.txt
rm /tmp/all_figures_alpha.txt
rm /tmp/all_figures_entanglement.txt
rm /tmp/all_figures_n_total.txt

echo "Finished copying figures to $DEST_DIR"