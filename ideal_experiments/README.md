# Toy Dataset experiments

This repository contains all the code to run the Toy Dataset experiments 

## Installing the experiment

We provide a YAML file to install the python environmet

```bash
conda env create -f concept-crl.yml
```

## Running the experiments

Before the experiments can be run, the database to store the results has to be created, which can be done by running
```bash
python -u scripts/create_db_table.py
```

The Toy Datasets are subdivided into 3 parts: the spline experiments, the random fourier experiments and the kernel experiments. Running all experiments can take up to 12 hours per feature set. 

### Spline Features

To run an individual experiment:
```bash
python -u experiment/spline_experiment.py \
    --seed [SEED] \
    --n_total [total_number_of_samples] \
    --test_frac [test_set_fraction] \
    --d_variables [number_of_dimensions] \
    --n_knots [number_of_knots_in_the_splines] \
    --degree [degree_of_the_splines] \
    --entanglement [correlation_between_the_variables] \
    --alpha [regularization_parameter] \
    --specification [miss_well_specified] \
    --regularizer [group/lasso/ridge]
```

We provide the settings used in the paper in the `experiments/spline_settings.py`, which can all be run using the `run_all_spline_parallel.py` command


### Random Fourier Features

To run an individual experiment:
```bash
python -u experiment/rff_experiment.py \
    --seed [SEED] \
    --n_total [total_number_of_samples] \
    --test_frac [test_set_fraction] \
    --d_variables [number_of_dimensions] \
    --p_features[number_of_n_components] \
    --entanglement [correlation_between_the_variables] \
    --alpha [regularization_parameter] \
    --specification [miss_well_specified] \
    --regularizer [group/lasso/ridge]
```

We provide the settings used in the paper in the `experiments/rff_settings.py`, which can all be run using the `run_all_rff_parallel.py` command

### Kernels

To run an individual experiment:
```bash
python -u experiment/kernel_experiment.py \
    --seed [SEED] \
    --n_total [total_number_of_samples] \
    --test_frac [test_set_fraction] \
    --d_variables [number_of_dimensions] \
    --kernel [rbf/laplacian/polynomial/cosine] \
    --entanglement [correlation_between_the_variables] \
    --alpha [regularization_parameter] \
    --specification [miss_well_specified] \
    --regularizer [group/lasso/ridge]
```

We provide the settings used in the paper in the `experiments/kernel_settings.py`, which can all be run using the `run_all_kernel_parallel.py` command