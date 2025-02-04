from itertools import product
import numpy as np

regularizer=["lasso", "group"] 
kernels=["rbf", "polynomial", "laplacian", "cosine"]

series_params = set()

## Collection of all the settings for plots, duplicates are handled by the set datastructure ##
# Dimension
d_variables=[5, 30, 60, 80, 100] 

alphas=[0.001, 0.01, 0.1]
entanglement=[0] 
spec=[False, True] # ["miss", "well"]
N_TOTAL = [1250]

dim_params = list(product(alphas, entanglement, spec, N_TOTAL))
series_params.update(
    product(
        regularizer, 
        d_variables,
        kernels,
        alphas,
        entanglement, 
        spec, 
        N_TOTAL
    )
)

# Regularizer param
alphas=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.] 

d_variables=[20, 60, 100] 
entanglement=[0] 
spec=[False, True] # ["miss", "well"]
N_TOTAL = [1250]

reg_params = list(product(d_variables, entanglement, spec, N_TOTAL))
series_params.update(
    product(
        regularizer, 
        d_variables,
        kernels,
        alphas,
        entanglement, 
        spec, 
        N_TOTAL
    )
)

# Correlated features
entanglement=[0., 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]

d_variables=[60] 
alphas=[0.001, 0.01, 0.1] 
spec=[False, True] # ["miss", "well"]
N_TOTAL = [1250]

entanglement_params = list(product(d_variables, alphas, spec, N_TOTAL))
series_params.update(
    product(
        regularizer, 
        d_variables,
        kernels,
        alphas,
        entanglement, 
        spec, 
        N_TOTAL
    )
)

# Nr data points
N_TOTAL = [65, 125, 1250, 2500, 5000]

d_variables=[60] 
alphas=[0.001, 0.01, 0.1] 
entanglement=[0]
spec=[False, True] # ["miss", "well"]

n_total_params = list(product(d_variables, alphas, entanglement, spec))
series_params.update(
    product(
        regularizer, 
        d_variables,
        kernels,
        alphas,
        entanglement, 
        spec, 
        N_TOTAL
    )
)