import duckdb

conn = duckdb.connect("data/experiments.duckdb")
conn.execute("""
CREATE TABLE experiments_features (
    regularizer VARCHAR, 
    n_total INTEGER,
    test_frac FLOAT,
    d_variables INTEGER,
    p_features INTEGER,
    entanglement FLOAT,
    alpha FLOAT,
    miss_well BOOL,
    seed INTEGER,
    perm_error_match FLOAT,
    perm_error_linear FLOAT,
    perm_error_corr FLOAT,
    perm_error_spear FLOAT,
    mse_match FLOAT,
    mse_linear FLOAT,
    r2_match FLOAT,
    r2_linear FLOAT,
    time_match FLOAT,
    time_linear FLOAT,
    time_corr FLOAT,
    time_spear FLOAT,
    performed TIMESTAMP,
    PRIMARY KEY (
            regularizer, 
            n_total, 
            test_frac,
            d_variables,
            p_features, 
            entanglement, 
            alpha, 
            miss_well,
            seed
            )
    )
""")

conn.execute("""
CREATE TABLE experiments_kernel (
    regularizer VARCHAR, 
    n_total INTEGER,
    test_frac FLOAT,
    d_variables INTEGER,
    kernel VARCHAR,
    n_kernel INTEGER,
    entanglement FLOAT,
    alpha FLOAT,
    miss_well BOOL,
    seed INTEGER,
    perm_error_match FLOAT,
    perm_error_linear FLOAT,
    perm_error_corr FLOAT,
    perm_error_spear FLOAT,
    mse_match FLOAT,
    mse_linear FLOAT,
    r2_match FLOAT,
    r2_linear FLOAT,
    time_match FLOAT,
    time_linear FLOAT,
    time_corr FLOAT,
    time_spear FLOAT,
    performed TIMESTAMP,
    PRIMARY KEY (
            regularizer, 
            n_total, 
            test_frac,
            d_variables,
            kernel, 
            n_kernel,
            entanglement, 
            alpha, 
            miss_well,
            seed
            )
    )
""")

conn.execute("""
CREATE TABLE experiments_spline (
    regularizer VARCHAR, 
    n_total INTEGER,
    test_frac FLOAT,
    d_variables INTEGER,
    n_knots INTEGER,
    degree INTEGER,
    entanglement FLOAT,
    alpha FLOAT,
    miss_well BOOL,
    seed INTEGER,
    perm_error_match FLOAT,
    perm_error_linear FLOAT,
    perm_error_corr FLOAT,
    perm_error_spear FLOAT,
    mse_match FLOAT,
    mse_linear FLOAT,
    r2_match FLOAT,
    r2_linear FLOAT,
    time_match FLOAT,
    time_linear FLOAT,
    time_corr FLOAT,
    time_spear FLOAT,
    performed TIMESTAMP,
    PRIMARY KEY (
            regularizer, 
            n_total, 
            test_frac,
            d_variables,
            n_knots, 
            degree,
            entanglement, 
            alpha, 
            miss_well,
            seed
            )
    )
""")

conn.execute("""
CREATE TABLE experiments_linear (
    regularizer VARCHAR, 
    n_total INTEGER,
    test_frac FLOAT,
    d_variables INTEGER,
    entanglement FLOAT,
    alpha FLOAT,
    miss_well BOOL,
    seed INTEGER,
    perm_error_match FLOAT,
    perm_error_corr FLOAT,
    perm_error_spear FLOAT,
    mse_match FLOAT,
    r2_match FLOAT,
    time_match FLOAT,
    time_corr FLOAT,
    time_spear FLOAT,
    performed TIMESTAMP,
    PRIMARY KEY (
            regularizer, 
            n_total, 
            test_frac,
            d_variables,
            entanglement, 
            alpha, 
            miss_well,
            seed
            )
    )
""")