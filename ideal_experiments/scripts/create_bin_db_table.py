import duckdb

conn = duckdb.connect("data/experiments_binary_ablation.duckdb")
conn.execute("""
CREATE TABLE experiments_features (
    regularizer VARCHAR, 
    n_total INTEGER,
    test_frac FLOAT,
    d_variables INTEGER,
    p_features INTEGER,
    entanglement FLOAT,
    alpha FLOAT,
    seed INTEGER,
    perm_error_match FLOAT,
    acc_label FLOAT,
    roc_label FLOAT, 
    acc_concept FLOAT,
    roc_concept FLOAT,
    ois_concept FLOAT, 
    nis_concept FLoat,
    time FLOAT,
    performed TIMESTAMP,
    PRIMARY KEY (
            regularizer, 
            n_total, 
            test_frac,
            d_variables,
            p_features, 
            entanglement, 
            alpha, 
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
    seed INTEGER,
    perm_error_match FLOAT,
    acc_label FLOAT,
    roc_label FLOAT, 
    acc_concept FLOAT,
    roc_concept FLOAT,
    ois_concept FLOAT, 
    nis_concept FLoat,
    time FLOAT,
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
    seed INTEGER,
    perm_error_match FLOAT,
    acc_label FLOAT,
    roc_label FLOAT, 
    acc_concept FLOAT,
    roc_concept FLOAT,
    ois_concept FLOAT, 
    nis_concept FLoat,
    time FLOAT,
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
    seed INTEGER,
    perm_error_match FLOAT,
    acc_label FLOAT,
    roc_label FLOAT, 
    acc_concept FLOAT,
    roc_concept FLOAT,
    ois_concept FLOAT, 
    nis_concept FLoat,
    time FLOAT,
    performed TIMESTAMP,
    PRIMARY KEY (
            regularizer, 
            n_total, 
            test_frac,
            d_variables,
            entanglement, 
            alpha, 
            seed
            )
    )
""")