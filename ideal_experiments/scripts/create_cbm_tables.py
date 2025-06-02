import duckdb

conn = duckdb.connect("data/experiments_cbm.duckdb")
conn.execute("""
CREATE TABLE experiments (
    N INTEGER,
    d_variables INTEGER,
    train_corr FLOAT,
    seed INTEGER,
    acc_label FLOAT,
    roc_label FLOAT, 
    acc_concept FLOAT,
    roc_concept FLOAT,
    ois_concept FLOAT, 
    nis_concept FLoat,
    time FLOAT,
    performed TIMESTAMP,
    PRIMARY KEY (
            N, 
            d_variables,
            train_corr, 
            seed
            )
    )
""")

conn = duckdb.connect("data/experiments_cem.duckdb")
conn.execute("""
CREATE TABLE experiments (
    N INTEGER,
    d_variables INTEGER,
    train_corr FLOAT,
    seed INTEGER,
    acc_label FLOAT,
    roc_label FLOAT, 
    acc_concept FLOAT,
    roc_concept FLOAT,
    ois_concept FLOAT, 
    nis_concept FLoat,
    time FLOAT,
    performed TIMESTAMP,
    PRIMARY KEY (
            N, 
            d_variables,
            train_corr, 
            seed
            )
    )
""")
conn = duckdb.connect("data/experiments_cbm_ar.duckdb")
conn.execute("""
CREATE TABLE experiments (
    N INTEGER,
    d_variables INTEGER,
    train_corr FLOAT,
    seed INTEGER,
    acc_label FLOAT,
    roc_label FLOAT, 
    acc_concept FLOAT,
    roc_concept FLOAT,
    ois_concept FLOAT, 
    nis_concept FLoat,
    time FLOAT,
    performed TIMESTAMP,
    PRIMARY KEY (
            N, 
            d_variables,
            train_corr, 
            seed
            )
    )
""")
