import duckdb

base_str_estimator = """
perm_error FLOAT,
    acc_label FLOAT, 
    roc_label FLOAT,
    acc_concept FLOAT, 
    roc_concept FLOAT,
    ois_concept FLOAT,
    nis_concept FLOAT,
    time FLOAT,
    seed INTEGER,
    performed TIMESTAMP,
    PRIMARY KEY (
            method, 
            alpha, 
            N,
            seed
        )
"""


base_str_cbm = """
acc_label FLOAT, 
    roc_label FLOAT,
    acc_concept FLOAT, 
    roc_concept FLOAT,
    ois_concept FLOAT,
    nis_concept FLOAT,
    time FLOAT,
    seed INTEGER,
    performed TIMESTAMP,
    PRIMARY KEY (
        N, 
        seed
    )
"""


# We create a db per model for ease of writing
conn = duckdb.connect("checkpoints/experiment_citris.duckdb")
conn.execute(f"""
CREATE TABLE experiments (
    method VARCHAR, 
    alpha FLOAT,
    N INTEGER, 
    {base_str_estimator}
    )
""")

conn = duckdb.connect("checkpoints/experiment_ivae.duckdb")
conn.execute(f"""
CREATE TABLE experiments (
    method VARCHAR, 
    alpha FLOAT,
    N INTEGER, 
    {base_str_estimator}
    )
""")

conn = duckdb.connect("checkpoints/experiment_cbm.duckdb")
conn.execute(f"""
CREATE TABLE experiments (
    N INTEGER,
    {base_str_cbm}
    )
""")

conn = duckdb.connect("checkpoints/experiment_cem.duckdb")
conn.execute(f"""
CREATE TABLE experiments (
    N INTEGER,
    {base_str_cbm}
    )
""")

conn = duckdb.connect("checkpoints/experiment_cbm_ar.duckdb")
conn.execute(f"""
CREATE TABLE experiments(
    N INTEGER,
    {base_str_cbm}
    )
""")


