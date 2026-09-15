"""Load stage: persist DataFrames to a layer and register them in the manifest.

    load_bronze -> write raw extractor output verbatim to data/bronze/
    load_silver -> validate cleaned/normalised output, write to data/silver/
    load_gold   -> build feature tables + train/val/test split, write to data/gold/

All three append a row to the local manifest (data/_manifest/manifest.jsonl) and
can emit SQL INSERTs with --push until a PostgreSQL driver is approved.
"""
