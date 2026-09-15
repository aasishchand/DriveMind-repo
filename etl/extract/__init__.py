"""Extraction stage: pull raw data into memory as DataFrames.

Extractors do *not* validate or reshape. They read whatever the source gives
(IDD annotation files, CARLA scenario logs) and hand back a raw DataFrame for
``load_bronze`` to persist verbatim.
"""
