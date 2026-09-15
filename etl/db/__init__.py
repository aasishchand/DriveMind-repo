"""PostgreSQL metadata layer: DDL, connection seam, and manifest helpers.

``schema.sql``    - table definitions (scenario_runs, experiments, data_manifest)
``manifest.py``   - build + persist data_manifest rows (local file for now)
``connection.py`` - single place a real driver will be plugged in (pending)
"""
