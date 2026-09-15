"""DriveMind ETL package.

Member 1 role: Data Engineering & ETL.

Pipeline layers (medallion architecture):

    bronze  -> raw, as-ingested data (no schema guarantees)
    silver  -> cleaned, validated, conformed to a common schema
    gold    -> ML / risk-engine feature tables with train/val/test split

Two data domains flow through these layers:

    idd             -> India Driving Dataset static labelled images (perception)
    carla_telemetry -> per-scenario CARLA simulation logs (trajectory / risk)
"""

__version__ = "0.1.0"
