"""Per-model pipeline scripts for DSS5104 CA2.

Each pipeline runs one model across all three datasets (M4, M5, Traffic)
and can be executed independently:

    python pipelines/run_patchtst.py              # full run
    python pipelines/run_patchtst.py --smoke-test # quick validation

To run all models sequentially:

    python pipelines/run_all.py
    python pipelines/run_all.py --smoke-test
"""
