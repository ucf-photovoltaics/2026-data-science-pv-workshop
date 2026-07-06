"""Thin wrapper so users can run `python run_pipeline.py` instead of `python -m rdstemplate`.

All logic is in the CLI entry point — this file just calls it.
"""

from rdstemplate.__main__ import main

if __name__ == "__main__":
    main()
