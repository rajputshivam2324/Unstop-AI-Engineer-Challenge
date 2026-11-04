#!/usr/bin/env python
"""Root runner (assignment deliverable 2).

    python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports

See `python run_integrity_check.py --help` and README.md for all flags.
"""

from integrity_engine.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
