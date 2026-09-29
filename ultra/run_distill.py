"""Deprecated alias of ``ultra/run.py``, which now handles the student tasks as well."""

import os
import runpy

if __name__ == "__main__":
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py"), run_name="__main__")
