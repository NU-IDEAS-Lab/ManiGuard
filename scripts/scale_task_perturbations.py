"""CLI wrapper for building published perturbation task-set trees."""

from __future__ import annotations

import os
import sys

from maniguard.data.perturbation_scaling import main

if __name__ == "__main__":
    exit_code = int(main())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
