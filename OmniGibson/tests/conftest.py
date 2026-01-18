import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
OG_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if OG_ROOT not in sys.path:
    sys.path.insert(0, OG_ROOT)

import omnigibson as og


def pytest_unconfigure(config):
    og.shutdown()
