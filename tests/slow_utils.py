import importlib.util
import os

import pytest


def require_slow_model_tests(*modules: str) -> None:
    if os.getenv("GIST_RUN_SLOW_MODEL_TESTS") != "1":
        pytest.skip("set GIST_RUN_SLOW_MODEL_TESTS=1 to run slow model tests")
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        pytest.skip(f"missing optional dependencies: {', '.join(missing)}")
