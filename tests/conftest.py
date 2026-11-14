import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RAGBLAME_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow: set RAGBLAME_SLOW=1 to run with real models")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
