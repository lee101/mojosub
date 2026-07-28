import os
import tempfile

import pytest

from mojosub import cache


@pytest.fixture(autouse=True, scope="session")
def isolated_cache():
    """Tests get their own compile cache.

    Verdicts (verified / retired) are persisted now, so a test that stubs a
    slow variant would otherwise write "this function is slower than CPython"
    into the developer's real cache and every later run - in any project on the
    machine - would believe it. Isolation costs a few real compiles per suite.
    """
    original = cache.DEFAULT_ROOT
    with tempfile.TemporaryDirectory(prefix="mojosub-tests-") as tmp:
        cache.DEFAULT_ROOT = tmp
        os.environ["MOJOSUB_CACHE"] = tmp
        yield tmp
    cache.DEFAULT_ROOT = original
