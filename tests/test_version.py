import re

import resona


def test_version_is_a_string() -> None:
    assert isinstance(resona.__version__, str)


def test_version_is_pep440_ish() -> None:
    # Loose check: at least MAJOR.MINOR, optionally more.
    assert re.match(r"^\d+\.\d+", resona.__version__)
