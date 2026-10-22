"""Setup shim for the FlaxSpeaker package.

All package metadata and the single runtime dependency list live in
``pyproject.toml`` (PEP 621). This file exists only so that legacy tooling that
invokes ``python setup.py`` keeps working. ``VERSION`` must match
``pyproject.toml`` and ``flaxspeaker.__version__`` (enforced by
``tests.py::test_version_consistency``).
"""

import setuptools

VERSION = "0.3.0"

setuptools.setup(version=VERSION)
