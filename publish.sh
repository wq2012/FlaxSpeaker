#!/bin/bash
set -o errexit

# This script requires these tools:
pip install setuptools wheel twine

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd ${PROJECT_PATH}

# clean up
rm -rf build
rm -rf dist
rm -rf flaxspeaker.egg-info

# build and upload
python setup.py sdist bdist_wheel
python -m twine upload dist/* --verbose

# clean up again
rm -rf build
rm -rf dist
rm -rf flaxspeaker.egg-info

popd
