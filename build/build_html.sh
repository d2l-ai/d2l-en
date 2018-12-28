#!/bin/bash
set -ex

conda activate d2l-en-build
rm -rf build/_build/
make html
