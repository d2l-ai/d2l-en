#!/bin/bash
set -e

conda env update -f build/env.yml
conda activate d2l-en-build

pip list

rm -rf build/_build/

make html EVAL=0
