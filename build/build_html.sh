#!/bin/bash
set -ex

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

conda activate d2l-en-build
rm -rf build/_build/
make html
