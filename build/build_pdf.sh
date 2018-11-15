#!/bin/bash
set -e

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
conda activate d2l-en-build

make pdf
cp build/_build/latex/d2l-en.pdf build/_build/html/
