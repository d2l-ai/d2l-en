#!/bin/bash
set -ex

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

conda activate d2l-en-build
make pdf
cp build/_build/latex/d2l-en.pdf build/_build/html/

[ -e build/_build/latex/d2l-en.aux ] && rm build/_build/latex/d2l-en.aux
[ -e build/_build/latex/d2l-en.idx ] && rm build/_build/latex/d2l-en.idx
