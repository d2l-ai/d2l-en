#!/bin/bash
set -ex

conda activate d2l-en-build
make pdf
cp build/_build/latex/d2l-en.pdf build/_build/html/

[ -e build/_build/latex/d2l-en.aux ] && rm build/_build/latex/d2l-en.aux
[ -e build/_build/latex/d2l-en.idx ] && rm build/_build/latex/d2l-en.idx
