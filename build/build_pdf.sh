#!/bin/bash
set -e

conda activate d2l-en-build

make pdf
cp build/_build/latex/d2l-en.pdf build/_build/html/
