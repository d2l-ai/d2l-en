#!/bin/bash
set -ex

git submodule update --init
conda env update -f build/env.yml

conda activate d2l-en-build
pip list
