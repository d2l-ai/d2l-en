#!/bin/bash
set -e
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
echo ${LD_LIBRARY_PATH}
echo ${PATH}

conda env update -f build/env.yml
conda activate d2l-en-build

pip list

rm -rf build/_build/

make html
