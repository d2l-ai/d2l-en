#!/bin/bash
set -e

# avoid putting data downloaded by scripts into the notebook package
mv build/data build/data-bak
make pkg
# backup build/data to avoid download the dataset each time and put the
rm -rf build/data
mv build/data-bak build/data

# For 1.0
cp build/_build/html/d2l-en.zip build/_build/html/d2l-en-1.0.zip
