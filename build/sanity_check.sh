#!/bin/bash
set -e
IPYNB=`find . -type f -name "*.ipynb"`
if [[ ! -z "$IPYNB" ]]; then
    echo "ERROR: Find the following .ipynb files. You should convert them into markdown files"
    echo $IPYNB
    exit -1
fi
