#!/bin/bash

for fn in img/*; do
    if grep -qrl $fn */*.md; then
        continue
    fi
    echo "Remove $fn"
    rm $fn
done
