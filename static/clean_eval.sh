#!/bin/bash
# a temp fix because of merged pytorch codes
for file in _build/eval/*.ipynb _build/eval/*/*.ipynb; do
    if grep -q "origin_pos" "$file"; then
        continue
    else
        echo "remove $file"
        rm -f $file
    fi
done
