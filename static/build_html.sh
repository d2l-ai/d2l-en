#!/bin/bash

rm -rf _build/rst _build/html
d2lbook build rst
cp static/frontpage/frontpage.html _build/rst/
d2lbook build html
cp static/frontpage/_images/* _build/html/_images/

for fn in `find _build/html/_images/ -iname '*.svg' `; do
    if [[ $fn == *'qr_'* ]] || [[ $fn == *'output_'* ]]; then
        continue
    fi
    echo "Zoom in $fn by 1.15x"
    rsvg-convert -z 1.15 -f svg -o tmp.svg $fn
    mv tmp.svg $fn
done
