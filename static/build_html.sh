#!/bin/bash

rm -rf _build/rst _build/html
d2lbook build rst
cp static/frontpage/frontpage.html _build/rst/
d2lbook build html
cp static/frontpage/_images/* _build/html/_images/

for fn in `find _build/html/_images/ -iname '*.svg' `; do
    if [[ $fn == *'qr_'* ]]; then
        continue
    fi
    echo "Zoom in $fn by 1.3x"
    rsvg-convert -z 1.3 -f svg -o tmp.svg $fn
    mv tmp.svg $fn
done
