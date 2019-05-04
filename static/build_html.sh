#!/bin/bash

rm -rf _build/rst _build/html
d2lbook build rst
cp static/frontpage/frontpage.html _build/rst/
d2lbook build html
cp static/frontpage/_images/* _build/html/_images/

for fn in `find _build/html/_images/ -iname '*.svg' `; do
    if [[ $fn == *'qr_'* ]] ; then # || [[ $fn == *'output_'* ]]
        continue
    fi
    # rsvg-convert installed on ubuntu changes unit from px to pt, so evening no
    # change of the size makes the svg larger...
    rsvg-convert -z 1 -f svg -o tmp.svg $fn
    mv tmp.svg $fn
done
