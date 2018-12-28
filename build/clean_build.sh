#!/bin/bash
set -ex

cd build

[ -e data-bak ] && rm -rf data-bak

# Clean build/chapter*/*ipynb and build/chapter*/*md that are no longer needed.
for ch in chapter*; do
    if ! [ -e "../$ch" ]; then
        rm -rf $ch
    else
        shopt -s nullglob
        for f in $ch/*.md $ch/*.ipynb; do
            base=$(basename $f)
            md=${base%%.*}.md
            if ! [ -e "../$ch/$md" ]; then
                rm $f
            fi
        done
    fi
done

# Clean images that are no longer needed.
shopt -s nullglob
for f in img/*.svg img/*.jpg img/*.png; do
    if ! [ -e "../$f" ]; then
        rm $f
    fi
done
