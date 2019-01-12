#!/bin/bash

DIR="build/_build/html"

for f in $DIR/chapter*/*html; do
	sed -i s/Ⓐ/\ /g $f
done
