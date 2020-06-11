#!/bin/bash

pip install setuptools wheel
python setup.py bdist_wheel

rm -rf whl.html
for fn in dist/*.whl; do
    echo "<a href=\"$fn\">$fn</a><br>" >>whl.html
done

rm -rf _build/html/dist
cp -r dist _build/html/
cp whl.html _build/html/
