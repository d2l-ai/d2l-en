#!/bin/bash
conda activate d2l-en-build
pip uninstall -y gluonbook
pip uninstall -y d2l
conda deactivate
make clean
