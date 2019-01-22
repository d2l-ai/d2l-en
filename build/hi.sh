#!/bin/bash
conda activate d2l-en-build
pip uninstall gluonbook
pip uninstall d2l
conda deactivate
make clean
