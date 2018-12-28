#!/bin/bash
set -ex

conda activate d2l-en-build

DIR=../d2l-en-notebooks
build/utils/notebooks_no_output.sh . ${DIR} d2l-ai/notebooks discuss.mxnet.io
rm -f ${DIR}/*.ipynb
cp environment.yml ${DIR}/
rm -rf ${DIR}/img/qr_* ${DIR}/img/frontpage

build/utils/upload_github.sh ${DIR} d2l-ai/notebooks
