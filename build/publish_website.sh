#!/bin/bash
set -ex

conda activate d2l-en-build

build/utils/upload_doc_s3.sh build/_build/html s3://en.d2l.ai
