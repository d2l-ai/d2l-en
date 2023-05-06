#!/bin/bash

set -ex

# Used to capture status exit of build eval command
ss=0

REPO_NAME="$1"  # Eg. 'd2l-en'
TARGET_BRANCH="$2" # Eg. 'master' ; if PR raised to master
CLEAR_CACHE="${3:-false}"  # Eg. 'true' or 'false'

pip3 install .
mkdir _build

# Move sanity check outside
d2lbook build outputcheck tabcheck

# Move aws copy commands for cache restore outside
if [ "$CLEAR_CACHE" = "false" ]; then
  echo "Retrieving pytorch build cache"
  aws s3 sync s3://preview.d2l.ai/ci_cache/"$REPO_NAME"-"$TARGET_BRANCH"/_build/eval _build/eval --delete --quiet --exclude 'data/*'
  echo "Retrieving pytorch slides cache"
  aws s3 sync s3://preview.d2l.ai/ci_cache/"$REPO_NAME"-"$TARGET_BRANCH"/_build/slides _build/slides --delete --quiet --exclude 'data/*'
fi

# Continue the script even if some notebooks in build fail to
# make sure that cache is copied to s3 for the successful notebooks
d2lbook build eval || ((ss=1))
d2lbook build slides --tab pytorch

# Move aws copy commands for cache store outside
echo "Upload pytorch build cache to s3"
aws s3 sync _build s3://preview.d2l.ai/ci_cache/"$REPO_NAME"-"$TARGET_BRANCH"/_build --acl public-read --quiet

# Exit with a non-zero status if evaluation failed
if [ "$ss" -ne 0 ]; then
  exit 1
fi
