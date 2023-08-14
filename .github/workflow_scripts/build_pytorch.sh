#!/bin/bash

set -ex

# Used to capture status exit of build eval command
ss=0

REPO_NAME="$1"  # Eg. 'd2l-en'
TARGET_BRANCH="$2" # Eg. 'master' ; if PR raised to master
CACHE_DIR="$3"  # Eg. 'ci_cache_pr' or 'ci_cache_push'

pip3 install .
mkdir _build

source $(dirname "$0")/utils.sh

# Move sanity check outside
d2lbook build outputcheck tabcheck

# Move aws copy commands for cache restore outside
if [ "$DISABLE_CACHE" = "false" ]; then
  echo "Retrieving pytorch build cache from "$CACHE_DIR""
  measure_command_time "aws s3 sync s3://preview.d2l.ai/"$CACHE_DIR"/"$REPO_NAME"-"$TARGET_BRANCH"/_build/eval _build/eval --delete --quiet --exclude 'data/*'"
  echo "Retrieving pytorch slides cache from "$CACHE_DIR""
  aws s3 sync s3://preview.d2l.ai/"$CACHE_DIR"/"$REPO_NAME"-"$TARGET_BRANCH"/_build/slides _build/slides --delete --quiet --exclude 'data/*'
fi

# Continue the script even if some notebooks in build fail to
# make sure that cache is copied to s3 for the successful notebooks
d2lbook build eval || ((ss=1))
d2lbook build slides --tab pytorch

# Move aws copy commands for cache store outside
echo "Upload pytorch build cache to s3"
measure_command_time "aws s3 sync _build s3://preview.d2l.ai/"$CACHE_DIR"/"$REPO_NAME"-"$TARGET_BRANCH"/_build --acl public-read --quiet --exclude 'eval*/data/*'"

# Exit with a non-zero status if evaluation failed
if [ "$ss" -ne 0 ]; then
  exit 1
fi
