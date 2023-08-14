#!/bin/bash

# Move all scripts related to html here!

set -ex

REPO_NAME="$1"  # Eg. 'd2l-en'
TARGET_BRANCH="$2" # Eg. 'master' ; if PR raised to master
JOB_NAME="$3" # Eg. 'd2l-en/master' or 'd2l-en/PR-2453/21be1a4'
LANG="$4" # Eg. 'en','zh' etc.
CACHE_DIR="$5"  # Eg. 'ci_cache_pr' or 'ci_cache_push'

pip3 install .
mkdir _build

source $(dirname "$0")/utils.sh

# Move aws copy commands for cache restore outside
measure_command_time "aws s3 sync s3://preview.d2l.ai/"$CACHE_DIR"/"$REPO_NAME"-"$TARGET_BRANCH"/_build _build --delete --quiet --exclude 'eval*/data/*'"

# Build D2L Website
./.github/workflow_scripts/build_html.sh $TARGET_BRANCH $JOB_NAME

# Build PDFs
d2lbook build pdf
d2lbook build pdf --tab mxnet


# Check if the JOB_NAME is either "$REPO_NAME/release" or "$REPO_NAME/classic"
if [[ "$JOB_NAME" == "$REPO_NAME/release" || "$JOB_NAME" == "$REPO_NAME/classic" ]]; then

  # Setup D2L Bot
  source $(dirname "$0")/setup_git.sh
  setup_git

  # Run d2lbook release deployment
  if [[ "$JOB_NAME" == *"/classic" ]]; then
    # Use classic s3 bucket for classic release
    LANG="classic"
  fi
  d2lbook build pkg
  d2lbook deploy html pdf pkg colab sagemaker slides --s3 "s3://${LANG}.d2l.ai/"

else
  # Run d2lbook preview deployment
  d2lbook deploy html pdf --s3 "s3://preview.d2l.ai/${JOB_NAME}/"
fi

# Move aws copy commands for cache store outside
measure_command_time "aws s3 sync _build s3://preview.d2l.ai/"$CACHE_DIR"/"$REPO_NAME"-"$TARGET_BRANCH"/_build --acl public-read --quiet --exclude 'eval*/data/*'"
