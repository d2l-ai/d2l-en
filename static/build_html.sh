#!/bin/bash

set -e

# Read arguments
BRANCH_NAME=$1
JOB_NAME=$2

STABLE_BASE_PATH="https://d2l.ai"
PREVIEW_BASE_PATH="http://preview.d2l.ai/$JOB_NAME"

# Generate Headers
if [[ "$BRANCH_NAME" == "release" ]]; then
    echo "Use release headers"
    alternate_text="Preview Version"
    alternate_base="${PREVIEW_BASE_PATH/release/master}"  # Substitute "release" with "master"
    current_base=$STABLE_BASE_PATH
else
    echo "Use ${JOB_NAME} headers"
    alternate_text="Stable Version"
    alternate_base=$STABLE_BASE_PATH
    current_base=$PREVIEW_BASE_PATH
fi


# Replace placeholders in ../config.ini
sed -i -e "s@###_ALTERNATE_VERSION_###@$alternate_text@g" ./config.ini
sed -i -e "s@###_ALTERNATE_VERSION_BASE_LINK_###@$alternate_base@g" ./config.ini
sed -i -e "s@###_CURRENT_VERSION_BASE_LINK_###@$current_base@g" ./config.ini


rm -rf _build/rst _build/html
d2lbook build rst --tab all
cp static/frontpage/frontpage.html _build/rst_all/
d2lbook build html --tab all
cp -r static/frontpage/_images/* _build/html/_images/

for fn in `find _build/html/_images/ -iname '*.svg' `; do
    if [[ $fn == *'qr_'* ]] ; then # || [[ $fn == *'output_'* ]]
        continue
    fi
    # rsvg-convert installed on ubuntu changes unit from px to pt, so evening no
    # change of the size makes the svg larger...
    rsvg-convert -z 1 -f svg -o tmp.svg $fn
    mv tmp.svg $fn
done

# Add SageMaker Studio Lab buttons
for f in _build/html/chapter*/*.html; do
    sed -i s/Open\ the\ notebook\ in\ Colab\<\\\/div\>\<\\\/div\>\<\\\/div\>\<\\\/h1\>/Open\ the\ notebook\ in\ Colab\<\\\/div\>\<\\\/div\>\<\\\/div\>\<a\ href=\"https:\\\/\\\/studiolab.sagemaker.aws\\\/import\\\/github\\\/d2l-ai\\\/d2l-pytorch-sagemaker-studio-lab\\\/blob\\\/main\\\/GettingStarted-D2L.ipynb\"\ onclick=\"captureOutboundLink\\\(\'https\:\\\/\\\/studiolab.sagemaker.aws\\\/import\\\/github\\\/d2l-ai\\\/d2l-pytorch-sagemaker-studio-lab\\\/blob\\\/main\\\/GettingStarted-D2L.ipynb\'\\\)\;\ return\ false\;\"\>\ \<button\ style=\"float\:right\",\ id=\"SageMaker\_Studio\_Lab\"\ class=\"mdl-button\ mdl-js-button\ mdl-button--primary\ mdl-js-ripple-effect\"\>\ \<i\ class=\"\ fas\ fa-external-link-alt\"\>\<\\\/i\>\ SageMaker\ Studio\ Lab\ \<\\\/button\>\<\\\/a\>\<div\ class=\"mdl-tooltip\"\ data-mdl-for=\"SageMaker\_Studio\_Lab\"\>\ Open\ the\ notebook\ in\ SageMaker\ Studio\ Lab\<\\\/div\>\<\\\/h1\>/g $f
done

