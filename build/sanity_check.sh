#!/bin/bash
set -e

# check no ipynb files
IPYNB=`find chapter* -type f -name "*.ipynb"`
if [[ ! -z "$IPYNB" ]]; then
    echo "ERROR: Find the following .ipynb files. "
    echo "$IPYNB"
    echo "ERROR: You should convert them into markdown files. "
    echo "ERROR: You can do it with by 'jupyter nbconvert --to markdown your.ipynb'"
    # exit -1
fi


# check no output
HAS_OUTPUT=0
for FNAME in `find . -type f -name "*.md"`; do
    OUTPUT=""
    IN_CODE=0
    IFS=''
    L=1
    while read LINE; do
        if [[ "$LINE" =~ ^\`\`\`.* ]]; then
            IN_CODE=$(($IN_CODE ^ 1))
        fi

        if [[ "$LINE" =~ ^\$\$.* ]] && ! [[ "$LINE" =~ .+\$\$$ ]]; then
            IN_CODE=$(($IN_CODE ^ 1))
        fi

        if [[ "$LINE" =~ ^\ \ \ .* ]] && [[ "$IN_CODE" == "0" ]]; then
            echo $FNAME $LINE
            OUTPUT=$(printf "$OUTPUT\nL$L: $LINE")
        fi
        L=$((L+1))
    done <"$FNAME"

    if [[ ! -z $OUTPUT ]]; then
        echo "ERROR: $FNAME contains the following cell outputs:"
        echo $OUTPUT
        HAS_OUTPUT=1
    fi
done

if [[ "$HAS_OUTPUT" == "1" ]]; then
    echo "ERROR: At least one .md file has cell outputs."
    echo "ERROR: You should remove them."
    exit -1
fi
