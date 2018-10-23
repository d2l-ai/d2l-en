#!/bin/bash
set -e

# Check no ipynb files.
IPYNB=`find chapter* -type f -name "*.ipynb"`
if [[ ! -z "$IPYNB" ]]; then
    echo "ERROR: Find the following .ipynb files. "
    echo "$IPYNB"
    echo "ERROR: You should convert them into markdown files. "
    echo "ERROR: You can do it with by 'jupyter nbconvert --to markdown your.ipynb'"
    exit -1
fi


# Check no output.
HAS_OUTPUT=0
for FNAME in `find chapter* -type f -name "*.md"`; do
    OUTPUT=""
    IN_CODE=0
	IN_JSON_OUTPUT=0
    IFS=''
    L=1
    while read LINE; do
        if [[ "$LINE" =~ ^\`\`\`\{\.json\ \.output ]]; then
			IN_JSON_OUTPUT=1
        elif [[ "$LINE" =~ ^\`\`\`.* ]]; then
			if [[ "$IN_JSON_OUTPUT" == "1" ]]; then
				IN_JSON_OUTPUT=0
			else
				IN_CODE=$(($IN_CODE ^ 1))
			fi
        fi

        # Skip one-line math.
        if ! [[ "$LINE" =~ ^\$\$.+ ]] || ! [[ "$LINE" =~ .+\$\$$ ]]; then
        # Math also allows multiple space at the beginning of a line.
            if [[ "$LINE" =~ ^\$\$.* ]] || [[ "$LINE" =~ .*\$\$$ ]]; then
                IN_CODE=$(($IN_CODE ^ 1))
            fi
        fi


        if [[ "$LINE" =~ ^\ \ \ .* ]] && [[ "$IN_CODE" == "0" ]] || [[ "$IN_JSON_OUTPUT" == "1" ]]; then
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
