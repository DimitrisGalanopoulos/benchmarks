#!/bin/bash

shopt -s -o pipefail
shopt -s extglob globstar dotglob nullglob 2>/dev/null
export GLOBIGNORE=.:..


for f in "$@"; do
    echo "$f"
    zip "$f".zip "$f"
    # gzip --keep "$f"
    # rm "$f"
done

