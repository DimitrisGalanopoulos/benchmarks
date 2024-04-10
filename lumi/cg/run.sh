#!/bin/bash

shopt -s -o pipefail
shopt -s extglob globstar dotglob nullglob 2>/dev/null
export GLOBIGNORE=.:..


script_dir="$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")"


for p in "${script_dir}"/*.out; do
    f="$(basename "$p")"
    "${script_dir}"/parse.awk "$p" > "${script_dir}/iterations/${f}" &
done

wait

