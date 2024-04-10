#!/bin/bash

shopt -s -o pipefail
shopt -s extglob globstar dotglob nullglob 2>/dev/null
export GLOBIGNORE=.:..

./1_div_packet_sizes.py
./2_perf_eval.py
./2_perf_eval_speedup.py
./3_perf_grouped_unique.py
./4_bottlenecks.py
./5_cv_block.py
./6_lossy.py
./6_lossy_tall.py
./7_cg.py
./8_cg_boxplot.py

