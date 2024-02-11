#!/usr/bin/env -S ${HOME}/lib/python/bin/python

import sys
import math as m
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns

import global_env as G
from util import *
from data_processing import *
from plot_util import *
from pallete import *
from boxplot import *
from barplot import *


dfs = []

# dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_fpc_d.csv', None, 'CVB_FPC('+str(G.num_packet_vals_keep)+')', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_fpc_d.csv', None, 'CVB_FPC', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_vector_lut_x86_d.csv', None, 'Dictionary(Custom)', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/csr5_d.csv', None, 'CSR5', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/sparsex_d.csv', None, 'SparseX', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/merge_d.csv', None, 'Merge', 'AMD-EPYC-64'))

# dfs.append(read_bench_file(G.bench_path + '/lumi/csr_d.csv', None, 'CSR', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/csr_vector_x86_d.csv', None, 'CSR avx2', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/mkl_ie_d.csv', None, 'MKL(64-bits)', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/mkl_ie_f.csv', None, 'MKL(32-bits)', 'AMD-EPYC-64'))

dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff/csr_cv_stream_23_no_initial_values.csv', None, 'DIV', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf/csr_cv_stream_rf.csv', None, 'DIV_RF', 'AMD-EPYC-64'))

# read_bench_file(G.bench_path + '/amd-epyc1/intermediate/rep2/amd-epyc1_synthetic_t24_d_rep2.csv', 'AMD-EPYC1'),
# read_bench_file(G.bench_path + '/amd-hawk/amd-hawk_synthetic_t64_d.csv', 'AMD-HAWK'),
# read_bench_file(G.bench_path + '/arm/intermediate/rep1/arm_synthetic_t80_d_rep1.csv', 'ARM-NEON'),
# read_bench_file(G.bench_path + '/power9-m100/intermediate/rep1/power9-m100_synthetic_t32_d_rep1.csv', 'POWER9'),
# read_bench_file(G.bench_path + '/intel-gold2/intermediate/rep1/intel-gold2_synthetic_t14_d_rep1.csv', 'INTEL-GOLD'),
# read_bench_file(G.bench_path + '/vulcan-P100/vulcan-P100_dtype-D_run_full_dataset.csv', 'Tesla-P100'),
# read_bench_file(G.bench_path + '/vulcan-V100/vulcan-V100_dtype-D_run_full_dataset.csv', 'Tesla-V100'),
# read_bench_file(G.bench_path + '/epyc5-A100/epyc5-A100_dtype-D_run_synthetic_matrices_small_dataset.csv', 'Tesla-A100'),
# read_bench_file(G.bench_path + '/alveo-u280/alveo-u280_spmv_4-2048_dtype-D.csv', 'Alveo-U280'),


df = concat_data_and_preprocess(dfs)

df = filter_num_packet_vals(df, G.num_packet_vals_keep)

df = calculate_gmeans(df)


file_out = 'figures/performance_barplot.png'
set_fig_size_scale(5, 2)
plt.rcParams['font.size'] = 14

p = Barplot(data=df, x='matrix_id', y='gflops',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('Matrix ID', 'Performance (GFLOPs)')
# sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df['format_name'].unique()), title=None, frameon=False)
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df['format_name'].unique()), title=None, frameon=False)


plt.ylim([0, 125])
(y_min, y_max) = p.ax.get_ylim()
boxes = p.get_bar_coords()
for (x0, y0, x1, y1) in boxes:
    text = '%d'%(y1,)
    if (y1 > y_max):
        p.ax.text(
            # x0, 
            x0 - 0.25, 
            y_max - 3, 
            text, 
            ha='center', 
            va='center', 
            # fontweight='bold', 
            size=10,
            color='black',
            # bbox=dict(facecolor='#445A64')
            )
        # print(y1)


p.plot(file_out)

