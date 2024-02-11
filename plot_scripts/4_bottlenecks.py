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

# - vectorization
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_d.csv', None, 'CSR\n(baseline)', 'AMD-EPYC-64'))
G.palette_format_dict['CSR\n(baseline)'] = G.palette_format_dict['CSR']
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_vector_x86_d.csv', None, 'CSR avx2\n(vectorization)', 'AMD-EPYC-64'))
G.palette_format_dict['CSR avx2\n(vectorization)'] = G.palette_format_dict['CSR avx2']

# short rows
dfs.append(read_bench_file(G.bench_path + '/lumi/csr5_d.csv', None, 'CSR5\n(short rows)', 'AMD-EPYC-64'))
G.palette_format_dict['CSR5\n(short rows)'] = G.palette_format_dict['CSR5']

# imbalance
dfs.append(read_bench_file(G.bench_path + '/lumi/merge_d.csv', None, 'Merge\n(imbalance)', 'AMD-EPYC-64'))
G.palette_format_dict['Merge\n(imbalance)'] = G.palette_format_dict['Merge']

# MLK
dfs.append(read_bench_file(G.bench_path + '/lumi/mkl_ie_d.csv', None, 'MKL(64-bits)\n(state of practice)', 'AMD-EPYC-64'))
G.palette_format_dict['MKL(64-bits)\n(state of practice)'] = G.palette_format_dict['MKL(64-bits)']

# latency - idx0
dfs.append(read_bench_file(G.bench_path + '/lumi/idx0/mkl_ie_d.csv', None, 'MKL_idx0(64-bits)\n(mem latency)', 'AMD-EPYC-64'))
G.palette_format_dict['MKL_idx0(64-bits)\n(mem latency)'] = G.palette_format_dict['MKL_idx0(64-bits)']


df = concat_data_and_preprocess(dfs)

# df = filter_num_packet_vals(df, G.num_packet_vals_keep)

df = calculate_gmeans(df)


file_out = 'figures/bottlenecks_barplot.png'
set_fig_size_scale(5, 2)
plt.rcParams['font.size'] = 14

p = Barplot(data=df, x='matrix_id', y='gflops',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('Matrix ID', 'Performance (GFLOPs)')
# sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df['format_name'].unique()), title=None, frameon=False)
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df['format_name'].unique()), title=None, frameon=False)


p.plot(file_out)

