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


format_labels = {
     'CSR':'CSR\n(baseline)',
     'CSR avx2':'CSR avx2 -\nvectorization\n(ILP)',
     'CSR5':'CSR5\n(short rows)',
     'Merge':'Merge\n(imbalance)',
     'MKL(64-bits)':'MKL\n(state of practice)',
     'MKL_idx0(64-bits)':'MKL_col_0\n(mem latency)',
     'LCM':'LCM\n(ILP, mem bandwidth)',
     'SparseX':'SparseX\n(mem bandwidth)',
}


dfs = []

# - vectorization
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_d.csv', None, 'CSR', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_vector_x86_d.csv', None, 'CSR avx2', 'AMD-EPYC-64'))

# short rows
dfs.append(read_bench_file(G.bench_path + '/lumi/csr5_d.csv', None, 'CSR5', 'AMD-EPYC-64'))

# imbalance
dfs.append(read_bench_file(G.bench_path + '/lumi/merge_d.csv', None, 'Merge', 'AMD-EPYC-64'))

# MLK
dfs.append(read_bench_file(G.bench_path + '/lumi/mkl_ie_d.csv', None, 'MKL(64-bits)', 'AMD-EPYC-64'))

# latency - idx0
dfs.append(read_bench_file(G.bench_path + '/lumi/idx0/mkl_ie_d.csv', None, 'MKL_idx0(64-bits)', 'AMD-EPYC-64'))

# bandwidth
dfs.append(read_bench_file(G.bench_path + '/lumi/lcm/lcm_d_numactl.csv', None, 'LCM', 'AMD-EPYC-64'))

# bandwidth
dfs.append(read_bench_file(G.bench_path + '/lumi/sparsex_d.csv', None, 'SparseX', 'AMD-EPYC-64'))


df = concat_data_and_preprocess(dfs, G.matrix_names_comression)

df = calculate_gmeans(df, 'gflops')


file_out = 'figures/4_bottlenecks_barplot_amd.pdf'
set_fig_size_scale(5, 2)
plt.rcParams['font.size'] = 14

p = Barplot(data=df, x='matrix_id', y='gflops',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('Matrix ID', 'Performance (GFLOPs)')
p.change_legend_labels(format_labels)
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df['format_name'].unique()), borderaxespad=0, labelspacing=0.2, columnspacing=1, handletextpad=0.2, title=None, frameon=False)
p.change_xticks_labels({'geomean':'\\textbf{geo}\n\\textbf{mean}'})

p.plot(file_out)



file_out = 'figures/4_bottlenecks_boxplot_amd.pdf'
set_fig_size_scale(3, 2)
plt.rcParams['font.size'] = 16

# p = Boxplot(data=df, x='matrix_id', y='gflops',  hue='format_name', palette=G.palette_format_dict)
p = Boxplot(data=df, x='format_name', y='gflops',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('', 'Performance (GFLOPs)')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=45)

medians = df.groupby(['format_name'], sort=False)['gflops'].median()
print(medians)
print(p.ax.get_xticks())
for xtick in p.ax.get_xticks():
    p.ax.text(xtick, medians.iloc[xtick] + 1, round(medians.iloc[xtick], 2), horizontalalignment='center', size=14, color='black', weight='semibold')


p.plot(file_out)


