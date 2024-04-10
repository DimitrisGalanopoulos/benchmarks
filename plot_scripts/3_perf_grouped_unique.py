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


set_fig_size_scale(2, 2)
plt.rcParams['font.size'] = 12


def do_plot(dfs, file_out):
    df = concat_data_and_preprocess(dfs, G.matrix_names_comression)

    df = filter_num_packet_vals(df, G.num_packet_vals_keep)

    # Enumerate formats and add names to palette dictionary.
    formats = df['format_name'].unique()
    print(formats)
    i = 0
    for f in formats:
        f_new = str(i + 1) + ') ' + f
        G.palette_format_dict[f_new] = G.palette_format_dict[f]
        df['format_name'] = df['format_name'].replace(f, f_new)
        i += 1

    df_features = read_features_file()

    UF = df_features.loc['vals unique fraction']
    print(UF)

    df['UF'] = UF[df['matrix_name']].values
    print(df)

    uf_ranges = np.array([0, 0.00001, 0.1, 0.3, 1])
    x_labels = ['[0, 0.001\\%]', '(0.001\\%, 10\\%]', '(10\\%, 30\\%]', '(30\\%, 100\\%]']

    p = Boxplot(data=df, x=pd.cut(df["UF"], uf_ranges), y='gflops', hue='format_name', palette=G.palette_format_dict)
    p.set_labels('Unique Values Percentage', 'Performance (GFLOPs)')
    p.ax.set_xticks(p.ax.get_xticks(), labels=x_labels)

    # sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df['format_name'].unique())//2, title=None, frameon=False)
    sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=m.ceil(len(df['format_name'].unique())/3), borderaxespad=0, labelspacing=0.2, columnspacing=1, handletextpad=0.2, title=None, frameon=False)

    return df, p


dfs = []

# dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_fpc_d.csv', None, 'CVB_FPC', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_vector_lut_x86_d.csv', None, 'Dictionary(Custom)', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/lcm/lcm_d_numactl.csv', None, 'LCM', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/csr5_d.csv', None, 'CSR5', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/sparsex_d.csv', None, 'SparseX', 'AMD-EPYC-64'))

# dfs.append(read_bench_file(G.bench_path + '/lumi/csr_d.csv', None, 'CSR', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/csr_vector_x86_d.csv', None, 'CSR avx2', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/mkl_ie_d.csv', None, 'MKL(64-bits)', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/mkl_ie_f.csv', None, 'MKL(32-bits)', 'AMD-EPYC-64'))

dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff/csr_cv_stream_d.csv', None, 'DIV', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf/csr_cv_stream_rf_d_genvec.csv', None, 'DIV_RF', 'AMD-EPYC-64'))

file_out = 'figures/3_performance_unique_boxplot_amd.pdf'
df, p = do_plot(dfs, file_out)
formats = df['format_name'].unique()
boxes = p.get_box_coords()
(x_min, x_max) = p.ax.get_xlim()
(y_min, y_max) = p.ax.get_ylim()
num_formats = len(formats)
i = 0
for (x0, y0, x1, y1) in boxes:
    lx = abs(x0 - x1)
    ly = abs(y0 - y1)
    x = x0 + lx/2
    # y = y0 - 5
    # text = int(y)
    text = i % num_formats + 1
    i += 1
    p.ax.text(
        x, 
        -4, 
        text, 
        ha='center', 
        va='center', 
        # fontweight='bold', 
        size=8,
        color='black',
        # bbox=dict(facecolor='#445A64')
        )
p.plot(file_out)


dfs = []

# dfs.append(read_bench_file(G.bench_path + '/icy/csr_cv_block_fpc_d.csv', None, 'CVB_FPC', 'INTEL-XEON-16'))
dfs.append(read_bench_file(G.bench_path + '/icy/csrrv_d.csv', None, 'Dictionary(CSR\\&RV)', 'INTEL-XEON-16'))
dfs.append(read_bench_file(G.bench_path + '/icy/csr_vector_lut_x86_d.csv', None, 'Dictionary(Custom)', 'INTEL-XEON-16'))
dfs.append(read_bench_file(G.bench_path + '/icy/lcm/lcm_d.csv', None, 'LCM', 'INTEL-XEON-16'))
# dfs.append(read_bench_file(G.bench_path + '/icy/csr5_d.csv', None, 'CSR5', 'INTEL-XEON-16'))
dfs.append(read_bench_file(G.bench_path + '/icy/sparsex_d.csv', None, 'SparseX', 'INTEL-XEON-16'))

# dfs.append(read_bench_file(G.bench_path + '/icy/csr_d.csv', None, 'CSR', 'INTEL-XEON-16'))
# dfs.append(read_bench_file(G.bench_path + '/icy/csr_vector_x86_d.csv', None, 'CSR avx2', 'INTEL-XEON-16'))
dfs.append(read_bench_file(G.bench_path + '/icy/mkl_ie_d.csv', None, 'MKL(64-bits)', 'INTEL-XEON-16'))
dfs.append(read_bench_file(G.bench_path + '/icy/mkl_ie_f.csv', None, 'MKL(32-bits)', 'INTEL-XEON-16'))

dfs.append(read_bench_file(G.bench_path + '/icy/sort_diff/csr_cv_stream_d.csv', None, 'DIV', 'INTEL-XEON-16'))
dfs.append(read_bench_file(G.bench_path + '/icy/sort_diff_rf/csr_cv_stream_d.csv', None, 'DIV_RF', 'INTEL-XEON-16'))

file_out = 'figures/3_performance_unique_boxplot_intel.pdf'
df, p = do_plot(dfs, file_out)
formats = df['format_name'].unique()
boxes = p.get_box_coords()
(x_min, x_max) = p.ax.get_xlim()
(y_min, y_max) = p.ax.get_ylim()
num_formats = len(formats)
i = 0
for (x0, y0, x1, y1) in boxes:
    lx = abs(x0 - x1)
    ly = abs(y0 - y1)
    x = x0 + lx/2
    # y = y0 - 5
    # text = int(y)
    text = i % num_formats + 1
    i += 1
    p.ax.text(
        x, 
        1, 
        text, 
        ha='center', 
        va='center', 
        # fontweight='bold', 
        size=8,
        color='black',
        # bbox=dict(facecolor='#445A64')
        )
p.plot(file_out)

