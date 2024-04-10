#!/usr/bin/env -S ${HOME}/lib/python/bin/python

import os
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


font_size = 14

set_fig_size_scale(5, 2)
plt.rcParams['font.size'] = font_size


def do_plot(dfs, file_out):
    df = concat_data_and_preprocess(dfs, G.matrix_names_comression)
    # print(df[df['format_name'] == 'LCM'])

    df = filter_num_packet_vals(df, G.num_packet_vals_keep)

    df = calculate_gmeans(df, 'gflops')

    df_tmp = df[df['format_name'] == 'MKL(64-bits)']
    df_tmp = df_tmp.set_index('matrix_id')
    df['gflops_ref'] = df_tmp.loc[df['matrix_id']]['gflops'].values
    df['speedup'] = df['gflops'] / df['gflops_ref']
    print(df['gflops_ref'])
    print(df['speedup'])
    df[df['matrix_id'] == 'geomean'][['format_name', 'System', 'gflops']].to_csv(os.path.splitext(file_out)[0] + '_gflops.csv', sep='\t', index=False, float_format='%g')
    df[df['matrix_id'] == 'geomean'][['format_name', 'System', 'speedup']].to_csv(os.path.splitext(file_out)[0] + '_speedup.csv', sep='\t', index=False, float_format='%g')

    p = Barplot(data=df, x='matrix_id', y='gflops',  hue='format_name', palette=G.palette_format_dict)
    p.set_labels('Matrix ID', 'Performance (GFLOPs)')
    sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df['format_name'].unique()), borderaxespad=0, labelspacing=0.2, columnspacing=1, handletextpad=0.2, title=None, frameon=False)
    p.change_xticks_labels({'geomean':'\\textbf{geo}\n\\textbf{mean}'})


    xlim = p.ax.get_xlim()
    total_width = xlim[1] - xlim[0]
    num_xticks = len(p.ax.get_xticks())
    xtick_width = total_width / num_xticks
    num_bars = len(p.ax.patches)
    # print(num_bars)

    for bar in p.ax.patches:
        x = bar.get_x()
        width = bar.get_width()
        if width == 0:
            continue
        ratio = 1 - 1/num_xticks
        new_width = width * ratio
        x_ratio = (x - xlim[0]) / total_width
        xtick_id = round(x / xtick_width)
        x_new = x * ratio
        bar.set_x(x_new)
        if xtick_id == num_xticks - 1:
            base = (num_xticks - 1) * (xtick_width * ratio) - (xtick_width * ratio) / 2
            bar.set_x(base + (x_new - base) * 2)
            bar.set_width(new_width * 2)

    xticks = [1] * num_xticks
    xlabels = p.ax.get_xticklabels()
    for i in range(len(xticks)):
        ratio = 1 - 1/num_xticks
        pos = xtick_width * i * ratio
        xticks[i] = xtick_width * i * ratio
    xticks[-1] = xticks[-1] + xtick_width * ratio / 2
    p.ax.set_xticks(xticks, labels=xlabels)

    return df, p


dfs = []

# dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_fpc_d.csv', None, 'CVB_FPC('+str(G.num_packet_vals_keep)+')', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_fpc_d.csv', None, 'CVB_FPC', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_vector_lut_x86_d.csv', None, 'Dictionary(Custom)', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/lcm/lcm_d.csv', None, 'LCM', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/lcm/lcm_d_unpinned.csv', None, 'LCM', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/lcm/lcm_d_c20.csv', None, 'LCM', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/lcm/lcm_d_parinit.csv', None, 'LCM', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/lcm/lcm_d_c20_unpinned.csv', None, 'LCM', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/lcm/lcm_d_numactl.csv', None, 'LCM', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/csr5_d.csv', None, 'CSR5', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/sparsex_d.csv', None, 'SparseX', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/merge_d.csv', None, 'Merge', 'AMD-EPYC-64'))

# dfs.append(read_bench_file(G.bench_path + '/lumi/csr_d.csv', None, 'CSR', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/csr_vector_x86_d.csv', None, 'CSR avx2', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/mkl_ie_d.csv', None, 'MKL(64-bits)', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/mkl_ie_f.csv', None, 'MKL(32-bits)', 'AMD-EPYC-64'))

dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff/csr_cv_stream_d.csv', None, 'DIV', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf/csr_cv_stream_rf_d_genvec.csv', None, 'DIV_RF', 'AMD-EPYC-64'))


file_out = 'figures/2_performance_barplot_amd.pdf'
df, p = do_plot(dfs, file_out)
plt.ylim([0, 125])
(y_min, y_max) = p.ax.get_ylim()
boxes = p.get_bar_coords()
for (x0, y0, x1, y1) in boxes:
    text = '%d'%(y1,)
    if (y1 > y_max):
        p.ax.text(
            # x0, 
            x0 - 0.35, 
            y_max - 4, 
            text, 
            ha='center', 
            va='center', 
            # fontweight='bold', 
            size=font_size,
            color='black',
            # bbox=dict(facecolor='#445A64')
            )
        # print(y1)
p.plot(file_out)


dfs = []

dfs.append(read_bench_file(G.bench_path + '/icy/csrrv_d.csv', None, 'Dictionary(CSR\\&RV)', 'INTEL-XEON-16'))

# dfs.append(read_bench_file(G.bench_path + '/icy/csr_cv_block_fpc_d.csv', None, 'CVB_FPC('+str(G.num_packet_vals_keep)+')', 'INTEL-XEON-16'))
# dfs.append(read_bench_file(G.bench_path + '/icy/csr_cv_block_fpc_d.csv', None, 'CVB_FPC', 'INTEL-XEON-16'))
dfs.append(read_bench_file(G.bench_path + '/icy/csr_vector_lut_x86_d.csv', None, 'Dictionary(Custom)', 'INTEL-XEON-16'))
dfs.append(read_bench_file(G.bench_path + '/icy/lcm/lcm_d.csv', None, 'LCM', 'INTEL-XEON-16'))
# dfs.append(read_bench_file(G.bench_path + '/icy/lcm/lcm_d_unpinned.csv', None, 'LCM', 'INTEL-XEON-16'))
# dfs.append(read_bench_file(G.bench_path + '/icy/csr5_d.csv', None, 'CSR5', 'INTEL-XEON-16'))
dfs.append(read_bench_file(G.bench_path + '/icy/sparsex_d.csv', None, 'SparseX', 'INTEL-XEON-16'))
# dfs.append(read_bench_file(G.bench_path + '/icy/merge_d.csv', None, 'Merge', 'INTEL-XEON-16'))

# dfs.append(read_bench_file(G.bench_path + '/icy/csr_d.csv', None, 'CSR', 'INTEL-XEON-16'))
# dfs.append(read_bench_file(G.bench_path + '/icy/csr_vector_x86_d.csv', None, 'CSR avx2', 'INTEL-XEON-16'))
dfs.append(read_bench_file(G.bench_path + '/icy/mkl_ie_d.csv', None, 'MKL(64-bits)', 'INTEL-XEON-16'))
dfs.append(read_bench_file(G.bench_path + '/icy/mkl_ie_f.csv', None, 'MKL(32-bits)', 'INTEL-XEON-16'))

dfs.append(read_bench_file(G.bench_path + '/icy/sort_diff/csr_cv_stream_d.csv', None, 'DIV', 'INTEL-XEON-16'))
dfs.append(read_bench_file(G.bench_path + '/icy/sort_diff_rf/csr_cv_stream_d.csv', None, 'DIV_RF', 'INTEL-XEON-16'))

file_out = 'figures/2_performance_barplot_intel.pdf'
df, p = do_plot(dfs, file_out)
p.plot(file_out)

