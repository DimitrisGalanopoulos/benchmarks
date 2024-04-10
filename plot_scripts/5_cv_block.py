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
    'CVB_ID':'CVB\nID',
    'CVB_d2f':'CVB\nD2F',
    'CVB_FPC':'CVB\nFPC',
    'CVB_FPZIP':'CVB\nFPZIP',
    'CVB_ZFP':'CVB\nZFP',
    'CVB_ZFP_lossy_1e-3':'CVB\nZFP\nlossy 1e-3',
}


dfs = []

dfs.append(read_bench_file(G.bench_path + '/lumi/mkl_ie_d.csv', None, 'MKL(64-bits)', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_id_d.csv', None, 'CVB_ID', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_d2f_d.csv', None, 'CVB_d2f', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_fpc_d.csv', None, 'CVB_FPC', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_fpzip_d.csv', None, 'CVB_FPZIP', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_zfp_d.csv', None, 'CVB_ZFP', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_zfp_d_tol_1e-3.csv', None, 'CVB_ZFP_lossy_1e-3', 'AMD-EPYC-64'))


df = concat_data_and_preprocess(dfs, G.matrix_names_comression)

df = filter_num_packet_vals(df, G.num_packet_vals_keep)



set_fig_size_scale(5, 2)
plt.rcParams['font.size'] = 14

df_bar = calculate_gmeans(df, 'gflops')

file_out = 'figures/5_cv_block_barplot_amd.pdf'

p = Barplot(data=df_bar, x='matrix_id', y='gflops',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('Matrix ID', 'Performance (GFLOPs)')
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df_bar['format_name'].unique()), borderaxespad=0, labelspacing=0.2, columnspacing=1, handletextpad=0.2, title=None, frameon=False)
p.change_xticks_labels({'geomean':'\\textbf{geo}\n\\textbf{mean}'})

p.plot(file_out)


set_fig_size_scale(3, 2)
plt.rcParams['font.size'] = 16

file_out = 'figures/5_cv_block_boxplot_amd.pdf'

p = Boxplot(data=df, x='format_name', y='gflops',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('', 'Performance (GFLOPs)')
p.change_xticks_labels(format_labels)

p.plot(file_out)

