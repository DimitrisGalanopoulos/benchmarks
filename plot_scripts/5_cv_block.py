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

dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_id_d.csv', None, 'CVB_ID', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_d2f_d.csv', None, 'CVB_d2f', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_fpc_d.csv', None, 'CVB_FPC', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_fpzip_d.csv', None, 'CVB_FPZIP', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_zfp_d.csv', None, 'CVB_ZFP', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_zfp_d_tol_1e-3.csv', None, 'CVB_ZFP_lossy_1e-3', 'AMD-EPYC-64'))


df = concat_data_and_preprocess(dfs)

df = filter_num_packet_vals(df, G.num_packet_vals_keep)

df = calculate_gmeans(df)


file_out = 'figures/cv_block_barplot.png'
set_fig_size_scale(5, 2)
plt.rcParams['font.size'] = 14

p = Barplot(data=df, x='matrix_id', y='gflops',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('Matrix ID', 'Performance (GFLOPs)')
# sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df['format_name'].unique()), title=None, frameon=False)
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df['format_name'].unique()), title=None, frameon=False)


p.plot(file_out)

