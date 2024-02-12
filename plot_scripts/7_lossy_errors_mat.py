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

dfs.append(read_bench_file(G.bench_path + '/lumi/mkl_ie_d.csv', None, 'MKL\n(64-bits)', 'AMD-EPYC-64'))
G.palette_format_dict['MKL\n(64-bits)'] = G.palette_format_dict['MKL(64-bits)']

# Use CVB_d2f for the matrix errors of MKL 32-bits.
dfs.append(read_bench_file(G.bench_path + '/lumi/csr_cv_block_d2f_d.csv', None, 'MKL\n(32-bits)', 'AMD-EPYC-64'))
G.palette_format_dict['MKL\n(32-bits)'] = G.palette_format_dict['MKL(32-bits)']

dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf/csr_cv_stream_rf.csv', None, 'DIV_RF\nLossless', 'AMD-EPYC-64'))
G.palette_format_dict['DIV_RF\nLossless'] = G.palette_format_dict['DIV_RF']

dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf_lossy/csr_cv_stream_d_1e-12.csv', None, 'DIV_RF\nTol. 1e-12', 'AMD-EPYC-64'))
G.palette_format_dict['DIV_RF\nTol. 1e-12'] = G.palette_format_dict['DIV_RF']
dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf_lossy/csr_cv_stream_d_1e-9.csv', None, 'DIV_RF\nTol. 1e-9', 'AMD-EPYC-64'))
G.palette_format_dict['DIV_RF\nTol. 1e-9'] = G.palette_format_dict['DIV_RF']
dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf_lossy/csr_cv_stream_d_1e-7.csv', None, 'DIV_RF\nTol. 1e-7', 'AMD-EPYC-64'))
G.palette_format_dict['DIV_RF\nTol. 1e-7'] = G.palette_format_dict['DIV_RF']
dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf_lossy/csr_cv_stream_d_1e-6.csv', None, 'DIV_RF\nTol. 1e-6', 'AMD-EPYC-64'))
G.palette_format_dict['DIV_RF\nTol. 1e-6'] = G.palette_format_dict['DIV_RF']
dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf_lossy/csr_cv_stream_d_1e-3.csv', None, 'DIV_RF\nTol. 1e-3', 'AMD-EPYC-64'))
G.palette_format_dict['DIV_RF\nTol. 1e-3'] = G.palette_format_dict['DIV_RF']


df = concat_data_and_preprocess(dfs)

df = filter_num_packet_vals(df, G.num_packet_vals_keep)

df['matrix_mae'] = df['matrix_mae'].fillna(0)
df['matrix_max_ae'] = df['matrix_max_ae'].fillna(0)
df['matrix_mse'] = df['matrix_mse'].fillna(0)
df['matrix_mape'] = df['matrix_mape'].fillna(0)
df['matrix_smape'] = df['matrix_smape'].fillna(0)
df['matrix_lnQ_error'] = df['matrix_lnQ_error'].fillna(0)
# df['matrix_mlare'] = df['matrix_mlare'].fillna(0)
df['matrix_gmare'] = df['matrix_gmare'].fillna(0)


# Enumerate formats and add names to palette dictionary.
formats = df['format_name'].unique()
print(formats)


file_out = 'figures/lossy_errors_matrix.png'
set_fig_size_scale(2, 2)
plt.rcParams['font.size'] = 12
p = Boxplot(data=df, x='format_name', y='matrix_mape', hue='format_name', palette=G.palette_format_dict, log_scale=True)
p.set_labels('', 'Matrix MAPE (%)')
p.plot(file_out)

