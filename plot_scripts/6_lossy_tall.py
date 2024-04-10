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


# set_fig_size_scale(3, 2)
# plt.rcParams['font.size'] = 16
set_fig_size_scale(0.3, 1.0)
plt.rcParams['font.size'] = 6


format_labels = {

    # 'MKL(64-bits)':'MKL(64-bits)',
    # 'MKL(32-bits)':'MKL(32-bits)',
    # 'DIV_RF':'DIV_RF (Lossless)',
    # 'DIV_RF_12':'DIV_RF Tol. 1e-12',
    # 'DIV_RF_9':'DIV_RF Tol. 1e-9',
    # 'DIV_RF_7':'DIV_RF Tol. 1e-7',
    # 'DIV_RF_6':'DIV_RF Tol. 1e-6',
    # 'DIV_RF_3':'DIV_RF Tol. 1e-3',

    'MKL(64-bits)':'64-bits',
    'MKL(32-bits)':'32-bits',
    'DIV_RF':'Tol. 0',
    'DIV_RF_12':'Tol. 1e-12',
    'DIV_RF_9':'Tol. 1e-9',
    'DIV_RF_7':'Tol. 1e-7',
    'DIV_RF_6':'Tol. 1e-6',
    'DIV_RF_3':'Tol. 1e-3',

}


dfs = []

dfs.append(read_bench_file(G.bench_path + '/lumi/mkl_ie_d.csv', None, 'MKL(64-bits)', 'AMD-EPYC-64'))
dfs[-1]['format_group'] = 'MKL'

# Use CVB_d2f for the matrix errors of MKL 32-bits.
df_mkl32 = read_bench_file(G.bench_path + '/lumi/mkl_ie_f.csv', None, 'MKL(32-bits)', 'AMD-EPYC-64')
df_cvb_d2f = read_bench_file(G.bench_path + '/lumi/csr_cv_block_d2f_d.csv', None, 'CVB_d2f', 'AMD-EPYC-64')
df_mkl32['matrix_mape'] = df_cvb_d2f['matrix_mape']
df_mkl32['matrix_smape'] = df_cvb_d2f['matrix_smape']
dfs.append(df_mkl32)
dfs[-1]['format_group'] = 'MKL'

dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf/csr_cv_stream_rf_d_genvec.csv', None, 'DIV_RF', 'AMD-EPYC-64'))
dfs[-1]['format_group'] = 'DIV_RF'

dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf_lossy/csr_cv_stream_d_1e-12.csv', None, 'DIV_RF_12', 'AMD-EPYC-64'))
dfs[-1]['format_group'] = 'DIV_RF'
dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf_lossy/csr_cv_stream_d_1e-9.csv', None, 'DIV_RF_9', 'AMD-EPYC-64'))
dfs[-1]['format_group'] = 'DIV_RF'
dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf_lossy/csr_cv_stream_d_1e-7.csv', None, 'DIV_RF_7', 'AMD-EPYC-64'))
dfs[-1]['format_group'] = 'DIV_RF'
dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf_lossy/csr_cv_stream_d_1e-6.csv', None, 'DIV_RF_6', 'AMD-EPYC-64'))
dfs[-1]['format_group'] = 'DIV_RF'
dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf_lossy/csr_cv_stream_d_1e-3.csv', None, 'DIV_RF_3', 'AMD-EPYC-64'))
dfs[-1]['format_group'] = 'DIV_RF'

df = concat_data_and_preprocess(dfs, G.matrix_names_comression)

df = filter_num_packet_vals(df, G.num_packet_vals_keep)


df['matrix_mae'] = df['matrix_mae'].fillna(0)
df['matrix_max_ae'] = df['matrix_max_ae'].fillna(0)
df['matrix_mse'] = df['matrix_mse'].fillna(0)
df['matrix_mape'] = df['matrix_mape'].fillna(0)
df['matrix_smape'] = df['matrix_smape'].fillna(0)
df['matrix_lnQ_error'] = df['matrix_lnQ_error'].fillna(0)
df['matrix_mlare'] = df['matrix_mlare'].fillna(-np.inf)
df['matrix_gmare'] = df['matrix_gmare'].fillna(0)


# Enumerate formats and add names to palette dictionary.
formats = df['format_name'].unique()
print(formats)


def add_x_label_groups(p):
    # return
    p.ax.set_xticks([1.5], minor=True)
    p.ax.tick_params(axis='x', which='minor', direction='out', length=50)
    p.ax.text(0.18, -0.2, 'MKL', horizontalalignment='center', size=6, color='black', weight='semibold', transform=plt.gcf().transFigure)
    p.ax.text(0.62, -0.2, 'DIV_RF', horizontalalignment='center', size=6, color='black', weight='semibold', transform=plt.gcf().transFigure)


file_out = 'figures/6_lossy_perf_amd_tall.pdf'
p = Boxplot(data=df, x='format_name', y='gflops', hue='format_name', palette=G.palette_format_dict, showfliers=False)
p.set_labels('', 'Performance (GFLOPs)')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=90)
add_x_label_groups(p)
p.plot(file_out)


df_medians = df
df_medians = calculate_aggregate(df_medians, 'gflops', np.median, 'median')
print(df_medians)

df_tmp = df_medians[df_medians['format_name'] == 'MKL(32-bits)']
df_tmp = df_tmp.set_index('matrix_id')
df_medians['gflops_ref'] = df_tmp.loc[df_medians['matrix_id']]['gflops'].values
df_medians['speedup'] = df_medians['gflops'] / df_medians['gflops_ref']
print(df_medians['gflops_ref'])
print(df_medians['speedup'])
df_medians[df_medians['matrix_id'] == 'median'][['format_name', 'System', 'gflops']].to_csv(os.path.splitext(file_out)[0] + '_gflops.csv', sep='\t', index=False, float_format='%g')
df_medians[df_medians['matrix_id'] == 'median'][['format_name', 'System', 'speedup']].to_csv(os.path.splitext(file_out)[0] + '_speedup.csv', sep='\t', index=False, float_format='%g')




file_out = 'figures/6_lossy_errors_matrix_amd_tall.pdf'
p = Boxplot(data=df, x='format_name', y='matrix_mape', hue='format_name', palette=G.palette_format_dict, showfliers=False, log_scale=True)
p.set_labels('', 'Matrix MAPE (\\%)')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=90)
add_x_label_groups(p)
p.ax.grid(axis='x', color='black', linestyle=(0, (30,30)), linewidth=0.1)
p.plot(file_out)


file_out = 'figures/6_lossy_errors_spmv_amd_tall.pdf'
p = Boxplot(data=df, x='format_name', y='spmv_mape', hue='format_name', palette=G.palette_format_dict, showfliers=False, log_scale=True)
p.set_labels('', 'SpMV MAPE (\\%)')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=90)
add_x_label_groups(p)
p.plot_custom(file_out, pad_inches=0.02)
