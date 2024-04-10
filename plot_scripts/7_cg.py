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


set_fig_size_scale(5, 2)
plt.rcParams['font.size'] = 14


format_labels = {
    'DIV_RF_12':'DIV_RF\nTol. 1e-12',
    'DIV_RF_9':'DIV_RF\nTol. 1e-9',
    'DIV_RF_7':'DIV_RF\nTol. 1e-7',
    'DIV_RF_6':'DIV_RF\nTol. 1e-6',
    'DIV_RF_3':'DIV_RF\nTol. 1e-3',
}


dfs = []
dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_d.csv', None, 'CSR', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_mkl_ie_d.csv', None, 'MKL(64-bits)', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_mkl_ie_f.csv', None, 'MKL(32-bits)', 'AMD-EPYC-64'))
dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_cv_stream_d.csv', None, 'DIV_RF', 'AMD-EPYC-64'))

# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_cv_stream_d_1e-12.csv', None, 'DIV_RF_12', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_cv_stream_d_1e-9.csv', None, 'DIV_RF_9', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_cv_stream_d_1e-7.csv', None, 'DIV_RF_7', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_cv_stream_d_1e-6.csv', None, 'DIV_RF_6', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_cv_stream_d_1e-3.csv', None, 'DIV_RF_3', 'AMD-EPYC-64'))

df = concat_data_and_preprocess(dfs, G.matrix_names_comression)


df_ref = df[df['format_name'] == 'MKL(64-bits)']
df_ref = df_ref.set_index('matrix_id')
df['time_ref'] = df_ref.loc[df['matrix_id']]['time'].values
df['speedup'] = df['time_ref'] / df['time']


df['compression_time'] = df['compression_time'].fillna(0)
df['time_total'] = df['time'] + df['compression_time']
df['speedup_total'] = df['time_ref'] / df['time_total']

df['time_per_iter'] = df['time'] / df['num_iterations']
df['time_total_per_iter'] = df['time_total'] / df['num_iterations']

df['compression_to_cg_iters'] = df['compression_time'] / df['time_per_iter']



df['error'] = df['error'].fillna(df['error'].max())
df['error_log'] = np.log10(df['error'])


file_out = 'figures/7_cg_comp_time_amd.pdf'
p = Barplot(data=df[df['format_name'] == 'DIV_RF'], x='matrix_id', y='compression_time',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('Matrix ID', 'Compression Time')
p.change_legend_labels(format_labels)
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df['format_name'].unique()), borderaxespad=0, labelspacing=0.2, columnspacing=1, handletextpad=0.2, title=None, frameon=False)
p.plot(file_out)

file_out = 'figures/7_cg_comp_to_cg_iters_amd.pdf'
p = Barplot(data=df[df['format_name'] == 'DIV_RF'], x='matrix_id', y='compression_to_cg_iters',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('Matrix ID', 'Compression Time to CG Iterations')
p.change_legend_labels(format_labels)
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df['format_name'].unique()), borderaxespad=0, labelspacing=0.2, columnspacing=1, handletextpad=0.2, title=None, frameon=False)
p.plot(file_out)

file_out = 'figures/7_cg_err_amd.pdf'
p = Barplot(data=df, x='matrix_id', y='error_log',  hue='format_name', palette=G.palette_format_dict)
# p = Barplot(data=df, x='matrix_id', y='error',  hue='format_name', palette=G.palette_format_dict, log_scale=True)
p.set_labels('Matrix ID', 'Error')
p.change_legend_labels(format_labels)
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df['format_name'].unique()), borderaxespad=0, labelspacing=0.2, columnspacing=1, handletextpad=0.2, title=None, frameon=False)
p.plot(file_out)

file_out = 'figures/7_cg_num_iters_amd.pdf'
p = Barplot(data=df, x='matrix_id', y='num_iterations',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('Matrix ID', 'Number of Iterations')
p.change_legend_labels(format_labels)
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df['format_name'].unique()), borderaxespad=0, labelspacing=0.2, columnspacing=1, handletextpad=0.2, title=None, frameon=False)
p.plot(file_out)



df_time = df
file_out = 'figures/7_cg_time_amd.pdf'
p = Barplot(data=df_time, x='matrix_id', y='time',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('Matrix ID', 'Time (sec)')
p.change_legend_labels(format_labels)
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df_time['format_name'].unique()), borderaxespad=0, labelspacing=0.2, columnspacing=1, handletextpad=0.2, title=None, frameon=False)
p.plot(file_out)


df_time = df
df_time = calculate_gmeans(df, 'time_per_iter')
file_out = 'figures/7_cg_tpi_amd.pdf'
p = Barplot(data=df_time, x='matrix_id', y='time_per_iter',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('Matrix ID', 'Time per Iteration (sec)')
p.change_legend_labels(format_labels)
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df_time['format_name'].unique()), borderaxespad=0, labelspacing=0.2, columnspacing=1, handletextpad=0.2, title=None, frameon=False)
p.change_xticks_labels({'geomean':'\\textbf{geo}\n\\textbf{mean}'})
p.plot(file_out)


df_time_total = calculate_gmeans(df, 'time_total_per_iter')
file_out = 'figures/7_cg_tpi_total_amd.pdf'
p = Barplot(data=df_time_total, x='matrix_id', y='time_total_per_iter',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('Matrix ID', 'Compression + Run Time per Iteration (sec)')
p.change_legend_labels(format_labels)
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df_time_total['format_name'].unique()), borderaxespad=0, labelspacing=0.2, columnspacing=1, handletextpad=0.2, title=None, frameon=False)
p.change_xticks_labels({'geomean':'\\textbf{geo}\n\\textbf{mean}'})
p.plot(file_out)


df_speedup = df
df_speedup = calculate_gmeans(df_speedup, 'speedup')
file_out = 'figures/7_cg_speedup_amd.pdf'
p = Barplot(data=df_speedup, x='matrix_id', y='speedup',  hue='format_name', palette=G.palette_format_dict)
p.set_labels('Matrix ID', 'Speedup vs MKL(64-bits)')
p.change_legend_labels(format_labels)
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df_speedup['format_name'].unique()), borderaxespad=0, labelspacing=0.2, columnspacing=1, handletextpad=0.2, title=None, frameon=False)
plt.ylim([0, 4])
(y_min, y_max) = p.ax.get_ylim()
boxes = p.get_bar_coords()
for (x0, y0, x1, y1) in boxes:
    text = '%d'%(y1,)
    if (y1 > y_max):
        p.ax.text(
            x0 - 0.1, 
            y_max - 0.1, 
            text, 
            ha='center', 
            va='center', 
            size=10,
            color='black',
            )
p.change_xticks_labels({'geomean':'\\textbf{geo}\n\\textbf{mean}'})
p.plot(file_out)

