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
from violinplot import *


set_fig_size_scale(0.8, 1)
plt.rcParams['font.size'] = 10


format_labels = {

    # 'MKL(64-bits)':'MKL\n(64-bits)',
    # 'MKL(32-bits)':'MKL\n(32-bits)',
    # 'DIV_RF_12':'DIV_RF\nTol. 1e-12',
    # 'DIV_RF_9':'DIV_RF\nTol. 1e-9',
    # 'DIV_RF_7':'DIV_RF\nTol. 1e-7',
    # 'DIV_RF_6':'DIV_RF\nTol. 1e-6',
    # 'DIV_RF_3':'DIV_RF\nTol. 1e-3',

    'MKL(64-bits)':'MKL(64-bits)',
    'MKL(32-bits)':'MKL(32-bits)',
    # 'DIV_RF':'DIV_RF (Lossless)',
    'DIV_RF':'DIV_RF',
    'DIV_RF_12':'DIV_RF Tol. 1e-12',
    'DIV_RF_9':'DIV_RF Tol. 1e-9',
    'DIV_RF_7':'DIV_RF Tol. 1e-7',
    'DIV_RF_6':'DIV_RF Tol. 1e-6',
    'DIV_RF_3':'DIV_RF Tol. 1e-3',

}


# os.system(G.bench_path + '/lumi/cg/run.sh')


dfs = []
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_d.csv', None, 'CSR', 'AMD-EPYC-64'))

dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_mkl_ie_d.csv', None, 'MKL(64-bits)', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_mkl_ie_f.csv', None, 'MKL(32-bits)', 'AMD-EPYC-64'))

dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_cv_stream_d.csv', None, 'DIV_RF', 'AMD-EPYC-64'))

# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_cv_stream_d_1e-12.csv', None, 'DIV_RF_12', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_cv_stream_d_1e-9.csv', None, 'DIV_RF_9', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_cv_stream_d_1e-7.csv', None, 'DIV_RF_7', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_cv_stream_d_1e-6.csv', None, 'DIV_RF_6', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_cv_stream_d_1e-3.csv', None, 'DIV_RF_3', 'AMD-EPYC-64'))

df = concat_data_and_preprocess(dfs, G.matrix_names_comression)


# dfs = []
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/iterations/cg_mkl_ie_d.out', None, 'MKL(64-bits)', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/iterations/cg_mkl_ie_f.out', None, 'MKL(32-bits)', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/iterations/cg_csr_cv_stream_d.out', None, 'DIV_RF', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/iterations/cg_csr_cv_stream_d_1e-12.out', None, 'DIV_RF_12', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/iterations/cg_csr_cv_stream_d_1e-9.out', None, 'DIV_RF_9', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/iterations/cg_csr_cv_stream_d_1e-7.out', None, 'DIV_RF_7', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/iterations/cg_csr_cv_stream_d_1e-6.out', None, 'DIV_RF_6', 'AMD-EPYC-64'))
# dfs.append(read_bench_file(G.bench_path + '/lumi/cg/cg_csr_cv_stream_d_1e-3.out', None, 'DIV_RF_3', 'AMD-EPYC-64'))
# df_iters = concat_data_and_preprocess(dfs, G.matrix_names_comression)
# print(df_iters)
# exit(0)


df_ref = df[df['format_name'] == 'MKL(64-bits)']
df_ref = df_ref.set_index('matrix_id')

df['time_ref'] = df_ref.loc[df['matrix_id']]['time'].values
df['num_iterations_ref'] = df_ref.loc[df['matrix_id']]['num_iterations'].values
df['time_per_iter_ref'] = df['time_ref'] / df['num_iterations_ref']

df['speedup'] = df['time_ref'] / df['time']


df['compression_time'] = df['compression_time'].fillna(0)
df['time_total'] = df['time'] + df['compression_time']
df['speedup_total'] = df['time_ref'] / df['time_total']

df['time_per_iter'] = df['time'] / df['num_iterations']
df['time_total_per_iter'] = df['time_total'] / df['num_iterations']

df['speedup_per_iter'] = df['time_per_iter_ref'] / df['time_per_iter']

df['compression_to_cg_iters'] = df['compression_time'] / df['time_per_iter']


df['error'] = df['error'].fillna(df['error'].max())
# df['error'] = np.sqrt(df['error'])

df['error_log'] = np.log10(df['error'])


showfliers = True
# showfliers = False

rotation = 0
# rotation = 90


df_comp = df[df['format_name'].isin(['DIV_RF', 'DIV_RF_12', 'DIV_RF_9', 'DIV_RF_7', 'DIV_RF_6', 'DIV_RF_3'])]
file_out = 'figures/8_cg_compression_time_amd.pdf'
p = Boxplot(data=df_comp, x='format_name', y='compression_time', hue='format_name', palette=G.palette_format_dict, legend=False, showfliers=showfliers)
p.ax.set_ylim(bottom=0)
p.set_labels('', 'Compression Time')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=rotation)
p.plot(file_out)

file_out = 'figures/8_cg_compression_to_cg_iters_amd.pdf'
p = Boxplot(data=df_comp, x='format_name', y='compression_to_cg_iters', hue='format_name', palette=G.palette_format_dict, legend=False, showfliers=showfliers)
p.ax.set_ylim(bottom=0)
p.set_labels('', 'Compression Time to CG Iterations')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=rotation)
p.plot(file_out)


file_out = 'figures/8_cg_err_amd.pdf'
p = Boxplot(data=df, x='format_name', y='error', hue='format_name', palette=G.palette_format_dict, showfliers=showfliers, log_scale=True)
p.set_labels('', 'Error (Residual)')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=rotation)
p.plot(file_out)

file_out = 'figures/8_cg_err_violin_amd.pdf'
p = Violinplot(data=df, x='format_name', y='error', hue='format_name', palette=G.palette_format_dict, cut=0, split=False, inner="quart", log_scale=True)
p.set_labels('', 'Error (Residual)')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=rotation)
p.plot(file_out)


file_out = 'figures/8_cg_num_iters_amd.pdf'
p = Boxplot(data=df, x='format_name', y='num_iterations', hue='format_name', palette=G.palette_format_dict, showfliers=showfliers)
p.set_labels('', 'Number of Iterations')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=rotation)
p.ax.set_ylim(bottom=0)
p.plot(file_out)

file_out = 'figures/8_cg_num_iters_violin_amd.pdf'
p = Violinplot(data=df, x='format_name', y='num_iterations', hue='format_name', palette=G.palette_format_dict, cut=0, split=False, inner="quart")
p.set_labels('', 'Number of Iterations')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=rotation)
p.ax.set_ylim(bottom=0)
p.plot(file_out)



file_out = 'figures/8_cg_time_amd.pdf'
p = Boxplot(data=df, x='format_name', y='time', hue='format_name', palette=G.palette_format_dict, showfliers=showfliers)
p.set_labels('', 'Time (sec)')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=rotation)
p.plot(file_out)


file_out = 'figures/8_cg_tpi_amd.pdf'
p = Boxplot(data=df, x='format_name', y='time_per_iter', hue='format_name', palette=G.palette_format_dict, showfliers=showfliers)
p.set_labels('', 'Time per Iteration (sec)')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=rotation)
p.plot(file_out)


file_out = 'figures/8_cg_tpi_total_amd.pdf'
p = Boxplot(data=df, x='format_name', y='time_total_per_iter', hue='format_name', palette=G.palette_format_dict, showfliers=showfliers)
p.set_labels('', 'Total Time per Iteration (sec)')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=rotation)
p.plot(file_out)


file_out = 'figures/8_cg_speedup_amd.pdf'
p = Boxplot(data=df, x='format_name', y='speedup', hue='format_name', palette=G.palette_format_dict, showfliers=showfliers)
p.set_labels('', 'Speedup')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=rotation)
p.plot(file_out)

file_out = 'figures/8_cg_speedup_total_amd.pdf'
p = Boxplot(data=df, x='format_name', y='speedup_total', hue='format_name', palette=G.palette_format_dict, showfliers=showfliers)
p.set_labels('', 'Total Speedup')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=rotation)
p.plot(file_out)


file_out = 'figures/8_cg_speedup_per_iter_amd.pdf'
p = Boxplot(data=df, x='format_name', y='speedup_per_iter', hue='format_name', palette=G.palette_format_dict, showfliers=showfliers)
p.set_labels('', 'Iteration Speedup')
p.change_xticks_labels(format_labels)
p.ax.tick_params(axis='x', labelrotation=rotation)
p.plot(file_out)



set_fig_size_scale(2, 1)
plt.rcParams['font.size'] = 10

df_speedup = df[df['format_name'] == 'DIV_RF']
df_speedup = calculate_gmeans(df_speedup, 'speedup_total')
print(df_speedup[df_speedup['matrix_id'] == 'geomean']['speedup_total'])

file_out = 'figures/8_cg_speedup_total_barplot_amd.pdf'
p = Barplot(data=df_speedup, x='matrix_id', y='speedup_total', hue='format_name', palette=G.palette_format_dict)
p.set_labels('Matrix ID', 'Total Speedup vs MKL(64-bits)')
p.change_legend_labels(format_labels)
sns.move_legend(p.ax, "lower center", bbox_to_anchor=(.5, 1), ncols=len(df_speedup['format_name'].unique()), borderaxespad=0, labelspacing=0.2, columnspacing=1, handletextpad=0.2, title=None, frameon=False)
# plt.ylim(bottom=0.9)
# plt.ylim(bottom=1)
# plt.ylim(bottom=df_speedup['speedup_total'].min())
p.change_xticks_labels({'geomean':'\\textbf{geo}\n\\textbf{mean}'})
p.ax.grid(axis='y', color='black', linestyle=(0, (10,10)), linewidth=0.3)

df_annot = df[df['format_name'] == 'DIV_RF']
new_record = pd.DataFrame([{'matrix_id':'geomean', 'num_iterations':df_annot['num_iterations'].mean(), 'compression_to_cg_iters':df_annot['compression_to_cg_iters'].mean()}])
df_annot = pd.concat([df_annot, new_record], ignore_index=True)

df_annot = df_annot[['matrix_id', 'num_iterations', 'compression_to_cg_iters']]
print(df_annot)


for i in range(len(p.ax.patches)):
    bar = p.ax.patches[i]
    if bar.get_width() == 0:
        continue
    v = bar.get_height()
    cg_iters = df_annot['num_iterations'].iloc[i]
    cg_iters = int(cg_iters / 1000 + 0.5)
    cg_iters = f'{cg_iters}K'
    comp_cg_iters = int(df_annot['compression_to_cg_iters'].iloc[i])
    p.ax.annotate(
            # '%d / %d'%(comp_cg_iters, cg_iters),
            f'{cg_iters}\n({comp_cg_iters})',
            # (bar.get_x() + bar.get_width() / 2., bar.get_height()),
            (bar.get_x() + bar.get_width() / 2., 0.1),
            # (bar.get_x() + bar.get_width() / 2., 1.1),
            ha='center', va='bottom',
            # xytext=(0, 9),
            # rotation=90,
            # textcoords='offset points',
            color='white',
            weight='bold',
            )


p.plot(file_out)


