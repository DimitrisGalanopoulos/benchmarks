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




dfs = []

dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf/csr_cv_stream_rf_d_genvec.csv', None, 'DIV_RF', 'AMD-EPYC-64'))

df = concat_data_and_preprocess(dfs, G.matrix_names_comression)

df = filter_num_packet_vals(df, G.num_packet_vals_keep)

df_features = read_features_file()
UF = df_features.loc['vals unique fraction']
df['UV'] = UF[df['matrix_name']].values * 100


df = df[['matrix_id', 'matrix_name', 'csr_mem_footprint', 'UV', 'mem_ratio']].reset_index(drop=True)
df['mem_ratio'] = df['mem_ratio'] * 100


df['matrix_name'] = df['matrix_name'].replace('_', '\\_', regex=True)

print("geomean =", sp.stats.gmean(df['mem_ratio']))



df['csr_mem_footprint'] = df['csr_mem_footprint'].map(lambda x: '%d' % x)
df['UV'] = df['UV'].map(lambda x: '%.2g' % x)
df['mem_ratio'] = df['mem_ratio'].map(lambda x: '%.2f' % x)

df.to_latex('matrix_suite.csv', index=False)
print(df)

exit()

