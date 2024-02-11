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

dfs.append(read_bench_file(G.bench_path + '/lumi/sort_diff_rf/csr_cv_stream_rf.csv', None, 'DIV_RF', 'AMD-EPYC-64'))

df = concat_data_and_preprocess(dfs)

file_out = 'figures/packet_size_boxplot.png'
set_fig_size_scale(2.5, 2)
plt.rcParams['font.size'] = 18

# p = Boxplot(data=df[df['format_name'] == 'DIV_RF'], x='CSRCV_NUM_PACKET_VALS', y='gflops', hue='CSRCV_NUM_PACKET_VALS')
p = Boxplot(data=df[df['format_name'] == 'DIV_RF'], x='CSRCV_NUM_PACKET_VALS', y='gflops', hue='format_name', palette=G.palette_format_dict, legend=False)
p.set_labels('Packet Size', 'Performance (GFLOPs)')
p.ax.tick_params(axis='x', labelrotation=45)
p.plot(file_out)

