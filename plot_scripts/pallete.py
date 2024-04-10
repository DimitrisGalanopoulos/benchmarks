import sys
import math as m
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.ticker as ticker
import seaborn as sns

import global_env as G
from util import *
from plot_util import *


# G.palette_default = sns.color_palette('colorblind')
G.palette_default = ['#FF6F61', '#FFD166', '#5E819D', '#8F3985', '#D5AB9C', '#3C6E71', '#A2A18F', '#7D9EA8', '#D9BF77', '#7A9D7E', '#EAD3C6', '#63774D', '#A8D0DB', '#FFE156', '#F0B8A6', '#6C5B7B', '#355C7D', '#ACD8AA', '#6B4226']

# plt.clf()
# sns.palplot(G.palette_default)
# plt.rcParams['figure.figsize'] = [figsize_base_width, figsize_base_height]
# plt.savefig("figures/palette_default.png", bbox_inches = 'tight')
# plt.show()


def print_palette():
    N = len(G.format_names)
    plt.rcParams['figure.figsize'] = [1, N]

    plt.clf()
    yticks = range(N)
    ax = plt.axes()
    ax.set_ylim(-0.5, N-0.5)
    ax.set_xticks([0, 1], labels=[])
    ax.tick_params(axis='x', bottom=False)
    ax.tick_params(axis='y')
    ax.set_yticks(yticks, labels=G.format_names)
    ax.invert_yaxis()
    for i in yticks:
        c = G.palette_default[i]
        plt.axhspan(i-0.5, i+0.5, color=c, alpha=1)
    plt.savefig("figures/palette_test.png", bbox_inches = 'tight')

    plt.rcParams['figure.figsize'] = [figsize_base_width, figsize_base_height]

print_palette()


sns.set_palette(G.palette_default)


def indexed_palette_create(palette, keys):
    return dict(zip(keys, palette[:len(keys)]))

G.palette_format_dict = indexed_palette_create(G.palette_default, G.format_names)
G.palette_format_dict['CVB_FPC('+str(G.num_packet_vals_keep)+')'] = G.palette_format_dict['CVB_FPC']
G.palette_format_dict['DIV('+str(G.num_packet_vals_keep)+')'] = G.palette_format_dict['DIV']
G.palette_format_dict['DIV_RF('+str(G.num_packet_vals_keep)+')'] = G.palette_format_dict['DIV_RF']
G.palette_format_dict['DIV_RF_12'] = G.palette_format_dict['DIV_RF']
G.palette_format_dict['DIV_RF_9'] = G.palette_format_dict['DIV_RF']
G.palette_format_dict['DIV_RF_7'] = G.palette_format_dict['DIV_RF']
G.palette_format_dict['DIV_RF_6'] = G.palette_format_dict['DIV_RF']
G.palette_format_dict['DIV_RF_3'] = G.palette_format_dict['DIV_RF']
print(G.palette_format_dict)

