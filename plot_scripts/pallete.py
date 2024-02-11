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
G.palette_default = ['#FF6F61', '#FFD166', '#5E819D', '#8F3985', '#A2A18F', '#3C6E71', '#D5AB9C', '#7D9EA8', '#D9BF77', '#7A9D7E', '#EAD3C6', '#63774D', '#A8D0DB', '#6B4226', '#F0B8A6', '#6C5B7B', '#355C7D', '#ACD8AA', '#FFE156']

plt.clf()
sns.palplot(G.palette_default)
plt.rcParams['figure.figsize'] = [figsize_base_width, figsize_base_height]
plt.savefig("figures/palette_default.png", bbox_inches = 'tight')
# plt.show()


sns.set_palette(G.palette_default)


def indexed_palette_create(palette, keys):
    return dict(zip(keys, palette[:len(keys)]))

G.palette_format_dict = indexed_palette_create(G.palette_default, G.format_names)
G.palette_format_dict['CVB_FPC('+str(G.num_packet_vals_keep)+')'] = G.palette_format_dict['CVB_FPC']
G.palette_format_dict['DIV('+str(G.num_packet_vals_keep)+')'] = G.palette_format_dict['DIV']
G.palette_format_dict['DIV_RF('+str(G.num_packet_vals_keep)+')'] = G.palette_format_dict['DIV_RF']
print(G.palette_format_dict)

