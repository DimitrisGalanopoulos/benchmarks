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


figsize_base_width = 3.487
figsize_base_height = figsize_base_width / 1.618


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

plt.rcParams['figure.dpi'] = 200


# plt.rcParams['figure.figsize'] = [16, 9]
# plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.figsize'] = [2.5*figsize_base_width, 2*figsize_base_height]
# plt.rcParams['figure.figsize'] = [3*figsize_base_width, 2*figsize_base_height]
# plt.rcParams['figure.figsize'] = [4*figsize_base_width, 2*figsize_base_height]


def set_fig_size_scale(x, y):
    plt.rcParams['figure.figsize'] = [x*figsize_base_width, y*figsize_base_height]

set_fig_size_scale(2.5, 2)

