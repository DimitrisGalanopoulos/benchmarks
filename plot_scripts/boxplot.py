import sys
import math as m
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.ticker as ticker
from matplotlib.cbook import boxplot_stats
import seaborn as sns

import global_env as G
from util import *
from plot_util import *
from pallete import *


# Seaborn warning:
#   Passing 'palette' without assigning 'hue' is deprecated and will be removed in v0.14.0.
#   Assign the 'x' variable to 'hue' and set 'legend=False' for the same effect.


class Boxplot:
    ax = None

    def __init__(self, **args):
        plt.clf()
        fig, ax = plt.subplots()   # Without passing the ax to seaborn, it doesn't inherit the pyplot rcParams.
        self.ax = sns.boxplot(
                ax=ax, **args,
                boxprops=dict(edgecolor='black'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                flierprops=dict(marker='d', markersize=3, markeredgecolor='black', markerfacecolor='black'),
                saturation=1,
                )

        # self.ax.xaxis.set_major_formatter(ticker.EngFormatter())
        # self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(human_number_format_binary))
        # ticks = self.ax.axes.get_xticks()
        # xlabels = ['$' + human_number_format_binary(x, 0) for x in ticks]
        # self.ax.set_xticklabels(xlabels)


    def set_labels(self, xlabel, ylabel):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)


    def plot(self, file_out):
        plt.savefig(file_out, bbox_inches='tight', pad_inches=0)

    def plot_custom(self, file_out, **args):
        plt.savefig(file_out, bbox_inches='tight', **args)


    def get_box_coords(self):
        i = 0
        boxes = []
        for c in self.ax.get_children():
            if type(c) != matplotlib.patches.PathPatch:
                continue
            transform = c.get_data_transform().inverted()
            box = c.get_extents()
            # lower left
            x0 = box.x0
            y0 = box.y0
            # upper right
            x1 = box.x1
            y1 = box.y1
            [x0, y0] = transform.transform((x0, y0))
            [x1, y1] = transform.transform((x1, y1))
            boxes.append((x0, y0, x1, y1))
        return sorted(boxes)


    def change_xticks_labels(self, lut):
        xticks = self.ax.get_xticks()
        xlabels = self.ax.get_xticklabels()
        for i in range(len(xlabels)):
            l = xlabels[i]
            t = l.get_text()
            if (t in lut):
                l = matplotlib.text.Text(text=lut[t])
            xlabels[i] = l
        self.ax.set_xticks(xticks, labels=xlabels)

