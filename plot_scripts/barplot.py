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


# saturation : float, optional
# Proportion of the original saturation to draw colors at. Large patches often look better with slightly desaturated colors, but set this to 1 if you want the plot colors to perfectly match the input color spec.


class Barplot:
    ax = None

    def __init__(self, **args):
        plt.clf()
        fig, ax = plt.subplots()   # Without passing the ax to seaborn, it doesn't inherit the pyplot rcParams.
        self.ax = sns.barplot(
                ax=ax, **args,
                linewidth=0.8, edgecolor='w',
                # order = ['male', 'female'],
                # capsize = 0.05,
                # saturation = 8,
                # errcolor = 'gray', errwidth = 2,
                errorbar=None,
                # ci=None,
                saturation=1,
                )


    def set_labels(self, xlabel, ylabel):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)


    def plot(self, file_out):
        plt.savefig(file_out, bbox_inches='tight', pad_inches=0)


    def get_bar_coords(self):
        boxes = []
        for bars in self.ax.containers:   # Each element can be multiple bars (e.g., grouped bars -> each hue group).
            for c in bars.get_children():
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


    def change_legend_labels(self, lut):
        labels = self.ax.get_legend().get_texts()
        for i in range(len(labels)):
            t = labels[i].get_text()
            if (t in lut):
                t = lut[t]
            labels[i] = t
        self.ax.legend(labels=labels)

