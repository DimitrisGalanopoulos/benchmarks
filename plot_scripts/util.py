import sys
import math as m
import numpy as np
import scipy as sp
import pandas as pd

import global_env as G


def human_number_format(num):
    if m.isnan(num):
        return(num)
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        # num /= 1000.0
        num //= 1000.0
    # add more suffixes if you need them
    # return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    return '%d%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

