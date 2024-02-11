import sys
import math as m
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns


bench_path = '..'


num_packet_vals_keep = 16384


format_names = ['CVB_ID', 'CVB_d2f', 'CVB_FPC', 'CVB_FPZIP', 'CVB_ZFP', 'CVB_ZFP_lossy_1e-3', 'Dictionary(Custom)', 'CSR5', 'Merge', 'SparseX', 'CSR', 'CSR avx2', 'MKL(64-bits)', 'MKL_idx0(64-bits)', 'MKL(32-bits)', 'DIV', 'DIV_RF']


# column_names = [
        # 'mtx_name', 'distribution', 'placement', 'seed',
        # 'nr_rows', 'nr_cols', 'nr_nzeros',
        # 'density', 'mem_footprint', 'mem_range',
        # 'avg_nnz_per_row', 'std_nnz_per_row',
        # 'avg_bw', 'std_bw',
        # 'avg_bw_scaled', 'std_bw_scaled',
        # 'avg_sc', 'std_sc',
        # 'avg_sc_scaled', 'std_sc_scaled',
        # 'skew',
        # 'avg_num_neighbours', 'cross_row_similarity',
        # 'implementation',  'time', 'gflops', 'W_avg', 'J_estimated'
        # ]

column_names = [
        'matrix_name',
        'num_threads',
        'csr_m',
        'csr_n',
        'csr_nnz',
        'time',
        'gflops',
        'csr_mem_footprint',
        'W_avg',
        'J_estimated',
        'format_name',
        'm',
        'n',
        'nnz',
        'mem_footprint',
        'mem_ratio',
        ]


column_names_packets = column_names + ['CSRCV_NUM_PACKET_VALS']


# all devices that will be used
ranges_dev = ['Tesla-P100', 'Tesla-V100', 'Tesla-A100', 'AMD-EPYC-24', 'AMD-EPYC-64', 'ARM-NEON', 'INTEL-XEON', 'INTEL-ICY', 'IBM-POWER9', 'Alveo-U280']

