import sys
import math as m
import numpy as np
import scipy as sp


bench_path = '..'


num_packet_vals_keep = 16384


format_names = ['CVB_ID', 'CVB_d2f', 'CVB_FPC', 'CVB_FPZIP', 'CVB_ZFP', 'CVB_ZFP_lossy_1e-3', 'Dictionary(Custom)', 'CSR5', 'SparseX', 'Merge', 'CSR', 'CSR avx2', 'MKL(64-bits)', 'MKL_idx0(64-bits)', 'MKL(32-bits)', 'DIV', 'DIV_RF', 'Dictionary(CSR\\&RV)', 'LCM']


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


matrix_names_comression = [
    'spal_004'          ,
    'ldoor'             ,
    'dielFilterV2real'  ,
    'nv2'               ,
    'af_shell10'        ,
    'boneS10'           ,
    'circuit5M'         ,
    'Hook_1498'         ,
    'Geo_1438'          ,
    'Serena'            ,
    'vas_stokes_2M'     ,
    'bone010'           ,
    'audikw_1'          ,
    'Long_Coup_dt0'     ,
    'Long_Coup_dt6'     ,
    'dielFilterV3real'  ,
    'nlpkkt120'         ,
    'cage15'            ,
    'ML_Geer'           ,
    'Flan_1565'         ,
    'Cube_Coup_dt0'     ,
    'Cube_Coup_dt6'     ,
    'Bump_2911'         ,
    'vas_stokes_4M'     ,
    'nlpkkt160'         ,
    'HV15R'             ,
    'Queen_4147'        ,
    'stokes'            ,
    'nlpkkt200'         ,
]


matrix_names_small = [
    'ts-palko',
    'neos',
    'stat96v3',
    'stormG2_1000',
    'xenon2',
    's3dkq4m2',
    'apache2',
    'Si34H36',
    'ecology2',
    'LargeRegFile',
    'largebasis',
    'Goodwin_127',
    'Hamrle3',
    'boneS01',
    'sls',
    'cont1_l',
    'CO',
    'G3_circuit',
    'degme',
    'atmosmodl',
    'SiO2',
    'tp-6',
    'af_shell3',
    'circuit5M_dc',
    'rajat31',
    'CurlCurl_4',
    'cage14',
    'nlpkkt80',
    'ss',
    'boneS10',
]



