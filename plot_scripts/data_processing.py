import sys
import math as m
import numpy as np
import scipy as sp
import pandas as pd

import global_env as G
from util import *


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def read_bench_file(path, column_names, format_name, system):
    if column_names != None:
        df = pd.read_csv(path, names=column_names, usecols=column_names)
    else:
        df = pd.read_csv(path)
    df['format_name'] = format_name
    df['System'] = system
    return df

def read_features_file():
    return pd.read_csv(G.bench_path + '/matrix_features.csv', sep='\t', index_col=0)



def concat_data_and_preprocess(dfs):
    df = pd.concat(dfs, ignore_index=True, sort=False)
    # df = df.dropna().reset_index(drop=True)
    df = df.reset_index(drop=True)
    print(df)
    if ('CSRCV_NUM_PACKET_VALS' in df):
        df['CSRCV_NUM_PACKET_VALS'] = df['CSRCV_NUM_PACKET_VALS'].apply(human_number_format)
        print(df)
        # print(df[df['CSRCV_NUM_PACKET_VALS'].isna()])
        # exit()
    matrix_names = [
        'spal_004'          ,
        'ldoor'             ,
        'dielFilterV2real'  ,
        'af_shell10'        ,
        'nv2'               ,
        'boneS10'           ,
        'circuit5M'         ,
        'Hook_1498'         ,
        'Geo_1438'          ,
        'Serena'            ,
        'vas_stokes_2M'     ,
        'bone010'           ,
        'audikw_1'          ,
        'Long_Coup_dt6'     ,
        'Long_Coup_dt0'     ,
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
    matrix_ids = {}
    i = 1
    for m in matrix_names:
        matrix_ids[m] = '(' + str(i) + ')'
        i+=1
    df['matrix_name'] = df['matrix_name'].replace({r'.*/' : '', '\.mtx$' : ''}, regex=True)
    df['matrix_id'] = df['matrix_name'].replace(matrix_ids, regex=True)
    print(df)
    return df


def filter_num_packet_vals(df, num_packet_vals):
    num_packet_vals = human_number_format(num_packet_vals)
    df['CSRCV_NUM_PACKET_VALS'] = df['CSRCV_NUM_PACKET_VALS'].fillna(num_packet_vals)
    df = df[df['CSRCV_NUM_PACKET_VALS'] == num_packet_vals]
    print(df)
    return df


def calculate_gmeans(df):
    df_gmeans = df.groupby(['format_name', 'System'], as_index=False)['gflops'].apply(sp.stats.gmean)
    df_gmeans['matrix_id'] = 'geo\nmean'
    df = pd.concat([df, df_gmeans], ignore_index=True, sort=False)
    print(df)
    return df


