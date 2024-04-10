#!/usr/bin/env -S ${HOME}/lib/python/bin/python

import sys
import numpy as np
import pandas as pd


df = pd.read_csv(sys.argv[1])
df['matrix_name'] = df['matrix_name'].replace({r'.*/' : '', r'\.mtx$' : ''}, regex=True)


df_gflops = df[['matrix_name', 'CSRCV_NUM_PACKET_VALS', 'gflops']]
df_gflops = df.pivot(index='matrix_name', columns='CSRCV_NUM_PACKET_VALS', values='gflops')
# print(df_gflops)


df_features = df
df_features = df_features[df_features['CSRCV_NUM_PACKET_VALS'] == 16384]
df_features = df_features.set_index('matrix_name')
df_features = df_features[['csr_m', 'csr_n', 'csr_nnz', 'csr_mem_footprint', 'mem_footprint', 'mem_ratio']]
# print(df_features)


df = pd.concat([df_features, df_gflops], axis=1)
df = df.sort_values(['csr_mem_footprint'])
# print(df)


# df.to_csv(sys.stdout, sep='\t', index=False)
df.to_csv(sys.stdout, sep='\t', header=False, float_format='%g')

