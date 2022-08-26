# Import library

import numpy as np
import seaborn as sns
import matplotlib as plt
# import pandas as pd
import sklearn
import vaex

# from tabulate import tabulate
# pdtabulate=lambda df:tabulate(df,headers='keys',tablefmt='psql')


# Read csv and convert to HDF5


train_path = '/mnt/b7b917e1-da96-4995-b7db-c30035d41dbe/Machine Learning Project/AMEX_2022/amex-default-prediction/train_data.csv'
test_path = '/mnt/b7b917e1-da96-4995-b7db-c30035d41dbe/Machine Learning Project/AMEX_2022/amex-default-prediction/test_data.csv'
hdf5_path = '/mnt/b7b917e1-da96-4995-b7db-c30035d41dbe/Machine Learning Project/'\
    'AMEX_2022/amex-default-prediction/train_data.csv.hdf5'

# df_5 = vaex.from_csv(train_path, chunk_size=10000)
df_5 = vaex.open(hdf5_path)

print('\n')
print(df_5.head(5))
print('\n')
print(df_5.describe())
print('\n')
print(df_5.info)
