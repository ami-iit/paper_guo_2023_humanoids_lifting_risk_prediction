import pandas as pd
import numpy as np

## concatate all 6 data sets
data_path01 = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/01_cheng_labeled.txt'
data_path02 = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/02_cheng_labeled.txt'
data_path03 = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/03_cheng_labeled.txt'
data_path04 = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/01_lorenzo_labeled.txt'
data_path05 = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/02_lorenzo_labeled.txt'
data_path06 = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/03_lorenzo_labeled.txt'

df_raw01 = pd.read_csv(data_path01, sep=' ')
df_raw02 = pd.read_csv(data_path02, sep=' ')
#df_raw02_nohead = df_raw02.iloc()
df_raw03 = pd.read_csv(data_path03, sep=' ')
df_raw04 = pd.read_csv(data_path04, sep=' ')
df_raw05 = pd.read_csv(data_path05, sep=' ')
df_raw06 = pd.read_csv(data_path06, sep=' ')

rows = df_raw01.shape[0]+df_raw02.shape[0]+df_raw03.shape[0]+df_raw04.shape[0]+df_raw05.shape[0]+df_raw06.shape[0]

df_combined = pd.concat([df_raw01, df_raw02, df_raw03, df_raw04, df_raw05, df_raw06].copy(), axis=0)
print(df_combined)
print('rows are: ', rows)
# save the df