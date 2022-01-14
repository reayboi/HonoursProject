import os.path
from os import scandir
import pandas as pd

benign_files = []
df = pd.DataFrame()

#This scirpts uses an Absolute Paths to files that have not been included with the code listing
#It will not be possible to run this script

path = r'D:\[CIC OUTPUT]\NORMAL'
save = r'D:ben.csv'

for file in os.listdir(path):
    if (file[-3:] == "csv"):
        benign_files.append(os.path.join(path, file))



for file in benign_files:
    df = df.append(pd.read_csv(file, sep=',', low_memory=False), sort=True)

print(df.shape)
df.loc[:,'Class'] = '0'
print(df.shape)
df.to_csv(save, index=False)