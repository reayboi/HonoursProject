import os.path
from os import scandir
import pandas as pd

#This script uses absolute paths to files that are not included within the code listing
#An error will be printed because the files are not available.

mal_files = []
read_path = r'D:\[CIC OUTPUT]'
print(read_path)
save_name = r'D:\Project_Files\[FINAL CSV]\malicious.csv'
print(save_name)

for files in os.listdir(read_path):
    if (files[-3:] == "csv"):
        #print(files)
        mal_files.append(os.path.join(read_path, files))

df = pd.DataFrame()

for f in mal_files:
    df = df.append(pd.read_csv(f, sep=',', low_memory=False), sort=True)

print(df.shape)
df.loc[:,'Class'] = 1
print(df.shape)
df.to_csv(save_name, index=False)