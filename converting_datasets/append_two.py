import pandas as pd 
import os

buffer = []

path = r'../csv_files/to_append/'
save = r'../csv_files/combined.csv'

df = pd.DataFrame()

for file in os.listdir(path):
    if (file[-3:] == "csv"):
        buffer.append(os.path.join(path, file))

for file in buffer:
    df = df.append(pd.read_csv(file, sep=','), sort=False)

df.pop('index')
df.pop('Unnamed: 0')

df.to_csv(save, index=False)
print("Appending completed.\nSuccessfully saved to file. ")

