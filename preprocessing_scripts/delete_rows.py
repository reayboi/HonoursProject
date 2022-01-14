import pandas as pd
import numpy as np

#The dataset that this file uses are not included within the code listing
#It is not possible to run this script

path = "csv_files/benign.csv"
df = pd.read_csv(path, sep=",")
df = df.sample(frac=1).reset_index()
df = df[df.index <= 250000]
print(df.shape)
df.to_csv("csv_files/benign_with_250000.csv", index=True)



