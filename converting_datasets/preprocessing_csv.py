import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


path = r'../csv_files/combined_250000.csv'

df = pd.read_csv(path, sep=',', low_memory=False)

#dropping duplicates:
print(f'before dropping duplicates: {df.shape}')
df = df.drop_duplicates()
print(f'after dropping duplicates: {df.shape}')

#manually dropping unwanted columns
#axis=1 tells Python that you want to apply function on columns instead of rows.
print('-------------------------------')
print(f'before dropping columns: {df.shape}')
df = df.drop(['Label', 'Timestamp', 'Flow ID', 'Idle Max', 'Idle Mean', 'Idle Min'], axis=1)
print(f'after dropping columns: {df.shape}')

#dropping infinite values:
df.replace([np.inf, -np.inf], np.nan, inplace=True)

#dropping nulls:
df = df.dropna()
#filling nulls with mean:
#df = df.fillna(df.mean(), inplace=True)
 
print('-------------------------------')
df[['Flow Bytes/s','Flow Packets/s']] = df[['Flow Bytes/s','Flow Packets/s']].astype('float')
#print(df.info(verbose=True))

le = LabelEncoder()

objList = df.select_dtypes(include = "object").columns
#print (objList)

for features in objList:
    df[features] = le.fit_transform(df[features].astype(str))

df.to_csv(r'../csv_files/final_preprocessed.csv', index=False)

'''
print(df.shape)

X = df.iloc[:, :-1]
Y = df.iloc[:, -1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

sc_x = StandardScaler()

X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

model = LogisticRegression()
model.fit(X_train, Y_train)
percentage = round(model.score(X_test, Y_test), 2)
print(f'logistic regression score: {percentage*100}%')
'''
