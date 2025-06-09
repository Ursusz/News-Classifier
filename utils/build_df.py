import pandas as pd

df1 = pd.read_csv('../dataset/Fake.csv')
df1.drop(columns=['subject', 'date'], inplace=True)
df1['label'] = 'FAKE'

df2 = pd.read_csv('../dataset/True.csv')
df2.drop(columns=['subject', 'date'], inplace=True)
df2['label'] = 'REAL'

df_concat = pd.concat([df1, df2], ignore_index=True)

df_randomized = df_concat.sample(frac=1, random_state=42).reset_index(drop=True)

df_randomized.to_csv('dataset/DataSet.csv', index=False)
