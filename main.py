import pandas as pd 

#data Processing (clean data)
df = pd.read_csv('loan_data.csv')
#change purpose from string to bool now on use df_data นะ
df_data = pd.get_dummies(df, columns=['purpose'], drop_first=True)
print(df_data.head())