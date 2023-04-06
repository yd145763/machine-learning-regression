# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:30:04 2023

@author: limyu
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split

#import csv
url_shareprice = "https://raw.githubusercontent.com/yd145763/machine-learning-regression/main/stock_price%20(1).csv"
df_shareprice = pd.read_csv(url_shareprice)

url_dow = "https://raw.githubusercontent.com/yd145763/machine-learning-regression/main/HistoricalPrices.csv"
df_dow = pd.read_csv(url_dow)

url_exchange = "https://raw.githubusercontent.com/yd145763/machine-learning-regression/main/exchange-rates%20(1).csv"
df_exchange = pd.read_csv(url_exchange)

df_exchange.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)


#convert and standardize datetime formate
df_shareprice["Date"] = pd.to_datetime(df_shareprice["Date"], format='%Y%m%d')
df_dow['Date'] = pd.to_datetime(df_dow['Date'], format='%m/%d/%y')
df_exchange['Date'] = df_exchange['Date'].str.replace(' ', '')
df_exchange['Date'] = pd.to_datetime(df_exchange['Date'], format='%d%b%Y', errors='coerce')


#remove all spaces
df_dow = df_dow.rename(columns=lambda x: x.replace(' ', ''))
merged_df = pd.merge(df_shareprice, df_dow, on='Date', how='inner', suffixes=('_share', '_dow'))
print(merged_df.columns)

# compute A - B, whether the share will up or drop at the end of the day
diff = merged_df['Close_share'] - merged_df['Open_share']

# create a new column C based on the condition A - B, whether the share will up or drop at the end of the day
cat_diff = pd.cut(diff, bins=[-float('inf'), 0, float('inf')], labels=[0, 1], include_lowest=True)
cat_diff = cat_diff.cat.add_categories([2])  # add missing category
cat_diff.fillna(2, inplace=True)  # fill missing values with 2
merged_df['UpDown'] = cat_diff.astype(int)

# sort the DataFrame by ascending date
merged_df = merged_df.sort_values('Date', ascending=True)

# add "Yesterday" column
merged_df['Yest_High_share'] = merged_df['High_share'].shift(1)
merged_df['Yest_Low_share'] = merged_df['Low_share'].shift(1)
merged_df['Yest_Close_share'] = merged_df['Close_share'].shift(1)
merged_df['Yest_Volume'] = merged_df['Volume'].shift(1)
merged_df['Yest_Open_dow'] = merged_df['Open_dow'].shift(1)
merged_df['Yest_High_dow'] = merged_df['High_dow'].shift(1)
merged_df['Yest_Low_dow'] = merged_df['Low_dow'].shift(1)
merged_df['Yest_Close_dow'] = merged_df['Close_dow'].shift(1)

print(merged_df.columns)

#set columns of parameters to be included in the model
col_norm = ['Open_share', 'Yest_High_share', 'Yest_Low_share', 'Yest_Close_share', 
            'Yest_Volume', 'Yest_Open_dow', 'Yest_High_dow', 'Yest_Low_dow', 'Yest_Close_dow']

#set feature of the model
feat_Open_share = tf.feature_column.numeric_column('Open_share')
feat_Yest_High_share = tf.feature_column.numeric_column('Yest_High_share')
feat_Yest_Low_share = tf.feature_column.numeric_column('Yest_Low_share')
feat_Yest_Close_share = tf.feature_column.numeric_column('Yest_Close_share')
feat_Yest_Volume = tf.feature_column.numeric_column('Yest_Volume')
feat_Yest_Open_dow  = tf.feature_column.numeric_column('Yest_Open_dow')
feat_Yest_High_dow = tf.feature_column.numeric_column('Yest_High_dow')
feat_Yest_Low_dow  = tf.feature_column.numeric_column('Yest_Low_dow')
feat_Yest_Close_dow  = tf.feature_column.numeric_column('Yest_Close_dow')

#lump all the features into a list
feat_cols = [feat_Open_share, feat_Yest_High_share, feat_Yest_Low_share, feat_Yest_Close_share,
             feat_Yest_Volume, feat_Yest_Open_dow, feat_Yest_High_dow, feat_Yest_Low_dow,
             feat_Yest_Close_dow] 

# normalize each column of the dataframe
df1_norm = merged_df[col_norm]
df1_norm = df1_norm.iloc[1:, :]

print(df1_norm.dtypes)

# create a MinMaxScaler object
scaler = MinMaxScaler()
df2_norm = pd.DataFrame(scaler.fit_transform(df1_norm), columns=df1_norm.columns)

#set manipulative variables and responding variable
Y_Data = merged_df["UpDown"][1:]
Y_Data = Y_Data.reset_index(drop=True)
X_Data = df2_norm

#split data into train data and test data
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Data, Y_Data, test_size = 0.3,random_state=101)

#set input function
input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_Train, 
                                                 y = Y_Train, 
                                                 batch_size = 10, 
                                                 num_epochs = 100,
                                                 shuffle = True)

#set test function
test_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_Test, 
                                                 y = Y_Test, 
                                                 batch_size = 10, 
                                                 num_epochs = 1,
                                                 shuffle = False)

#form model
model = tf.estimator.LinearClassifier(feature_columns = feat_cols, 
                                      n_classes = 2)

#train model 
model.train(input_fn = input_func, steps = 5000)

#evaluate the input function
eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_Test,
                                                      y=Y_Test,
                                                      batch_size=40,
                                                      num_epochs=1,
                                                      shuffle=False)
#evaluate the results
results = model.evaluate(eval_input_func)

#print the results
print(" ")
print(results)
print(" ")
    



