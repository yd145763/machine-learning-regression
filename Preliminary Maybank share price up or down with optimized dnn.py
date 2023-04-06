import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt

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

#set columns of parameters to be included in the model
col_norm = ['Open_share', 'Yest_High_share', 'Yest_Low_share', 'Yest_Close_share', 
            'Yest_Volume', 'Yest_Open_dow', 'Yest_High_dow', 'Yest_Low_dow', 'Yest_Close_dow', 'UpDown']

df1_norm = merged_df[col_norm]
df1_norm = df1_norm.iloc[1:, :]

# create a MinMaxScaler object
scaler = MinMaxScaler()
df2_norm = pd.DataFrame(scaler.fit_transform(df1_norm), columns=df1_norm.columns)

data = df2_norm

# Split the data into training and test sets using Pandas
train_data = data.sample(frac=0.6, random_state=0)
test_data = data.drop(train_data.index)

# Separate the input variables (features) from the output variable (target) using Pandas
train_features = train_data.drop('UpDown', axis=1)
train_labels = train_data['UpDown']
test_features = test_data.drop('UpDown', axis=1)
test_labels = test_data['UpDown']

# Define the model using Keras
model = keras.Sequential([
    layers.Dense(1000, activation='relu', input_shape=[len(train_features.keys())]),
    layers.Dense(800, activation='relu'),
    layers.Dense(500, activation='relu'),
    layers.Dense(200, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
opt = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_features, train_labels, epochs=200)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_features, test_labels, verbose=2)
results = model.evaluate(test_features, test_labels, verbose=2, batch_size=10)

predict_data_probs = model.predict(test_features)
predict_data = (predict_data_probs > 0.5).astype(int)

skplt.metrics.plot_confusion_matrix(test_labels, 
                                    predict_data,
                                   figsize=(6,6),
                                   title="Confusion Matrix")


print('\nTest accuracy:', test_acc)

cm = confusion_matrix(test_labels, predict_data)
accuracy = accuracy_score(test_labels, predict_data)
precision = precision_score(test_labels, predict_data)

print('Confusion matrix:')
print(cm)
print('Accuracy:', accuracy)
print('Precision:', precision)
