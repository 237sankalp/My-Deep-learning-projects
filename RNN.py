import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow
#DATA_PREPROSSESING
data = pd.read_csv('C:\\Users\\Dell\\Desktop\\Anaconda\\RNN\\Extracted\\Part 3 - Recurrent Neural Networks\\Google_Stock_Price_Train.csv')
train_set=data.iloc[:,1:2].values
data_test=pd.read_csv('C:\\Users\\Dell\\Desktop\\Anaconda\\RNN\\Extracted\\Part 3 - Recurrent Neural Networks\\Google_Stock_Price_Test.csv')
test_set=data_test.iloc[:,1:2].values
#FEATRUE_SCALING
SC=MinMaxScaler(feature_range=(0,1))
train_set_scaled=SC.fit_transform(train_set)
#Creating_a_datastructure_with_60_time_space
x_train=[]
y_train=[]
for i in range(60,1258):
    x_train.append(train_set_scaled[i-60:i,0])
    y_train.append(train_set_scaled[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)
#Reshaping
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
#RNN
regressor=tensorflow.keras.models.Sequential()
regressor.add(tensorflow.keras.layers.LSTM(units=100,return_sequences=True, input_shape=(x_train.shape[1],1)))
regressor.add(tensorflow.keras.layers.Dropout(0.2))
regressor.add(tensorflow.keras.layers.LSTM(units=100,return_sequences=True))
regressor.add(tensorflow.keras.layers.Dropout(0.2))
regressor.add(tensorflow.keras.layers.LSTM(units=100,return_sequences=True))
regressor.add(tensorflow.keras.layers.Dropout(0.2))
regressor.add(tensorflow.keras.layers.LSTM(units=100,return_sequences=True))
regressor.add(tensorflow.keras.layers.Dropout(0.2))
regressor.add(tensorflow.keras.layers.LSTM(units=100))
regressor.add(tensorflow.keras.layers.Dropout(0.2))
regressor.add(tensorflow.keras.layers.Dense(units=1,activation='sigmoid'))
regressor.compile(optimizer='adam',loss='mean_squared_error')
regressor.fit(x_train,y_train,batch_size=32,epochs=100)
dataset=pd.concat((data['Open'],data_test['Open']),axis=0)
inputs=dataset[len(dataset)-len(data_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=SC.transform(inputs)
x_test=[]
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])

x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predictions=regressor.predict(x_test)
predictions=SC.inverse_transform(predictions)
print(predictions)
plt.plot(test_set,color='red',label='real')
plt.plot(predictions,color='blue',label='prediction')
plt.show()