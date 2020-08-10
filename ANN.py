import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
import tensorflow as tf
from sklearn.metrics import confusion_matrix
##################################################################################PREPROSSESING
#Analysis and division
dataset = pd.read_csv('C:\\Users\\Dell\\Desktop\\Anaconda\\Machine Learning A-Z (Codes and Datasets)\\Part 8 - Deep Learning\\Section 39 - Artificial Neural Networks (ANN)\\Python\\data.csv')
X=dataset.iloc[:,3:-1].values
Y=dataset.iloc[:,-1].values
z=[1,0,0,600,1,40,3,60000,2,1,1,50000]
#label encoding from string to a integer for male and female
X[:,2]=LabelEncoder().fit_transform(X[:,2])

#label encoding for geography
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

#checking for missing data
impute=SimpleImputer(missing_values=np.nan,strategy='mean')
impute.fit(X[:,1:3])
X[:,1:3]=impute.transform(X[:,1:3])

#splitting dataset
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling(IMPORTANT)
SS=StandardScaler()
x_train=SS.fit_transform(x_train)
x_test=SS.fit_transform(x_test)
###########################################################################################
#Initialize a ANN
ann=tf.keras.models.Sequential()
#Adding input layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
#Training
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ann.fit(x_train,y_train,batch_size=32,epochs=50)
#Prediction
loss,acc=ann.evaluate(x_test,y_test)
print(loss,acc)
#prediction of actual person
p=ann.predict(SS.transform([z]))
print(p>0.5)
#prediction for x_test
pre=ann.predict(x_test)
pre=(pre>0.5)
print(np.concatenate((pre.reshape(len(pre),1),y_test.reshape(len(y_test),1)),1))
cm=confusion_matrix(y_test,pre)
print(cm)