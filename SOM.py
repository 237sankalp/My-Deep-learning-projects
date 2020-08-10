import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone,pcolor,colorbar,plot,show
dataset=pd.read_csv('E:\\Self Organizing Map\\Extracted\\Self_Organizing_Maps\\Credit_Card_Applications.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
SC=MinMaxScaler(feature_range=(0,1))
x_scaled=SC.fit_transform(x)
som=MiniSom(x=10,y=10,input_len=15,learning_rate=0.8)
som.random_weights_init(x)
som.train_random(data=x,num_iteration=200)
bone()
pcolor(som.distance_map().T)
colorbar()
MARKER=['o','s']
colors=['r','g']
for i,z in enumerate(x):
    w=som.winner(z)
    plot(w[0]+0.5,
         w[1]+0.5,
         MARKER[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()