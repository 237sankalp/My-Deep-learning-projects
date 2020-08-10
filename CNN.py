import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib
###################PREPROSSESING
train_data=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip='True')
train=train_data.flow_from_directory('C:\\Users\\Dell\\Desktop\\Anaconda\\Section 40 - Convolutional Neural Networks (CNN)\\dataset\\training_set',target_size=(64,64),batch_size=32,class_mode='binary')
test_data=ImageDataGenerator(rescale=1./255)
test=test_data.flow_from_directory('C:\\Users\\Dell\\Desktop\\Anaconda\\Section 40 - Convolutional Neural Networks (CNN)\\dataset\\test_set',target_size=(64,64),batch_size=32,class_mode='binary')
#Initialize
cnn=tf.keras.models.Sequential()
#convolutional
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#flattening
cnn.add(tf.keras.layers.Flatten())
#ANN
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Evaluation

cnn.fit(x=train ,validation_data=test, epochs=25)

test_img=image.load_img('C:\\Users\\Dell\\Desktop\\Anaconda\\Section 40 - Convolutional Neural Networks (CNN)\\dataset\\single_prediction\\cat_or_dog_2.jpg',target_size=(64,64))
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)
pre=cnn.predict(test_img)
train.class_indices
if pre[0][0] == 0:
    result='cat'
else:
    result='dog'

print(result)

f=open("C:\\Users\\Dell\\Desktop\\Anaconda\\CNN_MODEL\\mod","wb")
joblib.dump(cnn,f)
