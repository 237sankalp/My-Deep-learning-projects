#For Dataset#
#https://drive.google.com/file/d/1Q8KOOD6KN39mZf4DBG0cHTL96Xk2mM0Z/view?usp=sharing#


import tensorflow
from  tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
#IMAGE AUGMENTATION
train_data_gen=ImageDataGenerator(rescale=1./255,shear_range=0.2,rotation_range=0.2,zoom_range=0.2,horizontal_flip='True')
training_set=train_data_gen.flow_from_directory('E:\\FUTURE_API\\train',target_size=(64,64),color_mode="rgb",class_mode='categorical',batch_size=32)

test_data_gen=ImageDataGenerator(rescale=1./255)
test_set=test_data_gen.flow_from_directory('E:\\FUTURE_API\\test',target_size=(64,64),color_mode="rgb",class_mode='categorical',batch_size=32)

###CNN MODEL
cnn=tensorflow.keras.Sequential()
###
cnn.add(tensorflow.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
cnn.add(tensorflow.keras.layers.BatchNormalization(epsilon=0.005,fused='True'))
cnn.add(tensorflow.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tensorflow.keras.layers.Conv2D(filters=16,kernel_size=2,activation='relu'))
cnn.add(tensorflow.keras.layers.BatchNormalization(epsilon=0.005,fused='True'))
cnn.add(tensorflow.keras.layers.MaxPool2D(pool_size=3,strides=3))
#Flattening
cnn.add(tensorflow.keras.layers.Flatten())
##Dense
cnn.add(tensorflow.keras.layers.Dense(units=512,activation='relu'))
cnn.add(tensorflow.keras.layers.Dense(units=256,activation='relu'))
cnn.add(tensorflow.keras.layers.Dense(units=225,activation='softmax'))
cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#TESTING
history=cnn.fit(x=training_set,validation_data=test_set,epochs=10)

#PLOTTING
acx=history.history['accuracy']
val_acc=history.history['val_accuracy']
epochs = range(1,11)
plt.plot(epochs,acx,'g',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#PLOTTING
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,11)
plt.plot(epochs,loss,'g',label='Training accuracy')
plt.plot(epochs,val_loss,'b',label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
