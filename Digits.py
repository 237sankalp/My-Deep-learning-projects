import tensorflow as tf
import numpy as np
import cv2
data= tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=data.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)
val_loss,val_acc=model.evaluate(x_test,y_test)
print(val_acc,val_loss)
prediction=model.predict([x_test])
print(np.argmax(prediction[0]))
cv2.imshow("Predicted",x_test[0])
cv2.waitKey(0)
cv2.destroyAllWindows()