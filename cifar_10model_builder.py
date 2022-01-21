#%%
from keras.datasets import cifar10
#%%
#fotograflara yükledim
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#%%
#veriyi normalleştirme
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
#%%
#burdada one hat encoding(bilgisayara daha kolay olur diye)
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#%%
# cnn kullandım
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout,BatchNormalization
import tensorflow as tf
from keras import optimizers
import datetime, os
model = Sequential()
# Ağın katmanları aşağıdalar
model.add(Conv2D(input_shape=(32,32,3),filters=32,kernel_size=(2,2),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
#%%
# modelin eğitimi
model.fit(x_train , y_train,batch_size=35,epochs=20,verbose=1)
#%%
# son modeli test veri üzerinde değerlendirdim ve doğrulama sonucu (78%)
x1=model.evaluate(x_test,y_test)
print("testing data = ",x1)
x2=model.evaluate(x_train,y_train)
print("training data = ",x2)
#%%
model.save('CIFAR-10-MODEL.h5')
