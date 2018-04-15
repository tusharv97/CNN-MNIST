import numpy as np
import keras
from keras import layers
from keras.layers import Input, Dense,Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import Convolution2D,AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model,Sequential
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import pandas as pd
from sklearn.model_selection import train_test_split
import h5py

train=pd.read_csv('data.csv')



x_train=train.iloc[:38000,1:].values.reshape(38000,28,28,1)
x_test=train.iloc[38000:,1:].values.reshape(4000,28,28,1)

y_train=train.iloc[:38000,0].values
y_test=train.iloc[38000:,0].values
x_train=np.array(x_train)
x_test=np.array(x_test)

y_train=keras.utils.to_categorical(y_train,num_classes=10)
y_test=keras.utils.to_categorical(y_test,num_classes=10)
y_train=np.array(y_train)
y_test=np.array(y_test)

model=Sequential()
model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=64)
model.save('trained-model.h5')
preds=model.evaluate(x_test,y_test)

print('Accuracy:'+str(preds[1]))
