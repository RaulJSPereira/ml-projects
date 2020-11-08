# Import numpy and pandas
# Import categorical encoding, necessary layers and Model from keras

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model




# Load datasets into pandas

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
print(train.head(5))




# I set some parameters:
# m  = number of training examples,
# dt = total number of pixels,
# d  = number of pixels on dimension of the matrix (square root of d2),
# n  = number of classes

m=train.shape[0]
dt=train.shape[1]-1
d=int(np.sqrt(dt))
n=len(train.label.value_counts())
print([m,dt,d,n])




# I check that neither the training nor the test sets have missing values

print(test.columns[pd.concat([train.drop('label',axis=1),test]).isnull().sum()>0])




# I check that there is an even distribution of training examples across the different classes

print(train['label'].value_counts(sort=False))




# Some statistics on the pixel values. 
# I compute Pixel_stats for each pixel, using all training example. 
# Then I get statistics for each Pixel_stat.
# For example, the minimum value for each pixel is always 0, but the maximum value is not always 255.

print(train.drop('label',axis=1).describe().T.drop('count',axis=1).describe().drop('count',axis=0).add_prefix('Pixel_'))




# I get the dataset ready for usage with keras.
# Max: maximum pixel value in the training dataset
# x: array of pixel values from training dataset, normalized by Max
# y: labels transformed into n-dimensional array
# x_test: array of pixel values from test dataset, normalized by Max

Max=train.max().max()
x=train.iloc[:,1:].values/Max
y=to_categorical(train.label,n)
x_test=test.values/Max
print([x.shape,x.max(),y.shape])
print([x_test.shape,x_test.max()])




# I check that the categorical encoding of y with '1' at position 'i' of the n-dim array corresponds to label 'i'

print([set(train.label[np.argmax(y,axis=1)==i])=={i} for i in range(10)])




# I define a fully connected network.
# Each hidden layer uses ReLU as activation function, and I use dropout after each of them for regularization
# The number of neurons in each layer is given by the input array dims
# The output layer has n neurons and uses softmax activation

def DenseModel(dims,dropout_rate):
    X_input=Input(shape=(dt))
    X=X_input
    for dim in dims:
        X=Dense(dim,activation='relu')(X)
        X=Dropout(dropout_rate)(X)
    X_output=Dense(n,activation='softmax')(X)
    model=Model(inputs=X_input,outputs=X_output)
    return model




# The first model I create has 5 hidden layers with dimensions (392, 196, 196, 98, 98), and 35% dropout rate.
# I fit the model with 20 epochs of training, batch size of 128 examples, and 20% of the dataset left for validation

dims=[dt//2,dt//4,dt//4,dt//8,dt//8]
model1=DenseModel(dims,0.35)
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model1.summary())

model1.fit(x,y,batch_size=128,epochs=20,validation_split=0.2)




# I set the prediction to the the class with the highest probability in the output layer (softmax) 

pred1=pd.DataFrame({'ImageId':range(1,len(x_test)+1),'Label':model1.predict(x_test).argmax(axis=1)}).set_index('ImageId')
pred1.to_csv('pred1.csv')




# The second model has 7 hidden layers with dimensions (784, 784, 392, 392, 196, 196, 98), and 35% dropout rate.
# I fit the model with 25 epochs of training, batch size of 128 examples, and 20% of the dataset left for validation

dims=[dt,dt,dt//2,dt//2,dt//4,dt//4,dt//8]
model2=DenseModel(dims,0.35)
model2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model2.summary())

model2.fit(x,y,batch_size=128,epochs=25,validation_split=0.2)

pred2=pd.DataFrame({'ImageId':range(1,len(x_test)+1),'Label':model2.predict(x_test).argmax(axis=1)}).set_index('ImageId')
pred2.to_csv('pred2.csv')




# I check that almost 2% of the predictions differ compared with the previous model

print(pred2.join(pred1, lsuffix='1',rsuffix='2').apply(lambda x: x.Label1==x.Label2,axis=1).value_counts()/len(x_test))




# Third model has 9 hidden layers, dimensions are (784, 784, 784, 392, 392, 392, 196, 196, 196), and 35% dropout rate.
# I fit the model with 30 epochs of training, batch size of 128 examples, and 20% of the dataset left for validation

dims=[dt,dt,dt,dt//2,dt//2,dt//2,dt//4,dt//4,dt//4]
model3=DenseModel(dims,0.35)
model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model3.summary())

model3.fit(x,y,batch_size=128,epochs=30,validation_split=0.2)

pred3=pd.DataFrame({'ImageId':range(1,len(x_test)+1),'Label':model3.predict(x_test).argmax(axis=1)}).set_index('ImageId')
pred3.to_csv('pred3.csv')




# I check that almost 2.5% of the predictions differ compared with both model 1 and 2.

print(pred3.join(pred1, lsuffix='1',rsuffix='2').apply(lambda x: x.Label1==x.Label2,axis=1).value_counts()/len(x_test))
print(pred3.join(pred2, lsuffix='1',rsuffix='2').apply(lambda x: x.Label1==x.Label2,axis=1).value_counts()/len(x_test))




# Now I create the datasets for convolutional neural networks
# x_conv: 28 by 28 array of pixel values from training dataset, normalized by Max
# x_conv_test: 28 by 28 array of pixel values from test dataset, normalized by Max

x_conv=train.values[:,1:].reshape((m,d,d,1))/Max
x_conv_test=test.values.reshape((-1,d,d,1))/Max
print([x_conv.shape,x_conv.max()])
print([x_conv_test.shape,x_conv_test.max()])




# I define a convolutional neural network.
# Each hidden layer uses ReLU as activation function, and I use dropout after each of them for regularization
# The number of filters in each convolutional layer, the kernel sizes and strides are given as input arrays
# I use square filters, with the same stride on both dimensions. I always use 'same' padding
# After convolutional layers I flatten and apply a fully connected layer, with dimension given as input
# The output layer has n neurons and uses softmax activation

def ConvModel(dims,kernels,strides,dim_dense,dropout_rate):
    data=np.array([dims,kernels,strides]).T
    X_input=Input(shape=(d,d,1))
    X=X_input
    for dim,kernel,stride in data:
        X=Conv2D(dim,kernel_size=(kernel,kernel),strides=(stride,stride),padding='same',activation='relu')(X)
        X=Dropout(dropout_rate)(X)
    X=Flatten()(X)
    X=Dense(dim_dense,activation='relu')(X)
    X_output=Dense(n,activation='softmax')(X)
    model=Model(inputs=X_input,outputs=X_output)
    return model




# Generically I increase the number of filters as I reduce the size of the matrices. All kernels are 3x3.
# I start with 28x28x1 and go to 14x14x32 with a stride of 2.
# I then use 'same' padding and unit stride to stay with 14x14 window, but increase to 64 filters
# Finally I use stride of 2 again to go to 7x7x128.
# I use 128 neurons in the hidden dense layer.
# I fit the model with 15 epochs of training, 35% dropout, and 20% of the dataset left for validation

convmodel1=ConvModel([32,64,128],[3,3,3],[2,1,2],128,0.35)
convmodel1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(convmodel1.summary())

convmodel1.fit(x_conv,y_conv,batch_size=128,epochs=15,validation_split=0.2)

convpred1=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel1.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')
convpred1.to_csv('convpred1.csv')




# I use more hidden layers, but still all kernels are 3x3.
# Number of filters is 32,64 and 128, twice each. 
# I half the size of the window every two layers (to 14x14, 7x7 and 4x4)
# I use 256 neurons in the hidden dense layer.
# I fit the model with 20 epochs of training, 35% dropout, and 20% of the dataset left for validation

convmodel2=ConvModel([32,32,64,64,128,128],[3,3,3,3,3,3],[2,1,2,1,2,1],256,0.35)
convmodel2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(convmodel2.summary())

convmodel2.fit(x_conv,y_conv,batch_size=128,epochs=20,validation_split=0.2)

convpred2=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel2.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')
convpred2.to_csv('convpred2.csv')




# I try a simpler network now, with just two concolutional layers, dimensions are 32 and 64, windows 14x14 and 7x7.
# Dense layer with 128 neurons. I use 13 epochs of training, 35% dropout rate, and 20% train/dev partitioning

convmodel3=ConvModel([32,64],[3,3],[2,2],128,0.35)
convmodel3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(convmodel3.summary())

convmodel3.fit(x_conv,y_conv,batch_size=128,epochs=13,validation_split=0.2)

convpred3=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel3.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')
convpred3.to_csv('convpred3.csv')




# I define a new convolutional neural network, with batch normalization layers between convolution and activation

def ConvModel2(dims,kernels,strides,dim_dense,dropout_rate):
    data=np.array([dims,kernels,strides]).T
    X_input=Input(shape=(d,d,1))
    X=X_input
    for dim,kernel,stride in data:
        X=Conv2D(dim,kernel_size=(kernel,kernel),strides=(stride,stride),padding='same',use_bias=False)(X)
        X=BatchNormalization()(X)
        X=Activation('relu')(X)
        X=Dropout(dropout_rate)(X)
    X=Flatten()(X)
    X=Dense(dim_dense,use_bias=False)(X)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)
    X_output=Dense(n,activation='softmax')(X)
    model=Model(inputs=X_input,outputs=X_output)
    return model




# Like Convolutional model 1 (4 hidden layers), but with batch normalization

convmodel4=ConvModel2([32,64,128],[3,3,3],[2,1,2],128,0.35)
convmodel4.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(convmodel4.summary())

convmodel4.fit(x_conv,y_conv,batch_size=128,epochs=15,validation_split=0.2)

convpred4=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel4.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')
convpred4.to_csv('convpred4.csv')




# Like Convolutional model 2 (7 hidden layers), but with batch normalization

convmodel5=ConvModel2([32,32,64,64,128,128],[3,3,3,3,3,3],[2,1,2,1,2,1],256,0.35)
convmodel5.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(convmodel5.summary())

convmodel5.fit(x_conv,y_conv,batch_size=128,epochs=20,validation_split=0.2)

convpred5=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel5.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')
convpred5.to_csv('convpred5.csv')




# Finally, I create yet a third type of model, where there is a pooling layer between activation and dropout.
# The idea is that I will now be able to decrease the window size with pooling instead of using wider strides.

def ConvModel3(dims,kernels,strides,pools,dim_dense,dropout_rate):
    data=np.array([dims,kernels,strides,pools]).T
    X_input=Input(shape=(d,d,1))
    X=X_input
    for dim,kernel,stride,pool in data:
        X=Conv2D(dim,kernel_size=(kernel,kernel),strides=(stride,stride),padding='same',use_bias=False)(X)
        X=BatchNormalization()(X)
        X=Activation('relu')(X)
        X=MaxPooling2D(pool_size=(pool,pool))(X)
        X=Dropout(dropout_rate)(X)
    X=Flatten()(X)
    X=Dense(dim_dense,use_bias=False)(X)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)
    X_output=Dense(n,activation='softmax')(X)
    model=Model(inputs=X_input,outputs=X_output)
    return model




# Like Convolutional model 1 (4 hidden layers), but with batch normalization and pooling.
# All strides are set to 1, and I pool with 2x2 filters after each convolutional layer.

convmodel6=ConvModel3([32,64,128],[3,3,3],[1,1,1],[2,2,2],128,0.35)
convmodel6.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(convmodel6.summary())

convmodel6.fit(x_conv,y_conv,batch_size=128,epochs=15,validation_split=0.2)

convpred6=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel6.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')
convpred6.to_csv('convpred6.csv')


