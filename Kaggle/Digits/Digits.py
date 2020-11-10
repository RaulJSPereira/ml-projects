#!/usr/bin/env python
# coding: utf-8

# # Preliminaries

# ### Import data and packages

# Import numpy and pandas.
# Import categorical encoding, necessary layers and Model from keras.
# Import matplotlib for plotting accuracies

# In[1]:


import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt


# Load datasets into pandas

# In[2]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# In[3]:


train.head(5)


# ### Data Inspection

# I set some parameters:
# 
# m  = number of training examples,
# 
# dt = total number of pixels,
# 
# d  = number of pixels on dimension of the matrix (square root of d2),
# 
# n  = number of classes.

# In[4]:


m=train.shape[0]
dt=train.shape[1]-1
d=int(np.sqrt(dt))
n=len(train.label.value_counts())
print([m,dt,d,n])


# I check that neither the training nor the test sets have missing values

# In[5]:


test.columns[pd.concat([train.drop('label',axis=1),test]).isnull().sum()>0]


# I check that there is an even distribution of training examples across the different classes

# In[6]:


train['label'].value_counts(sort=False)


# Some statistics on the pixel values. 
# I compute Pixel_stats for each pixel, using all training example. 
# Then I get statistics for each Pixel_stat.
# For example, the minimum value for each pixel is always 0, but the maximum value is not always 255.

# In[7]:


train.drop('label',axis=1).describe().T.drop('count',axis=1).describe().drop('count',axis=0).add_prefix('Pixel_')


# # Dense Network

# ### Data Preparation

# I get the dataset ready for usage with keras.
# 
# Max: maximum pixel value in the training dataset
# 
# x: array of pixel values from training dataset, normalized by Max
# 
# y: labels transformed into n-dimensional array
# 
# x_test: array of pixel values from test dataset, normalized by Max

# In[8]:


Max=train.max().max()
x=train.iloc[:,1:].values/Max
y=to_categorical(train.label,n)
x_test=test.values/Max
print([x.shape,x.max(),y.shape])
print([x_test.shape,x_test.max()])


# I check that the categorical encoding of y with '1' at position 'i' of the n-dim array corresponds to label 'i'

# In[9]:


[set(train.label[np.argmax(y,axis=1)==i])=={i} for i in range(10)]


# ### Definition of the Model

# I define a fully connected network.
# Each hidden layer uses ReLU as activation function, and I use dropout after each of them for regularization.
# The number of neurons in each layer is given by the input array dims.
# The output layer has n neurons and uses softmax activation.

# In[10]:


def DenseModel(dims,dropout_rate):
    X_input=Input(shape=(dt))
    X=X_input
    for dim in dims:
        X=Dense(dim,activation='relu')(X)
        X=Dropout(dropout_rate)(X)
    X_output=Dense(n,activation='softmax')(X)
    model=Model(inputs=X_input,outputs=X_output)
    return model


# ### Model 1

# The first model I create has 5 hidden layers with dimensions (392, 196, 196, 98, 98), and 35% dropout rate.
# I fit the model with 20 epochs of training, batch size of 128 examples, and 20% of the dataset left for validation.

# In[13]:


dims=[dt//2,dt//4,dt//4,dt//8,dt//8]
model1=DenseModel(dims,0.35)
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model1.summary()


# In[14]:


hist1=model1.fit(x,y,batch_size=128,epochs=20,validation_split=0.2)


# Plot the training and validation costs for each epoch

# In[68]:


plt.plot(range(2,20),hist1.history['accuracy'][2:],range(2,20),hist1.history['val_accuracy'][2:])
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.legend(['training','validation'])
plt.show()


# I set the prediction to the the class with the highest probability in the output layer (softmax) 

# In[20]:


pred1=pd.DataFrame({'ImageId':range(1,len(x_test)+1),'Label':model1.predict(x_test).argmax(axis=1)}).set_index('ImageId')


# In[80]:


pred1.to_csv('pred1.csv')


# ### Model2

# The second model has 7 hidden layers with dimensions (784, 784, 392, 392, 196, 196, 98), and 35% dropout rate.
# I fit the model with 25 epochs of training, batch size of 128 examples, and 20% of the dataset left for validation.

# In[16]:


dims=[dt,dt,dt//2,dt//2,dt//4,dt//4,dt//8]
model2=DenseModel(dims,0.35)
model2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model2.summary()


# In[17]:


hist2=model2.fit(x,y,batch_size=128,epochs=25,validation_split=0.2)


# Plot the training and validation costs for each epoch

# In[69]:


plt.plot(range(2,25),hist2.history['accuracy'][2:],range(2,25),hist2.history['val_accuracy'][2:])
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.legend(['training','validation'])
plt.show()


# I check that almost 2% of the predictions differ compared with the previous model

# In[21]:


pred2=pd.DataFrame({'ImageId':range(1,len(x_test)+1),'Label':model2.predict(x_test).argmax(axis=1)}).set_index('ImageId')


# In[22]:


pred2.join(pred1, lsuffix='1',rsuffix='2').apply(lambda x: x.Label1==x.Label2,axis=1).value_counts()/len(x_test)


# In[81]:


pred2.to_csv('pred2.csv')


# ### Model 3

# Third model has 9 hidden layers, dimensions are (784, 784, 784, 392, 392, 392, 196, 196, 196), and 35% dropout rate.
# I fit the model with 30 epochs of training, batch size of 128 examples, and 20% of the dataset left for validation.

# In[23]:


dims=[dt,dt,dt,dt//2,dt//2,dt//2,dt//4,dt//4,dt//4]
model3=DenseModel(dims,0.35)
model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model3.summary()


# In[24]:


hist3=model3.fit(x,y,batch_size=128,epochs=30,validation_split=0.2)


# Plot the training and validation costs for each epoch

# In[70]:


plt.plot(range(2,30),hist3.history['accuracy'][2:],range(2,30),hist3.history['val_accuracy'][2:])
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.legend(['training','validation'])
plt.show()


# I check that around 2% of the predictions differ compared with both model 1 and 2.

# In[26]:


pred3=pd.DataFrame({'ImageId':range(1,len(x_test)+1),'Label':model3.predict(x_test).argmax(axis=1)}).set_index('ImageId')


# In[27]:


pred3.join(pred1, lsuffix='1',rsuffix='2').apply(lambda x: x.Label1==x.Label2,axis=1).value_counts()/len(x_test)


# In[28]:


pred3.join(pred2, lsuffix='1',rsuffix='2').apply(lambda x: x.Label1==x.Label2,axis=1).value_counts()/len(x_test)


# In[82]:


pred3.to_csv('pred3.csv')


# # Convolutional Networks

# ### Data Preparation

# Now I create the datasets for convolutional neural networks.
# 
# x_conv: 28 by 28 array of pixel values from training dataset, normalized by Max
# 
# x_conv_test: 28 by 28 array of pixel values from test dataset, normalized by Max

# In[29]:


x_conv=train.values[:,1:].reshape((m,d,d,1))/Max
x_conv_test=test.values.reshape((-1,d,d,1))/Max
y=to_categorical(train.label,n)
print([x_conv.shape,x_conv.max(),y.shape])
print([x_conv_test.shape,x_conv_test.max()])


# ### Definition of the Model

# I define a convolutional neural network.
# Each hidden layer uses ReLU as activation function, and I use dropout after each of them for regularization.
# The number of filters in each convolutional layer, the kernel sizes and strides are given as input arrays.
# I use square filters, with the same stride on both dimensions. I always use 'same' padding.
# After convolutional layers I flatten and apply a fully connected layer, with dimension given as input.
# The output layer has n neurons and uses softmax activation

# In[30]:


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


# ### Model 1

# Generically I increase the number of filters as I reduce the size of the matrices. All kernels are 3x3.
# I start with 28x28x1 and go to 14x14x32 with a stride of 2.
# I then use 'same' padding and unit stride to stay with 14x14 window, but increase to 64 filters.
# Finally I use stride of 2 again to go to 7x7x128.
# I use 128 neurons in the hidden dense layer.
# I fit the model with 15 epochs of training, 35% dropout, and 20% of the dataset left for validation.

# In[32]:


convmodel1=ConvModel([32,64,128],[3,3,3],[2,1,2],128,0.35)
convmodel1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
convmodel1.summary()


# In[33]:


convhist1=convmodel1.fit(x_conv,y,batch_size=128,epochs=15,validation_split=0.2)


# In[71]:


plt.plot(range(2,15),convhist1.history['accuracy'][2:],range(2,15),convhist1.history['val_accuracy'][2:])
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.legend(['training','validation'])
plt.show()


# In[36]:


convpred1=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel1.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')


# In[83]:


convpred1.to_csv('convpred1.csv')


# ### Model 2

# I use more hidden layers, but still all kernels are 3x3.
# Number of filters is 32,64 and 128, twice each. 
# I half the size of the window every two layers (to 14x14, 7x7 and 4x4).
# I use 256 neurons in the hidden dense layer.
# I fit the model with 20 epochs of training, 35% dropout, and 20% of the dataset left for validation.

# In[37]:


convmodel2=ConvModel([32,32,64,64,128,128],[3,3,3,3,3,3],[2,1,2,1,2,1],256,0.35)
convmodel2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
convmodel2.summary()


# In[38]:


convhist2=convmodel2.fit(x_conv,y,batch_size=128,epochs=20,validation_split=0.2)


# In[72]:


plt.plot(range(2,20),convhist2.history['accuracy'][2:],range(2,20),convhist2.history['val_accuracy'][2:])
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.legend(['training','validation'])
plt.show()


# In[40]:


convpred2=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel2.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')


# In[84]:


convpred2.to_csv('convpred2.csv')


# ### Model 3

# I try a simpler network now, with just two concolutional layers, dimensions are 32 and 64, windows 14x14 and 7x7.
# Dense layer with 128 neurons. I use 13 epochs of training, 35% dropout rate, and 20% train/dev partitioning.

# In[41]:


convmodel3=ConvModel([32,64],[3,3],[2,2],128,0.35)
convmodel3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
convmodel3.summary()


# In[42]:


convhist3=convmodel3.fit(x_conv,y,batch_size=128,epochs=13,validation_split=0.2)


# In[73]:


plt.plot(range(2,13),convhist3.history['accuracy'][2:],range(2,13),convhist3.history['val_accuracy'][2:])
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.legend(['training','validation'])
plt.show()


# In[44]:


convpred3=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel3.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')


# In[85]:


convpred3.to_csv('convpred3.csv')


# ### Definition of Model with Batch Normalization

# I define a new convolutional neural network, with batch normalization layers between convolution and activation

# In[45]:


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


# ### Model 4

# Like convolutional model 1 (4 hidden layers), but with batch normalization

# In[46]:


convmodel4=ConvModel2([32,64,128],[3,3,3],[2,1,2],128,0.35)
convmodel4.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
convmodel4.summary()


# In[47]:


convhist4=convmodel4.fit(x_conv,y,batch_size=128,epochs=15,validation_split=0.2)


# In[74]:


plt.plot(range(2,15),convhist4.history['accuracy'][2:],range(2,15),convhist4.history['val_accuracy'][2:])
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.legend(['training','validation'])
plt.show()


# In[49]:


convpred4=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel4.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')


# In[86]:


convpred4.to_csv('convpred4.csv')


# ### Model 5

# Like Convolutional model 2 (7 hidden layers), but with batch normalization

# In[50]:


convmodel5=ConvModel2([32,32,64,64,128,128],[3,3,3,3,3,3],[2,1,2,1,2,1],256,0.35)
convmodel5.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
convmodel5.summary()


# In[51]:


convhist5=convmodel5.fit(x_conv,y,batch_size=128,epochs=20,validation_split=0.2)


# In[75]:


plt.plot(range(2,20),convhist5.history['accuracy'][2:],range(2,20),convhist5.history['val_accuracy'][2:])
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.legend(['training','validation'])
plt.show()


# In[53]:


convpred5=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel5.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')


# In[87]:


convpred5.to_csv('convpred5.csv')


# ### Definition of Model with Batch Normalization and Pooling

# Finally, I create yet a third type of model, where there is a pooling layer between activation and dropout.
# The idea is that I will now be able to decrease the window size with pooling instead of using wider strides.

# In[54]:


def ConvModel3(dims,kernels,strides,pools,dim_dense,dropout_rate):
    data=np.array([dims,kernels,strides,pools]).T
    X_input=Input(shape=(d,d,1))
    X=X_input
    for dim,kernel,stride,pool in data:
        X=Conv2D(dim,kernel_size=(kernel,kernel),strides=(stride,stride),padding='same',use_bias=False)(X)
        X=BatchNormalization()(X)
        X=Activation('relu')(X)
        X=MaxPooling2D(pool_size=(pool,pool),strides=(pool,pool),padding='same')(X)
        X=Dropout(dropout_rate)(X)
    X=Flatten()(X)
    X=Dense(dim_dense,use_bias=False)(X)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)
    X_output=Dense(n,activation='softmax')(X)
    model=Model(inputs=X_input,outputs=X_output)
    return model


# ### Model 6

# Like Convolutional model 1 (4 hidden layers), but with batch normalization and pooling.
# All strides are set to 1. I pool with 2x2 filters after some of the convolutional layers. 

# In[55]:


convmodel6=ConvModel3([32,64,128],[3,3,3],[1,1,1],[2,1,2],128,0.35)
convmodel6.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
convmodel6.summary()


# In[56]:


convhist6=convmodel6.fit(x_conv,y,batch_size=128,epochs=15,validation_split=0.2)


# In[76]:


plt.plot(range(2,15),convhist6.history['accuracy'][2:],range(2,15),convhist6.history['val_accuracy'][2:])
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.legend(['training','validation'])
plt.show()


# In[58]:


convpred6=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel6.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')


# In[88]:


convpred6.to_csv('convpred6.csv')


# ### Model 7

# Like Convolutional model 2 (7 hidden layers), but with batch normalization and pooling.

# In[59]:


convmodel7=ConvModel3([32,32,64,64,128,128],[3,3,3,3,3,3],[1,1,1,1,1,1],[2,1,2,1,2,1],256,0.35)
convmodel7.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
convmodel7.summary()


# In[60]:


convhist7=convmodel7.fit(x_conv,y,batch_size=128,epochs=20,validation_split=0.2)


# In[77]:


plt.plot(range(2,20),convhist7.history['accuracy'][2:],range(2,20),convhist7.history['val_accuracy'][2:])
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.legend(['training','validation'])
plt.show()


# In[62]:


convpred7=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel7.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')


# In[89]:


convpred7.to_csv('convpred7.csv')


# Looking at the plot above, it seems I should train it a bit longer

# In[67]:


convhist7b=convmodel7.fit(x_conv,y,batch_size=128,epochs=8,validation_split=0.2)


# In[79]:


plt.plot(range(21,29),convhist7b.history['accuracy'],range(21,29),convhist7b.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.legend(['training','validation'])
plt.show()


# In[90]:


convpred7b=pd.DataFrame({'ImageId':range(1,len(x_conv_test)+1),'Label':convmodel7.predict(x_conv_test).argmax(axis=1)}).set_index('ImageId')


# In[91]:


convpred7b.to_csv('convpred7b.csv')


# In[ ]:




