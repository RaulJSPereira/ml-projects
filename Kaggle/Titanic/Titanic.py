#!/usr/bin/env python
# coding: utf-8

# # Preliminaries

# ### Import data and packages

# Import numpy, pandas, sklearn, matplotlib.
# Import lightgbm (gradient boosted decision tree algorithm) for implementing algorithm

# In[1]:


import numpy as np
import pandas as pd
import sklearn as sl
import matplotlib.pyplot as plt
import lightgbm as lgbm


# Load datasets into pandas

# In[2]:


data=pd.read_csv('train.csv',index_col=0)
test=pd.read_csv('test.csv',index_col=0)


# In[3]:


data.head(5)


# ### Data Inspection

# I join the training and test datasets so that I can investigate general features of the datasets

# In[6]:


alldata=pd.concat([data.drop('Survived',axis=1),test])


# I check that Age, Fare, Cabin and Embarked have missing values

# In[7]:


print(test.columns[alldata.isnull().sum()>0])


# I test some string manipulations to extract the title of the passenger from the Name feature.
# I find it surprising that 'the' appears as a title, so I check that it just stands for 'the Countess'

# In[8]:


print(alldata.Name.map(lambda x: x.split(',')[1].split(' ')[1]).unique())


# In[9]:


print(alldata.Name[alldata.Name.map(lambda x: x.split(',')[1].split(' ')[1])=='the'].values)


# I look at the type of categories in Embarked and Cabin.
# Around 77% of values in Cabin are missing, and still there are many distinct categories. 
# Therefore the frequency of each value is pretty low, so I will use only the first letter to distinguish categories

# In[10]:


print(alldata.Embarked.unique())


# In[11]:


print(alldata.Cabin.isnull().sum()/len(alldata))
print(alldata.Cabin.sort_values().unique())


# I plot the distribution of ages

# In[12]:


plt.hist(alldata.Age.fillna(alldata.Age.min()-1), range=(alldata.Age.min(), alldata.Age.max()), bins=50);


# I plot the distribution of Fares and find it is quite skewed. 
# I check that applying a square root makes it a little more even.

# In[13]:


plt.figure(figsize=(13,4))
plt.subplot(121)
plt.hist(alldata.Fare.fillna(alldata.Fare.min()-1), range=(alldata.Fare.min(),alldata.Fare.max()), bins=50);
plt.subplot(122)
plt.hist(np.sqrt(data.Fare.fillna(-1)), range=(np.sqrt(alldata.Fare.min()), np.sqrt(alldata.Fare.max())), bins=50);


# Finally, I see that both classes of labels are well represented in the training dataset

# In[14]:


print(data.Survived.value_counts().values/len(data))


# ### Data Preparation

# I apply the insights above to preprocess the data a bit before learning.
# I create new dataframes so that I keep the original data intact.
# I create a new category 'Family', given by 'SibSp'+'Parch', which denotes the total number of family members onboard.
# I fill missing values in the Age and Fare category by the median of the training dataset (avoid data leakage).
# I rescale the Fare values with a square root to uniformize the distribution somewhat.
# I fill missing values in both Cabin and Embarked with the letter 'Z', which does not exist in the datasets.
# For the Cabin category I take only the first letter

# In[15]:


Data=data[['Survived','Pclass','Sex','SibSp','Parch']]
Test=test[['Pclass','Sex','SibSp','Parch']]

Data=Data.assign(Family=data.SibSp+data.Parch)
Test=Test.assign(Family=test.SibSp+test.Parch)

AgeMedian=data.Age.median()
Data=Data.assign(Age=data.Age.fillna(AgeMedian))
Test=Test.assign(Age=test.Age.fillna(AgeMedian))

FareMedian=data.Fare.median()
Data=Data.assign(Fare=np.sqrt(data.Fare.fillna(FareMedian)))
Test=Test.assign(Fare=np.sqrt(test.Fare.fillna(FareMedian)))

Data=Data.assign(Title=data.Name.map(lambda x: x.split(',')[1].split(' ')[1]))
Test=Test.assign(Title=test.Name.map(lambda x: x.split(',')[1].split(' ')[1]))

Data=Data.assign(Embarked=data.Embarked.fillna('Z'))
Test=Test.assign(Embarked=test.Embarked.fillna('Z'))

Data=Data.assign(Cabin=data.Cabin.fillna('Z').map(lambda x: x[0]))
Test=Test.assign(Cabin=test.Cabin.fillna('Z').map(lambda x: x[0]))


# I use the label encoder of sklearn to encode the categorical features of the datasets

# In[16]:


from sklearn.preprocessing import LabelEncoder

cats=['Title','Sex','Cabin','Embarked']
AllData=pd.concat([Data,Test])

for cat in cats:
    encoder=LabelEncoder()
    encoder.fit(AllData[cat])
    Data[cat]=encoder.transform(Data[cat])
    Test[cat]=encoder.transform(Test[cat])


# In[17]:


Data.head(5)


# I set apart 20% of the data for validation of the training algorithm.
# The shuffling is random, but I set a seed so the partitioning is always the same.
# I check that the two classes are well distributed.

# In[18]:


train=Data.sample(frac=0.8,random_state=121)
valid=Data.drop(train.index,axis=0)
print(len(set(np.concatenate([train.index,valid.index])))==len(train)+len(valid)==len(Data))
print([train.Survived.value_counts().values/len(train),valid.Survived.value_counts().values/len(valid)])


# I get the data ready to use with lightgbm

# In[19]:


features=train.columns.drop('Survived')
Train=lgbm.Dataset(train[features],label=train['Survived'])
Valid=lgbm.Dataset(valid[features],label=valid['Survived'])


# ### Implementation of AUC metric

# In order to get a better understanding of the auc metric I implement it here.
# I start with a prerun function which partitions the interval of probabilities, [0,1], into a lattice of n values and finds the last one (i0) where the rate of false positives (FPR) is still 0, and the first one (i1) where FPR is 1.
# The idea is that I can scan the interval fast with a small value of n, and then attemp more precision later by looking only within the interval [ i0, i1 ], since values outside this interval do not contribute to the AUC metric.

# In[20]:


def prerun(labels,probs,n):
    Neg=(1-labels).sum()
    FP=0
    i=0
    while FP < Neg:
        i+=1
        predictions= probs > (1-i/n)
        FP=(1-labels[predictions]).sum()
        if FP==0:
            i0=1-i/n
    i1=1-i/n
    return i0,i1


# The function score divides the interval [ i0, i1 ] (found with prerun) into a lattice of n values, each setting a different value for the threshold that allows us to set the predictions to 0 or 1. At each value of the threshold I compute the rate of false positives (FPR) and the rate of true positives (FPR), which allow to obtain the plot the  curve defining the metric.
# I then use the trapezoidal rule for approximating the area under the curve (AUC), and also compute the accuracy at each value of the threshold.
# There is a parameter p which gives the degree of the polynomial used to scan the interval [ i0, i1], which I will analyze later.

# In[21]:


def MyAUC(labels,probs,nr,n,p):
    Pos=labels.sum()
    Neg=len(labels)-Pos 
    init,fin=prerun(labels,probs,nr)
    FPRs=[0.]
    TPRs=[0.]
    auc=[]
    accuracies=[]
    thresholds=[]
    for i in range(0,n+1):
        threshold=init-(init-fin)*(1-(1-i/n)**p)
        thresholds.append(threshold)
        predictions=probs>threshold
        TP=labels[predictions].sum()
        FP=predictions.sum()-TP
        FPR=FP/Neg
        TPR=TP/Pos
        auc.append((FPR-FPRs[-1])*(TPR+TPRs[-1])/2)
        FPRs.append(FPR)
        TPRs.append(TPR)
        accuracy=np.sum([i==j for i,j in np.array([labels,predictions]).T])/len(labels)
        accuracies.append(accuracy)
    posmax=np.argmax(accuracies)
    return FPRs,TPRs,auc,accuracies[posmax],thresholds[posmax]


# The function rescale takes the data found with MyAUC for some high value of n, and produces the AUC result that would be obtained for a smaller value of n. This is so that I can study the convergence of the AUC value with a single run.

# In[22]:


def rescale(x,y,n_old,n_new):
    selection=[1+int(n_old*i/n_new) for i in range(0,n_new+1)]
    x_new=np.array(x)[selection]
    y_new=np.array(y)[selection]
    return ((x_new[1:]-x_new[:-1])*(y_new[1:]+y_new[:-1])/2).sum()


# # Gradient-boosted Implementation

# ### Model 1

# I set the trees to have 16 leaves, and choose the AUC metric. I select 200 rounds of training, but with early stopping if the results do not improve after 100 rounds.

# In[40]:


param1={'num_leaves':16,'objective':'binary','metric':'auc','learning_rate':0.1}
model1=lgbm.train(param1,Train,200,valid_sets=[Valid],verbose_eval=True,early_stopping_rounds=100)


# I check the AUC score from the sklearn. It is quite different between the training and validation set.
# In order to find accuracy I would have to find what is the best threshold to use, I will do that later with MyAUC function which also outputs the accuracy score.

# In[41]:


print(sl.metrics.roc_auc_score(train['Survived'], model1.predict(train[features])))
print(sl.metrics.roc_auc_score(valid['Survived'], model1.predict(valid[features])))


# I check also the accuracy obtained with threshold=0.5

# In[42]:


print(sl.metrics.accuracy_score(train['Survived'], [int(i+0.5) for i in model1.predict(train[features])]))
print(sl.metrics.accuracy_score(valid['Survived'], [int(i+0.5) for i in model1.predict(valid[features])]))


# ### Model 2

# Same parameters as above, except that I use accuracy as a metric

# In[27]:


param2={'num_leaves':16,'objective':'binary','metric':'binary_error','learning_rate':0.1}
model2=lgbm.train(param2,Train,200,valid_sets=[Valid],verbose_eval=True,early_stopping_rounds=100)


# The AUC score is slightly lower than in the previous model, but that is expected as I used the accuracy metric for training instead.

# In[43]:


print(sl.metrics.roc_auc_score(train['Survived'], model2.predict(train[features])))
print(sl.metrics.roc_auc_score(valid['Survived'], model2.predict(valid[features])))


# I check the accuracy obtained with threshold=0.5 (in fact I output the error, to check that it matches the validation error output by the model).

# In[44]:


print(1-sl.metrics.accuracy_score(train['Survived'], [int(i+0.5) for i in model2.predict(train[features])]))
print(1-sl.metrics.accuracy_score(valid['Survived'], [int(i+0.5) for i in model2.predict(valid[features])]))


# The accuracy on the validation set is better with the second model than with the first. This makes sense as it was trained with the relevant metric. Since the evaluation of the model with through the accuracy, it makes sense to train with that metric.

# In[45]:


print(1-sl.metrics.accuracy_score(valid['Survived'], [int(i+0.5) for i in model1.predict(valid[features])]))


# I save the predictions for the test dataset

# In[32]:


PredF=[int(x+0.5) for x in model2.predict(Test)]
Final=Test[[]].assign(Survived=PredF)
Final.to_csv("Final.csv")


# ### Understanding the AUC metric

# I look at the evolution of the FPR, TPR and cumulative AUC values as I scan through possible threshold values. I observe that the contribution to the AUC value is highly skewed, coming mostly from the last threshold values. The plots here are obtained with a discretization of the threshold window through a polynomial of degree 1.
# At the end I plot also the AUC curve.

# In[33]:


AUCtest1=MyAUC(train['Survived'],model2.predict(train[features]),100,500,1)
AUCtest1b=np.cumsum(AUCtest1[2])
print(sum(AUCtest1[2]),AUCtest1[3],AUCtest1[4])
plt.figure(figsize=(18,9))
plt.subplot(231)
plt.plot(AUCtest1[0],'r',AUCtest1[1],'b',AUCtest1b,'g')
plt.xlabel('Threshold Lattice #')
plt.legend(['FPR','TPR','CumAUC'])
plt.subplot(232)
plt.yscale('log')
plt.xlabel('Threshold Lattice #')
plt.ylabel('CumAUC')
plt.plot(AUCtest1b)
plt.subplot(233)
plt.plot(AUCtest1[2])
plt.xlabel('Threshold Lattice #')
plt.ylabel('AUC increments')
plt.subplot(234)
plt.plot(AUCtest1[0],AUCtest1[1])
plt.xlabel('FPR')
plt.ylabel('TPR');


# Note that even with the second model, which was trained with the accuracy metric, the best accuracy appears at values other than 0.5. I shall use that threshold value nonetheless, as it is the one assumed when training.

# In[34]:


print([MyAUC(train['Survived'],model2.predict(train[features]),100,500,1)[3:],
       MyAUC(valid['Survived'],model2.predict(valid[features]),100,500,1)[3:]])


# I do the same analysis as above, but for polynomials for discretization of degree 2 and 3. These higher orders do seem to spread better the contribution to the AUC value.

# In[35]:


AUCtest2=MyAUC(train['Survived'],model2.predict(train[features]),100,500,2)
AUCtest2b=np.cumsum(AUCtest2[2])
print(sum(AUCtest2[2]),AUCtest2[3],AUCtest2[4])
plt.figure(figsize=(18,4))
plt.subplot(131)
plt.plot(AUCtest2[0],'r',AUCtest2[1],'b',AUCtest2b,'g')
plt.xlabel('Threshold Lattice #')
plt.legend(['FPR','TPR','CumAUC'])
plt.subplot(132)
plt.plot(AUCtest2[2])
plt.xlabel('Threshold Lattice #')
plt.ylabel('AUC increments')
plt.subplot(133)
plt.plot(AUCtest2[0],AUCtest2[1])
plt.xlabel('FPR')
plt.ylabel('TPR');


# In[36]:


AUCtest3=MyAUC(train['Survived'],model2.predict(train[features]),100,500,3)
AUCtest3b=np.cumsum(AUCtest3[2])
print(sum(AUCtest3[2]),AUCtest3[3],AUCtest3[4])
plt.figure(figsize=(18,4))
plt.subplot(131)
plt.plot(AUCtest3[0],'r',AUCtest3[1],'b',AUCtest3b,'g');
plt.xlabel('Threshold Lattice #')
plt.legend(['FPR','TPR','CumAUC'])
plt.subplot(132)
plt.plot(AUCtest3[2])
plt.xlabel('Threshold Lattice #')
plt.ylabel('AUC increments')
plt.subplot(133)
plt.plot(AUCtest3[0],AUCtest3[1])
plt.xlabel('FPR')
plt.ylabel('TPR');


# In order to better study the convergence of the 3 different discretizations, I obtain the data for the AUC curve with a tightly-knit lattice of 10000 values.

# In[37]:


AUCtest4=MyAUC(train['Survived'],model2.predict(train[features]),100,10000,1)
AUCtest5=MyAUC(train['Survived'],model2.predict(train[features]),100,10000,2)
AUCtest6=MyAUC(train['Survived'],model2.predict(train[features]),100,10000,3)
[sum(x[2]) for x in [AUCtest4,AUCtest5,AUCtest6]]


# I then use my rescale function to obtain the AUC values for a range of coarser lattices, and plot all the evolution of the AUC value for the 3 different polynomial orders. Despite the better spread found above, this plot does not present any evidence that higher orders provide better results.

# In[38]:


AUCtest4b=[rescale(AUCtest4[0],AUCtest4[1],10000,50*i) for i in range(4,100+1)]
AUCtest5b=[rescale(AUCtest5[0],AUCtest5[1],10000,50*i) for i in range(4,100+1)]
AUCtest6b=[rescale(AUCtest6[0],AUCtest6[1],10000,50*i) for i in range(4,100+1)]
x_axis=[50*i for i in range(4,100+1)]
plt.figure(figsize=(18,6))
plt.plot(x_axis,AUCtest4b,'r',x_axis,AUCtest5b,'b',x_axis,AUCtest6b,'g')
plt.legend(['p=1','p=2','p=3'])
plt.xlabel('n')
plt.ylabel('AUC');


# # Different validation sets

# ### Smaller

# Since the dataset is so small, I try to use a smaller validation set. This increases the available training data, but it is more prone to overfitting

# In[62]:


trainB=Data.sample(frac=0.9,random_state=121)
validB=Data.drop(trainB.index,axis=0)
print(len(set(np.concatenate([trainB.index,validB.index])))==len(trainB)+len(validB)==len(Data))
print([trainB.Survived.value_counts().values/len(trainB),validB.Survived.value_counts().values/len(validB)])


# In[63]:


features=trainB.columns.drop('Survived')
TrainB=lgbm.Dataset(trainB[features],label=trainB['Survived'])
ValidB=lgbm.Dataset(validB[features],label=validB['Survived'])


# I decrease the number of leaves to 16, and also decrease a little the learning rate, while naturally increasing the number of rounds. I also wait longer before allowing early stopping.

# In[64]:


param3={'num_leaves':16,'objective':'binary','metric':'binary_error','learning_rate':0.1}
model3=lgbm.train(param3,TrainB,200,valid_sets=[ValidB],verbose_eval=True,early_stopping_rounds=100)


# As expected, the accuracy has now decreased, as any further training would further worsen the validation results

# In[65]:


print(1-sl.metrics.accuracy_score(trainB['Survived'], [int(i+0.5) for i in model3.predict(trainB[features])]))
print(1-sl.metrics.accuracy_score(validB['Survived'], [int(i+0.5) for i in model3.predict(validB[features])]))
print(1-sl.metrics.accuracy_score(valid['Survived'], [int(i+0.5) for i in model2.predict(valid[features])]))


# In[56]:


PredFB=[int(x+0.5) for x in model3.predict(Test)]
FinalB=Test[[]].assign(Survived=PredFB)
FinalB.to_csv("FinalB.csv")


# ### Larger

# Now I try to use a larger validation set instead.

# In[66]:


trainC=Data.sample(frac=0.7,random_state=121)
validC=Data.drop(trainC.index,axis=0)
print(len(set(np.concatenate([trainC.index,validC.index])))==len(trainC)+len(validC)==len(Data))
print([trainC.Survived.value_counts().values/len(trainC),validC.Survived.value_counts().values/len(validC)])


# In[67]:


features=trainC.columns.drop('Survived')
TrainC=lgbm.Dataset(trainC[features],label=trainC['Survived'])
ValidC=lgbm.Dataset(validC[features],label=validC['Survived'])


# In[68]:


param4={'num_leaves':16,'objective':'binary','metric':'binary_error','learning_rate':0.1}
model4=lgbm.train(param4,TrainC,200,valid_sets=[ValidC],verbose_eval=True,early_stopping_rounds=100)


# As expected, the validation accuracy improved, even though the training error increased. It is unclear if the results of the test set will be better or worse, but there is potential for improvement

# In[69]:


print(1-sl.metrics.accuracy_score(trainC['Survived'], [int(i+0.5) for i in model4.predict(trainC[features])]))
print(1-sl.metrics.accuracy_score(validC['Survived'], [int(i+0.5) for i in model4.predict(validC[features])]))
print(1-sl.metrics.accuracy_score(valid['Survived'], [int(i+0.5) for i in model2.predict(valid[features])]))


# In[70]:


PredFC=[int(x+0.5) for x in model4.predict(Test)]
FinalC=Test[[]].assign(Survived=PredFC)
FinalC.to_csv("FinalC.csv")


# # Cross-validation

# Since the dataset is so small, it can be useful to use cross-validation, so that we actually train using the whole dataset.
# I start with a 5-fold split, yielding 20% data for validation in each iteration.

# In[72]:


KF=sl.model_selection.KFold(n_splits=5,shuffle=False)


# In[76]:


features=Data.columns.drop('Survived')
paramKF={'num_leaves':16,'objective':'binary','metric':'binary_error','learning_rate':0.09}
accuraciesA=[]
resultsA=[]
for train_ind,valid_ind in KF.split(Data):
    trainKF=Data.iloc[train_ind]
    validKF=Data.iloc[valid_ind]
    TrainKF=lgbm.Dataset(trainKF[features],label=trainKF['Survived'])
    ValidKF=lgbm.Dataset(validKF[features],label=validKF['Survived'])
    modelKF=lgbm.train(paramKF,TrainKF,200,valid_sets=[ValidKF],verbose_eval=False,early_stopping_rounds=100)
    accuracy=sl.metrics.accuracy_score(validKF['Survived'], [int(i+0.5) for i in modelKF.predict(validKF[features])])
    accuraciesA.append(accuracy)
    resultsA.append(modelKF.predict(Test))


# The accuracies obtained with each of the 5 folds

# In[79]:


print(accuraciesA)


# The final prediction is set by the average of the probabilities from each of the 5 folds

# In[86]:


PredKFA=[int(x+0.5) for x in (np.sum(resultsA,axis=0)/5)]
FinalKFA=Test[[]].assign(Survived=PredKFA)
FinalKFA.to_csv("FinalKFA.csv")


# Given my previous analysis on the size of the validation sets, I will attempt now cross-validation with only 3 folds.

# In[88]:


KF2=sl.model_selection.KFold(n_splits=3,shuffle=False)


# In[89]:


features=Data.columns.drop('Survived')
paramKF2={'num_leaves':16,'objective':'binary','metric':'binary_error','learning_rate':0.09}
accuraciesB=[]
resultsB=[]
for train_ind,valid_ind in KF2.split(Data):
    trainKF2=Data.iloc[train_ind]
    validKF2=Data.iloc[valid_ind]
    TrainKF2=lgbm.Dataset(trainKF2[features],label=trainKF2['Survived'])
    ValidKF2=lgbm.Dataset(validKF2[features],label=validKF2['Survived'])
    modelKF2=lgbm.train(paramKF2,TrainKF2,200,valid_sets=[ValidKF2],verbose_eval=False,early_stopping_rounds=100)
    accuracy=sl.metrics.accuracy_score(validKF2['Survived'], [int(i+0.5) for i in modelKF2.predict(validKF2[features])])
    accuraciesB.append(accuracy)
    resultsB.append(modelKF2.predict(Test))
    


# In[90]:


print(accuraciesB)


# In[91]:


PredKFB=[int(x+0.5) for x in (np.sum(resultsB,axis=0)/3)]
FinalKFB=Test[[]].assign(Survived=PredKFB)
FinalKFB.to_csv("FinalKFB.csv")

