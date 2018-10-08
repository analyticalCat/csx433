
# coding: utf-8

# In[1]:


import struct
import numpy as np
from array import array
import matplotlib.pyplot as plt
import numpy.linalg as LA
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import sklearn.tree as tree
import sklearn.metrics as metrics

#for Extreme Learning Machine
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer


# In[2]:


def load_mnist(dataset="training", selecteddigits=range(10)):

    if dataset == "training":
        fname_digits = '/Users/jess_/Downloads/Mnist/train-images.idx3-ubyte'
        fname_labels = '/Users/jess_/Downloads/Mnist/train-labels.idx1-ubyte'
    elif dataset == "testing":
        fname_digits = '/Users/jess_/Downloads/Mnist/t10k-images.idx3-ubyte'
        fname_labels = '/Users/jess_/Downloads/Mnist/t10k-labels.idx1-ubyte'
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
        
        
    #Import digits data
    digitsfileobject = open(fname_digits, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", digitsfileobject.read(16))
    digitsdata = array("B", digitsfileobject.read())
    digitsfileobject.close()

    #Import label data
    labelsfileobject = open(fname_labels, 'rb')
    magic_nr, size = struct.unpack(">II", labelsfileobject.read(8))
    labelsdata=array("B",labelsfileobject.read())
    labelsfileobject.close()
    
    #Find indices of selected digits
    indices=[k for k in range(size) if labelsdata[k] in selecteddigits]
    N=len(indices)
    
    #Create empty arrays for X and T
    X = np.zeros((N, rows*cols), dtype=np.uint8)
    T = np.zeros((N, 1), dtype=np.uint8)
    
    #Fill X from digitsdata
    #Fill T from labelsdata
    for i in range(N):
        X[i] = digitsdata[indices[i]*rows*cols:(indices[i]+1)*rows*cols]
        T[i] = labelsdata[indices[i]]
    
    return X,T


# In[3]:


def vectortoimg(v,show=True):
    plt.imshow(v.reshape(28, 28),interpolation='None', cmap='gray')
    plt.axis('off')
    if show:
        plt.show()


# In[4]:


X, T = load_mnist(dataset="training",selecteddigits=range(10))


# In[5]:


Xtest,Ttest=load_mnist("testing")


# In[6]:


len(T)


# In[7]:


Tunique, Tcounts = np.unique(T, return_counts=True)

print(np.asarray((Tunique, Tcounts)).T)


# In[8]:


print("Checking shape of matrix:", X.shape)
print("Checking min/max values:",(np.amin(X),np.amax(X)))
print("Checking unique labels in T:",list(np.unique(T)))


# In[9]:


print("Checking one training vector by plotting image:")
vectortoimg(X[-1])


# In[10]:


print("Checking multiple training vectors by plotting images.\nBe patient:")
plt.close('all')
fig = plt.figure()
nrows=10
ncols=10
for row in range(nrows):
    for col in range(ncols):
        plt.subplot(nrows, ncols, row*ncols+col + 1)
        vectortoimg(X[np.random.randint(len(T))],show=False)
plt.show()


# In[11]:


mu=np.mean(X,axis=0);print(mu)


# In[12]:


plt.plot(mu)


# In[13]:


print("Checking μ by plotting image:")
vectortoimg(mu)


# In[14]:


Z=X-mu;print(Z)


# In[15]:


C=np.cov(Z,rowvar=False);print(C)


# In[16]:


np.allclose(C,C.T)


# In[17]:


def vectortoimg_new(v,show=True):
    plt.imshow(v.reshape(784, 784),interpolation='None', cmap='gray')
    plt.axis('off')
    if show:
        plt.show()


# In[18]:


print("Checking C by plotting image:")
vectortoimg_new(C)


# In[19]:


[λ,V]=LA.eigh(C);print(λ,'\n\n',V)


# In[20]:


λ=np.flipud(λ);V=np.flipud(V.T);
plt.plot(λ)
plt.show()
row=V[0,:]; #Check once again
np.dot(C,row)-(λ[0]*row) #If the matrix product C.row is the same as λ[0]*row, this should evaluate to [0,0,0]
C.shape


# In[21]:


u, s, d = np.linalg.svd(C)
[q,p] = LA.eigh(C)
total = sum(s)
accumulate = 0
i = 0
per = np.zeros(C.shape[0])
acc = 0
while i < C.shape[0]:
    acc += s[i]/total
    per[i] = acc
    i += 1   
dim = 1
while per[dim - 1] < 0.8:
    dim += 1
print("The dimension that covers 80% of the variance is:",dim)
#variance coverage represents the representation of the original data.  It means that we can recover 80% of the data.
print(u)
print(p)
plt.plot(per)
plt.show()


# In[22]:


P=np.dot(Z,V.T);print(P) #Principal components
P.shape


# In[23]:


Xrec44=(np.dot(P[:,0:44],V[0:44,:]))+mu;print(Xrec44) #Reconstruction using 3 components (X is recovered)


# In[24]:


#Decision tree training for pca2
#clf = tree.DecisionTreeClassifier()
#https://github.com/efebozkir/handwrittendigit-recognition/blob/master/decisiontreefile.py
# clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=32, max_features=2)

# clf = clf.fit(P[:,0:2], T)
# Ztest=Xtest-mu
# Ptest=np.dot(Ztest,V.T)

# y2=clf.predict(Ptest[:,0:2])
# print('y.shape:',y2.shape)
# print(metrics.classification_report(Ttest, y2, digits=4))
# print(metrics.precision_score(Ttest, y2, average='weighted'))


# In[35]:


#calculate the performance of given classifier
def classifierPerformance(confusionMatrix, caption, totalc):
    accuracy=np.zeros((10), dtype=float)
    sensitivity=np.zeros((10), dtype=float)
    specificity=np.zeros((10), dtype=float)
    ppv=np.zeros((10),dtype=float)

    for i,value in enumerate(confusionMatrix):
        colsum=np.sum(confusionMatrix[:,i])
        rowsum=np.sum(confusionMatrix[i,:])
        TP=confusionMatrix[i,i]
        TN=(totalc-rowsum-colsum+confusionMatrix[i,i])
        FP=colsum-confusionMatrix[i,i]
        FN=rowsum-confusionMatrix[i,i]
        accuracy[i]=(TP+TN)/totalc
        sensitivity[i]=TP/(TP+FN)
        specificity[i]=TN/(FP+TN)
        ppv[i]=TP/(TP+FP)
    
    print('classifier performance:', caption, accuracy, sensitivity, specificity, ppv)
    
    return (accuracy, sensitivity,specificity, ppv)


# In[26]:


# #calculating the confusion matrix for 2 dimensions
# confusionMatrix=np.zeros((10,10),dtype=int)
# for i, value in enumerate(y2):
#     confusionMatrix[Ttest[i],value]+=1
# print(confusionMatrix)          

# performance2 = np.zeros((4), dtype=np.float)
# performance2=classifierPerformance(confusionMatrix,'confusion2',len(Ttest))


# In[36]:


#calculating the random classifier
import random as random
randomMatrix=np.zeros((10,10),dtype=int)
k=0
while k<len(Ttest):
    k+=1
    i=random.randint(0,9)
    j=random.randint(0,9)
    randomMatrix[i,j]+=1

performancerandom=np.zeros((4),dtype=float)
performancerandom=classifierPerformance(randomMatrix,'random',len(Ttest))


# In[28]:


# #decision tree with 10 pca dimensions
# clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=10, max_features=10)
# clf = clf.fit(P[:,0:10], T)
# y10=clf.predict(Ptest[:,0:10])
# print('y.shape:',y10.shape)

# confusionMatrix=np.zeros((10,10),dtype=int)
# for i, value in enumerate(y10):
#     confusionMatrix[Ttest[i],value]+=1
# print(confusionMatrix)          
# performance10=np.zeros((4),dtype=float)
# performance10=classifierPerformance(confusionMatrix,'confusion10',len(Ttest))


# In[37]:


#decision tree with  all 784 demensions. The algorithm will reduce the dimension to 1
clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=10, max_features=784)
clf = clf.fit(P, T)
yfull=clf.predict(Ptest)
print('y.shape:',yfull.shape)

confusionMatrix=np.zeros((10,10),dtype=int)
for i, value in enumerate(yfull):
    confusionMatrix[Ttest[i],value]+=1
print(confusionMatrix)          

performancefull=np.zeros((4),dtype=float)
performancefull=classifierPerformance(confusionMatrix,'full',len(Ttest))


# In[38]:


#compare the four measurements for the full run to see the differences between the measurements. which measurement makes the most sense.abs
#use performancefull as the data input
xdigits=np.arange(0,10)
plt.clf()
plt.plot(xdigits, performancefull[0], 'r-',xdigits,performancefull[1], 'b-',xdigits,performancefull[2], 'y-',xdigits,performancefull[3],'g-')
plt.show()

#we see that the specificity is much lower than others. the true positive rate.  This means that each time the program identify a digit as 5 or 8, it's only right about 75% of the time.



# In[39]:


#make the classifier
def make_classifier():
    # use internal transfer funcs
    srhl_tanh = MLPRandomLayer(n_hidden=10, activation_func='tanh')

    return GenELMClassifier(hidden_layer=srhl_tanh)


# In[40]:


# use Extreme Learning Machine for digit recognition
#http://wdm0006.github.io/sklearn-extensions/extreme_learning_machines.html


clf = make_classifier()
clf.fit(X, T)
yelm=clf.predict(Xtest)
print(yelm.shape)
#score = clf.score(Xtest, Ttest)
score = clf.score(X,T)
print('Model %s score: %s' % ('elm', score))


confusionMatrix=np.zeros((10,10),dtype=int)
for i, value in enumerate(yelm):
    confusionMatrix[Ttest[i],value]+=1
print(confusionMatrix)          

performanceelm=np.zeros((4),dtype=float)
performanceelm=classifierPerformance(confusionMatrix,'elm',len(Ttest))
print(performanceelm)


# In[41]:


#pick the measurements with the most difference and compare the 5 runs.
plt.clf()
plt.plot(xdigits, performancerandom[2], 'r-',xdigits, performancefull[2], 'g-',xdigits, performanceelm[2],'p-')
plt.xlabel('digits')
plt.ylabel('Specificity')
plt.show()

#The graph below tells us that 10 dimension really gives us a good result. 


# In[46]:


#pick the measurements with the most difference and compare the 4 runs.
plt.clf()
plt.plot(xdigits, performancerandom[0], 'r-',xdigits, performancefull[0], 'g-',xdigits, performanceelm[0],'p-')
plt.xlabel('digits')
plt.ylabel('Accuracy')
plt.show()

plt.clf()
plt.plot(xdigits, performancerandom[1], 'r-',xdigits, performancefull[1], 'g-',xdigits, performanceelm[1],'p-')
plt.xlabel('digits')
plt.ylabel('Sensitivity')
plt.show()

plt.clf()
plt.plot(xdigits, performancerandom[3], 'r-',xdigits, performancefull[3], 'g-',xdigits, performanceelm[3],'p-')
plt.xlabel('digits')
plt.ylabel('PPV')
plt.show()

#The graph below tells us that 10 dimension really gives us a good result. 
averageP=np.average(performanceelm,1)
print(averageP)
print(2*averageP[1]*averageP[3]/(averageP[1]+averageP[3]))

