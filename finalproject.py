import struct
import numpy as np
from array import array
import matplotlib.pyplot as plt
import numpy.linalg as LA
import matplotlib.cm as cm
import pandas as pd
import sklearn.tree as tree
import sklearn.metrics as metrics
import graphviz
import ConfusionMatrix as cm
import random

#%matplotlib inline 

excelfile='c:\\Users\\jess_\\Downloads\\finalproject.xlsx'

def load_mnist(dataset="training", selecteddigits=range(10), path=r"/Users/jess_/Downloads/Mnist"):

    #Check training/testing specification. Must be "training" (default) or "testing"
    if dataset == "training":
        fname_digits = path + '/' + 'train-images.idx3-ubyte'
        fname_labels = path + '/' + 'train-labels.idx1-ubyte'
    elif dataset == "testing":
        fname_digits = path + '/' + 't10k-images.idx3-ubyte'
        fname_labels = path + '/' + 't10k-labels.idx1-ubyte'
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

def vectortoimg(v,show=True):
    plt.imshow(v.reshape(28, 28),interpolation='None', cmap='gray')
    plt.axis('off')
    if show:
        plt.show()

X,T=load_mnist()
len(T)


Tunique, Tcounts = np.unique(T, return_counts=True)

print(np.asarray((Tunique, Tcounts)).T)
print("Checking shape of matrix:", X.shape)
print("Checking min/max values:",(np.amin(X),np.amax(X)))
print("Checking unique labels in T:",list(np.unique(T)))
print("Checking one training vector by plotting image:")
#vectortoimg(X[-20])

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

mu=np.mean(X,axis=0);print(mu)
plt.plot(mu)
print("Checking μ by plotting image:")
#vectortoimg(mu)
Z=X-mu;print(Z)
C=np.cov(Z,rowvar=False);print(C)
np.allclose(C,C.T)

def vectortoimg_new(v,show=True):
    plt.imshow(v.reshape(784, 784),interpolation='None', cmap='gray')
    plt.axis('off')
    if show:
        plt.show()


print("Checking C by plotting image:")
vectortoimg_new(C)

[λ,V]=LA.eigh(C);print(λ,'\n\n',V)

λ=np.flipud(λ);V=np.flipud(V.T)
row=V[0,:] #Check once again
np.dot(C,row)-(λ[0]*row) #If the matrix product C.row is the same as λ[0]*row, this should evaluate to [0,0,0]
P=np.dot(Z,V.T);print(P) #Principal components
R=np.dot(P,V);print(R-Z)
Xrec=R+mu;print(Xrec-X)

R2=np.dot(P[:,0:2], V[0:2,:]);R2
Xrec2=R2+mu;Xrec2
#vectortoimg(Xrec2[-20])

R10=np.dot(P[:,0:10], V[0:10,:]);R10
Xrec10=R10+mu;Xrec10
#vectortoimg(Xrec10[-20])

#Draw Scatter Plot for all 10 digits using P2
pandasP2=pd.DataFrame(P[:,0:2])
pandasT=pd.DataFrame(T)
pandasT.columns=['labels']

print(pandasT.columns)

scatter=pandasT.join(pandasP2,lsuffix='_labels',rsuffix='_P')

#revisit
for i in np.arange(0,10):
    plt.scatter(scatter.loc[scatter['labels'] == i,0], scatter.loc[scatter['labels'] == i,1], color= np.random.rand(3,),  alpha=.2)
    
plt.legend(np.arange(0,10).astype('object'))
#plt.show()


#Decision tree training
#clf = tree.DecisionTreeClassifier()
#https://github.com/efebozkir/handwrittendigit-recognition/blob/master/decisiontreefile.py
clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=32, max_features=2)

clf = clf.fit(P[:,0:2], T)
dot_data = tree.export_graphviz(clf, out_file='mnistdotdata') 
graph = graphviz.Source(dot_data) 
#graph.render("Mnist") 
dot_data = tree.export_graphviz(clf, out_file='mnistoutput', 
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
#graph 

Xtest,Ttest=load_mnist("testing")
Ztest=Xtest-mu
Ptest=np.dot(Ztest,V.T)

y2=clf.predict(Ptest[:,0:2])
print('y.shape:',y2.shape)
#writeexcel.writeExcelData(, excelfile,'sheet1',1,1)
print(metrics.classification_report(Ttest, y2, digits=4))


#calculating the confusion matrix for 2 dimensions
confusionMatrix=np.zeros((10,10),dtype=int)
for i, value in enumerate(y2):
    confusionMatrix[Ttest[i],value]+=1
print(confusionMatrix)          

performance2 = np.zeros((4), dtype=np.float)
performance2=cm.classifierPerformance(confusionMatrix,'confusion2',len(Ttest))

#calculating the random classifier
randomMatrix=np.zeros((10,10),dtype=int)
k=0
while k<len(Ttest):
    k+=1
    i=random.randint(0,9)
    j=random.randint(0,9)
    randomMatrix[i,j]+=1

performancerandom=np.zeros((4),dtype=float)
performancerandom=cm.classifierPerformance(randomMatrix,'random',len(Ttest))

#decision tree with 10 pca dimensions
clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=32, max_features=10)
clf = clf.fit(P[:,0:10], T)
y10=clf.predict(Ptest[:,0:10])
print('y.shape:',y10.shape)

confusionMatrix=np.zeros((10,10),dtype=int)
for i, value in enumerate(y10):
    confusionMatrix[Ttest[i],value]+=1
print(confusionMatrix)          
performance10=np.zeros((4),dtype=float)
performance10=cm.classifierPerformance(confusionMatrix,'confusion10',len(Ttest))


#decision tree with  all 784 demensions. The algorithm will reduce the dimension to 1
clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=32, max_features=784)
clf = clf.fit(P, T)
yfull=clf.predict(Ptest)
print('y.shape:',yfull.shape)

confusionMatrix=np.zeros((10,10),dtype=int)
for i, value in enumerate(yfull):
    confusionMatrix[Ttest[i],value]+=1
print(confusionMatrix)          

performancefull=np.zeros((4),dtype=float)
performancefull=cm.classifierPerformance(confusionMatrix,'full',len(Ttest))

cm.writeExcelData(pd.DataFrame(list(performancerandom)), excelfile,'Sheet1',1,1)
cm.writeExcelData(pd.DataFrame(list(performance2)), excelfile,'Sheet1',11,1)
cm.writeExcelData(pd.DataFrame(list(performance10)), excelfile,'Sheet1',21,1)
cm.writeExcelData(pd.DataFrame(list(performancefull)), excelfile,'Sheet1',31,1)

xdigits=np.arange(0,10)

plt.clf()
plt.plot(xdigits, performancerandom[0], 'r--',xdigits,performance2[0], 'bo')
plt.show()



    


