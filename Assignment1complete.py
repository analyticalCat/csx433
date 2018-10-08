import os
import csv
import numpy as np
import math


os.chdir("c:\users\jessicazh\downloads")
print(os.getcwd())

#initialize all variables
X=[]
T=[]
dataF=[]
dataM=[]
hF=range(32)
hM=range(32)
B=32
xmin=0
xmax=0


def Readdatafile(fileName, X,T, xmin,xmax):
    count=0
    with open('Assignment_1_CSV.csv') as f:
        reader = csv.reader(f)
        for row in reader:
        #assign each row to array and convert the first two columns to inches
            if count<>0:
                height=int(row[0])*12+int(row[1])
                if row[2]=="Female":
                    dataF.append(height)
                else:
                    dataM.append(height)
                X.append(height)
                T.append(row[2])
                if height>xmax:
                    xmax=height
                if (height<xmin or xmin==0):
                    xmin=height
            count=count+1
    f.close

    return X,T,xmin,xmax

    
def findb(xHeight, xB, xxmin, xxmax):
    up=float(xHeight)-float(xxmin)
    bot=float(xxmax)-float(xxmin)
    xx=int(round(up/bot*(xB-1)))
    return xx

def BuildHistogramClassifier(X,T,B,xmin,xmax):
    HF=np.zeros(32).astype('int32')
    HM=np.zeros(32).astype('int32')
    
    for i, height in enumerate(X): 
        b=findb(height,B,xmin,xmax)
        if T[i]=='Female':
            HF[b]=HF[b]+1
        else:
            HM[b]=HM[b]+1
    return HF,HM

def writeDatafile(HF,HM,filename):
    with open(filename,'wb') as f:
        dataw=csv.writer(f, delimiter=',')
        dataw.writerow(HF)
        dataw.writerow(HM)
    f.close

def predictGender(height, HF,HM,B,xmin,xmax):
    bin=findb(height,B,xmin,xmax)
    return float(HF[bin])/(float(HF[bin])+float(HM[bin]))

def normalizelist(HF, HM, B, xmin, xmax):
    PHF=np.zeros(32).astype('float') 
    PHM=np.zeros(32).astype('float')
    height=52
    i=0

def calBayesianFormulas(N,sigma,X,meanX):
    p1=N/(sigma*np.sqrt(2*math.pi))
    p2=-0.5*((X-meanX)/sigma)**2
    p3=p1*math.e**p2
    return p3

X,T,xmin,xmax=Readdatafile('Assignment_1_CSV.csv', X,T,xmin,xmax)
HF, HM=BuildHistogramClassifier(X,T,B,xmin,xmax)

writeDatafile(HF,HM,'temp.csv')

hDataset=[55,60,65,70,75,80]

#calcualte the mean and standard deviation of female and male
meanF=np.mean(dataF)
meanM=np.mean(dataM)
stdF=np.std(dataF)
stdM=np.std(dataM)

for i in hDataset:
    bF=calBayesianFormulas(8900,stdF,i,meanF)
    bM=calBayesianFormulas(7800,stdM,i,meanM)
    PF=bF/(bF+bM)
    print(str(i) + '=' + str(PF))






