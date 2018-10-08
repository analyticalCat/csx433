import numpy as np
import pandas as pd
import decimal as dcm
import math
import numpy.linalg as LA

def readExcelSheet1(excelfile):
    from pandas import read_excel
    return (read_excel(excelfile)).values

def readExcelRange(excelfile,sheetname="Sheet1",startrow=1,endrow=1,startcol=1,endcol=1):
    from pandas import read_excel
    values=(read_excel(excelfile, sheetname,header=None)).values
    return values[startrow-1:endrow,startcol-1:endcol]

def readExcel(excelfile,**args):
    if args:
        data=readExcelRange(excelfile,**args)
    else:
        data=readExcelSheet1(excelfile)
    if data.shape==(1,1):
        return data[0,0]
    elif (data.shape)[0]==1:
        return data[0]
    else:
        return data

def writeExcelData(x,excelfile,sheetname,startrow,startcol):
    from pandas import DataFrame, ExcelWriter
    from openpyxl import load_workbook
    df=DataFrame(x)
    book = load_workbook(excelfile)
    writer = ExcelWriter(excelfile,  engine='openpyxl') 
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.to_excel(writer, sheet_name=sheetname,startrow=startrow-1, startcol=startcol-1, header=False, index=False)
    writer.save()
    writer.close()

def getSheetNames(excelfile):
    from pandas import ExcelFile
    return (ExcelFile(excelfile)).sheet_names

excelfile='c:\\Users\\jess_\\Downloads\\Assignment_4_Data_and_Template.xlsx'

sheets=getSheetNames(excelfile)

arg={"sheetname":'Training Data',"startrow":1, "endrow":6601, "startcol":1, "endcol":17}
data=readExcel(excelfile,**arg)

X=np.array(data[1:,:-2],dtype=float) 
TB=np.array(data[1:,-2],dtype=int)
TM=np.array(data[1:,-1],dtype=int)
TMM=np.zeros((len(TM),len(np.unique(TM))))
TMM=TMM-1

#Keslerized the result
for i, row in enumerate(TMM):
    row[TM[i]]=1
    

#Prepend 1 to matrix X to form Xa
Xa=np.insert(X,0,1,axis=1)

#Get the Pseudoinverse of Xa ans Xap
Xap=LA.pinv(Xa,1e-5)
print(Xap.shape, Xap[0])

#Calculate W
W=np.dot(Xap, TB)
WM=np.dot(Xap, TMM)
print('classifier:', WM)
    
writeExcelData(W,excelfile,'Classifiers',5,1)
writeExcelData(WM,excelfile,'Classifiers',5,5)

#classifier application on one data
index=0
xa=Xa[index]

est=np.dot(xa,W)
if est>0: 
    ta=1 
else: 
    ta=-1
print(index, xa, 'TB[index]=',TB[index],'ta=',ta)
estM=np.dot(xa,WM)
print('C classifiers result:', np.where(estM==np.amax(estM))[0])


#calculating the confusion matrix for the training set
TP=0;TN=0;FP=0;FN=0

for i, row in enumerate(Xa):
    if np.dot(row,W)>0: #positive
        if TB[i]==1: #true positive
            TP+=1
        else: #false positive
            FP+=1
    else:
        if TB[i]==1: # false negative
            FN+=1
        else:# true negative
            TN+=1

print('i=',i)
print('TP:', TP,'FN:', FN)
print('FP:', FP, 'TN:', TN)
#calculating Accuracy, sensitivity, Specificity and PPV
performanceB=np.zeros(4,dtype=float)
performanceB[0]=(TP+TN)/(TP+TN+FP+FN) # accuracy
performanceB[1]=TP/(TP+FN) #sensitivity
performanceB[2]=TN/(FP+TN) #specificity
performanceB[3]=TP/(FP+TP) #PPV

writeExcelData([(TN,FP),(FN,TP)],excelfile, 'Performance',10,3)
writeExcelData(performanceB,excelfile, 'Performance',8,7)

#calculating the confusion matrix for 6-class classifiers
confusion6=np.zeros((6,6),dtype=int)
estM=np.dot(Xa,WM)
for i, row in enumerate(estM):
    index=np.where(np.amax(row)==row)
    confusion6[TM[i],index]+=1

print('confusion matrix for 6:', confusion6)
writeExcelData(confusion6,excelfile, 'Performance',19,3)


#import the testing set
arg={"sheetname":'To be classified',"startrow":5, "endrow":54, "startcol":1, "endcol":15}
data=readExcel(excelfile,**arg)

testX=np.insert(data,0,1,axis=1)
testTB=np.zeros(len(testX),dtype=int)
testTM=np.zeros(len(testX),dtype=int)

for i, row in enumerate(testX):
    est=np.dot(row,W)
    if est>0:
        testTB[i]=1
    else:
        testTB[i]=-1

writeExcelData(testTB,excelfile,'To be classified', 5,16)

estM=np.dot(testX, WM)
for i, row in enumerate(estM):
    testTM[i]=int(np.where(np.amax(row)==row)[0])

writeExcelData(testTM,excelfile,'To be classified', 5,17)


# notes from the professor during lecture 6:
# TB=data[:,-2]
# TM=data[:,-1]
# augment=lambda X: np.c_[np.ones]
# Calssifier = lambda X,T: np.dot(np.linalg.pinv(Augment(X)), T if le(np.unique(T))==2 else Keslerize(T))
# Applyclassifier=lambda X,W: unkeslerize(np.dot(augment(X),W)) if np.ndim(W)==2 else np.sign(np.dot(Augment(X),W)).astype(int)
# def confunsionMatrix(T,R):
#     labels=np.unique(T);
#     C=len(labels)
    
# Ask Vignesh for pictures.
# You always start with linear classifiers.  If not working or some special result. then logistic regression.

