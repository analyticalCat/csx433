import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans


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

excelfile='c:\\Users\\jess_\\Downloads\\Assignment_5_Data_and_Template.xlsx'

sheets=getSheetNames(excelfile)

arg={"sheetname":'Data',"startrow":1, "endrow":951, "startcol":1, "endcol":2}
data=readExcel(excelfile,**arg)


X_train=np.array(data[1:],dtype=float)
T=np.zeros((len(X_train),3),dtype=int)
print(X_train.shape, T.shape)

#Apply Gaussian mixture

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=3, covariance_type='full')
print(clf.n_components)
clf.fit(X_train)


# # display predicted scores by the model as a contour plot
x = np.linspace(00., 120.,240)
y = np.linspace(0., 50., 100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X,Y,Z, norm=LogNorm(vmin=1.0, vmax=100.0),
                  levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.5, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1],.5)
plt.title('Original training data')
plt.axis('tight')
plt.show()

#predict the class lable, confidence and count
clabels=clf.predict(X_train)
confs=clf.predict_proba(X_train)
means=clf.means_
if max(means[:,0])==means[0,0]:
    indexM=0
        
    if min(means[:,0])==means[1,0]:
        indexC=1
        indexF=2
    else:
        indexC=2
        indexF=1
else:
    if max(means[:,0])==means[1,0]:
        indexM=1
        if min(means[:,0])==means[0,0]:
            indexC=0
            indexF=2
        else:
            indexC=2
            indexF=0
    else:
        indexM=2
        if min(means[:,0])==means[0,0]:
            indexC=0
            indexF=1
        else:
            indexC=1
            indexF=0


countM=0;countF=0;countC=0

R=[]

for i,value in enumerate(clabels):
    if value==indexM:
        label='M'
        countM+=1
    else:
        if value==indexF:
            label='F'
            countF+=1
        else:
            label='C'
            countC+=1

    conf=max(confs[i])
    R.append((label,conf))

R=np.array(R)
count=[countM,countF,countC]
writeExcelData(R,excelfile,'Results',2,1)
writeExcelData(count, excelfile, 'Results',2,6)

