import numpy as np
import pandas as pd
import decimal as dcm
import math



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

excelfile='c:\\Users\\jess_\\Downloads\\Assignment_2_Data_and_Template.xlsx'

sheets=getSheetNames(excelfile)

arg={"sheetname":'Data',"startrow":1, "endrow":168, "startcol":1, "endcol":3}
data=readExcel(excelfile,**arg)

X=np.array(data[1:,1:3],dtype=float) #why 1:3?  I thought it should be 1:2???
T=np.array(data[1:,0],dtype=str)


# Female bucket and male bucket
NF=np.array(data[(data[:,0]=='Female'),1:],dtype=float)
NM=np.array(data[(data[:,0]=='Male'),1:],dtype=float)

fmheight=np.mean(NF[:,0])
mmheight=np.mean(NM[:,0])
print ('Female mean height'); print (np.mean(NF[:,0]))
print ('Male mean height'); print (np.mean(NM[:,0]))

fmspan=np.mean(NF[:,1])
mmspan=np.mean(NM[:,1])
print ('Female mean span'); print (np.mean(NF[:,1]))
print ('Male mean span'); print (np.mean(NM[:,1]))

fcov=np.cov(NF[:,0],NF[:,1])
mcov=np.cov(NM[:,0],NM[:,1])
writeExcelData(fcov, excelfile,'Bayesian',4,3)
writeExcelData(mcov, excelfile,'Bayesian',6,3)

print ('Female Covariance'); print (fcov)
print ('Male Covariance'); print (mcov)


queries=(readExcel(excelfile,
                  sheetname='Queries',
                  startrow=3,
                  endrow=6,
                  startcol=1,
                  endcol=2)).astype(float);queries

htmin=0.0;htmax=0.0;hsmin=0.0;hsmax=0.0


B=8


def find2Dbin(xHeight, xSpan, B, hmin, hmax, smin, smax):
    up=float(xHeight)-float(hmin)
    bot=float(hmax)-float(hmin) 
    r=int(round(up/bot*(B-1)))

    up=float(xSpan)-float(smin)
    bot=float(smax)-float(smin) 
    c=int(round(up/bot*(B-1)))
    
    return r,c

htmin=min(X[:,0])
htmax=max(X[:,0])
hsmin=min(X[:,1])
hsmax=max(X[:,1])


#construct 2-D histogram for female and male
def BuildHistogramClassifier(X,B,htmin,htmax,hsmin,hsmax):
    HF=np.zeros((B,B),dtype=int)
    HM=np.zeros((B,B),dtype=int)

    
    for i in range(X.shape[0]):
        r,c=find2Dbin(X[i,0], X[i,1], B, htmin, htmax, hsmin,hsmax)
        if T[i]=='Female':
            HF[r,c]=HF[r,c]+1
        else:
            HM[r,c]=HM[r,c]+1

    writeExcelData(HF, excelfile, "Female Histogram", 7,2)


    writeExcelData(HM, excelfile, "Male Histogram", 7,2)

    return HF,HM


HF,HM=BuildHistogramClassifier(X, B, htmin, htmax,hsmin,hsmax)

#Histogram'
for row in queries:
    r,c=find2Dbin(row[0], row[1],B,htmin,htmax,hsmin,hsmax)
    print ('queries'); print (row[0], row[1])
    if HF[r,c]+HM[r,c]==0:
        print (row); print ('N/A')
    else:
        print('for '); print(row[0],row[1]); print(':'); print(float(HF[r,c])/float(HF[r,c]+HM[r,c]))
        

#Bayesian claasifier 
def Bayesianformula(xrow,d,xmean,xcov,samplenum):
    det=np.linalg.det(xcov)
    det=det**.5
    beforee=1.0/(2*np.pi*det)
    #print ('beforee'); print( beforee)

    covinverse=np.linalg.inv(xcov)
    xminusmean=np.array(xrow)-np.array(xmean)
    # print 'step1';print xminusmean
    # print 'step2'; print xminusmeanT
    # print 'covinverse'; print covinverse
    #aftere=-0.5*xminusmean*covinverse*xminusmeanT
    aftere=xminusmean.dot(covinverse)
    afteree=aftere.dot(xminusmean.T)
    aftereee=-0.5*afteree
    
    result=beforee*math.e**aftereee*samplenum
    return result


xfmean=[fmheight,fmspan]
xmmean = [mmheight, mmspan]
startrow=3;startcol=4
for row in queries:
    PDFfemale=Bayesianformula(row,2,xfmean, fcov,89)
    PDFmale=Bayesianformula(row,2,xmmean, mcov,78)
    print (PDFfemale,PDFmale)
    print( "PDF");print(row);print( PDFfemale/(PDFfemale+PDFmale))
    

#Extra Credit: reconstruct the female histogram using female model parameters
def reconstructHistogram(HF,HM,B,hmin,hmax,smin,smax):
    #first, find the h and s value in each H[r,c]
    HFRec=np.zeros((B,B),dtype=float)
    HMRec=np.zeros((B,B),dtype=float)
    sumf=0.0
    summ=0.0
    bw=(hmax-hmin)/B* (smax-smin)/B

  #initialize 
    recfhmax=0.0
    recfhmin=0.0
    recfsmax=0.0
    recfsmin=0.0
    recmsmax=0.0
    recmsmin=0.0
    recmhmin=0.0
    recmhmax=0.0

    for i,row in enumerate(HF):
        h=i/B*(hmax-hmin)+hmin+ 0.5*(hmax-hmin)/B
        for j,cell in enumerate(row):
            s=j/B*(smax-smin)+smin+0.5*(smax-smin)/B
            rec=[(h,s)]
            PDFF=Bayesianformula(rec,2,xfmean, fcov,89)
            PDFM=Bayesianformula(rec,2,xmmean, mcov,78)
            print (PDFF*bw, PDFM*bw)
            HFRec[i,j]=np.round(PDFF*bw)
            HMRec[i,j]=np.round(PDFM*bw)

            if HFRec[i,j]!=0:
                recfhmax=max(h,recfhmax)
                if recfhmin==0: recfhmin=h
                recfhmin=min(h, recfhmin)
                recfsmax=max(s, recfsmax)
                if recfsmin==0: recfsmin=s
                recfsmin=min(s, recfsmin)
            
            if HMRec[i,j]!=0:
                recmhmax=max(h,recmhmax)
                if recmhmin==0: recmhmin=h
                recmhmin=min(h, recmhmin)
                recmsmax=max(s, recmsmax)
                if recmsmin==0: recmsmin=s
                recmsmin=min(s, recmsmin)

            sumf+=PDFF*bw
            summ+=PDFM*bw
 
    print("pdf sum", sumf, summ)

    #finally, write to excel
    writeExcelData([round(recfhmin,1),round(recfhmax,1),round(recfsmin,1),round(recfsmax,1)], excelfile, 'Reconstructed Female Histogram',1,2)
    writeExcelData([round(recmhmin,1),round(recmhmax,1),round(recmsmin,1),round(recmsmax,1)],excelfile,'Reconstructed Male Histogram',1,2)
    writeExcelData(HFRec,excelfile, 'Reconstructed Female Histogram',7,2)
    writeExcelData(HMRec,excelfile, 'Reconstructed Male Histogram',7,2)

reconstructHistogram(HF,HM,B,htmin,htmax,hsmin,hsmax)