import numpy as np
import pandas as pd
import decimal as dcm
import math
import matplotlib.pyplot as plt

def classifierPerformance(confusionMatrix, caption, totalc):
    accuracy=np.zeros((10), dtype=float)
    sensitivity=np.zeros((10), dtype=float)
    specificity=np.zeros((10), dtype=float)
    ppv=np.zeros((10),dtype=float)

    for i,value in enumerate(confusionMatrix):
        rowsum=np.sum(confusionMatrix[:,i])
        colsum=np.sum(confusionMatrix[i,:])
        TN=confusionMatrix[i,i]
        TP=(totalc-rowsum-colsum+confusionMatrix[i,i])
        FN=sum(confusionMatrix[:,i])-confusionMatrix[i,i]
        FP=sum(confusionMatrix[i,:])-confusionMatrix[i,i]
        accuracy[i]=(TP+TN)/totalc
        sensitivity[i]=TP/(TP+FN)
        specificity[i]=TN/(FP+TN)
        ppv[i]=TP/(TP+FP)
    
    print('classifier performance:', caption, accuracy, sensitivity, specificity, ppv)
    
    return (accuracy, sensitivity,specificity, ppv)

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

excelfile='c:\\Users\\jess_\\Downloads\\finalproject.xlsx'

