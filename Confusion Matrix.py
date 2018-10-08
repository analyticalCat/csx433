import numpy as np


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
    
    return accuracy, sensitivity,specificity, ppv
