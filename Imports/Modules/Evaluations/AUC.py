from sklearn.metrics import auc

def calculateAUC(fpr, tpr):

    return auc(fpr, tpr)
