# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:03:16 2020

@author: twsle
"""
#Gaussian File Reader and Structural Descriptor Anaylsis
#Updated on 11.2.2020
#Trevor Sleight and Caitlin Sexton
#****************************************************

#Clear the workspace
get_ipython().magic('reset -sf')

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import metrics

from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Load Data Files
dir_path, filename = os.path.split(os.path.abspath(__file__))
os.path.join(dir_path, 'currentDatawithIP.xlsx')

 
data = pd.read_excel(os.path.join(dir_path, 'ta98Data.xlsx'), 
                     sheet_name = 'Sheet1')

#loads descriptors set
bestData = pd.read_excel(os.path.join(dir_path,"best_descriptors.xlsx"), 
                         sheet_name = 'Sheet1')

#loads descriptor set into a list
finalIDS = list(bestData['Index'][0:21])#  + ['HOMO', 'LUMO', 'Gibbs']

# Classification and ROC analysis
tprs = []
aucs = []
confs = np.array([[0,  0],[ 0, 0]])
mean_fpr = np.linspace(0, 1, 50)
    
for n in range(0,10):
    #get a new random seed from the clock
    cValue = int(time.perf_counter()*1000)

    #shuffle the order of the data rows at the n level
    data2 = data.sample(frac=1).reset_index(drop=True)
    
    finaldata = data2[finalIDS]
    finaldata[finalIDS] = StandardScaler().fit_transform(finaldata[finalIDS] )
    y = data2['TA98']
    
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=3)

    for i, (train, test) in enumerate(cv.split(finaldata, y)):
          
        X_train = finaldata.iloc[train,:]
        X_test = finaldata.iloc[test,:]    
        y_train = y[train]
        y_test = y[test]  

        logmodel = LogisticRegression(multi_class = 'ovr',solver = 'liblinear', 
                                      random_state=cValue, max_iter = 1000000)
        logmodel.fit(X_train, y_train)
        
        columnNames = list(finaldata.columns)   
        [a] = logmodel.coef_
        rankedColumnns_Raw = pd.DataFrame(data = {'Coef':a, 'Name':columnNames})
        
        #this threshold can be adjusted
        y_test_probabilities = logmodel.predict_proba(X_test)
        predictions = (y_test_probabilities[:,1] > 0.3).astype(int)
       
        #print off the confusion matrixes
        print(classification_report(y_test, predictions))
        conf = confusion_matrix(y_test, predictions)
        print(conf)
        confs += conf
        print(accuracy_score(y_test, predictions))
        
        y_pred_proba = logmodel.predict_proba(X_test)[::,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test,  y_pred_proba, pos_label = 1)
        auc1 = metrics.roc_auc_score(y_test, y_pred_proba)
    
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        
        tprs.append(interp_tpr)
        aucs.append(auc1)

fig, ax = plt.subplots()
    
tn = confs[0,0]
tp = confs[1,1]
fn = confs[1,0]
fp = confs[0,1]

acc = (tp+tn) / (tp + tn + fn + fp)

#compute the final QSAR metrix
print('accuracy is ', acc) 
print('Precision is ', tp/(tp+fp))
prec = tp/(tp+fp)
print('Recall/Sensitivity ', tp/(tp+fn))  
rec =  tp/(tp+fn)
print('Specificity is ', tn/(tn+fp))
print('F1 is ', 2*(prec*rec)/(prec+rec))

#set the 50% guess line
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)


#print the ROC plot
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey',
                alpha=.2, label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic example")
ax.legend(loc="lower right")
plt.show()


