# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Proteomics and machine learning identify a distinct blood biomarker panel to detect Parkinson’s disease up to 
# 7 years before motor disease

# *Jenny Hällqvist1,2#, *Michael Bartl3,4#, Mohammed Dakna3, Sebastian Schade5, Paolo Garagnani6, Maria-Giulia 
# Bacalini7, Chiara Pirazzini6, Kailash Bhatia8, Sebastian Schreglmann8, Mary Xylaki3, Sandrina Weber3, Marielle Ernst9, 
# Maria-Lucia Muntea5, Friederike Sixel-Döring5,10, Claudio Franceschi6, Ivan Doykov1, Justyna Śpiewak1, Héloїse 
# Vinette1,11, Claudia Trenkwalder5,12 Wendy E. Heywood1, PROPAGE-AGEING Consortium, Kevin Mills2*, Brit Mollenhauer3,5*

# 1	  UCL Institute of Child Health and Great Ormond Street Hospital, London, UK
# 2	  UCL Queen Square Institute of Neurology, Clinical and Movement Neurosciences, London, UK
# 3	  Department of Neurology, University Medical Center Goettingen, Germany
# 4	  Institute for Neuroimmunology and Multiple Sclerosis Research, University Medical Center Goettingen, Germany
# 5	  Paracelsus-Elena-Klinik, Kassel, Germany
# 6	  Department of Experimental, Diagnostic, and Specialty Medicine (DIMES), University of Bologna, Bologna, Italy
# 7	  IRCCS Istituto delle Scienze Neurologiche di Bologna, Bologna, Italy
# 8	  National Hospital for Neurology & Neurosurgery, Queen Square, London, WC1N3BG
# 9	  Institute of Diagnostic and Interventional Neuroradiology, University Medical Center Goettingen, Germany
# 10  Department of Neurology, Philipps-University, Marburg, Germany
# 11  UCL: Food, Microbiomes and Health Institute Strategic Programme, Quadram Institute Bioscience, Norwich Research Park, Norwich, UK
# 12  Department of Neurosurgery, University Medical Center Goettingen, Goettingen, Germany

# *Both first and last authors contributed equally
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:58:18 2024

@author: Jenny Hallqvist
"""

# Import packages and functions
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC as SVM
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from JH_function_zScore import zScore
import warnings 
warnings.filterwarnings("ignore")
plt.rcParams.update({'font.sans-serif':'Century Gothic'})

# =============================================================================
# Load the main dataset, z-score variables, replace NaN with median and define x and y of dataset
# =============================================================================
dataFile = r'...'                                               # Path to Excel file
sheetName = '...'                                               # Name of Excel sheet

dataFile_2 = r'...'                            # Path to Excel file containing new data
sheetName_2 = '...'                                             # Name of Excel sheet

DF_Import = pd.read_excel(dataFile, sheet_name = sheetName, header = 0, 
                  skipfooter = 0, index_col = 0)

DF_Import_2 = pd.read_excel(dataFile_2, sheet_name = sheetName_2, header = 0, 
                  index_col = 0)

DF_Import = zScore(DF_Import, 2)                                # z-score data
DF_DNP_DKK = DF_Import[DF_Import['Class'].isin(['DKK', 'DNP'])] # Collect controls and de novo PD samples
x_DNP_DKK = DF_DNP_DKK.iloc[:,2:]                               # Define x
y = DF_DNP_DKK['DKK_0_DNP_1']                                   # Define y
x_DNP_DKK.fillna(x_DNP_DKK.median(), inplace = True)            # Replace NaNs with median
# =============================================================================

# =============================================================================
# Select a random set from the dataframe to use as a training set
# =============================================================================
x_train, x_test, y_train, y_test = train_test_split(x_DNP_DKK, y, 
                                                        test_size = 0.3, 
                                                        random_state = 0)
# =============================================================================

# =============================================================================
# Determine the number of features to select for SVM
# =============================================================================
estimator_SVM = SVM(dual = True)
selector_SVM = RFECV(estimator_SVM, step = 1, cv = StratifiedKFold(n_splits = 5, 
                                                                   shuffle = True, 
                                                                   random_state = 0))
selector_SVM = selector_SVM.fit(x_train, y_train)
numberFeatures_SVM = np.count_nonzero(selector_SVM.support_)
# =============================================================================

# =============================================================================
# Select features and train the model
# =============================================================================
selector_SVM = RFE(estimator_SVM, n_features_to_select = numberFeatures_SVM, step = 1)
selector_SVM = selector_SVM.fit(x_train, y_train)
varSelection_SVM = selector_SVM.support_
varRank_SVM = selector_SVM.ranking_
varList_SVM = pd.DataFrame(list(zip(DF_DNP_DKK.columns[2:], varSelection_SVM, varRank_SVM)), 
                        columns = ['Variable_SVM', 'BooleanSelector', 'Variable_Ranking'])
varList_SVM = varList_SVM[varList_SVM.BooleanSelector != False]
varList_SVM = list(varList_SVM['Variable_SVM'])

svm = SVM(dual = True, max_iter = 5000)
regModel_SVM = svm.fit(x_train[[*varList_SVM]], y_train)
# =============================================================================

# =============================================================================
# Predict y in the reduced models
# =============================================================================
class_pred_SVM = regModel_SVM.predict(x_test[[*varList_SVM]])               # Predicted classes
modelScore_SVM = regModel_SVM.score(x_train[[*varList_SVM]], y_train)       # Scores of the training model
cv_score_SVM = cross_val_score(regModel_SVM, x_train[[*varList_SVM]],       # CV scores of the training model
                               y_train, cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)).mean() 
predClasses = pd.DataFrame(list(zip(y_test, class_pred_SVM)),
                           index = y_test.index, columns = ['Actual class', 'Predicted class SVM']).sort_values(by = ['Actual class'])
# =============================================================================

# =============================================================================
# Predict the groups of iRBD and other neuro. disorder samples
# =============================================================================
DF_OND_DKS = DF_Import[DF_Import['Class'].isin(['DKS', 'OND'])]             # Collect iRBD and other neuro. samples from initial dataset (already z-scored)
x_OND_DKS = DF_OND_DKS.iloc[:,2:]                                           # Define x
x_OND_DKS.fillna(x_OND_DKS.median(), inplace = True)                        # Replace NaNs with median
class_pred_SVM_OND_DKS = regModel_SVM.predict(x_OND_DKS[[*varList_SVM]])    # Predicted classes

predClasses_OND_DKS = pd.DataFrame(list(zip(DF_OND_DKS['Class'], class_pred_SVM_OND_DKS)),
                           index = x_OND_DKS.index, columns = ['Actual class', 'Predicted class SVM'])

DF_New_DKS = zScore(DF_Import_2, 2)                                         # Collect iRBD samples from new and additional dataset and z-score
x_New_DKS = DF_New_DKS.iloc[:,2:]                                           # Define x
x_New_DKS.fillna(x_New_DKS.median(), inplace = True)                        # Replace NaNs with median
class_pred_New_iRBD = regModel_SVM.predict(x_New_DKS[[*varList_SVM]])       # Predicted classes

predClasses_New_DKS = pd.DataFrame(list(zip(DF_New_DKS['Class'], class_pred_New_iRBD)),
                           index = x_New_DKS.index, columns = ['Actual class', 'Predicted class SVM'])
# =============================================================================

# =============================================================================
# Repeated k-fold cross validation of the model, applied to all data
# Collect classification metrics
# =============================================================================
# # # Define empty lists/dataframes to populate
list_precision = []
list_recall = []
list_F1 = []
list_MCC = []
list_BAC = []
df_precision_CV = pd.DataFrame()
df_recall_CV = pd.DataFrame()
df_fpr_CV = pd.DataFrame()
df_tpr_CV = pd.DataFrame()

# # # Define cross validation strategy
rsKFold = RepeatedStratifiedKFold(n_splits = 6, n_repeats = 40, random_state = 0)

# # # Loop through obervations according to CV split
for i, (train_index, test_index) in enumerate(rsKFold.split(np.array(x_DNP_DKK), y)):
    # # # Define training model for each iteration
    estimator_SVM_iter = SVM(dual = True)
    selector_SVM_iter = RFECV(estimator_SVM_iter, step = 1, cv = StratifiedKFold(n_splits = 3, 
                                                                       shuffle = True, 
                                                                       random_state = 0))
    selector_SVM_iter = selector_SVM_iter.fit(x_DNP_DKK.iloc[train_index], y.iloc[train_index])
    varSelection_SVM_iter = selector_SVM_iter.support_
    numberFeatures_SVM_iter = np.count_nonzero(selector_SVM_iter.support_)
    
    varRank_SVM_iter = selector_SVM_iter.ranking_
    varList_SVM_iter = pd.DataFrame(list(zip(x_DNP_DKK.columns[2:], varSelection_SVM_iter, varRank_SVM_iter)), 
                            columns = ['Variable_SVM', 'BooleanSelector', 'Variable_Ranking'])
    varList_SVM_iter = varList_SVM_iter[varList_SVM_iter.BooleanSelector != False]
    varList_SVM_iter = list(varList_SVM_iter['Variable_SVM'])
    svm = SVM(dual = True, max_iter = 5000)
    regModel_SVM_iter = svm.fit(x_DNP_DKK.iloc[train_index][[*varList_SVM_iter]], y.iloc[train_index])
    # # # Calculate precision, recall and F1-score
    precision_val, recall_val, fbeta_score_val, __ = precision_recall_fscore_support( 
        y.iloc[test_index], regModel_SVM_iter.predict(x_DNP_DKK.iloc[test_index][[*varList_SVM_iter]]), 
        average = 'weighted')
    list_precision.append(precision_val)
    list_recall.append(recall_val)
    list_F1.append(fbeta_score_val)
    # # # Calculate the balanced accuracy score
    bac = balanced_accuracy_score(y.iloc[test_index], regModel_SVM_iter.predict(x_DNP_DKK.iloc[test_index][[*varList_SVM_iter]]))
    list_BAC.append(bac)  
    # # # Precision-recall and receiver operating characteristics (ROC)
    y_test_pred_CV = regModel_SVM_iter.decision_function(x_DNP_DKK.iloc[test_index][[*varList_SVM_iter]]) # Collect decision function
    # # # Precision-recall
    precision_CV, recall_CV, __ = precision_recall_curve(y.iloc[test_index], y_test_pred_CV) # P-R
    df_precision_CV = pd.concat(([df_precision_CV, pd.DataFrame(precision_CV)]), axis = 1, ignore_index = True) # P-R
    df_recall_CV = pd.concat(([df_recall_CV, pd.DataFrame(recall_CV)]), axis = 1, ignore_index = True) # P-R
    
    # ROC
    roc_curve(y.iloc[test_index], y_test_pred_CV)
    fpr_CV, tpr_CV, __ = roc_curve(y.iloc[test_index], y_test_pred_CV) # ROC
    df_fpr_CV = pd.concat(([df_fpr_CV, pd.DataFrame(fpr_CV)]), axis = 1, ignore_index = True) # ROC
    df_tpr_CV = pd.concat(([df_tpr_CV, pd.DataFrame(tpr_CV)]), axis = 1, ignore_index = True) # ROC
    
# # # Averages of repeaded metrics
avg_Precision = np.average(list_precision)
avg_Recall = np.average(list_recall)
avg_F1 = np.average(list_F1)
avg_MCC = np.average(list_MCC)
avg_BAC = np.average(list_BAC)
# =============================================================================

# =============================================================================
# Plots
# =============================================================================
# # # Important features
featImport = pd.DataFrame(regModel_SVM.coef_, index = ['Coeff'], 
                          columns = varList_SVM).transpose()
featImport = abs(featImport).sort_values(by = ['Coeff'])

fig, ax = plt.subplots(figsize = [10,10], tight_layout = True) 
plt.hlines(featImport.index, xmin = 0, xmax = featImport['Coeff'], 
           linewidth = 5, color = 'lightcoral', alpha = 0.5)
plt.plot(featImport['Coeff'], range(0, len(featImport.index)), 'o', 
         markersize = 30, 
         color = 'lightcoral')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.xlabel("Feature importance")
plt.show()

# # # Results of final predictions
fig, ax = plt.subplots(figsize = [10,10], tight_layout = True) 
j = 0
for j in range(len(predClasses.columns)):
    colours = predClasses[predClasses.columns[j]].replace(0, 'grey')
    colours = colours.replace(1, 'lightcoral')
    plt.scatter(np.linspace(0, len(predClasses), len(predClasses)), 
                predClasses[predClasses.columns[j]] + j*5, 
                c = colours, s = 400, alpha = 0.8, lw = 3)
    j = j + 1
    
ax.set_yticks([0.5, 5.5])
ax.set_xticks([])
ax.set_yticklabels(predClasses.columns, fontsize = 18)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.show()

# # # Histograms of repeated k-fold cross validation metrics
fig, ax = plt.subplots(1,4, figsize = [15, 5], layout = 'constrained', sharey = True) 

sns.histplot(ax = ax[0], x = list_precision, color = '#E50000', alpha = 0.4, edgecolor = '#E50000', linewidth = 2)
ax[0].set_xlabel('Precision')
sns.histplot(ax = ax[1], x = list_recall, color = '#E50000', alpha = 0.4, edgecolor = '#E50000', linewidth = 2)
ax[1].set_xlabel('Recall')
sns.histplot(ax = ax[2], x = list_F1, color = '#E50000', alpha = 0.4, edgecolor = '#E50000', linewidth = 2)
ax[2].set_xlabel('F1')
sns.histplot(ax = ax[3], x = list_BAC, color = '#E50000', alpha = 0.4, edgecolor = '#E50000', linewidth = 2)
ax[3].set_xlabel('Balanced accuracy score')

for axis in range(4):
    ax[axis].spines['top'].set_visible(False)
    ax[axis].spines['right'].set_visible(False)
    ax[axis].spines['bottom'].set_visible(True)
    ax[axis].spines['left'].set_visible(True)
    ax[axis].grid(which = 'major', axis = 'y', 
                  color = 'lightgrey', linestyle = '-.', linewidth = 0.5)
    ax[axis].xaxis.label.set_fontsize(18)
    ax[axis].yaxis.label.set_fontsize(18)
    ax[axis].tick_params(axis='both', which='major', labelsize=18)
plt.show()

# # # Plot precision-recall curve of test set
regModel_SVM = svm.fit(x_train[[*varList_SVM]], y_train)
y_test_pred = regModel_SVM.decision_function(x_test[[*varList_SVM]])    
precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred)
fig, ax = plt.subplots(figsize = (15,15))
display = PrecisionRecallDisplay(precision, recall)

plt.plot(recall, precision, label = ' Combined SVM predictors (AUC = ' + str(round(auc(recall, precision),2)) + ')', 
          alpha = 0.5, lw = 12, ls = 'solid', c = 'grey')

myPalette = ['#E50000', '#FF4653', '#EE7879', '#F4ABAA',
              '#E50000', '#FF4653', '#EE7879', '#F4ABAA']
line = ['solid','solid','solid','solid','dashed',
        'dashdot','dotted','dashed']

col = 0
for var in varList_SVM:
    svmSingle_model = svm.fit(x_train[var].values.reshape(-1,1), y_train)
    y_test_pred_s = svmSingle_model.decision_function(x_test[var].values.reshape(-1,1))    
    precision_s, recall_s, __ = precision_recall_curve(y_test, y_test_pred_s)
    plt.plot(recall_s, precision_s, label = var + ' (AUC = ' + str(round(auc(recall_s, precision_s),2)) + ')', 
              ls = line[col], alpha = 0.9, lw = 6, c = myPalette[col])
    col = col + 1
    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.label.set_fontsize(26)
ax.yaxis.label.set_fontsize(26)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(fontsize = 20)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# # # Plot ROC of test set
regModel_SVM = svm.fit(x_train[[*varList_SVM]], y_train)
y_test_pred = regModel_SVM.decision_function(x_test[[*varList_SVM]])    
roc_curve(y_test, y_test_pred)
fig, ax = plt.subplots(figsize = (15,15))
test_fpr, test_tpr, tr_thresholds = roc_curve(y_test, y_test_pred)
plt.plot(test_fpr, test_tpr, label = ' Combined SVM predictors (AUC = ' + str(auc(test_fpr, test_tpr)) + ')',  
                                  alpha = 0.5, lw = 12, ls = 'solid', c = 'grey')
myPalette = ['#E50000', '#FF4653', '#EE7879', '#F4ABAA','#E50000', '#FF4653', '#EE7879', '#F4ABAA']
line = ['solid','solid','solid','solid','dashed','dashdot','dotted','dashed']

col = 0
for var in varList_SVM:
    svmSingle_model = svm.fit(x_train[var].values.reshape(-1,1), y_train)
    y_test_pred_s = svmSingle_model.decision_function(x_test[var].values.reshape(-1,1))    
    test_fpr_s, test_tpr_s, tr_thresholds_s = roc_curve(y_test, y_test_pred_s)
    plt.plot(test_fpr_s, test_tpr_s, label = var + ' (AUC = ' + str(round(auc(test_fpr_s, test_tpr_s), 2)) + ')',  
              ls = line[col], alpha = 0.9, lw = 6, c = myPalette[col])
    col = col + 1

plt.plot([0,1],[0,1], ls = 'dashed', c = 'grey', alpha = 0.8)
   
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.label.set_fontsize(26)
ax.yaxis.label.set_fontsize(26)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(fontsize = 20)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
# =============================================================================


