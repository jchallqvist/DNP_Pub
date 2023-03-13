# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 17:05:15 2021

@author: Jenny Hallqvist
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import LinearSVC as SVM
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 
from JH_function_zScore import zScore

plt.rcParams.update({'font.sans-serif':'Century Gothic'}) # new


#dataFile = r'C:\Users\Jenny Halqvist\OneDrive - University College London\PhD_Thesis\Data\Targeted_DNP_Plasma\Targeted_GottingenPlasma_Outlier_Age_Sex_Corrected.xlsx'
#dataFile = r'C:\Users\Jenny Halqvist\OneDrive - University College London\PhD_Thesis\Data\Targeted_GottingenPlasma_Outlier_Age_Sex_Corrected.xlsx'
#sheetName = 'Prediction_Age_Sex_Corr_DKK_DNP'

dataFile = r'file:///C:\Users\Jenny%20Halqvist\OneDrive%20-%20University%20College%20London\Paper_Predictive_PD_Markers\20210716_DNP_For_Paper.xlsx'
sheetName = 'Non_Age_Sex_Corrected_DNP_IDs'

DF_Valid = pd.read_excel(dataFile, sheet_name=sheetName, header=0, skiprows=None, 
                  skipfooter=0, index_col=0, names=None, 
                  parse_dates=False, date_parser=None, 
                  na_values=None, thousands=None, convert_float=True, 
                  converters=None, true_values=None, 
                  false_values=None, engine=None, squeeze=False)

DF_Valid = zScore(DF_Valid, 2) # z-score to get true size of variable importance

# Start of feature columns
fCol = 2

figSize = [10,10]

# =============================================================================
 # Model variables (remember to check for outliers before)
x = DF_Valid.iloc[:,fCol:]       # independent var(s)
y = DF_Valid['DKK_0_DNP_1']    # dependent var
# =============================================================================

# =============================================================================
 # Replace NaNs with median
x_Imput = x.copy()
x_Imput.fillna(x_Imput.median(), inplace = True)
# =============================================================================

# =============================================================================
 # Alternatively, drop all NaN-containing rows
# x_dropNaN = DF_Valid.copy()
# x_dropNaN = x_dropNaN.dropna()
# y_dropNaN = x_dropNaN[y.name]
# x_dropNaN = x_dropNaN.iloc[:,fCol:] 
# =============================================================================

# =============================================================================
 # Define which NaN handling method to use -> x_train = ... and y = ...
# Drop NaN 
# x_matrix = x_dropNaN 
# y = y_dropNaN

# Imput NaN with median
x_matrix = x_Imput
y = y
# =============================================================================

# =============================================================================
# Select a random set from the dataframe to use as a training set
x_train, x_test, y_train, y_test = train_test_split(x_matrix, y, 
                                                        test_size = 0.3, random_state = 0) 
# =============================================================================

# =============================================================================
# Determine the number of features to select for SVM
estimator_SVM = SVM()
selector_SVM = RFECV(estimator_SVM, step = 1, cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0))
selector_SVM = selector_SVM.fit(x_train, y_train)
check = selector_SVM.support_ # new
numberFeatures_SVM = np.count_nonzero(selector_SVM.support_)
# =============================================================================

# =============================================================================
# Plot number of features vs cross-validation scores for SVM
fig, ax = plt.subplots(figsize = figSize, tight_layout = True) 
plt.plot(range(1, len(selector_SVM.grid_scores_) + 1), selector_SVM.grid_scores_, 
         color = 'lightpink', lw = 4)
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score \n (no of correct classifications)")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.show()
# =============================================================================

# =============================================================================
# Feature selection by RFE
selector_SVM = RFE(estimator_SVM, n_features_to_select = numberFeatures_SVM, step = 1)
# selector_SVM = selector_SVM.fit(x_matrix, y)
selector_SVM = selector_SVM.fit(x_train, y_train)
varSelection_SVM = selector_SVM.support_
varRank_SVM = selector_SVM.ranking_
varList_SVM = pd.DataFrame(list(zip(DF_Valid.columns[fCol:], varSelection_SVM, varRank_SVM)), 
                       columns = ['Variable_SVM', 'BooleanSelector', 'Variable_Ranking'])
varList_SVM = varList_SVM[varList_SVM.BooleanSelector != False]
varList_SVM = list(varList_SVM['Variable_SVM'])
# =============================================================================

# =============================================================================
# Multiple linear discriminant regression using the selected variables
svm = SVM()
regModel_SVM = svm.fit(x_train[[*varList_SVM]], y_train) # Fit the model
# =============================================================================

# =============================================================================
# Plot the important features
featImport = pd.DataFrame(regModel_SVM.coef_, index = ['Coeff'], columns = varList_SVM).transpose()
featImport = abs(featImport).sort_values(by = ['Coeff'])

fig, ax = plt.subplots(figsize = figSize, tight_layout = True) 
# ax = plt.barh(featImport.index, featImport['Coeff'], color = 'lightcoral')
plt.hlines(featImport.index, xmin = 0, xmax = featImport['Coeff'], linewidth = 5, color = 'lightcoral', alpha = 0.5)
plt.plot(featImport['Coeff'], range(0, len(featImport.index)), 'o', markersize = 30, 
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
# =============================================================================

# =============================================================================
# CV scores for total datasets - to use as initial CV overview
sKf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
cv_SVM = cross_validate(regModel_SVM, x_matrix[[*varList_SVM]], y, cv = sKf, return_train_score = True)
     
testScore = [cv_SVM['test_score'].mean()]
trainScore = [cv_SVM['train_score'].mean()]
cv_Summary = pd.DataFrame(list(zip(cv_SVM['test_score'], cv_SVM['train_score'])), columns = ['test', 'train'])#, index = ['SVM'])
# =============================================================================

# =============================================================================
# Predict y in the reduced models
class_pred_SVM = regModel_SVM.predict(x_test[[*varList_SVM]]) # Predicted classes
modelScore_SVM = regModel_SVM.score(x_train[[*varList_SVM]], y_train) # Scores of the training model
cv_score_SVM = cross_val_score(regModel_SVM, x_train[[*varList_SVM]], y_train, cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)).mean() # CV scores of the training model

predClasses = pd.DataFrame(list(zip(y_test, class_pred_SVM)),
                           index = y_test.index, columns = ['Actual class', 'Predicted class SVM']).sort_values(by = ['Actual class'])
# =============================================================================

# =============================================================================
wrongClass_SVM = predClasses['Predicted class SVM'] != predClasses['Actual class'] # Find the wrongly predicted
correctlyPred_SVM = (len(wrongClass_SVM) - np.sum(wrongClass_SVM))/len(wrongClass_SVM)
# =============================================================================

# =============================================================================
# Plot results of final predictions
fig, ax = plt.subplots(figsize = figSize, tight_layout = True) 
j = 0
for j in range(len(predClasses.columns)):
    colours = predClasses[predClasses.columns[j]].replace(0, 'grey')
    colours = colours.replace(1, 'lightcoral')
    plt.scatter(np.linspace(0, len(predClasses), len(predClasses)), 
                predClasses[predClasses.columns[j]] + j*5, 
                c = colours, s = 400, alpha = 0.8, lw = 3)
    j = j + 1
    
ax.set_yticks([0.5, 5.5])#, 10.5, 15.5])
ax.set_xticks([])
ax.set_yticklabels(predClasses.columns, fontsize = 18)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.show()
# =============================================================================

# =============================================================================
# Plot ROC
fig, ax = plt.subplots(figsize = figSize, tight_layout = True) 
fullROC = plot_roc_curve(regModel_SVM, x_test[[*varList_SVM]], y_test, ax = ax, name = 'Combined', 
                         alpha = 1, lw = 6, ls = 'solid', c = 'grey')

# Plot individual ROC curves of predictor values
individual_predictions_SVM = []
svmSingle = SVM()
line = ['dashdot', 'solid', 'dotted', 'dashdot', 'solid', 'dashed', 'dotted',
        'dashdot', 'solid', 'dotted', 'dashdot']

colPalette = sns.color_palette('cubehelix', n_colors = len(varList_SVM)).as_hex()

col = 0

for var in varList_SVM:
    single_regModel_SVM = svmSingle.fit(pd.DataFrame(x_train[var]), y_train) # Fit the model
    single_class_pred_SVM = single_regModel_SVM.predict(pd.DataFrame(x_test[var])) # Predicted
    individual_predictions_SVM.append(single_class_pred_SVM)
    plot_roc_curve(single_regModel_SVM, pd.DataFrame(x_test[var]), y_test, ax = ax, name = var, 
                   alpha = 0.5 + col/5, lw = 4, ls = line[col], c = colPalette[col])
    col = col + 1
    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(fontsize = 14)

plt.show()

# =============================================================================
# Predict OND and DKS

# Import data
dataFile = r'file:///C:\Users\Jenny%20Halqvist\OneDrive%20-%20University%20College%20London\Paper_Predictive_PD_Markers\20210716_DNP_For_Paper.xlsx'
sheetName = 'Non_Age_Sex_Corr_iRBD_OND_IDs'

DF_Valid_2 = pd.read_excel(dataFile, sheet_name=sheetName, header=0, skiprows=None, 
                  skipfooter=0, index_col=0, names=None, 
                  parse_dates=False, date_parser=None, 
                  na_values=None, thousands=None, convert_float=True, 
                  converters=None, true_values=None, 
                  false_values=None, engine=None, squeeze=False)

x_OND_DKS = DF_Valid_2.iloc[:,1:]       # independent var(s)
# =============================================================================

# =============================================================================
 # Replace NaNs with median
x_Imput_OND_DKS = x_OND_DKS.copy()
x_Imput_OND_DKS.fillna(x_Imput_OND_DKS.median(), inplace = True)

# Predict y in the reduced models
class_pred_SVM_OND_DKS = regModel_SVM.predict(x_Imput_OND_DKS[[*varList_SVM]]) # Predicted classes

predClasses_OND_DKS = pd.DataFrame(list(zip(DF_Valid_2['Class'], class_pred_SVM_OND_DKS)),
                           index = DF_Valid_2.index, columns = ['Actual class', 'Predicted class SVM'])
# =============================================================================

