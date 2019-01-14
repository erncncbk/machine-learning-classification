# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:45:14 2019

@author: Erencan
"""
import numpy as np
import pandas as pd 

df = pd.read_csv('weatherAUS.csv')
#%%
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# As we can see the first four columns have less than 60% data, we can ignore these four columns
# We don't need the location column because 
# we are going to find if it will rain in Australia(not location specific)
# We are going to drop the date column too.
# We need to remove RISK_MM because we want to predict 'RainTomorrow' and RISK_MM can leak some info to our model
df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)
df.shape

#%%   Missing Values
#Let us get rid of all null values in df
df = df.dropna(how='any')
df.shape

#%% Z-score
#its time to remove the outliers in our data - we are using Z-score to detect and remove the outliers.
from scipy import stats
z = np.abs(stats.zscore(df._get_numeric_data()))
print(z)
df= df[(z < 3).all(axis=1)]
print(df.shape)

#%%
#Lets deal with the categorical cloumns now
# simply change yes/no to 1/0 for RainToday and RainTomorrow
df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

#See unique values and convert them to int using pd.getDummies()
categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
for col in categorical_columns:
    print(np.unique(df[col]))
# transform the categorical columns
df = pd.get_dummies(df, columns=categorical_columns)
df.iloc[4:9]

# %%  Min-Max Scaler
#next step is to standardize our data - using MinMaxScaler
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
df.iloc[4:10]

#%%

X = df.drop(columns = ["RainTomorrow"],axis=1)
y = df[['RainTomorrow']]


#%% Train - Test Split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#%% Logistic Regression 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,roc_auc_score
clf_logreg = LogisticRegression()
clf_logreg.fit(X_train,y_train)
y_pred = clf_logreg.predict(X_test)
clf_logreg_accuracy_score = accuracy_score(y_test,y_pred)
clf_logreg_precision_score = precision_score(y_test,y_pred)
clf_logreg_recall_score  = recall_score(y_test,y_pred)
clf_logreg_f1_score      = f1_score(y_test,y_pred)
print('Accuracy :',clf_logreg_accuracy_score)
print('Precision :',clf_logreg_precision_score )
print('Recall :',clf_logreg_recall_score )
print('F1_score :',clf_logreg_f1_score )

#%% Confusion Matrix
from sklearn.metrics import confusion_matrix
clf_logreg_cm = confusion_matrix(y_test,y_pred)
sns.heatmap(clf_logreg_cm,annot=True,fmt="d",cmap=sns.cubehelix_palette(start=-5.8,rot=-.4) )
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.title("LR Classification - Confusion Matrix")
plt.show()
#%%
# calculate ROC curve
#y_score = clf_logreg.fit(X_train, y_train).decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#calculate AUC
auc = roc_auc_score(y_test, y_pred)
print('AUC: %.3f' % auc)

plt.plot(fpr, tpr, label='Logistic Regression-ROC curve (area = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic ')
plt.legend(loc="lower right")

#%% Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=120, max_depth=4)
clf_rf.fit(X_train,y_train)
y_pred1 = clf_rf.predict(X_test)
clf_rf_accuracy_score = accuracy_score(y_test,y_pred1)
clf_rf_precision_score = precision_score(y_test,y_pred1)
clf_rf_recall_score = recall_score(y_test,y_pred1)
clf_rf_f1_score      = f1_score(y_test,y_pred1)

print('Accuracy :',clf_rf_accuracy_score)
print('Precision :',clf_rf_precision_score )
print('Recall :',clf_rf_recall_score )
print('F1_score :',clf_rf_f1_score )

#%% Confusion Matrix
from sklearn.metrics import confusion_matrix
clf_rf_cm = confusion_matrix(y_test,y_pred1)
sns.heatmap(clf_rf_cm,annot=True,fmt="d",cmap= sns.cubehelix_palette(start=1,rot=-.4)) 
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.title("RF Classification - Confusion Matrix")
plt.show()

#%%
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
#calculate AUC
auc = roc_auc_score(y_test, y_pred1)
print('AUC: %.3f' % auc)

plt.plot(fpr, tpr, label='Random Forest-ROC curve (area = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic ')
plt.legend(loc="lower right")

#%% Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train,y_train)
y_pred2 = clf_dt.predict(X_test)
clf_dt_accuracy_score = accuracy_score(y_test,y_pred2)
clf_dt_precision_score = precision_score(y_test,y_pred2)
clf_dt_recall_score = recall_score(y_test,y_pred2)
clf_dt_f1_score      = f1_score(y_test,y_pred2)

print('Accuracy :',clf_dt_accuracy_score)
print('Precision :',clf_dt_precision_score )
print('Recall :',  clf_dt_recall_score )
print('F1_score :',clf_dt_f1_score )

#%% Confusion Matrix
from sklearn.metrics import confusion_matrix
clf_rf_cm = confusion_matrix(y_test,y_pred2)
sns.heatmap(clf_rf_cm,annot=True,fmt="d",cmap= sns.cubehelix_palette(start=-3,rot=.4)) 
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.title("DT Classification - Confusion Matrix")
plt.show()

#%%
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
#calculate AUC
auc = roc_auc_score(y_test, y_pred2)
print('AUC: %.3f' % auc)

plt.plot(fpr, tpr, label='Decision Tree-ROC curve (area = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

#%%   Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)
y_pred3 = clf_nb.predict(X_test)
clf_nb_accuracy_score = accuracy_score(y_test,y_pred3)
clf_nb_precision_score = precision_score(y_test,y_pred3)
clf_nb_recall_score = recall_score(y_test,y_pred3)
clf_nb_f1_score      = f1_score(y_test,y_pred3)

print('Accuracy :',clf_nb_accuracy_score)
print('Precision :',clf_nb_precision_score )
print('Recall :',  clf_nb_recall_score )
print('F1_score :',clf_nb_f1_score )


#%% Confusion Matrix
from sklearn.metrics import confusion_matrix
clf_nb_cm = confusion_matrix(y_test,y_pred3)
sns.heatmap(clf_nb_cm,annot=True,fmt="d",cmap= sns.cubehelix_palette(start=0,rot=-.6)) 
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.title("NB Classification - Confusion Matrix")
plt.show()
#%%
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
#calculate AUC
auc = roc_auc_score(y_test, y_pred3)
print('AUC: %.3f' % auc)

plt.plot(fpr, tpr, label='Naive Bayes-ROC curve (area = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic ')
plt.legend(loc="lower right")

#%% Support Vector Machine  Classifier
from sklearn import svm
clf_svc = svm.SVC(kernel='linear')
clf_svc.fit(X_train,y_train)
y_pred4 = clf_svc.predict(X_test)
clf_svc_accuracy_score = accuracy_score(y_test,y_pred4)
clf_svc_precision_score = precision_score(y_test,y_pred4)
clf_svc_recall_score = recall_score(y_test,y_pred4)
clf_svc_f1_score      = f1_score(y_test,y_pred4)

print('Accuracy :',clf_svc_accuracy_score)
print('Precision :',clf_svc_precision_score)
print('Recall :',  clf_svc_recall_score )
print('F1_score :',clf_svc_f1_score )

#%% Confusion Matrix
from sklearn.metrics import confusion_matrix
clf_nb_cm = confusion_matrix(y_test,y_pred4)
sns.heatmap(clf_nb_cm,annot=True,fmt="d",cmap= sns.cubehelix_palette(start=-1.9,rot=0)) 
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.title("SVM Classification - Confusion Matrix")
plt.show()
#%%
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred4)
#calculate AUC
auc = roc_auc_score(y_test, y_pred4)
print('AUC: %.3f' % auc)

plt.plot(fpr, tpr, label='SVM Classifier-ROC curve (area = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic ')
plt.legend(loc="lower right")

#%%  K-NN

from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(n_neighbors=9)
clf_knn.fit(X_train,y_train)
y_pred5 = clf_knn.predict(X_test)
clf_knn_accuracy_score = accuracy_score(y_test,y_pred5)
clf_knn_precision_score = precision_score(y_test,y_pred5)
clf_knn_recall_score = recall_score(y_test,y_pred5)
clf_knn_f1_score      = f1_score(y_test,y_pred5)

print('Accuracy :',clf_knn_accuracy_score)
print('Precision :',clf_knn_precision_score)
print('Recall :',  clf_knn_recall_score )
print('F1_score :',clf_knn_f1_score )

#%% Confusion Matrix
from sklearn.metrics import confusion_matrix
clf_knn_cm = confusion_matrix(y_test,y_pred5)
sns.heatmap(clf_nb_cm,annot=True,fmt="d") 
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.title("KNN Classification - Confusion Matrix")
plt.show()

#%%
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred5)
#calculate AUC
auc = roc_auc_score(y_test, y_pred5)
print('AUC: %.3f' % auc)

plt.plot(fpr, tpr, label='K-NN-ROC curve (area = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic ')
plt.legend(loc="lower right")

#%% K-nn score
#import matplotlib.pyplot as plt
#score_list = []
#for each in range(1,15):
#    clf_knn2 = KNeighborsClassifier(n_neighbors=each)
#    clf_knn2.fit(X_train,y_train)
#    y_pred = clf_knn2.predict(X_test)
#    score_list.append(accuracy_score(y_test,y_pred5))
#
#plt.plot(range(1,15),score_list)
#plt.xlabel("k_values")
#plt.ylabel("Accuracy")
#plt.show()


#%% Visualization

#  Classification Accuracy
x = [clf_logreg_accuracy_score,clf_rf_accuracy_score,clf_dt_accuracy_score,clf_nb_accuracy_score,clf_svc_accuracy_score,clf_knn_accuracy_score]
y = ['Logistic Regression','Random Forest Classifier','Decision Tree Classifier','Naive Bayes Classifier','Support Vector Machine','K-NN Classifier']
x,y = list(x),list(y)
plt.figure(figsize=(15,10)) 
ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Accuracy Score')
plt.title('Comparison of accuracy score all classification ')

#%%  Classification Precision
x1 = [clf_logreg_precision_score,clf_rf_precision_score,clf_dt_precision_score,clf_nb_precision_score,clf_svc_precision_score,clf_knn_precision_score]
y = ['Logistic Regression','Random Forest Classifier','Decision Tree Classifier','Naive Bayes Classifier','Support Vector Machine','K-NN Classifier']
x,y = list(x),list(y)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=x1, y=y,palette = sns.cubehelix_palette(start=2.8, rot=.1))
plt.xlabel('Presicion Score')
plt.title('Comparison of precision score all classification ')
#%% Classification Recall
x2 = [clf_logreg_recall_score,clf_rf_recall_score,clf_dt_recall_score,clf_nb_recall_score,clf_svc_recall_score,clf_knn_recall_score]
y = ['Logistic Regression','Random Forest Classifier','Decision Tree Classifier','Naive Bayes Classifier','Support Vector Machine','K-NN Classifier']
x,y = list(x),list(y)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=x2, y=y,palette = sns.cubehelix_palette(rot=-.4))
plt.xlabel('Recall Score')
plt.title('Comparison of recall score all classification ')
#%%  Classification F1_score
x3 = [clf_logreg_f1_score,clf_rf_f1_score,clf_dt_f1_score,clf_nb_f1_score,clf_svc_f1_score,clf_knn_f1_score]
y = ['Logistic Regression','Random Forest Classifier','Decision Tree Classifier','Naive Bayes Classifier','Support Vector Machine','K-NN Classifier']
x,y = list(x),list(y)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=x3, y=y,palette = sns.cubehelix_palette(start=-5.8,rot=-.4))
plt.xlabel('F1 Score')
plt.title('Comparison of f1_score score all classification ')

























#