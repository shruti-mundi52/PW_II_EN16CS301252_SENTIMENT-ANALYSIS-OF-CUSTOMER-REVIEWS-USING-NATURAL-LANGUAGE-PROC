# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:51:05 2020

@author: varun
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import hotel_review_script
# Getting Dataset
data = pd.read_csv("hotel_review_final.csv")

#getting info
data.info()

#Description of dataset
description = data.describe()

#checking for null values
data.isnull().sum()

#getting unique value count of the columns
data.nunique()

# append the positive and negative text reviews
data_review = data.copy()

#class object
hr = hotel_review()


X_train, X_test, y_train, y_test = hr.splitting(data_review)

print('Naive Bayes')
# train a naive bayes classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
nb = GaussianNB() 
nb.fit(X_train,y_train)
y_pred_nb = nb.predict(X_test)
print(accuracy_score(y_test,y_pred_nb))
print(classification_report(y_test,y_pred_nb))
# show feature importance
#feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)
#feature_importances_df.head(20)

print('Decision Tree')
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred_dtc=dtc.predict(X_test)
print(accuracy_score(y_test,y_pred_dtc))
print(classification_report(y_test,y_pred_dtc))
print(confusion_matrix(y_test,y_pred_dtc))


print('Logistic Regression')
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(penalty='l2',C=10, max_iter=10000)
lr.fit(X_train,y_train)
y_pred_lr=lr.predict(X_test)
print(accuracy_score(y_test,y_pred_lr))
print(classification_report(y_test,y_pred_lr))
print(confusion_matrix(y_test,y_pred_lr))


print('Gradient Boosting')
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
y_pred_gbc=gbc.predict(X_test)
print(accuracy_score(y_test,y_pred_lr))
print(classification_report(y_test,y_pred_lr))


print('Random Forest')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(accuracy_score(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))

print('SVM')
from sklearn.svm import SVC
svm = SVC(C=100, kernel='rbf', gamma='auto')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
# Use accuracy_score function to get the accuracy
print(accuracy_score(y_test,y_pred_svm))
print(classification_report(y_test,y_pred_svm))


#model={"name":"Logistic Regression","object":lr}
#plot_curve(model,data_selected[features],data_selected[label],cv=10)#user defined function
#
#models=[
#        {
#             "model":lr,
#             "label":"Logistic Regression"
#                
#                
#        },
#        {
#              "model": dtc,
#              "label":"Decision Tree Classifier"
#                        
#        },
#        
#        {
#              "model": gbc,
#              "label":"Gradient Boosting Classifier"
#                        
#        },
#        ]
#Roc_Auc_curve(models,X_test,y_test)#user defined function


# checking on input
li= ["The hotel was bad.everything was messy"]
X=hr.input_pipeline(li)
y_inp_=lr.predict(X)
if y_inp_==0:
    print("seems you are satisfied thank you")
else:
    print("Seems you are not satisfied. We will surely report this to the Hotel. We are extremely sorry")

