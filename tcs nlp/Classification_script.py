# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:17:52 2020

@author: varun
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def dataset(string):
    data=pd.read_csv(string)
    return data

#getting names of columns
def get_column_name(data):
    li=[]
    for columns in data.columns:
        li.append(columns)
    return li

def get_features(data,target):
    columnname=get_column_name(data)
    features=[]
    for columns in columnname:
        if target in columns:
            pass
            
        else:
            features.append(columns)
    return features

#getting description of data
def get_description(data):
    return data.describe()

#getting index of outliers using IQR score as key value pair
def quantile_outlier_detection(data):    
    q1=data.quantile(0.25)
    q3=data.quantile(0.75)
    iqr=q3-q1
    dic={}
    for columns in data.columns:
        li=[]
        for i in range (0,len(data[columns])):
            if (data[columns][i]<(q1[columns]-1.5*iqr[columns]))|(data[columns][i]>(q3[columns]+1.5*iqr[columns])):
                li.append(i)
        dic[columns]=li
    return dic

#checking correlation using pearson test
def pearson(data,features,target):
    dic_corr={}
    from scipy.stats import pearsonr
    for feature in features:
        stats,p=pearsonr(data[feature],data[target])
        dic_corr[feature]=('%.3f' %p)
    dic_corr=pd.DataFrame(dic_corr,index=[0])
    return dic_corr
 #feature scaling
def feature_scaling(X,features):
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    scaled_X=sc.fit_transform(X)
    scaled_X=pd.DataFrame(scaled_X,columns=features)
    return scaled_X

# feature normalizing 
def normalizing(data):
    from sklearn.preprocessing import Normalizer
    nm=Normalizer()
    x_normalized=nm.fit_transform(data)
    return x_normalized

#train test split function
def train_test_split(X,Y):
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
    return X_train,X_test,Y_train,Y_test

#detecting outliers using boxplot
def boxplot(data):
    for columns in data.columns:
        sns.boxplot(data[columns])
        plt.title(columns)
        plt.show()
#making distplots
def distplot(data):
    for columns in data.columns:
        sns.distplot(data[columns])
        plt.title(columns)
        plt.show()

#making correlation heatmap
def heatmap(data):
    fig,ax=plt.subplots(figsize=(10,10))
    sns.heatmap(data.corr(),fmt='0.2f',annot=True,cmap = "YlGnBu")
    plt.show()


#making ROC-AUC curve
def Roc_Auc_curve(models,X_test,Y_test):   
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    
    for m in models:
        model = m['model'] # select the model
        #model.fit(x_train, y_train) # train the model
        y_pred=model.predict(X_test) # predict the test data
        # Compute False postive rate, and True positive rate
        fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:,1])
        # Calculate Area under the curve to display on the plot
        auc = roc_auc_score(Y_test,model.predict(X_test))
        # Now, plot the computed values
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
# Custom settings for the plot 
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()   # Display


#plot Learning Curve
def plot_curve(model,X,Y,cv,scoring='accuracy'):
    classifier_name=model['name']+' '+'classifier'
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(model['object'], X, Y, n_jobs=1, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), verbose=0,scoring=scoring)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(4,4))
    plt.title(classifier_name)
    
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    
    # box-like grid
    plt.grid()
    
    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    # sizes the window for readability and displays the plot
    # shows error from 0 to 1.1
    plt.ylim(-.1,1.1)
    plt.legend(loc="best")
    #plt.savefig('learningcurve.png')
    plt.show()
#select features via K best f_regression
def k_best_f_classif(X,Y,K):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    skb=SelectKBest(k=K)
    selected=skb.fit(X,Y)
    featuresselected=selected.transform(X)
    selected_column_index=list(skb.get_support())
    column_name_X_modeled=get_column_name(X)
    selected_column_name=[]
    for i in range(0,len(selected_column_index)):
        if selected_column_index[i]==True:
            selected_column_name.append(column_name_X_modeled[i])
        
    X_modeled_2=X[selected_column_name]
    return X_modeled_2

#checking correlation using pearson test
def pearson(data,features,target):
    dic_corr={}
    from scipy.stats import pearsonr
    for feature in features:
        stats,p=pearsonr(data[feature],data[target])
        dic_corr[feature]=('%.3f' %p)
    dic_corr=pd.DataFrame(dic_corr,index=[0])
    return dic_corr

#checking correlation using pearson test
def Shapiro(data,features):
    dic_corr={}
    from scipy.stats import shapiro
    for feature in features:
        stats,p=shapiro(data[feature])
        dic_corr[feature]=('%.3f' %p)
    dic_shapiro=pd.DataFrame(dic_corr,index=[0])
    return dic_shapiro


