import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import seaborn as sns
from sklearn.metrics import roc_curve, auc,accuracy_score,confusion_matrix,make_scorer
import collections
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold,RandomizedSearchCV
from xgboost import XGBClassifier,plot_importance
from matplotlib import pyplot
from catboost import CatBoostClassifier

def LR(train,test,params):
    X_Train,X_Val,Y_Train,Y_Val = train_test_split(train,train.stroke,random_state=101)
    model = LogisticRegression(class_weight='balanced',random_state=101)
    model.fit(X_Train[params],Y_Train)

    prediction = model.predict_proba(X_Val[params])
    prediction = prediction[:,1]
    print('Validation Accuracy: ')
    fpr,tpr,_ = roc_curve(Y_Val, prediction)
    print(auc(fpr,tpr))

    prediction = model.predict_proba(X_Train[params])
    prediction = prediction[:, 1]
    print('Train Accuracy: ')
    fpr, tpr, _ = roc_curve(Y_Train, prediction)
    print(auc(fpr, tpr))

    model = LogisticRegression(class_weight='balanced',random_state=101)
    model.fit(train[params], train.stroke)
    prediction = model.predict_proba(test[params])
    prediction = prediction[:,1]
    test.loc[:,'stroke'] = prediction
    test[['id','stroke']].to_csv('output_lr.csv',index=False)

def XGB(train,test,params):
    parameters = {
        'min_child_weight': [5, 10],
        'gamma': [0.5, 2, 5],
        'subsample': [0.4, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5]
    }
    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    model = XGBClassifier(n_estimators=100, learning_rate=0.02, objective='binary:logistic',
                          silent=True, nthread=1)

    # define scoring function
    def custom_auc(ground_truth, predictions):
        # I need only one column of predictions["0" and "1"]. You can get an error here
        # while trying to return both columns at once
        fpr, tpr, _ = roc_curve(ground_truth, predictions[:, 1], pos_label=1)
        return auc(fpr, tpr)

        # to be standart sklearn's scorer

    my_auc = make_scorer(custom_auc, greater_is_better=True, needs_proba=True)

    search = RandomizedSearchCV(model, parameters,
                          cv=skf.split(train[params], train.stroke),scoring=my_auc,n_iter=15)
    search.fit(train[params],train.stroke)

    print('Best Score: ' + str(search.best_score_))
    print('Best Params: ')
    print(search.best_params_)

    prediction = search.predict_proba(test[params])
    prediction = prediction[:, 1]
    test.loc[:,'stroke'] = prediction
    test[['id','stroke']].to_csv('output_xgb.csv',index=False)

def CB(train,test,params,idx):

    parameters = {
        'iterations': [20,50,100],
        'learning_rate': [0.1, 0.01, 0.001],
        'depth': [3, 5, 7],
    }
    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    model = CatBoostClassifier(loss_function='Logloss')

    # define scoring function
    def custom_auc(ground_truth, predictions):
        # I need only one column of predictions["0" and "1"]. You can get an error here
        # while trying to return both columns at once
        fpr, tpr, _ = roc_curve(ground_truth, predictions[:, 1], pos_label=1)
        return auc(fpr, tpr)

        # to be standart sklearn's scorer

    my_auc = make_scorer(custom_auc, greater_is_better=True, needs_proba=True)

    search = RandomizedSearchCV(model, parameters,
                          cv=skf.split(train[params], train.stroke),scoring=my_auc,n_iter=15)
    search.fit(train[params],train.stroke,cat_features=idx)

    print('Best Score: ' + str(search.best_score_))
    print('Best Params: ')
    print(search.best_params_)

    prediction = search.predict_proba(test[params])
    prediction = prediction[:, 1]
    test.loc[:,'stroke'] = prediction
    test[['id','stroke']].to_csv('output_cb.csv',index=False)


def preprocess_data(train,test):

    combined = pd.concat([train,test])
    combined.loc[:, 'age2'] = combined.age * combined.age
    combined.loc[:, 'age3'] = combined.age * combined.age * combined.age
    combined.loc[:, 'HTHD'] = combined.heart_disease * combined.hypertension
    combined.loc[:, 'HTpHD'] = combined.heart_disease + combined.hypertension

    combined.loc[:, 'age_group'] = 'Low'
    combined.loc[combined.age < 40,'age_group'] = 'Children'
    combined.loc[combined.age < 40, 'age_group'] = 'Children'
    combined.loc[(combined.age >= 40) & (combined.age < 65), 'age_group'] = 'Adults'
    combined.loc[combined.age >= 65, 'age_group'] = 'Elderly'
    combined.loc[combined.age < 10,'smoking_status'] = 'never smoked'
    smoketab = pd.crosstab([combined.gender, np.round(combined.age, -1)], combined.smoking_status).apply(lambda x: x / x.sum(), axis=1)
    combined.smoking_status = combined.apply(lambda x: np.argmax(smoketab.loc[x['gender']].loc[int(np.round(x['age'],-1))]),axis=1)

    bmiByAge = combined[~combined.bmi.isnull()].pivot_table(index=np.round(combined[~combined.bmi.isnull()].age,-1),values='bmi',aggfunc='median')
    combined.loc[combined.bmi.isnull(),'bmi'] = combined[combined.bmi.isnull()].apply(lambda x: bmiByAge.loc[int(np.round(x['age'],-1))],axis=1)
    combined.loc[:, 'bmi2'] = combined.bmi * combined.bmi

    train,test = combined[:len(train)],combined[len(train):]
    return train,test

def OHE(train,test):
    combined = pd.concat([train,test])
    combined.Residence_type = LabelEncoder().fit_transform(combined.Residence_type)
    combined.smoking_status = LabelEncoder().fit_transform(combined.smoking_status)
    combined.age_group = LabelEncoder().fit_transform(combined.age_group)
    combined.work_type = LabelEncoder().fit_transform(combined.work_type)
    combined.HTpHD = LabelEncoder().fit_transform(combined.HTpHD)
    combined.gender = LabelEncoder().fit_transform(combined.gender)
    combined.ever_married = LabelEncoder().fit_transform(combined.ever_married)
    return combined[:len(train)],combined[len(train):]


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train,test = preprocess_data(train,test)

CB(train,test,['age','heart_disease','hypertension','HTpHD','avg_glucose_level','gender','smoking_status'],[1,2,3,5,6])

train,test = OHE(train,test)
LR(train,test,['smoking_status','age','heart_disease','hypertension','HTpHD','avg_glucose_level','gender'])
XGB(train,test,['smoking_status','age','heart_disease','hypertension','HTpHD','avg_glucose_level','gender'])


cbPredictions = pd.read_csv('output_cb.csv').stroke
lrPredictions = pd.read_csv('output_lr.csv').stroke
xgbPredictions = pd.read_csv('output_xgb.csv').stroke
finalPrediction = (cbPredictions + lrPredictions + xgbPredictions)/3.0
test.loc[:,'stroke'] = finalPrediction
test[['id','stroke']].to_csv('final_output.csv',index=False)

