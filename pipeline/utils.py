import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

#====================================================================================================================

# Funtion "to_snake_case" 
def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


#====================================================================================================================


# Function "grid_search" for search best parameters
def grid_search_cv(estimator, params, X, y):
    '''
    parameters:
    "estimator": algorithm to train
    "params": hyperparameters
    "x": features train
    "y": target train
    return: the best parameters
    '''
  
    grid_search= GridSearchCV(estimator=estimator, param_grid=params, scoring='roc_auc', cv=5, error_score='raise')
    grid_search.fit(X, y)
    best_params  = grid_search.best_params_

    print(f'Best hiperparameters: {best_params}')
    return best_params


#====================================================================================================================


# Function model_evaluator for to evaluate each model and to Graph results

def model_evaluator(model, X_train, y_train, X_test, y_test):

    '''
    parameters:    
      model: model previously trained and with hyperparameter tuning
      X_train, y_train: features and target dataset train
      X_test, y_test: features and target dataset test
    calculate metrics accuracy and roc_auc
    show graphs
    return the values for accuracy and ROC-AUC
    '''
    
    results_metrics = pd.DataFrame(columns=['group', 'accuracy', 'f1', 'roc_auc']).set_index(keys=['group'])
    
    for group, X, y in (('train', X_train, y_train), ('test', X_test, y_test)):
        
        # Predictions
        y_prediction = model.predict(X)
        y_prediction_proba = model.predict_proba(X)[:, 1] # Probabilities of the positive class

        # METRICS Accuracy and ROC AUC
        accuracy = round(accuracy_score(y, y_prediction), 3)
        roc_auc = round(roc_auc_score(y, y_prediction_proba),3)  
        f1      = round(f1_score(y, y_prediction), 3)

        results_metrics.loc[group] = {'accuracy':accuracy, 'f1':f1, 'roc_auc':roc_auc}
            
    return results_metrics


#====================================================================================================================

def get_name_model (model):
    name = str(model).split('(')[0] # model name
    return name

#====================================================================================================================