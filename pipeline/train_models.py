import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.neural_network  import MLPClassifier
from lightgbm                import LGBMClassifier
from xgboost                 import XGBClassifier
from utils                   import model_evaluator, get_name_model


#====================================================================================================================

summary_models = pd.DataFrame(columns=['model', 'accuracy', 'f1', 'roc_auc' , 'estimator']).set_index(keys=['model'])

#====================================================================================================================

# Hyperparameter dictionaries

params_lr = {'penalty': ['l1', 'l2'],
             'C': [0.01, 0.1, 1.0],
             'solver': ['liblinear', 'saga'],
             'max_iter': [500, 1000]}

params_dt = {'criterion': ['gini', 'entropy', 'log_loss'],
             'max_depth': [None, 5, 10],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4]}
params_rf = {'n_estimators': [100, 200, 500],
             'max_depth': [5, 10, 20],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4],
             'max_features': ['sqrt', 'log2']}

params_xgb = {'n_estimators': [100, 200, 500],
              'max_depth': [5, 10, 20],
              'learning_rate': [0.01, 0.1, 1.0],
              'subsample': [0.5, 0.8, 1.0],
              'colsample_bytree': [0.5, 0.8, 1.0]}

params_lgbm = {'n_estimators': [100, 200, 500],
               'max_depth': [5, 10, 20],
               'learning_rate': [0.01, 0.1, 1.0],
               'subsample': [0.5, 0.8, 1.0],
               'colsample_bytree': [0.5, 0.8, 1.0]}

params_cb = {'iterations': [100, 200, 500],
             'depth': [5, 10, 16],
             'learning_rate': [0.01, 0.2],
             'subsample': [0.5, 0.8, 1.0],
             'colsample_bylevel': [0.5, 0.8]}


params_knb = {'n_neighbors': [3, 5, 7, 9, 11],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}


params_mlp = {
    'hidden_layer_sizes': [(50,100,50),(100, 200, 100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.04],
    'learning_rate': ['constant','adaptive'],
    'max_iter': [1000]
}

#====================================================================================================================
def select_best_model(X_train, y_train, X_test, y_test):

    print('Starting models training')
    score = 0
    
    # Model 1: LogisticRegresion   

    # Training 
    params_lr = {'C': 1.0, 'max_iter': 500, 'penalty': 'l1', 'solver': 'saga'}
    model_lr = LogisticRegression(random_state= 12345, penalty=params_lr['penalty'], C=params_lr['C'], solver=params_lr['solver'])
    model_lr.fit(X_train, y_train)
    result_1 = model_evaluator(model_lr, X_train, y_train, X_test, y_test)
    print('  ...')

    # Save Result

    summary_models.loc[get_name_model(model_lr)] = {'accuracy': result_1.loc['test'].accuracy, 'f1': result_1.loc['test'].f1, 
                                'roc_auc': result_1.loc['test'].roc_auc, 'estimator': model_lr}
    
    roc_auc = result_1.loc['test'].roc_auc
    
    if roc_auc > score:
        model = model_lr
        score = roc_auc

    #====================================================================================================================

    # Model 2: DesicionTreeClassifier

    # Training 
    params_dt = {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10}
    model_dt = DecisionTreeClassifier(random_state=12345, criterion=params_dt['criterion'], max_depth=params_dt['max_depth'], 
                                        min_samples_leaf=params_dt['min_samples_leaf'], min_samples_split=params_dt['min_samples_split'])
    model_dt.fit(X_train, y_train)

    result_2 = model_evaluator(model_dt, X_train, y_train, X_test, y_test)
    print('  ...')

    # Save Result

    summary_models.loc[get_name_model(model_dt)] = {'accuracy': result_2.loc['test'].accuracy, 'f1': result_2.loc['test'].f1, 'roc_auc': result_2.loc['test'].roc_auc, 'estimator': model_dt}

    roc_auc = result_2.loc['test'].roc_auc

    if roc_auc > score:
        model = model_dt
        score = roc_auc

    #====================================================================================================================

    # Model 3: RandomForestClassifier

    # Training 
    params_rf = {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
    model_rf = RandomForestClassifier(random_state=12345, max_depth=params_rf['max_depth'], max_features=params_rf['max_features'], 
                                    min_samples_leaf=params_rf['min_samples_leaf'], min_samples_split=params_rf['min_samples_split'],
                                    n_estimators=params_rf['n_estimators'])
    model_rf.fit(X_train, y_train)
    result_3 = model_evaluator(model_rf, X_train, y_train, X_test, y_test )
    print('  ...')

    # Save Result

    summary_models.loc[get_name_model(model_rf)] = {'accuracy': result_3.loc['test'].accuracy, 'f1': result_3.loc['test'].f1, 'roc_auc': result_3.loc['test'].roc_auc, 'estimator': model_rf}

    roc_auc = result_3.loc['test'].roc_auc

    if roc_auc > score:
        model = model_rf
        score = roc_auc


    #====================================================================================================================

    # Model 4: XGBClassifier

    # Training 
    params_xgb = {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 20, 'n_estimators': 200, 'subsample': 1.0}
    model_xgb = XGBClassifier(random_state=12345, colsample_bytree= params_xgb['colsample_bytree'], 
                            learning_rate=params_xgb['learning_rate'], max_depth=params_xgb['max_depth'], 
                            n_estimators=params_xgb['n_estimators'], subsample=params_xgb['subsample'])

    model_xgb.fit(X_train, y_train)
    result_4 = model_evaluator(model_xgb, X_train, y_train, X_test, y_test)
    print('  ...')


    # Save Result

    summary_models.loc[get_name_model(model_xgb)] = {'accuracy': result_4.loc['test'].accuracy, 'f1':result_4.loc['test'].f1, 'roc_auc': result_4.loc['test'].roc_auc, 'estimator': model_xgb}

    roc_auc = result_4.loc['test'].roc_auc

    if roc_auc > score:
        model = model_xgb
        score = roc_auc


    #====================================================================================================================

    # Model 5: LGBMClassifier

    # Training 
    params_lgbm = {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 20, 'n_estimators': 500, 'subsample': 0.5}
    model_lgbm = LGBMClassifier(random_state=12345, colsample_bytree=params_lgbm['colsample_bytree'], learning_rate=params_lgbm['learning_rate'],
                                max_depth=params_lgbm['max_depth'], n_estimators=params_lgbm['n_estimators'], subsample=params_lgbm['subsample'])
                                
    model_lgbm.fit(X_train, y_train)
    result_5 = model_evaluator(model_lgbm, X_train, y_train, X_test, y_test)
    print('  ...')


    # Save Result
    
    summary_models.loc[get_name_model(model_lgbm)] = {'accuracy': result_5.loc['test'].accuracy, 'f1':result_5.loc['test'].f1, 'roc_auc': result_5.loc['test'].roc_auc, 'estimator': model_lgbm}

    roc_auc = result_5.loc['test'].roc_auc

    if roc_auc > score:
        model = model_lgbm
        score = roc_auc


    #====================================================================================================================

    # Model 6: KNeighborsClassifier

    # Training 
    params_knb = {'algorithm': 'ball_tree', 'n_neighbors': 11, 'weights': 'distance'}
    model_knb = KNeighborsClassifier(algorithm=params_knb['algorithm'], n_neighbors=params_knb['n_neighbors'], weights=params_knb['weights'])
    model_knb.fit(X_train, y_train)

    result_6 = model_evaluator(model_knb, X_train, y_train, X_test, y_test)
    print('  ...')


    # Save Result
    
    summary_models.loc[get_name_model(model_knb)] = {'accuracy': result_6.loc['test'].accuracy, 'f1':result_6.loc['test'].f1, 'roc_auc': result_6.loc['test'].roc_auc, 'estimator': model_knb}

    roc_auc = result_6.loc['test'].roc_auc

    if roc_auc > score:
        model = model_knb
        score = roc_auc


    #====================================================================================================================

    # Model 7: MLPClassifier (ANN)

    # Training 
    params_mlp = {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (100, 200, 100), 'learning_rate': 'constant', 'max_iter': 1500, 'solver': 'adam'}

    model_mlp = MLPClassifier(random_state=12345, activation=params_mlp['activation'], alpha=params_mlp['alpha'], 
                            hidden_layer_sizes=params_mlp['hidden_layer_sizes'], learning_rate=params_mlp['learning_rate'],
                            max_iter=params_mlp['max_iter'], solver=params_mlp['solver'], warm_start=True)

    model_mlp.fit(X_train, y_train)
    result_7 = model_evaluator(model_mlp, X_train, y_train, X_test, y_test)
    print('  ...')

    
    # Save Result

    summary_models.loc[get_name_model(model_mlp)] = {'accuracy': result_7.loc['test'].accuracy, 'f1':result_7.loc['test'].f1, 'roc_auc': result_7.loc['test'].roc_auc, 'estimator': model_mlp}

    roc_auc = result_7.loc['test'].roc_auc

    if roc_auc > score:
        model = model_mlp
        score = roc_auc

    #====================================================================================================================

    print(summary_models)
    print(f'\nBest model: "{get_name_model(model)}" roc_auc={score:.2f} "\n')
    print('... executed training models. \n')

    return model, score, summary_models

#====================================================================================================================
#====================================================================================================================
# REVISION KATLHYN REYES
#====================================================================================================================

# El desarrollo de tu función para abordar el entrenamiento de modelos es muy agradable por los comentarios y ordenado. Felicitaciones.
# En este caso, mi sugerencia es que se creen funciones para encapsular tareas repetitivas, como la del entrenamiento y evaluación de modelos.
# Esto puede contribuir a que el código sea más claro y conciso.
# Mi propuesta de mejora sería la siguiente

# def select_best_model(X_train, y_train, X_test, y_test):
#     print('Inicio del entrenamiento de modelos')
#     score = 0
#     summary_models = {}

#     # Configuraciones del modelo y parameter grids
#     model_configs = [
#         (LogisticRegression, params_lr),
#         (DecisionTreeClassifier, params_dt),
#         (RandomForestClassifier, params_rf),
#         (XGBClassifier, params_xgb),
#         (LGBMClassifier, params_lgbm),
#         (KNeighborsClassifier, params_knb),
#         (MLPClassifier, params_mlp)
#     ]

#     # Bucle de entrenamiento
#     for estimator, param_grid in model_configs:
#         model, roc_auc = train_model(estimator, X_train, y_train, X_test, y_test, summary_models)
#         if roc_auc > score:
#             best_model = model
#             score = roc_auc

#     print(summary_models)
#     print(f'\nBest model: "{get_name_model(best_model)}" roc_auc={score:.2f}\n')
#     print('... executed training models.\n')

#     return best_model, score, summary_models
