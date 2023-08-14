from utils import model_evaluator, get_name_model

def predict_model(model, X_train, y_train, X_valid, y_valid):

    scores_valid  = model_evaluator(model, X_train, y_train, X_valid, y_valid)
    roc_auc_valid = scores_valid.loc['test'].roc_auc

    name = get_name_model(model)
    print(f"Evaluation best model, roc_auc={roc_auc_valid:.2f}\n")

    print(f'... executed predictions with best model {name}.\n')

    return roc_auc_valid

#====================================================================================================================
#====================================================================================================================
# REVISION KATLHYN REYES
#====================================================================================================================
# tu funcion cumple con el objetivo de evaluar un modelo entrenado en un conjunto de datos de validación y retornar la mejor puntuación ROC AUC. 

