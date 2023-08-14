import pandas as pd
import os
from preprocessing import preprocessing, processing
from train_models  import select_best_model
from predict       import predict_model


#====================================================================================================================

# Generate paths
current_dir   = os.path.dirname(os.path.realpath('../customer_churn_prediction_model_final_project/dataset/contract.csv')) 

contract_path = os.path.join(current_dir, "contract.csv") 
internet_path = os.path.join(current_dir, "internet.csv") 
personal_path = os.path.join(current_dir, "personal.csv") 
phone_path    = os.path.join(current_dir, "phone.csv") 

# Load data
data_contract = pd.read_csv(contract_path)
data_internet = pd.read_csv(internet_path)
data_personal = pd.read_csv(personal_path)
data_phone    = pd.read_csv(phone_path)


# Preprocessing data
customers = preprocessing(data_contract, data_internet, data_personal, data_phone)

# Processing data
X_train, y_train, X_test, y_test, X_valid, y_valid = processing(customers)

# Training models
model, score, summary_models = select_best_model(X_train, y_train, X_test, y_test)

# Predictions
roc_auc_valid = predict_model(model, X_train, y_train, X_valid, y_valid)



#====================================================================================================================
# REVISION KATLHYN REYES
# Hola Angélica, un gusto, los comentarios los realizaré al final de tu código


#====================================================================================================================
# Muy buen inicio al generar las rutas y cargar los datos.
# podemos mejorar el codigo, esto mediante el uso de un diccionario para almacenar las rutas de los archivos y un bucle para la carga de datos. mi porpuesto seria la siguiente:
# # Definir rutas de archivos
# file_paths = {
#     'contract': 'contract.csv',
#     'internet': 'internet.csv',
#     'personal': 'personal.csv',
#     'phone': 'phone.csv'
# }
# # Generar rutas
# current_dir = os.path.dirname(os.path.realpath('../customer_churn_prediction_model_final_project/dataset/contract.csv'))
# # Carga
# data = {}
# for key, file_name in file_paths.items():
#     file_path = os.path.join(current_dir, file_name)
#     data[key] = pd.read_csv(file_path)

#====================================================================================================================
# Felicitaciones Angélica, me gustó mucho el orden que matuviste en tu desarrollo, hay muchas funciones que se encuentran perfectamente optimizadas.Te deje al final de cada .py  unas ideas comentadas a ver si te sirven
# me voy con buenos ejemplos e ideas gracias a tu proyecto :D . Gracias !!