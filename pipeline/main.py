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

# cambios generados para revisión