import pandas as pd
import numpy as np
from sklearn.preprocessing   import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling  import RandomOverSampler
from utils                   import to_snake_case

#====================================================================================================================

# Rename columns
def rename_columns(data_contract, data_internet, data_personal, data_phone):

    data_contract = data_contract.rename(columns = lambda x: to_snake_case(x))
    data_internet = data_internet.rename(columns = lambda x: to_snake_case(x))
    data_personal = data_personal.rename(columns = lambda x: to_snake_case(x))
    data_phone    =    data_phone.rename(columns = lambda x: to_snake_case(x))

    return (data_contract, data_internet, data_personal, data_phone)

#====================================================================================================================

# Change data type
def change_data_type(data_contract, data_internet, data_personal, data_phone):

    data_contract.begin_date = pd.to_datetime(data_contract.begin_date, format="%Y-%m-%d")
    data_contract['paperless_billing'] = data_contract['paperless_billing'].apply({'Yes': 1, 'No': 0}.get)
    data_contract[['type','payment_method']] = data_contract[['type', 'payment_method']].astype('category')
    data_contract.total_charges = data_contract.total_charges.replace(' ', np.nan).astype('float')

    data_internet.internet_service = data_internet.internet_service.astype('category')
    data_internet[['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']] = data_internet[[
                'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']].applymap({'Yes': 1, 'No': 0}.get)

    data_personal.gender = data_personal.gender.astype('category')
    data_personal[['partner', 'dependents']] = data_personal[['partner', 'dependents']].applymap({'Yes': 1, 'No': 0}.get)

    data_phone.multiple_lines = data_phone.multiple_lines.astype('category')

    return(data_contract, data_internet, data_personal, data_phone)


#====================================================================================================================

# Create "customers" merge 4 dataframes
def generate_customer_df(data_contract, data_internet, data_personal, data_phone):
    customers = pd.merge(data_contract, data_personal, on='customer_id', how='outer')
    customers = pd.merge(    customers, data_internet, on='customer_id', how='outer')
    customers = pd.merge(    customers,    data_phone, on='customer_id', how='outer')

    return customers

#====================================================================================================================


def preprocessing(data_contract, data_internet, data_personal, data_phone):

    # Rename columns
    data_contract, data_internet, data_personal, data_phone = rename_columns(data_contract, data_internet, data_personal, data_phone)

    # Change datatype
    data_contract, data_internet, data_personal, data_phone = change_data_type(data_contract, data_internet, data_personal, data_phone)
    
    # Fill NaN
    data_contract['total_charges'] = data_contract['total_charges'].fillna(data_contract['monthly_charges'])
    
    # Merge dataframes in a dataframe named "customers"
    customers = generate_customer_df(data_contract, data_internet, data_personal, data_phone)

    # Fill NaN in "customers"
    services = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
    customers[services] = customers[services].fillna(0).astype(int)
    customers['internet_service'] = customers['internet_service'].cat.add_categories(['Neither']).fillna('Neither')
    customers['multiple_lines'] = customers['multiple_lines'].cat.add_categories(['No line']).fillna('No line')


    # Add new columns in "customers"
    customers['phone_line'] = customers['multiple_lines'].apply(lambda x: 0 if x == 'No line' else 1)
    customers['internet'] = customers['internet_service'].apply(lambda x: 0 if x == 'Neither' else 1)
    customers['churn_customer'] = customers['end_date'].apply(lambda x: 0 if x == 'No' else 1)
    customers['tenure_month'] = customers.apply(lambda x: ((customers['begin_date'].max() - x['begin_date']).days / 30.44) if x['end_date'] == "No" 
                                                else ((pd.to_datetime(x['end_date']) - x['begin_date']).days / 30.44), axis=1).astype(int)

    print('... executed prepocessing data.\n')

    return customers


#====================================================================================================================
# Data processing
#====================================================================================================================

def X_y (customers):
    # identify features and target
    X = customers.drop(['begin_date','customer_id','end_date','churn_customer'], axis=1)
    y = customers['churn_customer']

    return X, y

#====================================================================================================================


def X_encoder (X):

    # Apply OneHotEncoder for categorical features

    col_cat = X.select_dtypes(include=['category']).columns.tolist()
    
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    col_ohe = ohe.fit_transform(X[col_cat])

    # Transform to dataframe the OHE features
    X_ohe = pd.DataFrame(data=col_ohe, columns=ohe.get_feature_names_out()).astype(int)

    # Remove categorical columns "cat_col" in features
    X.drop(labels=col_cat, axis=1, inplace=True)

    # Add OHE features transformed into "X" features
    X = X.merge(right=X_ohe, left_index=True, right_index=True)
    
    return X

#====================================================================================================================

def remove_features (X):
    # Select features with a correlation less than 0.9

    corr_matrix = round(X.corr(),2)

    selected_features = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                colname = corr_matrix.columns[i]
                selected_features.append(colname)

    X = X.drop(selected_features, axis=1)

    return X

#====================================================================================================================

def split_data (X, y):
    # Split: train 70% / test 15% / valid 15%

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=12345)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.50, random_state=12345)

    return X_train, y_train, X_test, y_test, X_valid, y_valid

#====================================================================================================================

def scaler_data (X_train, X_test, X_valid):

    # Numerical features 
    col_numeric = ['monthly_charges', 'total_charges', 'tenure_month']

    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(X_train[col_numeric])

    # transform
    X_train[col_numeric] = scaler.transform(X_train[col_numeric])
    X_test[col_numeric]  = scaler.transform(X_test[col_numeric])
    X_valid[col_numeric] = scaler.transform(X_valid[col_numeric])

    return X_train, X_test, X_valid

#====================================================================================================================

def balanced_classes (X_train, y_train):
    # Under-sampling
    rus = RandomUnderSampler(random_state=12345, sampling_strategy=0.50)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    # Over-sampling
    ros = RandomOverSampler(random_state=12345)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    return X_train, y_train

#====================================================================================================================


def processing (customers):
    
    X, y = X_y(customers)
    X = X_encoder(X)
    X = remove_features (X)
    X_train, y_train, X_test, y_test, X_valid, y_valid = split_data (X, y)
    X_train, X_test, X_valid = scaler_data (X_train, X_test, X_valid)
    X_train, y_train = balanced_classes (X_train, y_train)

    print('... executed processing data.\n')
    return X_train, y_train, X_test, y_test, X_valid, y_valid 


#====================================================================================================================

# cambios generados para revisión