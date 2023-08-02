import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
palette = ['#1FE315', '#F56600', '#618DFA', '#A111A8', '#F5A700', '#AEFFE7', '#42C0A7', '#FFA195']


#====================================================================================================================



# Funtion "to_snake_case" 
def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


#====================================================================================================================

# Function "plots" to format graphs within a grid line 

def plots(data, columns, kind=None, titles=None, figsize=None):

    '''
    Input parameters:
    "data": dataset
    "columns": columns to use from the dataset
    "kind": type of graphics to use
    "titles": titles to use

    Return:
     Show the graphs in the same line
    '''
    
    # INITIAL SETUP *****************************************************************************************************
    
    # Convert each item in the list to a list
    columns = [x if isinstance(x, list) else [x] for x in columns]

    # Convert in the list to data if it's not
    data_list = data if isinstance(data, list) else [data] # ADD LINE-------
    
    # Graph style 
    sns.set_theme(style="whitegrid")

    # Palette 
    palette = ['#1FE315', '#F56600', '#618DFA', '#A111A8', '#F5A700', '#AEFFE7', '#42C0A7', '#FFA195']
   
    # Number of graphs 
    n_plots = len(columns)

    # Figure size 
    if not figsize:
        figsize = (n_plots * 6, 5)

    # Figure with set of subplots
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)


    # BUCLE ***************************************************************************************************************
    
    # Create the graphs on each axis
    for i, col in enumerate(columns):
        
        data = data_list[i]

        # Count of values in col
        n_col = len(col)

        # Define ax
        if n_plots > 1: # mayor a 1
            ax = axes[i]
        elif n_plots == 1:
            ax = axes

        # SELECT COLOR(S) -------------------------------------------------------------------------------------------------

        if n_col > 1 or kind[i] == 'count' or kind[i] == 'bar':
            #  Count of unique values in categorical column
            n_cat = data[col[-1]].nunique()

            # Create a new color list with only the number of values needed
            colors = palette[i:i+n_cat]
        
        elif n_col == 1:
            color = palette[i]
        
        
        # GRAPH TYPE ------------------------------------------------------------------------------------------------------ 

        if kind:

            #  H I S T O G R A M -------------------------------------------------------------------------------------------
            # x=numerical variable, hue=categorical variable (optional)
            
            if kind[i] == 'hist':
                if n_col == 2:
                    sns.histplot(data, x=col[0], hue=col[1], kde=True, ax=ax, palette=colors, edgecolor="black", alpha=0.5)
                else:
                    sns.histplot(data, x=col[0], kde=True, ax=ax, color=color, edgecolor="black")


            # C O U N T  (BAR GRAPHIC) --------------------------------------------------------------------------
            # x=categorical variable, hue=categorical variable (optional)
            
            elif kind[i] == 'count':
                if n_col == 2:
                    sns.countplot(data=data, x=col[0], ax=ax, palette=colors, hue=col[1])
                else:
                    sns.countplot(data=data, x=col[0], ax=ax, palette=colors)

                # XLabels rotation
                ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
                                               
                # Add labels of percent to bars
                for p in ax.patches:
                    ax.text(p.get_x() + p.get_width()/2, p.get_height() + 3, f'{(p.get_height()/len(data)):.0%}', ha="center")

                # Legends:
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(fontsize=10, loc="best", title=col[-1])


            # B A R  GRAPHIC ------------------------------------------------------------------------------------------
            # x: numerical variable, y: categorical variable, hue= categorical variable (optional)

            elif kind[i] == 'bar': 
                if n_col == 3:
                    sns.barplot(data=customers, x=col[1], y=col[0], hue=col[-1], estimator='sum', ax=ax, palette=colors)

                elif  n_col == 2:
                    sns.barplot(data=customers, x=col[1], y=col[0], estimator='sum', ax=ax, palette=colors)
                
                else:
                    print('Verify information to create the bar chart ')
                    print('x: numerical variable, y: categorical variable, hue= categorica variable (optional)' )
            
            
            # B O X P L O T  -------------------------------------------------------------------------------------------
            # x=numerical variable, y=categorical variable (optional)

            elif kind[i] == 'boxplot':

                # Boxplot with 2 variables: numerical and categorical
                if n_col == 2:
                    sns.boxplot(data, x=col[0], y=col[1], orient='h', ax=ax, palette=colors)
                    ax.set_ylabel(col[1])

                # Boxplot simple
                else:
                    sns.boxplot(data, x=col[0], ax=ax, color=color)


            # S C A T T E R --------------------------------------------------------------------------------------------- 
            # x= numerical variable, y= numerical variable, hue=categorical variable (optional)
            
            elif kind[i] == 'scatter':

                if n_col == 3:  # add "hue" 
                    sns.scatterplot(data, x=col[0], y=col[1], hue=col[2], ax=ax, palette=colors)
            
                elif n_col == 2:
                    sns.scatterplot(data, x=col[0], y=col[1], sizes=(20, 200), ax=ax)


        # -----------------------------------------------------------------------------------------------------------------
        
        # Size labels   
        ax.tick_params(axis="both", labelsize=10)
        # ax.tick_params(axis="y", labelsize=10)

        # Titles:
        ax.set_title(titles[i], fontsize=14)

    # END BUCLE ************************************************************************************************************


    # Position and axes size
    fig.tight_layout()

    # Distance between the axes
    fig.subplots_adjust(wspace=0.40)

    # Show graph
    plt.show()



#====================================================================================================================


# function "find_outliers"

def outliers(df, col):

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    min = (Q1 - 1.5 * IQR).astype('int')
    max = (Q3 + 1.5 * IQR).astype('int')

    indexes = df[(df[col] <= min) | (df[col] >= max)].index

    return indexes

#====================================================================================================================



# Function grid_search for search best parameters

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
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5)) 

    
    for group, X, y in (('train', X_train, y_train), ('test', X_test, y_test)):
        
        # Predictions
        y_prediction = model.predict(X)
        y_prediction_proba = model.predict_proba(X)[:, 1] # Probabilities of the positive class

        # METRICS Accuracy and ROC AUC
        accuracy = round(accuracy_score(y, y_prediction), 2)
        roc_auc = round(roc_auc_score(y, y_prediction_proba),2)  
        f1      = round(f1_score(y, y_prediction), 2)

        results_metrics.loc[group] = {'accuracy':accuracy, 'f1':f1, 'roc_auc':roc_auc}
        
        # ----------------------------------------------------------------------------------------------------------------------

        # PLOTS
        color = 'blue' if group == 'train' else 'orange'

        # axs[0] = Accuracy
        ax= axs[0]

        accuracy_thresholds = np.arange(0, 1.01, 0.05) # [0.0, 0.05, 0.10, 0.15, 0.20, ... , 1.0]
        accuracy_scores = [accuracy_score(y, y_prediction_proba>=threshold) for threshold in accuracy_thresholds]

        max_accuracy_score_idx = np.argmax(accuracy_scores)
        ax.plot(accuracy_thresholds, accuracy_scores, color=color, 
                label=f'{group}, max={accuracy_scores[max_accuracy_score_idx]:.2f} @ {accuracy_thresholds[max_accuracy_score_idx]:.2f}')
        
        # Defining crosses for some thresholds    
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(accuracy_thresholds-threshold))
            marker_color = 'green' if threshold != 0.5 else 'red'
            ax.plot(accuracy_thresholds[closest_value_idx], accuracy_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('Accuracy')
        ax.legend(loc='lower center')
        ax.set_title(f'Accuracy') 

        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [f1_score(y, y_prediction_proba >= threshold) for threshold in f1_thresholds]

        # axs[1] = F1
        ax = axs[1]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{group}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        
        # Defining crosses for some thresholds         
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds - threshold))
            marker_color = 'green' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'F1') 



        # axs[2] = ROC-CURVE
        ax = axs[2]   
        fpr, tpr, roc_thresholds = roc_curve(y, y_prediction_proba)

        ax.plot(fpr, tpr, color=color, label=f'{group}, ROC AUC={roc_auc:.2f}')
        
        # Defining crosses for some thresholds  
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'green' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        
        ax.plot([0, 1], [0, 1], color='black', linestyle='--')
        
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.legend(loc='lower center')        
        ax.set_title(f'ROC Curve')

    display(results_metrics)
   
    return results_metrics


#====================================================================================================================

