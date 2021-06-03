import math
import numpy as np
import pandas as pd
import h2o
from tabular_toolbox.metrics import calc_score

binaryclass_metrics = ['auc','f1', 'accuracy', 'recall','precision']
multiclass_metircs = ['auc','f1', 'accuracy', 'recall','precision']
regression_metrics = ['rmse', 'r2', 'mse', 'mae','msle']

            
def eval_socres(data, label, model, task_type, method_name=None):

    if task_type in ['binary', 'multiclass']:
        if method_name != 'h2o':
            X_data = data.copy()
            y_ture = X_data.pop(label)
            y_preds = model.predict(X_data)
            if method_name == 'autogluon':
                y_probs = model.predict_proba(X_data).values
            else:
                if method_name != 'autokeras':
                    y_probs = model.predict_proba(X_data)
                else:
                    y_preds = pd.Series(y_preds)
                    y_probs = None
        else:
            test = h2o.H2OFrame(data.copy())
            preds_df = model.predict(test).as_data_frame(use_pandas=True)
            y_ture = data[label]
            y_preds = preds_df.iloc[:, 0].values
            y_probs = preds_df.iloc[:, 1:].values 
        if task_type == 'binary':
            metrics = binaryclass_metrics
        elif task_type == 'multiclass':
            metrics = multiclass_metircs
        score = calc_score(y_ture, y_preds, y_probs, task=task_type, metrics=metrics)
    else: # regression
        if method_name != 'h2o':
            X_data = data.copy()
            y_ture = X_data.pop(label)
            y_preds = nozeros_process(model.predict(X_data))
        else:
            test = h2o.H2OFrame(data.copy())
            preds_df = model.predict(test).as_data_frame(use_pandas=True)
            y_ture = data[label]
            y_preds = nozeros_process(preds_df.iloc[:, 0].values)
        metrics = regression_metrics
        score = calc_score(y_ture, y_preds, task=task_type, metrics=metrics)

    return score

def nozeros_process(data):
    return np.array(list(map(lambda x: max(x, 0), data))).reshape(-1, 1)