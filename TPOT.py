import gc
import numpy as np
import pandas as pd
import datetime, time
from sklearn.preprocessing import LabelEncoder
from utils import eval_socres

import warnings
warnings.filterwarnings('ignore')

from tpot import TPOTClassifier, TPOTRegressor


file_res = open(f'/home/zhangxj/program/AUTOML/res_log/tpot_results.csv', 'a+')
data_dir = '/home/zhangxj/program/AUTOML/data'

dataset = {
    # 'Titanic': ['binary', 'Survived'],
    # 'Bank': ['binary', 'deposit'],
    # 'Blood': ['binary', 'Class'],
    # 'Diabetes': ['binary', 'Class'],
    # 'Electrical_grid': ['binary', 'Class'],
    # 'Hepatitis': ['binary', 'Class'],
    # 'Adult': ['binary', 'Class'], 
    # 'West_Nile_Virus': ['binary', 'WnvPresent'],
    # 'Santander_Customer': ['binary', 'TARGET'],
    # 'Income': ['binary', 'income >50K'],

    # 'Splice': ['multiclass', 'Class'],
    # 'Thyroid': ['multiclass', 'Class'],
    # 'Yeast': ['multiclass', 'Class'],
    # 'Marketing': ['multiclass', 'Income'],
    # 'Movement_Libras': ['multiclass', 'Class'], 
    # 'Races': ['multiclass', 'config'],   
    # 'Mobility': ['multiclass', 'transportation_type'], 
     
    # 'House_Prices': ['regression', 'SalePrice'],
    # 'Mercedes_Benz_Greener_Manufacturing': ['regression', 'y'],
    # 'Allstate_Claims_Severity': ['regression', 'loss'],
    # 'Playground_Series_Regression': ['regression', 'target'],
    'avocado': ['regression', 'AveragePrice'],
    'puma32h': ['regression', 'thetadd6'],
    'ailerons': ['regression', 'Goal'],
    'forestFires': ['regression', 'Area'],
    'california': ['regression', 'MedianHouseValue'],
    'pole': ['regression', 'Output'],
}


if __name__ == '__main__':

    for data_name, data_info in zip(dataset.keys(), dataset.values()):
        train_data = pd.read_pickle(f'{data_dir}/{data_name}/train.pkl')
        test_data = pd.read_pickle(f'{data_dir}/{data_name}/test.pkl')
        task_type = data_info[0]
        label_name = data_info[1]

        if task_type == 'binary':
            file_res.write('data, task, auc, f1, accuracy, recall, precision, time\n')
            metric = 'roc_auc'
        elif task_type == 'multiclass':
            file_res.write('data, task, auc, f1, accuracy, recall, precision, time\n') 
            metric = 'roc_auc_ovo' 
        elif task_type == 'regression':
            file_res.write('data, task, rmse, r2, mse, mae, msle, time\n')
            metric = 'neg_mean_squared_error' # 'r2' 

        # Label preprocess for scores
        if task_type != 'regression':
            le = LabelEncoder()
            train_data[label_name] = le.fit_transform(train_data[label_name])
            test_data[label_name] = le.transform(test_data[label_name])
        else:
            # pass
            maxs, mins = train_data[label_name].max(), train_data[label_name].min()
            train_data[label_name] = (train_data[label_name] - mins) / (maxs - mins)
            test_data[label_name] = (test_data[label_name] - mins) / (maxs - mins)
        X_train = train_data.copy()
        y_train = X_train.pop(label_name)
        print('[{}] Data Prepare Finished!'.format(data_name))

        training_params = {
            'generations': 3, #20
            'population_size': 100,  # 100
            'max_time_mins': 10,
            'scoring': metric,
            'cv': 3,
            'random_state': 2021,
            'verbosity': 2
        }

        time_start = time.time()
        if task_type in ['binary', 'multiclass']:
            model = TPOTClassifier(**training_params)
        elif task_type == 'regression':
            model = TPOTRegressor(**training_params)
        model.fit(X_train, y_train)

        scores = eval_socres(test_data.copy(), label_name, model, task_type, method_name='tpot')
        score_values = list(scores.values())
        print(scores)
        time_end = time.time()
        total_time = time_end - time_start
        print('total time: {}'.format(total_time)) 
        file_res.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(data_name, task_type, \
                score_values[0], score_values[1], score_values[2], score_values[3], score_values[4], total_time))

        del model, train_data, test_data
        gc.collect()

    file_res.close()