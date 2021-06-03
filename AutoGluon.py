import gc
import numpy as np
import pandas as pd
import datetime, time
from sklearn.preprocessing import LabelEncoder
from utils import eval_socres

import warnings
warnings.filterwarnings('ignore')

from autogluon.tabular import TabularPredictor

file_res = open(f'/home/zhangxj/program/AUTOML/res_log/autogluon_results.csv', 'a+')
data_dir = '/mnt/nfsroot/pub/datasets'

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
    # 'Digit_Recognizer': ['multiclass', 'label'],
     
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
        task_type = data_info[0]
        label_name = data_info[1]
        train_data = pd.read_csv(f'{data_dir}/{task_type}/{data_name}/train.csv')
        test_data = pd.read_csv(f'{data_dir}/{task_type}/{data_name}/test.csv')

        if task_type == 'binary':
            file_res.write('data, task, auc, f1, accuracy, recall, precision, time\n')
            metric = 'roc_auc'
        elif task_type == 'multiclass':
            file_res.write('data, task, auc, f1, accuracy, recall, precision, time\n') 
            metric = 'roc_auc_ovo_macro' 
        elif task_type == 'regression':
            file_res.write('data, task, rmse, r2, mse, mae, msle, time\n')
            metric = 'r2'  #'root_mean_squared_error' #'r2'

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
        print('[{}] Data Prepare Finished!'.format(data_name))

        time_start = time.time()
        model = TabularPredictor(label=label_name, eval_metric=metric, path='/home/zhangxj/program/AUTOML/AutoGluonModels')
        model.fit(train_data, time_limit=600)  
        leaderboard = model.leaderboard(test_data)

        scores = eval_socres(test_data.copy(), label_name, model, task_type, method_name='autogluon')
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