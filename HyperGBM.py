import gc
import numpy as np
import pandas as pd
import datetime, time
from sklearn.preprocessing import LabelEncoder
from utils import eval_socres

import warnings
warnings.filterwarnings('ignore')

from hypergbm import make_experiment
from hypernets.searchers import EvolutionSearcher
from hypergbm.search_space import search_space_general_with_class_balancing
from hypergbm.search_space import GeneralSearchSpaceGenerator

search_space_general = GeneralSearchSpaceGenerator(n_estimators=500,
                                                   enable_lightgbm=True, 
                                                   enable_xgb=True, 
                                                   enable_catboost=True)

file_res = open(f'/home/zhangxj/program/AUTOML/res_log/hypergbm_results.csv', 'a+')
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
     
    # 'House_Prices': ['regression', 'SalePrice'],
    # 'Mercedes_Benz_Greener_Manufacturing': ['regression', 'y'],
    # 'Allstate_Claims_Severity': ['regression', 'loss'],
    # 'Playground_Series_Regression': ['regression', 'target'],
    # 'avocado': ['regression', 'AveragePrice'],
    'puma32h': ['regression', 'thetadd6'],
    # 'ailerons': ['regression', 'Goal'],
    # 'forestFires': ['regression', 'Area'],
    # 'california': ['regression', 'MedianHouseValue'],
    # 'pole': ['regression', 'Output'],
}


if __name__ == '__main__':

    for data_name, data_info in zip(dataset.keys(), dataset.values()):
        task_type = data_info[0]
        label_name = data_info[1]
        train_data = pd.read_csv(f'{data_dir}/{task_type}/{data_name}/train.csv')
        test_data = pd.read_csv(f'{data_dir}/{task_type}/{data_name}/test.csv')

        if task_type == 'binary':
            file_res.write('data, task, auc, f1, accuracy, recall, precision, time\n')
            metric = 'auc'
        elif task_type == 'multiclass':
            file_res.write('data, task, auc, f1, accuracy, recall, precision, time\n') 
            metric = 'auc' 
        elif task_type == 'regression':
            file_res.write('data, task, rmse, r2, mse, mae, msle, time\n')
            metric =  'r2'  #'rmse' #'r2' 

        if task_type != 'regression':
            le = LabelEncoder()
            train_data[label_name] = le.fit_transform(train_data[label_name])
            test_data[label_name] = le.transform(test_data[label_name])
        else:
            pass
            maxs, mins = train_data[label_name].max(), train_data[label_name].min()
            train_data[label_name] = (train_data[label_name] - mins) / (maxs - mins)
            test_data[label_name] = (test_data[label_name] - mins) / (maxs - mins)
        print('[{}] Data Prepare Finished!'.format(data_name))

        time_start = time.time()
        model = make_experiment(train_data.copy(),
                            target=label_name,
                            task=task_type,
                            reward_metric=metric,
                            cv=False,
                            num_folds=3,
                            max_trials=500,
                            early_stopping_rounds=50, #100
                            early_stopping_time_limit=3600, #3600
                            collinearity_detection=False,
                            drift_detection=False,
                            drift_detection_variable_shift_threshold=0.97,
                            drift_detection_threshold=0.7,
                            drift_detection_min_features= int(train_data.shape[1]*0.25),
                            pseudo_labeling=False,
                            pseudo_labeling_proba_threshold=0.8,
                            clear_cache=True,
                            class_balancing='sample_weight',
                            searchers=EvolutionSearcher(search_space_general, population_size=500, sample_size=20, candidates_size=20),
                            ensemble_size=20,
                            log_level='info',
                            seed=2021,
                            ).run() 
        
        scores = eval_socres(test_data.copy(), label_name, model, task_type, method_name='hypergbm')
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