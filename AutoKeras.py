import numpy as np
import pandas as pd
import datetime, time
from sklearn.preprocessing import LabelEncoder
from utils import eval_socres

from autokeras import StructuredDataClassifier, StructuredDataRegressor

file_res = open(f'/home/zhangxj/program/AUTOML/res_log/autokeras_results.csv', 'a+')

data_dir = '/home/zhangxj/program/AUTOML/data'

dataset = {
    'Titanic': ['binary', 'Survived'],
    # 'Bank_Marketing_Data': ['binary', 'y'],
    # 'Magic': ['binary', 'Class'],
    # 'Spambase': ['binary', 'Spam'],
    # 'Adult': ['binary', 'Class'],
    # 'Splice': ['multiclass', 'Class'],
    # 'Thyroid': ['multiclass', 'Class'],
    # 'Yeast': ['multiclass', 'Class'],
    # 'House_prices': ['regression', 'SalePrice'],
    # 'Bike_Sharing': ['regression', 'count']
}


if __name__ == '__main__':

    for data_name, data_info in zip(dataset.keys(), dataset.values()):
        train_data = pd.read_pickle(f'{data_dir}/{data_name}/train.pkl')
        test_data = pd.read_pickle(f'{data_dir}/{data_name}/test.pkl')
        task_type = data_info[0]
        label_name = data_info[1]
        if task_type == 'binary':
            file_res.write('data, task, auc, f1, accuracy, recall, precision, time\n')
        elif task_type == 'multiclass':
            file_res.write('data, task, logloss, f1, accuracy, recall, precision, time\n')  
        elif task_type == 'regression':
            file_res.write('data, task, r2, mae, msle, mse, rmse, time\n')                      

        le = LabelEncoder()
        train_data[label_name] = le.fit_transform(train_data[label_name])
        test_data[label_name] = le.transform(test_data[label_name])

        X_train = train_data.copy()
        y_train = X_train.pop(label_name)
        print('\n[{}] Data Prepare Finished!\n'.format(data_name))

        time_start = time.time()
        model = StructuredDataClassifier(max_trials=5, seed=2021)
        model.fit(X_train, y_train, epochs=5, batch_size=32)  
        
        scores = eval_socres(test_data, label_name, model, task_type, method_name='autokeras')
        score_values = list(scores.values())
        print(scores)
        time_end = time.time()
        total_time = time_end - time_start 
        print('total time: {}'.format(total_time)) 
        file_res.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(data_name, task_type, \
                score_values[0], score_values[1], score_values[2], score_values[3], score_values[4], total_time))

        del model, train_data, test_data

    file_res.close()