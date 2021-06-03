import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from process_data import processData
from sklearn.model_selection import train_test_split

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
    'avocado': ['regression', 'AveragePrice'],
    'puma32h': ['regression', 'thetadd6'],
    'ailerons': ['regression', 'Goal'],
    # 'forestFires': ['regression', 'Area'],
    # 'california': ['regression', 'MedianHouseValue'],
    # 'pole': ['regression', 'Output'],
}

def Titanic_transform(data):
    data['Sex'] = data['Sex'].astype('category')
    data['Embarked'] = data['Embarked'].astype('category')
    data['Sex'] = data['Sex'].cat.codes
    data['Embarked'] = data['Embarked'].cat.codes
    data_c = data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
    return data_c


if __name__ == '__main__':

    for data_name, data_info in zip(dataset.keys(), dataset.values()):
        task_type = data_info[0]
        label_name = data_info[1]
        train_data = pd.read_csv(f'{data_dir}/{task_type}/{data_name}/train.csv')
        test_data = pd.read_csv(f'{data_dir}/{task_type}/{data_name}/test.csv')

        # full_data = pd.read_csv(f'{data_dir}/{task_type}/{data_name}/forestFires.csv')
        # print(full_data.info())
        # train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=9527)
        # train_data.to_csv(f'{data_dir}/{task_type}/{data_name}/train.csv')
        # test_data.to_csv(f'{data_dir}/{task_type}/{data_name}/test.csv')
        
        train_data, ag_predictor = processData(train_data, label_column=label_name, problem_type=task_type)
        test_data, _ = processData(test_data, label_column=label_name, ag_predictor=ag_predictor)
        train_data.dropna(axis=0, how='any', inplace=True)
        test_data.dropna(axis=0, how='any', inplace=True)
        
        pd.to_pickle(train_data, '/home/zhangxj/program/AUTOML/data/' + data_name + '/train.pkl')
        pd.to_pickle(test_data, '/home/zhangxj/program/AUTOML/data/' + data_name + '/test.pkl')

        print('Data Process Finished!')

