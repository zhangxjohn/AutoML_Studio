""" Convert data from pandas DataFrame to numeric matrix (required for autoML methods like auto-sklearn, TPOT) """

import warnings
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

def processData(data, label_column=None, ag_predictor=None, problem_type=None, eval_metric=None):
    """ Converts pandas Dataframe to matrix of entirely numerical values (stored in DataFrame).
        Performs same data preprocessing as used for AutoGluon's tabular neural network model, 
        to deal with issues such as: missing value imputation, one-hot encoding of categoricals, 
        handling of high-cardinality categoricals, handling unknown categorical feature-levels at test-time, etc.
        
        If ag_predictor is not None, uses existing autogluon predictor object to process data (must have tabularNN as first model).
        To process training data, ag_predictor should = None. For test data, should != None.
        Returns:
            Tuple (X, y, ag_predictor)
            where y may be None if labels are not present in test data.
    """
    
    # fit dummy neural network model just to preprocess data. Here we ensure no embedding layers are used.
    if ag_predictor is None:
        if label_column is None:
            raise ValueError("when processing training data, label_column cannot be None")
        elif not label_column in data.columns:
            raise ValueError("label_column cannot be missing from training data")

        ag_predictor = TabularPredictor(label=label_column, problem_type=problem_type, eval_metric=eval_metric, 
                            path='/home/zhangxj/program/AUTOML/AutoGluonModels').fit(
                            train_data=data, tuning_data=data,
                            hyperparameters={'NN': {'num_epochs': 0, 'proc.embed_min_categories': np.inf}}, 
                            num_bag_folds=0, num_stack_levels=0, verbosity=0)

    model = ag_predictor._trainer.load_model(ag_predictor._trainer.get_model_names()[0]) # This must be the neural net model which contains data processor

    if 'NeuralNetMXNet' not in model.name:
        raise ValueError("Data preprocessing error. This model should be the NeuralNet, not the: %s" % model.name)
    bad_inds = [] # row-indices to remove from dataset
    if label_column is not None and label_column in data.columns:
        label_cleaner = ag_predictor._learner.label_cleaner
        y = data[label_column].values
        data = data.drop([label_column], axis=1, inplace=False)
        y = label_cleaner.transform(y)
        if np.sum(y.isna()) > 0:
            bad_inds = y.index[y.apply(np.isnan)].tolist() # remove these inds as label is NaN (due to very rare classes)
            warnings.warn("Dropped these rows from data in preprocessing, due to missing labels: " + str(bad_inds))
    else:
        y = None
    data_initial_processed = ag_predictor._learner.transform_features(data) # general autogluon data processing.
    # data_fg = ag_predictor._learner.general_data_processing(X=data, X_test=data, holdout_frac=0.0, num_bagging_folds=0)
    X = model.preprocess(data_initial_processed, is_test=True) # neural net-specific autogluon data processing required to turn tabular data into numerical matrix.
    if len(bad_inds) > 0:
        y.drop(index=bad_inds, inplace=True)
        X.drop(index=bad_inds, axis=0, inplace=True)

    X[label_column] = y

    return X, ag_predictor
