# Importing the libraries
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.losses import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dropout
from sklearn.utils import shuffle
from keras.layers import Dense, Bidirectional
from keras.layers.recurrent import LSTM, GRU
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import time
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from keras import backend as K




# Importing the libraries
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.python.ops.metrics_impl import root_mean_squared_error
#import mysql.connector as connection
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Selecting indicators that will be used as model variables
METRIC_KEYS_SDK4ED = ['bugs', 'duplicated_blocks', 'code_smells', #'comment_lines', 'ncloc', 'uncovered_lines', 'vulnerabilities', 'complexity',
                      'sqale_index', 'reliability_remediation_effort', 'security_remediation_effort']
                      #'AMC', 'WMC', 'DIT', 'NOC', 'RFC', 'CBO', 'Ca', 'Ce', 'CBM', 'IC', 'LCOM', 'LCOM3', 'CAM', 'NPM', 'DAM', 'MOA']
                      #'Security Index', 'blocker_violations', 'critical_violations', 'major_violations', 'minor_violations', 'info_violations']
DATASET = ['igniterealtime_openfire_measures']

WINDOW_SIZE = 2 # choose based on error minimization for different forecasting horizons
VERSIONS_AHEAD = [1, 5, 10, 20, 40]
results = {}
results['_info'] = {'metrics': METRIC_KEYS_SDK4ED, 'window_size': WINDOW_SIZE, 'versions_ahead': VERSIONS_AHEAD}

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def create_regressor(reg_type, X, Y, project, versions_ahead):
    # Splitting the dataset into the Training set and Test set
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle = False)
    
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
        
    # Create the regressor model
    if reg_type == 'LinearRegression':
        # Fitting Multiple Linear Regression to the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        pipeline = Pipeline([('regressor', regressor)])
    if reg_type == 'LassoRegression':
        # Fitting Lasso Regression to the Training set
        from sklearn.linear_model import Lasso
        regressor = Lasso(alpha = 100000)
        pipeline = Pipeline([('regressor', regressor)])
    if reg_type == 'RidgeRegression':
        # Fitting Ridge Regression to the Training set
        from sklearn.linear_model import Ridge
        regressor = Ridge(alpha = 1000000)
        pipeline = Pipeline([('regressor', regressor)])
    if reg_type == 'SGDRegression':
        # Fitting SGD Regression to the Training set
        from sklearn.linear_model import SGDRegressor
        regressor = SGDRegressor(max_iter=1000, tol=1e-3)
        pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
    elif reg_type == 'SVR_linear':
        # Fitting linear SVR to the dataset
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'linear', C = 10000)
        pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
    elif reg_type == 'SVR_rbf':
        # Fitting SVR to the dataset
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'rbf', gamma = 0.01, C = 10000)
        pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
    elif reg_type == 'RandomForestRegressor':
        # Fitting Random Forest Regression to the dataset
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
        pipeline = Pipeline([('regressor', regressor)])
    elif reg_type == 'ANN':
        # Fitting ANN to the dataset
        from keras.wrappers.scikit_learn import KerasRegressor
        regressor = KerasRegressor(build_fn = baseline_model, epochs = 1000, batch_size = 5, verbose = True)
        pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])

    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=4)

    from sklearn.metrics.scorer import make_scorer
    def mean_absolute_percentage_error(y_true, y_pred): 
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def root_mean_squared_error(y_true, y_pred): 
        return sqrt(mean_squared_error(y_true, y_pred))
        
    # Applying TimeSeriesSplit Validation
    from sklearn.model_selection import cross_validate
    scorer = {'neg_mean_absolute_error': 'neg_mean_absolute_error', 'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2', 'mean_absolute_percentage_error': make_scorer(mean_absolute_percentage_error, greater_is_better=False), 'root_mean_squared_error': make_scorer(root_mean_squared_error, greater_is_better=False)}
    scores = cross_validate(estimator = pipeline, X = X, y = Y.ravel(), scoring = scorer, cv = tscv, return_train_score = False)

    # Fill results dict object
    for key, value in scores.items():
        scores[key] = value.tolist()
    results[project][versions_ahead][reg_type] = scores
#     regressor.fit(X_train, Y_train)
#     results[project][versions_ahead][reg_type]['test_set_r2'] = regressor.score(X_test, Y_test)
    
# For every project in dataset
for project in DATASET:
    # Initialize results dict object
    results[project] = {}
    for versions_ahead in VERSIONS_AHEAD:
        # Initialize results dict object
        results[project][versions_ahead] = {}

        print(results[project][versions_ahead])
                
        # Importing the dataset
        dataset = pd.read_csv(project + '.csv', sep=";", usecols = METRIC_KEYS_SDK4ED)
        dataset['total_principal'] = dataset['reliability_remediation_effort'] + dataset['security_remediation_effort'] + dataset['sqale_index']
        dataset = dataset.drop(columns=['sqale_index', 'reliability_remediation_effort', 'security_remediation_effort'])
        
        # Adding time-shifted prior and future period
        data = series_to_supervised(dataset.values, n_in = WINDOW_SIZE)
        
        # Append dependend variable column with value equal to next version's total_principal
        data['forecasted_total_principal'] = data['var4(t)'].shift(-versions_ahead)
        data = data.drop(data.index[-versions_ahead:])
            
        # Remove total_cost from indpependent variables set
        data = data.drop(columns=['var4(t-2)', 'var4(t-1)'])
        
        # Set X, Y
        X = data.iloc[:, data.columns != 'forecasted_total_principal'].values
        Y = data.iloc[:, data.columns == 'forecasted_total_principal'].values
        
        for reg_type in ['LinearRegression', 'LassoRegression', 'RidgeRegression', 'SGDRegression', 'SVR_rbf', 'SVR_linear', 'RandomForestRegressor']:
            regressor = create_regressor(reg_type, X, Y, project, versions_ahead)
          
    for project in ['igniterealtime_openfire_measures']:
            print('**************** %s ****************' % project)
            for reg_type in ['LinearRegression', 'LassoRegression', 'RidgeRegression', 'SGDRegression', 'SVR_rbf', 'SVR_linear', 'RandomForestRegressor']:
                print('================ %s ================' % reg_type)
                for versions_ahead in VERSIONS_AHEAD:
                    # Print scores
                    mae_mean = np.asarray(results[project][versions_ahead][reg_type]['test_neg_mean_absolute_error']).mean()
                    mae_std = np.asarray(results[project][versions_ahead][reg_type]['test_neg_mean_absolute_error']).std()
                    mse_mean = np.asarray(results[project][versions_ahead][reg_type]['test_neg_mean_squared_error']).mean()
                    mse_std = np.asarray(results[project][versions_ahead][reg_type]['test_neg_mean_squared_error']).std()
                    mape_mean = np.asarray(results[project][versions_ahead][reg_type]['test_mean_absolute_percentage_error']).mean()
                    mape_std = np.asarray(results[project][versions_ahead][reg_type]['test_mean_absolute_percentage_error']).std()
                    r2_mean = np.asarray(results[project][versions_ahead][reg_type]['test_r2']).mean()
                    r2_std = np.asarray(results[project][versions_ahead][reg_type]['test_r2']).std()
                    rmse_mean = np.asarray(results[project][versions_ahead][reg_type]['test_root_mean_squared_error']).mean()
                    rmse_std = np.asarray(results[project][versions_ahead][reg_type]['test_root_mean_squared_error']).std()
                    #test_set_r2 = results[project][versions_ahead][reg_type]['test_set_r2']
                    print('%0.3f;%0.3f;%0.3f;%0.3f' % (abs(mae_mean), abs(rmse_mean), abs(mape_mean), r2_mean))
