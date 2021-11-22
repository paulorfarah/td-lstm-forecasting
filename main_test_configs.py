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
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import optimizers
import itertools
#import csv
from csv import writer


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


def score_lstm(estimator, X, y, scoring, cv):
    from sklearn.model_selection import train_test_split

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    scoring['neg_mean_absolute_error'] = mean_absolute_error(y_test, y_pred)
    scoring['neg_mean_squared_error'] = mean_squared_error(y_test, y_pred)
    scoring['r2'] = r2_score(y_test, y_pred)
    # scoring['mean_absolute_percentage_error'] = mean_absolute_percentage_error(y_test, y_pred)
    # scoring['root_mean_squared_error'] = root_mean_squared_error(y_test, y_pred)
    return scoring

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X). (WINDOW_SIZE)
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    print("series_to_supervised")

    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
      
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def main(DATASET, WINDOW_SIZE, VERSIONS_AHEAD,EXP,comb):

    METRIC_KEYS_SDK4ED = ['bugs', 'duplicated_blocks', 'code_smells',
                          # 'comment_lines', 'ncloc', 'uncovered_lines', 'vulnerabilities', 'complexity',
                          'sqale_index', 'reliability_remediation_effort', 'security_remediation_effort']
    results = {}
    results['_info'] = {'metrics': METRIC_KEYS_SDK4ED, 'window_size': WINDOW_SIZE, 'versions_ahead': VERSIONS_AHEAD}

    def create_regressor(reg_type, X, Y, project, versions_ahead):
        # Splitting the dataset into the Training set and Test set
        #     from sklearn.model_selection import train_test_split
        #     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle = False)

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
     
        # define base model
        def baseline_model():
            # create model
            model = Sequential()
            model.add(Dense(10, input_shape=(10,), input_dim=10, kernel_initializer='normal', activation='relu'))
            model.add(Dense(1, kernel_initializer='normal'))
            # Compile model
            model.compile(loss='mean_squared_error', optimizer='adam')
            return model

       
        def lstm_model(optimizer='adam', activation="relu", neurons = 50,learn_rate = 0.1, dropout_rate=0.2, layers = 1):
            # LSTM layer expects inputs to have shape of (batch_size, timesteps, input_dim).
            # In keras you need to pass (timesteps, input_dim) for input_shape argument.
            K.clear_session()
            opt = optimizers.Adam(learning_rate=learn_rate)
            if layers == 1:
                model = Sequential()
                model.add(LSTM(neurons, input_shape=(9, 1), kernel_initializer='normal', activation=activation))
                model.add(Dropout(dropout_rate))
                model.add(Dense(1, kernel_initializer='normal'))
                model.compile(loss='mean_squared_error', optimizer=opt)
                return model
            else:
                model = Sequential()
                model.add(LSTM(neurons, input_shape=(9, 1), return_sequences=True, kernel_initializer='normal', activation=activation))
                model.add(Dropout(dropout_rate))
                model.add(LSTM(neurons, input_shape=(9, 1), return_sequences=False, kernel_initializer='normal', activation=activation))
                model.add(Dense(1, kernel_initializer='normal'))
                model.compile(loss='mean_squared_error', optimizer=opt)
                return model


        # Create the regressor model
        if reg_type == 'LinearRegression':
            # Fitting Multiple Linear Regression to the Training set
            from sklearn.linear_model import LinearRegression
            regressor = LinearRegression()
            pipeline = Pipeline([('regressor', regressor)])
        if reg_type == 'LassoRegression':
            # Fitting Lasso Regression to the Training set
            from sklearn.linear_model import Lasso
            regressor = Lasso(alpha=100000)
            pipeline = Pipeline([('regressor', regressor)])
        if reg_type == 'RidgeRegression':
            # Fitting Ridge Regression to the Training set
            from sklearn.linear_model import Ridge
            regressor = Ridge(alpha=1000000)
            pipeline = Pipeline([('regressor', regressor)])
        if reg_type == 'SGDRegression':
            # Fitting SGD Regression to the Training set
            from sklearn.linear_model import SGDRegressor
            regressor = SGDRegressor(max_iter=1000, tol=1e-3)
            pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
        elif reg_type == 'SVR_linear':
            # Fitting linear SVR to the dataset
            from sklearn.svm import SVR
            regressor = SVR(kernel='linear', C=10000)
            pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
        elif reg_type == 'SVR_rbf':
            # Fitting SVR to the dataset
            from sklearn.svm import SVR
            regressor = SVR(kernel='rbf', gamma=0.01, C=10000)
            pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
        elif reg_type == 'RandomForestRegressor':
            # Fitting Random Forest Regression to the dataset
            from sklearn.ensemble import RandomForestRegressor
            regressor = RandomForestRegressor(n_estimators=100, random_state=0)
            pipeline = Pipeline([('regressor', regressor)])
        elif reg_type == 'ANN':
            # Fitting ANN to the dataset
            from keras.wrappers.scikit_learn import KerasRegressor
            regressor = KerasRegressor(build_fn=baseline_model, epochs=1000, batch_size=5, verbose=False)
            pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
        elif reg_type == 'LSTM':
            from keras.wrappers.scikit_learn import KerasRegressor
            regressor = KerasRegressor(build_fn=lstm_model,batch_size=comb[0],epochs=comb[1],verbose=False)
            # pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
            pipeline = Pipeline([('regressor', regressor)])

        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)

        from sklearn.metrics import make_scorer
        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        def root_mean_squared_error(y_true, y_pred):
            return sqrt(mean_squared_error(y_true, y_pred))

        # Applying TimeSeriesSplit Validation
        from sklearn.model_selection import cross_validate
        scorer = {'neg_mean_absolute_error': 'neg_mean_absolute_error',
                  'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2',
                  'mean_absolute_percentage_error': make_scorer(mean_absolute_percentage_error,
                                                                greater_is_better=False),
                  'root_mean_squared_error': make_scorer(root_mean_squared_error, greater_is_better=False)}

        if reg_type == 'LSTM':
            # reshape from [samples, timesteps] into [samples, timesteps, features]
            n_features = 1
            X = X.reshape((X.shape[0], X.shape[1], n_features))
           
        scores = cross_validate(estimator=pipeline, X=X, y=Y.ravel(), scoring=scorer, cv=tscv, return_train_score=False)
        

        # Fill results dict object
        for key, value in scores.items():
            scores[key] = value.tolist()
        results[project][versions_ahead][reg_type] = scores
  
   # For every project in dataset
    for project in DATASET:
        # Initialize results dict object
        results[project] = {}
        for versions_ahead in VERSIONS_AHEAD:
            print("Version ahead " +  str(versions_ahead) + " " + project)
            # Initialize results dict object
            results[project][versions_ahead] = {}

            # Importing the dataset
            #coloca 3 colunas em uma so e dpois dropa essas 3 colunas
            dataset = pd.read_csv(project + '.csv', sep=";", usecols=METRIC_KEYS_SDK4ED)
            dataset['total_principal'] = dataset['reliability_remediation_effort'] + dataset[
                'security_remediation_effort'] + dataset['sqale_index']
            dataset = dataset.drop(
                columns=['sqale_index', 'reliability_remediation_effort', 'security_remediation_effort'])

            # Adding time-shifted prior and future period
            data = series_to_supervised(dataset.values, n_in=WINDOW_SIZE)
            # Append dependend variable column with value equal to next version's total_principal
            data['forecasted_total_principal'] = data['var4(t)'].shift(-versions_ahead)
            data = data.drop(data.index[-versions_ahead:])
            # Remove total_cost from indpependent variables set
            #drop pra nao alienar o algoritmo (creio eu)
            data = data.drop(columns=['var4(t-2)', 'var4(t-1)', 'var4(t)'])
            
          

            # Set X, Y
            X = data.iloc[:, data.columns != 'forecasted_total_principal'].values
            # Y = data.iloc[:, data.columns == 'forecasted_total_principal'].values
            Y = data.forecasted_total_principal.values
            #for reg_type in ['LinearRegression', 'LassoRegression', 'RidgeRegression', 'SGDRegression', 'SVR_rbf', 'SVR_linear', 'RandomForestRegressor', 'LSTM']:
            #teste com estrutura do melo
            for reg_type in ['LSTM']:
                regressor = create_regressor(reg_type, X, Y, project, versions_ahead)

    print_forecasting_errors(VERSIONS_AHEAD, reg_type, results, versions_ahead, DATASET, EXP,comb)


def read_dataset(VERSIONS_AHEAD, WINDOW_SIZE):
    return read_td_dataset(VERSIONS_AHEAD, WINDOW_SIZE)


def read_td_dataset(VERSIONS_AHEAD, WINDOW_SIZE):
    METRIC_KEYS_SDK4ED = ['bugs', 'duplicated_blocks', 'code_smells',
                          # 'comment_lines', 'ncloc', 'uncovered_lines', 'vulnerabilities', 'complexity',
                          'sqale_index', 'reliability_remediation_effort', 'security_remediation_effort']

    # Importing the dataset
    dataset = pd.read_csv('apache_kafka_measures.csv', sep=";", usecols=METRIC_KEYS_SDK4ED)
    dataset['total_principal'] = dataset['reliability_remediation_effort'] + dataset['security_remediation_effort'] + \
                                 dataset['sqale_index']
    dataset = dataset.drop(columns=['sqale_index', 'reliability_remediation_effort', 'security_remediation_effort'])
    # Adding time-shifted prior and future period
    data = series_to_supervised(dataset.values, n_in=WINDOW_SIZE)
    # Append dependend variable column with value equal to next version's total_principal
    data['forecasted_total_principal'] = data['var4(t)'].shift(-VERSIONS_AHEAD)
    data = data.drop(data.index[-VERSIONS_AHEAD:])
    # Include/remove TD as independent variable
    data = data.drop(columns=['var4(t-2)', 'var4(t-1)'])
    X = data.iloc[:, data.columns != 'forecasted_total_principal'].values
    Y = data.iloc[:, data.columns == 'forecasted_total_principal'].values
   
    return X, Y, METRIC_KEYS_SDK4ED

def append_new_line(file_name, row):
    with open(file_name, 'a+') as f_object:
        writer_object = writer(f_object,delimiter=";")
        writer_object.writerow(row)
        f_object.close()

def print_forecasting_errors(VERSIONS_AHEAD, reg_type, results, versions_ahead, DATASET, EXP,comb):
    for project in DATASET:
        print('**************** %s ****************' % project)
        #print(results[project][versions_ahead][reg_type])
        #for reg_type in ['LinearRegression', 'LassoRegression', 'RidgeRegression', 'SGDRegression', 'SVR_rbf', 'SVR_linear', 'RandomForestRegressor', 'LSTM']:
       
        filename = 'results/' + project + '-model-find' + '.csv'

        for reg_type in ['LSTM']:
            print('*************** %s **************' % reg_type)
            average_mae = 0
            average_rmse = 0
            average_mape = 0
            average_r2_mean2 = 0
            average_mean_fit_time2 = 0
            average_std_fit_time2 = 0
            average_mean_score_time2 = 0
            average_std_score_time2 = 0
            for versions_ahead in VERSIONS_AHEAD:
                # Print scores
                mae_mean = np.asarray(results[project][versions_ahead][reg_type]['test_neg_mean_absolute_error']).mean()
                mae_std = np.asarray(results[project][versions_ahead][reg_type]['test_neg_mean_absolute_error']).std()
                mse_mean = np.asarray(results[project][versions_ahead][reg_type]['test_neg_mean_squared_error']).mean()
                mse_std = np.asarray(results[project][versions_ahead][reg_type]['test_neg_mean_squared_error']).std()
                mape_mean = np.asarray(
                    results[project][versions_ahead][reg_type]['test_mean_absolute_percentage_error']).mean()
                mape_std = np.asarray(
                    results[project][versions_ahead][reg_type]['test_mean_absolute_percentage_error']).std()
                r2_mean = np.asarray(results[project][versions_ahead][reg_type]['test_r2']).mean()
                r2_std = np.asarray(results[project][versions_ahead][reg_type]['test_r2']).std()
                rmse_mean = np.asarray(results[project][versions_ahead][reg_type]['test_root_mean_squared_error']).mean()
                rmse_std = np.asarray(results[project][versions_ahead][reg_type]['test_root_mean_squared_error']).std()
                # test_set_r2 = results[project][versions_ahead][reg_type]['test_set_r2']
                fit_mean = np.asarray(results[project][versions_ahead][reg_type]['fit_time']).mean()
                fit_std = np.asarray(results[project][versions_ahead][reg_type]['fit_time']).std()
                score_mean = np.asarray(results[project][versions_ahead][reg_type]['score_time']).mean()
                score_std = np.asarray(results[project][versions_ahead][reg_type]['score_time']).std()

                print('%0.3f;%0.3f;%0.3f;%0.3f' % (abs(mae_mean), abs(rmse_mean), abs(mape_mean), r2_mean))
            
                line = []
                line.append(EXP)
                line.append(project)
                #horizon
                line.append(str(versions_ahead))
                #config
                line.append(str(comb))
                line.append(fit_mean)
                line.append(fit_std)
                line.append(score_mean)
                line.append(score_std)
                average_mean_fit_time2 += fit_mean
                average_std_fit_time2 += fit_std
                average_mean_score_time2 += score_mean
                average_std_score_time2 += score_std
                line.append(abs(mae_mean))
                average_mae += abs(mae_mean)
                line.append(abs(rmse_mean))
                average_rmse += abs(rmse_mean)
                line.append(abs(mape_mean))
                average_mape += abs(mape_mean)
                line.append(r2_mean)
                average_r2_mean2 += r2_mean
                with open(filename, 'a+') as f_object:
                    writer_object = writer(f_object,delimiter=";")
                    writer_object.writerow(line)
                    f_object.close()

                #print(abs(mae_mean2[i]),abs(rmse_mean2[i]),abs(mape_mean2[i]),r2_mean2[i],sep=';')                   
                #print('%0.3f,%0.3f,%0.3f,%0.3f' % (abs(mae_mean), abs(rmse_mean), abs(mape_mean), r2_mean))
            line = []
            line.append(EXP)
            line.append(project)
            #horizon
            line.append("average")
            #config
            line.append(str(comb))
            line.append(average_mean_fit_time2/5)
            line.append(average_std_fit_time2/5)
            line.append(average_mean_score_time2/5)
            line.append(average_std_score_time2/5)
            line.append(average_mae/5)
            line.append(average_rmse/5)
            line.append(average_mape/5)
            line.append(average_r2_mean2/5)
            with open(filename, 'a+') as f_object:
                writer_object = writer(f_object,delimiter=";")
                writer_object.writerow(line)
                f_object.close()

if __name__ == '__main__':
    # Selecting indicators that will be used as model variables

    # 'AMC', 'WMC', 'DIT', 'NOC', 'RFC', 'CBO', 'Ca', 'Ce', 'CBM', 'IC', 'LCOM', 'LCOM3', 'CAM', 'NPM', 'DAM', 'MOA']
    # 'Security Index', 'blocker_violations', 'critical_violations', 'major_violations', 'minor_violations', 'info_violations']
    DATASET_5_SPLIT = [
               'apache_kafka_measures',
               'apache_groovy_measures',
               'apache_systemml_measures',
               'commonsio_measures',
               'company_projectA_measures',
               'company_projectB_measures',
               'google_guava_measures',
               'jenkinsci_jenkins_measures',
               'square_okhttp_measures',
                ]
    DATASET_4_split = ['apache_ofbiz_measures', 'apache_nifi_measures', 'apache_incubator_dubbo_measures',
                       'square_retrofit_measures', 'spring-projects_spring-boot_measures', 'java_websocket_measures',
                       'zxing_zxing_measures', 'igniterealtime_openfire_measures']
        
                

    WINDOW_SIZE = 2  # choose based on error minimization for different forecasting horizons
    EXPERIMENTS = 5
    CONFIGS = 3
    VERSIONS_AHEAD = [1, 5, 10, 20, 40]
    #VERSIONS_AHEAD = [1,5]
    #main(DATASET, WINDOW_SIZE, VERSIONS_AHEAD,1,1)
    #for config in range(1,CONFIGS+1):
    batch_size = [5,10,15]
    epochs = [100,500,1000,1500]
    optimizer = ['adam']
    learn_rate = [0.01,0.1,0.2]
    activation = ['relu', 'tanh', 'sigmoid']
    dropout_rate = [0.1,0.2,0.25]
    neurons = [10,50,100,150,200]
    layers = [1,2]

    parameters_array = [batch_size,epochs,optimizer,learn_rate,activation,dropout_rate,neurons,layers]
    list_paremeters = []
    for p in itertools.product(*parameters_array):
        list_paremeters.append(p)
    print("combinacoes " + str(len(list_paremeters)))
    for exp in range(1,EXPERIMENTS+1):
        for dataset in DATASET_5_SPLIT:
            listDataset = []
            listDataset.append(dataset)
            for comb in list_paremeters:
                  #  main(listDataset, WINDOW_SIZE, VERSIONS_AHEAD,exp,comb)
                try:
                    main(listDataset, WINDOW_SIZE, VERSIONS_AHEAD,exp,comb)
                except:
                   print("An exception occurred " + str(comb))
            

