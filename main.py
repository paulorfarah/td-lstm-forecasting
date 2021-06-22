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


def score_lstm(estimator, X, y, scoring, cv):
    from sklearn.model_selection import train_test_split

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    # scorer = {'neg_mean_absolute_error': 'neg_mean_absolute_error',
    #           'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2',
    #           'mean_absolute_percentage_error': make_scorer(mean_absolute_percentage_error,
    #                                                         greater_is_better=False),
    #           'root_mean_squared_error': make_scorer(root_mean_squared_error, greater_is_better=False)}

    print('X_test: ', X_test)
    print('y:', y.shape)
    print('y_train:', y_train.shape)
    print('ytest: ', y_test.shape)
    print('y_pred:', y_pred.shape)
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
    print("dataset")

    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    print(df.head(5))
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

def series_to_supervised_recurrent(data, n_in=1, n_out=1, dropnan=True):
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
    print("dataset")

    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    print(df.head(5))
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

def main(DATASET, WINDOW_SIZE, VERSIONS_AHEAD):

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

        def lstm_model():
            # LSTM layer expects inputs to have shape of (batch_size, timesteps, input_dim).
            # In keras you need to pass (timesteps, input_dim) for input_shape argument.

            model = Sequential()
            model.add(LSTM(250, input_shape=(3, 3), kernel_initializer='normal', activation='relu'))
            model.add(Dense(1, kernel_initializer='normal'))
            # Compile model
            model.compile(loss='mean_squared_error', optimizer='adam')
            return model
            #print("model dim: ", input_shape, output_dim)
            ##model = Sequential()
            #model.add(GRU(256, input_shape=input_shape, return_sequences=True))
            #model.add(Dropout(dropout))
            #model.add(GRU(128, return_sequences=True))
            #model.add(Dropout(dropout))
            #model.add(GRU(64))
            #model.add(Dropout(dropout))
            #model.add(Dense(output_dim, activation='softmax'))
            #model.summary()
            #model.compile(loss='mse', optimizer='adam')
            #return model

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
            regressor = KerasRegressor(build_fn=lstm_model, epochs=1000, batch_size=5, verbose=True)
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

       # if reg_type == 'LSTM':
            # reshape from [samples, timesteps] into [samples, timesteps, features]
            #n_features = 1
            #X = X.reshape((X.shape[0], X.shape[1], n_features))

        print(X.shape)
        print(Y.shape)
        # if reg_type == 'LSTM':
        #     scores = score_lstm(estimator=pipeline, X=X, y=Y.ravel(), scoring=scorer, cv=tscv)
        # else:
        scores = cross_validate(estimator=pipeline, X=X, y=Y.ravel(), scoring=scorer, cv=tscv, return_train_score=False)

        # Fill results dict object
        for key, value in scores.items():
            print(key, value)
            scores[key] = value.tolist()
        results[project][versions_ahead][reg_type] = scores

    #     regressor.fit(X_train, Y_train)
    #     results[project][versions_ahead][reg_type]['test_set_r2'] = regressor.score(X_test, Y_test)

    # For every project in dataset
    for project in DATASET:
        # Initialize results dict object
        results[project] = {}
        for versions_ahead in VERSIONS_AHEAD:
            print("Version ahead " +  str(versions_ahead))
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
           # print(data.head(5))
            # Append dependend variable column with value equal to next version's total_principal
            data['forecasted_total_principal'] = data['var4(t)'].shift(-versions_ahead)
           # print("depois do append")
           # print(data.head(5))
            data = data.drop(data.index[-versions_ahead:])
           # print("depois do drop")
           # print(data.head(5))

            # Remove total_cost from indpependent variables set
            #drop pra nao alienar o algoritmo (creio eu)
            data = data.drop(columns=['var4(t-2)', 'var4(t-1)', 'var4(t)'])
          #  print(data.head(5))

            #para transformar na representação do melo é preciso splitar as linhas 
            # em colunas baseado no tamanho da janela eno número de variaveis
            #print(data.shape[0])
            timeseries_list = list()
            timeseries_labels = list()
            for i in range(0,data.shape[0]):
                window = list()
                for j in range(0,data.shape[1]-1,3):
                     window.append(data.iloc[i,j:j+3 ].values.astype(np.float64))
                timeseries_list.append(window)
                timeseries_labels.append(data.iloc[i, data.shape[1]-1])
            
            timeseries_tensor = np.array(timeseries_list)

            timeseries_tensor = timeseries_tensor.transpose((0,2,1))
            
            timeseries_labels = np.array(timeseries_labels).astype(np.float64)
            timeseries_labels = timeseries_labels.reshape(-1, 1)
            
            # Set X, Y
            X = data.iloc[:, data.columns != 'forecasted_total_principal'].values
            # Y = data.iloc[:, data.columns == 'forecasted_total_principal'].values
            Y = data.forecasted_total_principal.values
            # for i in range(len(X)):
            #    print(X[i], Y[i])

            

            #for reg_type in ['LinearRegression', 'LassoRegression', 'RidgeRegression', 'SGDRegression', 'SVR_rbf', 'SVR_linear', 'RandomForestRegressor', 'LSTM']:
            for reg_type in ['LSTM']:
                regressor = create_regressor(reg_type, timeseries_tensor, timeseries_labels, project, versions_ahead)

    print_forecasting_errors(VERSIONS_AHEAD, reg_type, results, versions_ahead, DATASET)


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

def print_forecasting_errors(VERSIONS_AHEAD, reg_type, results, versions_ahead, DATASET):
    for project in DATASET:
        print('**************** %s ****************' % project)
        print(results[project][versions_ahead][reg_type])

       # 'RandomForestRegressor','LinearRegression', 'LassoRegression', 'RidgeRegression', 'SGDRegression', 'SVR_rbf',
       #                  'SVR_linear',

        for reg_type in ['LSTM']:
            print('================ %s ================' % reg_type)
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

                print('%0.3f;%0.3f;%0.3f;%0.3f' % (abs(mae_mean), abs(rmse_mean), abs(mape_mean), r2_mean))

def make_GRU(input_shape, output_dim, dropout=0.3):
        #print("model dim: ", input_shape, output_dim)
        model = Sequential()
        model.add(GRU(256, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(GRU(128, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(GRU(64))
        model.add(Dropout(dropout))
        model.add(Dense(output_dim, activation='softmax'))
        #model.summary()
        return model
def evaluation_metrics(self,y_test, y_pred, weights_t):
       # print("ROC_AUC: " + str(roc_auc_score(y_test, y_pred, sample_weight=weights_t)))
        print("F1-Score: " + str(f1_score(y_test, y_pred, sample_weight=weights_t)))
        print("Precision: " + str(precision_score(y_test, y_pred, sample_weight=weights_t)))
        print("Recall: " + str(recall_score(y_test, y_pred, sample_weight=weights_t)))
        print("Accuracy: " + str(accuracy_score(y_test, y_pred, sample_weight=weights_t)))
def f1(y_true, y_pred):
            def recall(y_true, y_pred):
                """Recall metric.

                Only computes a batch-wise average of recall.

                Computes the recall, a metric for multi-label classification of
                how many relevant items are selected.
                """
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                recall = true_positives / (possible_positives + K.epsilon())
                return recall

            def precision(y_true, y_pred):
                """Precision metric.

                Only computes a batch-wise average of precision.

                Computes the precision, a metric for multi-label classification of
                how many selected items are relevant.
                """
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision
            precision = precision(y_true, y_pred)
            recall = recall(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))

def auc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

def funRemovida():
    print("tensor input X:", timeseries_tensor.shape)
    print("Timeserie labe y:", timeseries_labels.shape)
    print("Splitting dataset into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(timeseries_tensor, timeseries_labels, test_size=0.30, random_state=42)
    X_train = X_train.transpose((0,2,1))
    X_test = X_test.transpose((0,2,1))
    print("Tensor X train:", X_train.shape)
    print("Tensor y train:", y_train.shape)
    print("Tensor X test:", X_test.shape)
    print("Tensor y test:", y_test.shape)

    print("computing weights...")
    
    fractions = 1-y_train.sum(axis=0)/len(y_train)
    weights = fractions[y_train.argmax(axis=1)]
    print("... DONE!")

    print("setting stratified k-fold...")
    k=2
    print("number of k:",k)
    skf = StratifiedKFold(n_splits=k, shuffle=True,  random_state=1337)
    print("... DONE!")
    
    print("Executing algorithm...")
    lastModels = {}
    history_general = {}
    val_history_general = {}
    
    set_number_of_epochs=20
    set_batch_size=512
    print("number of epochs:", set_number_of_epochs)
    print("number of batch:", set_batch_size)
    
    start = time.time()
    #for index, (train_indices, val_indices) in enumerate(skf.split(X_train, y_train)):
    """  print ("Training on fold " + str(index + 1) + "/"+str(k)+"...") 
        xtrain, xval = X_train[train_indices], X_train[val_indices]
        ytrain, yval = y_train[train_indices], y_train[val_indices] 
        weights_train = weights[train_indices]
        weights_val = weights[val_indices]

        #model = None
        model = make_GRU((xtrain.shape[1], xtrain.shape[2]), 2)
        print("compilando")
        #model.compile(loss='mse', optimizer='adam', metrics=['acc', f1])
        model.compile(loss='mse', optimizer='adam', metrics=['acc'])
        print("fit..")
        history = model.fit(xtrain, ytrain, validation_data=(xval, yval, weights_val), epochs=set_number_of_epochs, batch_size=set_batch_size, sample_weight=weights_train, verbose=False)

        print("preditct")
        output = model.predict_classes(xval)

        print(confusion_matrix(yval.argmax(axis=1), output))

        print(classification_report(yval.argmax(axis=1), output))

        end = time.time()
        time_in_seconds =  end - start
        print("time (in seconds)", time_in_seconds)
        
        lastModel = model
        history_general = history.history
        
        print("... DONE!")
    """
    
if __name__ == '__main__':
    # Selecting indicators that will be used as model variables

    # 'AMC', 'WMC', 'DIT', 'NOC', 'RFC', 'CBO', 'Ca', 'Ce', 'CBM', 'IC', 'LCOM', 'LCOM3', 'CAM', 'NPM', 'DAM', 'MOA']
    # 'Security Index', 'blocker_violations', 'critical_violations', 'major_violations', 'minor_violations', 'info_violations']
    DATASET = ['_benchmark_repository_measures',
  #              'apache_groovy_measures',
   #             'apache_incubator_dubbo_measures',
   #             'apache_kafka_measures',
    #            'apache_nifi_measures',
     #           'apache_ofbiz_measures',
      #          'apache_systemml_measures',
       #         'commonsio_measures',
        #        'company_projectA_measures',
         #       'company_projectB_measures',
          #      'google_guava_measures',
           #     'igniterealtime_openfire_measures',
            #    'java_websocket_measures',
             #   'jenkinsci_jenkins_measures',
              #  'spring-projects_spring-boot_measures',
               # 'square_okhttp_measures',
                #'square_retrofit_measures',
                #'zxing_zxing_measures']
                ]

    WINDOW_SIZE = 2  # choose based on error minimization for different forecasting horizons
    VERSIONS_AHEAD = [2, 5, 10, 20, 40]

    main(DATASET, WINDOW_SIZE, VERSIONS_AHEAD)

