# Importing the libraries
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.losses import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM
# Importing the libraries
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.python.ops.metrics_impl import root_mean_squared_error
import mysql.connector as connection

df_params = pd.read_csv('parameters.csv')

def plot_results():
    # Initialize the lists for X and Y

    data = pd.read_csv('mape.csv')

    df = pd.DataFrame(data)
    # print(df.groupby('Weeks ahead').head())

    for i in [1, 5, 10, 20, 40]:
        df_1 = df[df['Horizon'] == i]
        print(df_1.head())
        #
        labels = list(df_1.iloc[:, 1])
        # Y = [list(df_1.iloc[:, 2]), list(df_1.iloc[:, 3])]
        x = np.arange(len(labels))  # the label locations

        width = 0.35  # the width of the bars
        new = list(df_1.iloc[:, 18])
        original = list(df_1.iloc[:, 19])

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, original, width, label='Original study')
        rects2 = ax.bar(x + width / 2, new, width, label='New study')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('MAPE(%)')
        # ax.set_title('Scores by group and gender')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.xticks(rotation=30)
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        # plt.show()
        plt.savefig('results/mape' + str(i) + '.pdf')



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

    # print('X_test: ', X_test)
    # print('y:', y.shape)
    # print('y_train:', y_train.shape)
    # print('ytest: ', y_test.shape)
    # print('y_pred:', y_pred.shape)
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

def main(DATASET, WINDOW_SIZE, VERSIONS_AHEAD):


    # X, y, METRIC_KEYS_SDK4ED = read_dataset(VERSIONS_AHEAD, WINDOW_SIZE)

    # Applying Grid Search to find the best model and the best parameters
    # from sklearn.model_selection import GridSearchCV
    # from sklearn.model_selection import TimeSeriesSplit

    # from sklearn.preprocessing import StandardScaler
    # sc_X = StandardScaler()
    # X = sc_X.fit_transform(X)

    # tscv = TimeSeriesSplit(n_splits=5)

    # from sklearn.linear_model import LinearRegression
    # regressor = LinearRegression()

    # from sklearn.linear_model import Lasso
    # regressor = Lasso(random_state = 0)
    # parameters = {'alpha' : [100, 1000, 100000, 1000000, 10000000]}

    # from sklearn.linear_model import Ridge
    # regressor = Ridge(random_state = 0)
    # parameters = {'alpha' : [1000, 100000, 1000000, 10000000]}

    # from sklearn.linear_model import SGDRegressor
    # regressor = SGDRegressor(random_state = 0)

    # from sklearn.svm import SVR
    # regressor = SVR(kernel = 'linear')
    # parameters = {'C' : [1000, 10000, 100000]}

    # from sklearn.svm import SVR
    # regressor = SVR(kernel = 'rbf')
    # parameters = {'C' : [1000, 10000, 100000, 1000000, 10000000], 'gamma' : [0.1, 0.01, 0.001]}

    # from sklearn.ensemble import RandomForestRegressor
    # regressor = RandomForestRegressor(random_state=0)
    # parameters = {'n_estimators': [5, 10, 100, 1000], 'max_depth': [2, 5, 10, 20, 50, 100]}
    #
    # grid_search = GridSearchCV(estimator=regressor,
    #                            param_grid=parameters,
    #                            cv=tscv)
    # grid_search = grid_search.fit(X, y.ravel())
    # best_accuracy = grid_search.best_score_
    # best_parameters = grid_search.best_params_
    #
    # print(best_accuracy)
    # print(best_parameters)

    METRIC_KEYS_SDK4ED = ['bugs', 'duplicated_blocks', 'code_smells',
                          # 'comment_lines', 'ncloc', 'uncovered_lines', 'vulnerabilities', 'complexity',
                          'sqale_index', 'reliability_remediation_effort', 'security_remediation_effort']
    results = {}
    results['_info'] = {'metrics': METRIC_KEYS_SDK4ED, 'window_size': WINDOW_SIZE, 'versions_ahead': VERSIONS_AHEAD}

    # def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    #     """
    #     Frame a time series as a supervised learning dataset.
    #     Arguments:
    #         data: Sequence of observations as a list or NumPy array.
    #         n_in: Number of lag observations as input (X).
    #         n_out: Number of observations as output (y).
    #         dropnan: Boolean whether or not to drop rows with NaN values.
    #     Returns:
    #         Pandas DataFrame of series framed for supervised learning.
    #     """
    #     n_vars = 1 if type(data) is list else data.shape[1]
    #     df = pd.DataFrame(data)
    #     cols, names = list(), list()
    #     # input sequence (t-n, ... t-1)
    #     for i in range(n_in, 0, -1):
    #         cols.append(df.shift(i))
    #         names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    #     # forecast sequence (t, t+1, ... t+n)
    #     for i in range(0, n_out):
    #         cols.append(df.shift(-i))
    #         if i == 0:
    #             names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
    #         else:
    #             names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    #     # put it all together
    #     agg = pd.concat(cols, axis=1)
    #     agg.columns = names
    #     # drop rows with NaN values
    #     if dropnan:
    #         agg.dropna(inplace=True)
    #     return agg

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

            neurons = [200]

            model = Sequential()
            model.add(LSTM(250, input_shape=(10, 1), kernel_initializer='normal', activation='relu'))
            model.add(Dense(1, kernel_initializer='normal'))
            # Compile model
            model.compile(loss='mean_squared_error', optimizer='adam')
            return model

        # Create the regressor model
        name_project = project.replace('data/', '')
        if reg_type == 'LinearRegression':
            # Fitting Multiple Linear Regression to the Training set
            from sklearn.linear_model import LinearRegression
            regressor = LinearRegression()
            pipeline = Pipeline([('regressor', regressor)])
        if reg_type == 'LassoRegression':
            # Fitting Lasso Regression to the Training set
            from sklearn.linear_model import Lasso
            alpha =  df_params[(df_params['project'] == name_project) & (df_params['model'] == reg_type) & (df_params['parameter'] == 'alpha')]['value']
            regressor = Lasso(alpha=int(alpha))
            pipeline = Pipeline([('regressor', regressor)])
        if reg_type == 'RidgeRegression':
            # Fitting Ridge Regression to the Training set
            from sklearn.linear_model import Ridge
            alpha = df_params[(df_params['project'] == name_project) & (df_params['model'] == reg_type) & (
                        df_params['parameter'] == 'alpha')]['value']
            # print("alpha"+ str(alpha))
            regressor = Ridge(alpha=int(alpha))
            pipeline = Pipeline([('regressor', regressor)])
        if reg_type == 'SGDRegression':
            # Fitting SGD Regression to the Training set
            from sklearn.linear_model import SGDRegressor
            max_iter = df_params[(df_params['project'] == name_project) & (df_params['model'] == reg_type) & (
                        df_params['parameter'] == 'max_iter')]['value']

            # print(max_iter)
            regressor = SGDRegressor(max_iter=int(max_iter), tol=1e-3)
            pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
        elif reg_type == 'SVR_linear':
            # Fitting linear SVR to the dataset
            from sklearn.svm import SVR
            C = df_params[(df_params['project'] == name_project) & (df_params['model'] == reg_type) & (
                    df_params['parameter'] == 'C')]['value']
            regressor = SVR(kernel='linear', C=int(C))
            pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
        elif reg_type == 'SVR_rbf':
            # Fitting SVR to the dataset
            from sklearn.svm import SVR
            gamma = df_params[(df_params['project'] == name_project) & (df_params['model'] == reg_type) & (
                    df_params['parameter'] == 'gamma')]['value']
            C = df_params[(df_params['project'] == name_project) & (df_params['model'] == reg_type) & (
                    df_params['parameter'] == 'C')]['value']
            # print("C = df_params[(df_params['project'] == " + name_project + ") & (df_params['model'] == " + reg_type + ") & (df_params['parameter'] == 'C')]['value']")
            regressor = SVR(kernel='rbf', gamma=float(gamma), C=int(C))
            pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
        elif reg_type == 'RandomForestRegressor':
            # Fitting Random Forest Regression to the dataset
            from sklearn.ensemble import RandomForestRegressor
            n_estimators = df_params[(df_params['project'] == name_project) & (df_params['model'] == reg_type) & (
                    df_params['parameter'] == 'n_estimators')]['value']
            random_state = df_params[(df_params['project'] == name_project) & (df_params['model'] == reg_type) & (
                    df_params['parameter'] == 'random_state')]['value']
            max_depth = df_params[(df_params['project'] == name_project) & (df_params['model'] == reg_type) & (
                    df_params['parameter'] == 'max_depth')]['value']
            # print("n_est: " + n_estimators)
            # print("max_d: ")
            # print(max_depth)
            if max_depth.empty:
                regressor = RandomForestRegressor(n_estimators=int(n_estimators), random_state=0)
            else:
                regressor = RandomForestRegressor(n_estimators=int(n_estimators), random_state=0, max_depth=int(max_depth))
            pipeline = Pipeline([('regressor', regressor)])
        elif reg_type == 'ANN':
            # Fitting ANN to the dataset
            from keras.wrappers.scikit_learn import KerasRegressor
            regressor = KerasRegressor(build_fn=baseline_model, epochs=1000, batch_size=5, verbose=True)
            pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
        elif reg_type == 'LSTM':
            from keras.wrappers.scikit_learn import KerasRegressor
            regressor = KerasRegressor(build_fn=lstm_model, epochs=1000, batch_size=5, verbose=True)
            # pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
            pipeline = Pipeline([('regressor', regressor)])

        from sklearn.model_selection import TimeSeriesSplit

        n_splits = 5
        list_split4 = ['data/apache_ofbiz_measures', 'data/apache_nifi_measures', 'data/apache_incubator_dubbo_measures',
                       'data/square_retrofit_measures', 'data/spring-projects_spring-boot_measures', 'data/java_websocket_measures',
                       'data/zxing_zxing_measures', 'data/igniterealtime_openfire_measures']

        if project in list_split4:
            n_splits = 4
        tscv = TimeSeriesSplit(n_splits=n_splits)

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

        # print(X.shape)
        # print(Y.shape)
        # if reg_type == 'LSTM':
        #     scores = score_lstm(estimator=pipeline, X=X, y=Y.ravel(), scoring=scorer, cv=tscv)
        # else:
        scores = cross_validate(estimator=pipeline, X=X, y=Y.ravel(), scoring=scorer, cv=tscv, return_train_score=False)

        # Fill results dict object
        for key, value in scores.items():
            # print(key, value)
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

            # Importing the dataset
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
            data = data.drop(columns=['var4(t-2)', 'var4(t-1)'])

            # Set X, Y
            X = data.iloc[:, data.columns != 'forecasted_total_principal'].values

            # Y = data.iloc[:, data.columns == 'forecasted_total_principal'].values
            Y = data.forecasted_total_principal.values
            # for i in range(len(X)):
            #     print(X[i], Y[i])
            for reg_type in ['LinearRegression', 'LassoRegression', 'RidgeRegression', 'SGDRegression', 'SVR_rbf', 'SVR_linear', 'RandomForestRegressor']:
            # for reg_type in ['LinearRegression']:
                regressor = create_regressor(reg_type, X, Y, project, versions_ahead)

    print_forecasting_errors(VERSIONS_AHEAD, reg_type, results, versions_ahead, DATASET)


# def read_dataset(VERSIONS_AHEAD, WINDOW_SIZE):
#     return read_td_dataset(VERSIONS_AHEAD, WINDOW_SIZE)


# def read_td_dataset(VERSIONS_AHEAD, WINDOW_SIZE):
#     METRIC_KEYS_SDK4ED = ['bugs', 'duplicated_blocks', 'code_smells',
#                           # 'comment_lines', 'ncloc', 'uncovered_lines', 'vulnerabilities', 'complexity',
#                           'sqale_index', 'reliability_remediation_effort', 'security_remediation_effort']
#
#     # Importing the dataset
#     dataset = pd.read_csv('apache_kafka_measures.csv', sep=";", usecols=METRIC_KEYS_SDK4ED)
#     dataset['total_principal'] = dataset['reliability_remediation_effort'] + dataset['security_remediation_effort'] + \
#                                  dataset['sqale_index']
#     dataset = dataset.drop(columns=['sqale_index', 'reliability_remediation_effort', 'security_remediation_effort'])
#     # Adding time-shifted prior and future period
#     data = series_to_supervised(dataset.values, n_in=WINDOW_SIZE)
#     # Append dependend variable column with value equal to next version's total_principal
#     data['forecasted_total_principal'] = data['var4(t)'].shift(-VERSIONS_AHEAD)
#     data = data.drop(data.index[-VERSIONS_AHEAD:])
#     # Include/remove TD as independent variable
#     data = data.drop(columns=['var4(t-2)', 'var4(t-1)'])
#     X = data.iloc[:, data.columns != 'forecasted_total_principal'].values
#     Y = data.iloc[:, data.columns == 'forecasted_total_principal'].values
#     return X, Y, METRIC_KEYS_SDK4ED

def read_refactoring_dataset():
    try:
        mydb = connection.connect(host="localhost", database='Student', user="root", passwd="root", use_pure=True)
        query = "Select * from studentdetails;"
        result_dataFrame = pd.read_sql(query, mydb)
        mydb.close()  # close the connectionexcept Exception as e:
        mydb.close()
        # print(str(e))
    except:
        print(sys.exc_info())

def print_forecasting_errors(VERSIONS_AHEAD, reg_type, results, versions_ahead, project_list):
    df_res = pd.DataFrame(columns=['Model', 'Horizon', 'apache_groovy', 'apache_incubator_dubbo', 'apache_kafka', 'apache_nifi', 'apache_ofbiz', 'apache_systemml',
                          'commonsio', 'google_guava', 'igniterealtime_openfire', 'java_websocket', 'jenkinsci_jenkins',
                          'spring-projects_spring-boot', 'square_okhttp', 'square_retrofit', 'zxing_zxing'])

    for project in project_list:
        # print('**************** %s ****************' % project)
        # print(results[project][versions_ahead][reg_type])
        for reg_type in ['LinearRegression', 'LinearRegression', 'LassoRegression', 'RidgeRegression', 'SGDRegression', 'SVR_rbf',
                         'SVR_linear', 'RandomForestRegressor']:
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

                print('%d: %0.3f;%0.3f;%0.3f;%0.3f' % (versions_ahead, abs(mae_mean), abs(rmse_mean), abs(mape_mean), r2_mean))
                project_name = project.replace('data/', '').replace('_measures', '')
                filterinDataframe = df_res[(df_res['Model'] == reg_type) & (df_res['Horizon'] == versions_ahead)]
                if not filterinDataframe.empty:
                    df_res.loc[(df_res['Model'] == reg_type) & (df_res['Horizon'] == versions_ahead), project_name] = abs(mape_mean)

                else:
                    res = {'Model': reg_type, 'Horizon': versions_ahead, project_name: abs(mape_mean)}
                    df_res.loc[len(df_res)] = res

    df_res['mean'] = df_res.iloc[:, 2:17].mean(axis=1)

    original_study = [3.47, 1.39, 1.44, 3.19, 6.97, 2.04, 5.18, 8.62, 4.11, 3.91, 7.45, 12.92, 6.76, 5.94, 18.39, 9.24, 8.34, 15.80, 17.01, 15.03, 8.00, 18.65, 10.57, 10.66, 15.56, 21.93, 14.01, 7.38, 10.56, 8.61, 9.34, 8.98, 11.18, 8.31, 5.94]
    df_res['original_study'] = original_study
    df_res.to_csv('mape.csv')
    # plot_results()



if __name__ == '__main__':
    # Selecting indicators that will be used as model variables

    # 'AMC', 'WMC', 'DIT', 'NOC', 'RFC', 'CBO', 'Ca', 'Ce', 'CBM', 'IC', 'LCOM', 'LCOM3', 'CAM', 'NPM', 'DAM', 'MOA']
    # 'Security Index', 'blocker_violations', 'critical_violations', 'major_violations', 'minor_violations', 'info_violations']

    DATASET = ['data/apache_kafka_measures', 'data/apache_groovy_measures', 'data/apache_incubator_dubbo_measures', 'data/apache_nifi_measures',
                    'data/apache_ofbiz_measures', 'data/apache_systemml_measures', 'data/commonsio_measures', 'data/google_guava_measures', 'data/igniterealtime_openfire_measures'
                    , 'data/java_websocket_measures', 'data/jenkinsci_jenkins_measures', 'data/spring-projects_spring-boot_measures'
                    , 'data/square_okhttp_measures', 'data/square_retrofit_measures', 'data/zxing_zxing_measures']

    # DATASET = ['data/igniterealtime_openfire_measures']


    WINDOW_SIZE = 2  # choose based on error minimization for different forecasting horizons
    VERSIONS_AHEAD = [1, 5, 10, 20, 40]

    # testing parameters
    # alpha = df_params[
    #     (df_params['project'] == 'zxing_zxing_measures') & (df_params['model'] == 'SGDRegression') & (df_params['parameter'] == 'max_iter')][
    #     'value']
    # print(alpha)

    # max_iter = df_params[(df_params['project'] == 'apache_kafka_measures') & (df_params['model'] == 'SGDRegression') & (
    #             df_params['parameter'] == 'max_iter')]['value']
    # print(max_iter)

    # C = df_params[(df_params['project'] == 'apache_kafka_measures') & (df_params['model'] == 'SVR_rbf') & (
    #             df_params['parameter'] == 'C')]['value']
    # print(C)

    # gamma = df_params[(df_params['project'] == 'apache_kafka_measures') & (df_params['model'] == 'SVR_rbf') & (df_params['parameter'] == 'gamma')]['value']
    # print(gamma)
    main(DATASET, WINDOW_SIZE, VERSIONS_AHEAD)

    plot_results()
