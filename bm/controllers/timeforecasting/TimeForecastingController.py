#  Copyright (c) 2022. Slonos Labs. All rights Reserved.
import math
import pickle
from _socket import herror
from datetime import datetime, timedelta
from random import randint

import dateutil
import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import xlabel
from pandas import DataFrame, Series
#from scipy.linalg import pinv2
#import scipy.linalg.pinv2
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from datetime import datetime
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller

from app.base.db_models.ModelForecastingResults import ModelForecastingResults

from app import db
from app.base.db_models.ModelProfile import ModelProfile
from bm.datamanipulation.AdjustDataFrame import convert_data_to_sample
from bm.db_helper.AttributesHelper import add_features, add_labels, add_api_details, update_api_details_id, add_forecasting_results
from bm.utiles.CVSReader import get_only_file_name
from bm.utiles.Helper import Helper
import scipy.stats as stats





class TimeForecastingController:
    test_value = ''

    def __init__(self):
        self.test_value = '_'

    def analyize_dataset(self, file_location):

        # 1- is there any date column
        ds = pd.read_csv(file_location)
        forecasting_columns_arr = []
        depended_columns_arr = []
        datetime_columns_arr = []
        for col in ds.columns:
            if ds[col].dtype == 'object':
                try:
                    #ds[col] = pd.to_datetime(ds[col], dayfirst=True, format="%d/%m/%Y %H:%M")
                    ds[col] = pd.to_datetime(ds[col], yearfirst=True, format="%Y/%m/%d")
                    datetime_columns_arr.append(col)
                except ValueError:
                    forecasting_columns_arr.append(col)
                    pass
            elif (ds[col].dtype == 'float64' or ds[col].dtype == 'int64'):
                depended_columns_arr.append(col)
            else:
                forecasting_columns_arr.append(col)

        # 2- Suggested forcasting values
        return forecasting_columns_arr, depended_columns_arr, datetime_columns_arr;


    def create_forecating_model_(self, csv_file_location, forecasting_factor, depended_factor, time_factor):
        try:
            # Prepare training and testing data
            data = pd.read_csv(csv_file_location, usecols=[forecasting_factor, depended_factor, time_factor])

            # Prepare training and testing data
            data[time_factor] = pd.to_datetime(data[time_factor])
            data = data.set_index(time_factor)
            data = data.sort_index()
            forecasting_categories = numpy.array(pd.Categorical(data[forecasting_factor]).categories)
            forecasting_category = forecasting_categories[0]
            forecasting_dataframe = data[[forecasting_factor, depended_factor]].copy()
            forecasting_dataframe = forecasting_dataframe[
                forecasting_dataframe[forecasting_factor] == forecasting_category].copy()
            forecasting_dataframe = forecasting_dataframe.drop(forecasting_factor, 1)

            if not forecasting_dataframe.index.is_unique:  # Remove duplicated index
                forecasting_dataframe = forecasting_dataframe.loc[~forecasting_dataframe.index.duplicated(), :]
            # forecasting_dataframe = forecasting_dataframe.asfreq('MS') #--- To be uncomment later
            # forecasting_dataframe = forecasting_dataframe.astype(float) #--- To be uncomment later
            forecasting_dataframe[time_factor] = forecasting_dataframe.index

            X = forecasting_dataframe[time_factor].values
            y = forecasting_dataframe[depended_factor].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

            # Fitting Random Forest Regression to the dataset
            regressor = RandomForestRegressor(n_estimators=10, random_state=0)
            regressor.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

            y_pred = regressor.predict(X_test.reshape(-1, 1))

            # # Visualising the Random Forest Regression Results
            # Prepare thick of X axis
            last_3_months = ['0 days', '18 days', '36 days', '58 days', '72 days',
                             '90 days']  # pd.DatetimeIndex(forecasting_dataframe[time_factor]).month
            # last_3_months = last_3_months.drop_duplicates()
            # no_of_intervals = math.floor(100 / len(last_3_months))
            interval_thick = [0, 20, 40, 60, 80, 100]
            # for i in range(len(last_3_months)):
            # interval_thick.append(i * no_of_intervals)

            X = pd.to_datetime(X)
            X_grid = np.arange(X.min(), X.max(), dtype='datetime64[h]')
            X_grid = X_grid.reshape((len(X_grid), 1))
            plt.plot(y_test, color='blue', label="Data")
            plt.plot(y_pred, color='red', label="Prediction")
            plt.title('Random Forest Regression')
            plt.xticks(interval_thick, last_3_months, rotation='vertical')
            plt.xlabel(time_factor.replace('_', ' '))
            plt.legend(loc='best')
            plt.ylabel(depended_factor.replace('_', ' '))
            plt.show()

            return forecasting_category, X_test, y_test, y_pred;
        except  Exception as e:
            print('Ohh -get_model_status...Something went wrong.')
            print(e.__str__())

    def relu(self, x):  # hidden layer activation function
        return np.maximum(x, 0, x)

    def hidden_nodes(self, X, input_weights, biases):
        G = np.dot(X, input_weights)
        G = G + biases
        H = self.relu(G)
        return H

    def predict(self, X, input_weights, biases, output_weights):
        out = self.hidden_nodes(X, input_weights, biases)
        out = np.dot(out, output_weights)
        return out

    def parser(x):
        return datetime.strptime(x, "%Y-%m-%d")

    def get_useful_model_info(self, data: DataFrame):
        df_start_date = data.first_valid_index()
        df_start_date = pd.to_datetime(df_start_date)
        df_end_date = data.last_valid_index()
        df_end_date = pd.to_datetime(df_end_date)
        forecasting_start_date = df_end_date - timedelta(days=90)  # last three month (90 days)
        training_end_date = forecasting_start_date - timedelta(days=1)
        training_data_rows = data.loc[pd.to_datetime(data.index.values) <= training_end_date]
        forecasting_dates = pd.to_datetime(data.index[pd.to_datetime(data.index.values) > training_end_date]).tolist()
        # forecasting_dates_freq = pd.date_range(forecasting_dates[0], forecasting_dates[-1], freq='M')
        optimized_forecasting_dates_freq = Helper.previous_n_months(4)
        # for i in forecasting_dates_freq:
        #    str_1 = str(i)
        #    optimized_forecasting_dates_freq.append(str_1[0:10])
        # forecasting_dates = forecasting_dates.year
        # months_list = Helper.previous_n_months(4)
        # print(months_list)

        return len(training_data_rows.index), forecasting_dates, optimized_forecasting_dates_freq

    def get_forecasting_results(self):
        try:
            forecasting_results = ModelForecastingResults.query.all()
            period_dates=[]
            actual=[]
            predicted=[]
            for profile in forecasting_results:
                predicted.append(profile.predicted)
                actual.append(profile.actual)
                period_dates.append(profile.period_dates)
            return actual, predicted, period_dates
        except  Exception as e:
            print('Ohh -get_model_status...Something went wrong.')
            print(e)
            return 0

