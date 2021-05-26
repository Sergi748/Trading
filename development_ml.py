# -*- coding: utf-8 -*-

import os
import math
import pickle
import joblib
import datetime
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from scipy import stats
from pylab import rcParams
from tqdm.notebook import tqdm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from yahoofinancials import YahooFinancials
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


class trainingModel_ml():

    def __init__(self, idYahoo, path, varPredict, startHistorical, endHistorical):
    
        self.idYahoo = idYahoo
        self.path = path
        self.varPredict = varPredict
        self.startHistorical = startHistorical
        self.endHistorical = endHistorical
        self.cols = [self.varPredict, 'range_hl', 'range_oc', 'volume']
        self.merging_keys = ['order_day']
        self.df = pd.DataFrame()
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.testPred = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_train_scaled = pd.DataFrame()
        self.y_train_scaled = pd.DataFrame()
        self.X_test_scaled = pd.DataFrame()
        self.prediction = pd.DataFrame()
        self.dfMetricsTrainTest = pd.DataFrame()
        self.models = []
        self.scaler = ""
             
            
    def __createDirectory(self):
        
        listDir = ['output', 'governance']
        for i in listDir:
            if os.path.exists(self.path + '/' + self.idYahoo + '/' + i) == False:
                os.makedirs(self.path + '/' + self.idYahoo + '/' + i)

        listDirOut = ['modelos', 'metricas', 'predicciones', 'graficas']
        for i in listDirOut:
            if os.path.exists(self.path + '/' + self.idYahoo + '/output/' + i) == False:
                os.makedirs(self.path + '/' + self.idYahoo + '/output/' + i)

            
    def __saveModels(self, models, scaler, path, idYahoo):

        namesModels = ['xgb', 'lgb', 'lr', 'GradBoost', 'RandForest']
        for i, model in enumerate(models):    
            fileName = self.path + '/' + self.idYahoo + '/output/modelos/model_' + self.varPredict + '_' + namesModels[i] + '_ml.pkl'
            pickle.dump(model, open(fileName, 'wb'))

        scaler_filepath = self.path + '/' + self.idYahoo + '/output/modelos/scaler_' + self.varPredict + '_ml.pkl'
        joblib.dump(scaler, scaler_filepath)


    def load_data(self):
        
        '''
        idYahoo: Indicated of company from yahoo finance
        startHistorical: Start date of the history in format 'yyyy-MM-dd'
        endHistorical: End date of the history in format 'yyyy-MM-dd'
        
        return: dataframe with data history
        '''
        
        def __create_new_vars(df):

            # Get difference between high and low of each day and get difference between open and close of each day
            df['range_hl'] = df['high'] - df['low']
            df['range_oc'] = df['open'] - df['close']
            # Add a column 'order_day' to indicate the order of the rows by date
            df['order_day'] = [x for x in list(range(len(df)))]

            return df

        trainingModel_ml.__createDirectory(self)
        yahoo_financials = YahooFinancials(self.idYahoo)
    
        data = yahoo_financials.get_historical_price_data(start_date=self.startHistorical, 
                                                          end_date=self.endHistorical, 
                                                          time_interval='daily')
        
        df = pd.DataFrame(data[self.idYahoo]['prices'])
        df['Date'] = df.formatted_date
        df = df.drop(['date', 'formatted_date'], axis=1)
        # convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')
        df['month'] = df['Date'].dt.month
        df = df.drop_duplicates()
    
        # sort by datetime
        df.sort_values(by='Date', inplace=True, ascending=True)        
        df = __create_new_vars(df)
    
        self.df = df

        
    def create_datasets(self, N, test_size, plotSave):
        
        '''
        N: Number of previous records considered
        test_size: Value to divide data set into train and test
        plot: Parameter to make the distribution graph or not.

        Function to create the data sets to be used in the development of the model
        '''
        
        def __get_mov_avg_std(df, col, N):
            """
            Given a dataframe, get mean and std dev at timestep t using values from t-1, t-2, ..., t-N.
            Inputs
                df         : dataframe. Can be of any length.
                col        : name of the column you want to calculate mean and std dev
                N          : get mean and std dev at timestep t using values from t-1, t-2, ..., t-N
            Outputs
                df_out     : same as df but with additional column containing mean and std dev
            """
            mean_list = df[col].rolling(window = N, min_periods=1).mean() # len(mean_list) = len(df)
            std_list = df[col].rolling(window = N, min_periods=1).std()   # first value will be NaN, because normalized by N-1

            # Add one timestep to the predictions
            mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
            std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))

            # Append mean_list to df
            df_out = df.copy()
            df_out[col + '_mean'] = mean_list
            df_out[col + '_std'] = std_list

            return df_out
        
        def __scale_row(row, feat_mean, feat_std):
            """
            Given a pandas series in row, scale it to have 0 mean and var 1 using feat_mean and feat_std
            Inputs
                row      : pandas series. Need to scale this.
                feat_mean: mean  
                feat_std : standard deviation
            Outputs
                row_scaled : pandas series with same length as row, but scaled
            """
            # If feat_std = 0 (this happens if adj_close doesn't change over N days), 
            # set it to a small number to avoid division by zero
            feat_std = 0.001 if feat_std == 0 else feat_std

            row_scaled = (row-feat_mean) / feat_std

            return row_scaled

        def __shift_range(df, merging_keys, cols, N):

            shift_range = [x+1 for x in range(N)]

            for shift in tqdm(shift_range):
                train_shift = df[merging_keys + cols].copy()

                # E.g. order_day of 0 becomes 1, for shift = 1.
                # So when this is merged with order_day of 1 in df, this will represent lag of 1.
                train_shift['order_day'] = train_shift['order_day'] + shift

                foo = lambda x: '{}_lag_{}'.format(x, shift) if x in cols else x
                train_shift = train_shift.rename(columns=foo)

                df = pd.merge(df, train_shift, on=merging_keys, how='left') #.fillna(0)

            # Remove the first N rows which contain NaNs
            df = df[N:]    

            for col in cols:
                df = __get_mov_avg_std(df, col, N)

            return df

        def __split_X_y(df, cols, target, test_size, plotSave):

            def __split_scaled(df, test_size, cols, target):

                num_test = int(test_size*len(df))
                num_train = len(df) - num_test

                # Split into train, cv, and test
                train = df[:num_train]
                test = df[num_train:]

                cols_to_scale = [target]

                for i in range(1,N+1):
                    for col in cols:
                        cols_to_scale.append(col + '_lag_' + str(i))

                # Do scaling for train set
                # Here we only scale the train dataset, and not the entire dataset to prevent information leak
                scaler = StandardScaler().fit(train[cols_to_scale])
                train_scaled = scaler.transform(train[cols_to_scale])

                # Convert the numpy array back into pandas dataframe
                train_scaled = pd.DataFrame(train_scaled, columns=cols_to_scale)
                train_scaled[['Date', 'month']] = train.reset_index()[['Date', 'month']]

                # Do scaling for test set
                test_scaled = test[['Date']]
                for col in tqdm(cols):
                    feat_list = [col + '_lag_' + str(shift) for shift in range(1, N+1)]
                    temp = test.apply(lambda row: __scale_row(row[feat_list], row[col+'_mean'], row[col+'_std']), axis=1)
                    test_scaled = pd.concat([test_scaled, temp], axis=1)

                return train, test, train_scaled, test_scaled, scaler

            train, test, train_scaled, test_scaled, scaler =  __split_scaled(df, test_size, cols, target)

            features = []
            for i in range(1,N+1):
                for col in cols:
                    features.append(col + '_lag_' + str(i))

            # Split into X and y
            X_train = train[features]
            y_train = train[target]
            X_test = test[features]
            y_test = test[target]

            # Split into X and y
            X_train_scaled = train_scaled[features]
            y_train_scaled = train_scaled[target]
            X_test_scaled = test_scaled[features]

            if plotSave == True:
                plt.figure(figsize = (10,8))
                plt.plot(train['Date'], train[self.varPredict], label='train')
                plt.plot(test['Date'], test[self.varPredict], label='test')
                plt.legend(loc = "upper left")
                plt.xlabel('Year')
                plt.ylabel('Stock Price ($)')
                plt.savefig(self.path + '/' + self.idYahoo + '/output/graficas/plot_train_test_' + self.varPredict + '_ml.png')

            return train, test, X_train, y_train, X_test, y_test, X_train_scaled, y_train_scaled, X_test_scaled, scaler

        df_1 = __shift_range(self.df, self.merging_keys, self.cols, N)
        train, test, X_train, y_train, X_test, y_test, X_train_scaled, y_train_scaled, X_test_scaled, scaler = __split_X_y(df_1, self.cols, self.varPredict, test_size, plotSave)

        self.train = train
        self.test = test
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_scaled = X_train_scaled
        self.y_train_scaled = y_train_scaled
        self.X_test_scaled = X_test_scaled
        self.scaler = scaler
        
        
    def development_models_pred_test(self, plotTest):
        
        '''
        plotTest: Parameter to make the distribution plot of test or not
        This function develops the models and makes the predictions
        '''
        
        def __get_mape(y_true, y_pred): 
            """
            Compute mean absolute percentage error (MAPE)
            """
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 4)

        estimatorXGB = {'random_state':[22],
                        'max_depth':stats.randint(1, 100), 
                        'max_leaves':stats.randint(1, 100), 
                        'learning_rate':stats.uniform(0.1, 0.8),
                        'min_child_weight':stats.randint(1, 100),
                        'subsample':stats.uniform(0.1, 1),
                        'n_estimators':stats.randint(1, 100)}
        model_xgbRandomSearch = RandomizedSearchCV(XGBRegressor(), estimatorXGB, scoring='neg_mean_squared_error', n_jobs=5, cv=5, random_state=22).fit(self.X_train_scaled, self.y_train_scaled)

        estimatorLGB = {'random_state':[22],
                        'max_depth':stats.randint(1, 50), 
                        'num_leaves':stats.randint(1, 25), 
                        'max_leaves':stats.randint(1, 25), 
                        'learning_rate':stats.uniform(0.1, 1),
                        'min_child_weight':stats.randint(1, 50),
                        'subsample':stats.uniform(0.1, 1),
                        'n_estimators':stats.randint(1, 100)}
        model_lgbRandomSearch = RandomizedSearchCV(LGBMRegressor(), estimatorLGB, scoring='r2', n_jobs=5, cv=5, random_state=22).fit(self.X_train_scaled, self.y_train_scaled)

        estimatorLR = {'n_jobs':stats.randint(1, 5)}
        model_lrRandomSearch = RandomizedSearchCV(LinearRegression(), estimatorLR, scoring='r2', n_jobs=5, cv=5, random_state=22).fit(self.X_train_scaled, self.y_train_scaled)

        estimatorGradBoost = {'random_state':[22],
                              'n_estimators':stats.randint(1, 100),
                              'max_depth':stats.randint(1, 50), 
                              'learning_rate':stats.uniform(0.1, 1),
                              'min_weight_fraction_leaf':stats.uniform(0.1, 1),
                              'min_samples_split':stats.randint(1, 100)}
        model_GradBoostRandomSearch = RandomizedSearchCV(GradientBoostingRegressor(), estimatorGradBoost, scoring='neg_mean_squared_error', n_jobs=5, cv=5, random_state=22).fit(self.X_train_scaled, self.y_train_scaled)

        estimatorRandForest = {'random_state':[22],
                               'n_estimators':stats.randint(1, 100),
                               'max_depth':stats.randint(1, 50), 
                               'min_samples_split':stats.randint(1, 100),
                               'min_samples_leaf':stats.randint(1, 100),
                               'max_leaf_nodes':stats.randint(1, 100)}
        model_RandForestRandomSearch = RandomizedSearchCV(RandomForestRegressor(), estimatorRandForest, scoring='neg_mean_squared_error', n_jobs=5, cv=5, random_state=22).fit(self.X_train_scaled, self.y_train_scaled)

        pred_train_xgb_scaled = model_xgbRandomSearch.predict(self.X_train_scaled)
        pred_train_xgb = pred_train_xgb_scaled * math.sqrt(self.scaler.var_[0]) + self.scaler.mean_[0]
        pred_train_lgb_scaled = model_lgbRandomSearch.predict(self.X_train_scaled)
        pred_train_lgb = pred_train_lgb_scaled * math.sqrt(self.scaler.var_[0]) + self.scaler.mean_[0]
        pred_train_lr_scaled = model_lrRandomSearch.predict(self.X_train_scaled)
        pred_train_lr = pred_train_lr_scaled * math.sqrt(self.scaler.var_[0]) + self.scaler.mean_[0]
        pred_train_GradBoost_scaled = model_GradBoostRandomSearch.predict(self.X_train_scaled)
        pred_train_GradBoost = pred_train_GradBoost_scaled * math.sqrt(self.scaler.var_[0]) + self.scaler.mean_[0]
        pred_train_RandForest_scaled = model_RandForestRandomSearch.predict(self.X_train_scaled)
        pred_train_RandForest = pred_train_RandForest_scaled * math.sqrt(self.scaler.var_[0]) + self.scaler.mean_[0]

        models = [model_xgbRandomSearch, model_lgbRandomSearch, model_lrRandomSearch, model_GradBoostRandomSearch, model_RandForestRandomSearch]
        namesModels = ['xgb', 'lgb', 'lr', 'GradBoost', 'RandForest']

        def __pred_test(test, models, namesModels):

            for i, model in enumerate(models):
                var = 'pred_' + namesModels[i]
                pred = model.predict(self.X_test_scaled)
                test[var + '_scaled'] = pred
                test[var] = test[var + '_scaled'] * test[self.varPredict + '_std'] + test[self.varPredict + '_mean']
                test.drop([var + '_scaled'], axis=1, inplace=True)

            return test

        test_copy = self.test.copy()
        test_copy = __pred_test(test_copy, models, namesModels)
        varsPred = [elem for elem in test_copy.columns if elem.__contains__('pred')]
        test_copy['pred_ensamble'] = test_copy[varsPred].mean(axis=1)

        dfMetricsTrainTest = pd.DataFrame({'model':namesModels
                                           , 'RMSE':[round(math.sqrt(mean_squared_error(self.y_train, pred_train_xgb)), 3), round(math.sqrt(mean_squared_error(self.y_train, pred_train_lgb)), 3)
                                                     , round(math.sqrt(mean_squared_error(self.y_train, pred_train_lr)), 3), round(math.sqrt(mean_squared_error(self.y_train, pred_train_GradBoost)), 3)
                                                     , round(math.sqrt(mean_squared_error(self.y_train, pred_train_RandForest)), 3)]
                                           , 'MAPE (%)':[__get_mape(self.y_train, pred_train_xgb), __get_mape(self.y_train, pred_train_lgb), __get_mape(self.y_train, pred_train_lr)
                                                         , __get_mape(self.y_train, pred_train_GradBoost), __get_mape(self.y_train, pred_train_RandForest)]
                                           , 'RMSE_pred_test':[round(math.sqrt(mean_squared_error(test_copy[[self.varPredict]], test_copy[['pred_xgb']])), 3)
                                                               , round(math.sqrt(mean_squared_error(test_copy[[self.varPredict]], test_copy[['pred_lgb']])), 3)
                                                               , round(math.sqrt(mean_squared_error(test_copy[[self.varPredict]], test_copy[['pred_lr']])), 3)
                                                               , round(math.sqrt(mean_squared_error(test_copy[[self.varPredict]], test_copy[['pred_GradBoost']])), 3)
                                                               , round(math.sqrt(mean_squared_error(test_copy[[self.varPredict]], test_copy[['pred_RandForest']])), 3)]
                                           , 'MAPE_pred_test (%)':[__get_mape(test_copy[[self.varPredict]], test_copy[['pred_xgb']]), __get_mape(test_copy[[self.varPredict]], test_copy[['pred_lgb']])
                                                                   , __get_mape(test_copy[[self.varPredict]], test_copy[['pred_lr']]), __get_mape(test_copy[[self.varPredict]], test_copy[['pred_GradBoost']])
                                                                   , __get_mape(test_copy[[self.varPredict]], test_copy[['pred_RandForest']])]})

        if plotTest == True:
            rcParams['figure.figsize'] = 10, 8 # width 10, height 8
            ax = test_copy.plot(x='Date', y=[self.varPredict] + varsPred + ['pred_ensamble'], style=['g-', 'y-', 'b-'], grid=True)
            ax.legend(['test'] + varsPred + ['pred_ensamble'])
            ax.set_xlabel("Date")
            ax.set_ylabel("USD")
            ax.set_title("Zoom in to test set")

        fileSave = self.path + '/' + self.idYahoo + '/output/metricas/metrics_train_' + self.varPredict + '_ml.csv'
        dfMetricsTrainTest.to_csv(fileSave, sep=';', index=False)
        self.testPred, self.dfMetricsTrainTest, self.models = test_copy, dfMetricsTrainTest, models
        trainingModel_ml.__saveModels(self, models, self.scaler, self.path, self.idYahoo)


class predictions_ml():
    
    def __init__(self, idYahoo, path, varPredict, datePred):
        self.idYahoo = idYahoo
        self.path = path
        self.varPredict = varPredict
        self.datePred = datePred
        self.cols = [self.varPredict, 'range_hl', 'range_oc', 'volume']        
        self.merging_keys = ['order_day']
        self.models = []
        self.df = pd.DataFrame()
        self.dfScaled = pd.DataFrame()
        self.dfShift = pd.DataFrame()
        self.dfPred = pd.DataFrame()
        
        
    def __load_data_pred(self):

        def __create_new_vars(df):

            # Get difference between high and low of each day and get difference between open and close of each day
            df['range_hl'] = df['high'] - df['low']
            df['range_oc'] = df['open'] - df['close']
            # Add a column 'order_day' to indicate the order of the rows by date
            df['order_day'] = [x for x in list(range(len(df)))]

            return df

        dateInit = str(datetime.datetime.strptime(self.datePred, '%Y-%m-%d') - datetime.timedelta(days=30))[0:10]
        yahoo_financials = YahooFinancials(self.idYahoo)

        data = yahoo_financials.get_historical_price_data(start_date=dateInit, 
                                                          end_date=self.datePred, 
                                                          time_interval='daily')

        df = pd.DataFrame(data[self.idYahoo]['prices'])
        df['Date'] = df.formatted_date
        df = df.drop(['date', 'formatted_date'], axis=1)
        # We add a new row with information about date we want to predict
        df.loc[len(df)] = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, self.datePred]
        # convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')
        df['month'] = df['Date'].dt.month
        df = df.drop_duplicates()

        # sort by datetime
        df.sort_values(by='Date', inplace=True, ascending=True)        
        df = __create_new_vars(df)
                    
        self.df = df
                
            
    def __create_dateset(self, N):

        def __scale_row(row, feat_mean, feat_std):
            """
            Given a pandas series in row, scale it to have 0 mean and var 1 using feat_mean and feat_std
            Inputs
                row      : pandas series. Need to scale this.
                feat_mean: mean  
                feat_std : standard deviation
            Outputs
                row_scaled : pandas series with same length as row, but scaled
            """
            # If feat_std = 0 (this happens if adj_close doesn't change over N days), 
            # set it to a small number to avoid division by zero
            feat_std = 0.001 if feat_std == 0 else feat_std

            row_scaled = (row-feat_mean) / feat_std

            return row_scaled

        def __get_mov_avg_std(df, col, N):
            """
            Given a dataframe, get mean and std dev at timestep t using values from t-1, t-2, ..., t-N.
            Inputs
                df         : dataframe. Can be of any length.
                col        : name of the column you want to calculate mean and std dev
                N          : get mean and std dev at timestep t using values from t-1, t-2, ..., t-N
            Outputs
                df_out     : same as df but with additional column containing mean and std dev
            """
            mean_list = df[col].rolling(window = N, min_periods=1).mean() # len(mean_list) = len(df)
            std_list = df[col].rolling(window = N, min_periods=1).std()   # first value will be NaN, because normalized by N-1

            # Add one timestep to the predictions
            mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
            std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))

            # Append mean_list to df
            df_out = df.copy()
            df_out[col + '_mean'] = mean_list
            df_out[col + '_std'] = std_list

            return df_out

        def shift_range(df, merging_keys, cols, N):

            shift_range = [x+1 for x in range(N)]

            for shift in tqdm(shift_range):
                df_shift = df[merging_keys + cols].copy()

                # E.g. order_day of 0 becomes 1, for shift = 1.
                # So when this is merged with order_day of 1 in df, this will represent lag of 1.
                df_shift['order_day'] = df_shift['order_day'] + shift

                foo = lambda x: '{}_lag_{}'.format(x, shift) if x in cols else x
                df_shift = df_shift.rename(columns=foo)

                df = pd.merge(df, df_shift, on=merging_keys, how='left') #.fillna(0)

            for col in cols:
                df = __get_mov_avg_std(df, col, N)

            return df

        dfPred = shift_range(self.df, self.merging_keys, self.cols, N)
        dfPred = dfPred[N:]
        # Do scaling for predict set
        df_scaled = dfPred[['Date']]
        for col in tqdm(self.cols):
            feat_list = [col + '_lag_' + str(shift) for shift in range(1, N+1)]
            temp = dfPred.apply(lambda row: __scale_row(row[feat_list], row[col+'_mean'], row[col+'_std']), axis=1)
            df_scaled = pd.concat([df_scaled, temp], axis=1)

        features = []
        for i in range(1,N+1):
            for col in self.cols:
                features.append(col + '_lag_' + str(i))

        df_scaled = df_scaled[features]
        self.dfShift, self.dfScaled = dfPred, df_scaled
        

    def pred_closing_price(self, N):
        
        def __load_models(path, idYahoo, varPredict, namesModels):

            models = []
            for model in namesModels:
                fileName = path + '/' + idYahoo + '/output/modelos/model_' + varPredict + '_' + model + '_ml.pkl'
                loaded_model = pickle.load(open(fileName, 'rb'))
                models.append(loaded_model)

            return models
        
        predictions_ml.__load_data_pred(self)
        predictions_ml.__create_dateset(self, N)

        namesModels = ['xgb', 'lgb', 'lr', 'GradBoost', 'RandForest']
        models = __load_models(self.path, self.idYahoo, self.varPredict, namesModels)
        self.models = models
        dfPred = self.dfShift
        
        for i, model in enumerate(models):
            var = 'pred_ml_' + namesModels[i]
            pred = model.predict(self.dfScaled)
            dfPred[var + '_scaled'] = pred
            dfPred[var] = dfPred[var + '_scaled'] * dfPred[self.varPredict + '_std'] + dfPred[self.varPredict + '_mean']
            dfPred.drop([var + '_scaled'], axis=1, inplace=True)

        varsPred = [elem for elem in dfPred.columns if elem.__contains__('pred_ml')]
        dfPred['pred_ensamble_ml'] = dfPred[varsPred].mean(axis=1)
        dfPred = dfPred[dfPred.Date == self.datePred][['Date', 'pred_ensamble_ml'] + varsPred]
        dfPred.Date = dfPred.Date.astype(str)
        self.dfPred = dfPred
        
        fileSummary = self.path + '/' + self.idYahoo + '/output/predicciones/predictionsMade_' + self.varPredict + '_ml.csv'
        if os.path.exists(fileSummary):
            dfAllPreds = pd.read_csv(fileSummary, sep=';')
            dfAllPreds.loc[len(dfAllPreds)] = dfPred.values.tolist()[0]
            dfAllPreds.to_csv(fileSummary, sep=';', index=False)
        else:
            dfPred.to_csv(fileSummary, sep=';', index=False)
        