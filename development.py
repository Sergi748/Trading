# -*- coding: utf-8 -*-

import os
import joblib
import datetime 
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import model_from_json


class trainingModel():
    
    '''
    Class to make a complete trading model.
    The required arguments are:
        idYahoo: this is the market id in Yahoo finance
        pathProject: path where the model and predictions will be saved
        startDate: Start date to train model
        endDate: End date to train model
    '''
       
    np.random.seed(1234)
    
    def __init__(self, idYahoo, pathProject, varPredict, startHistorical, endHistorical):
    
        self.idYahoo = idYahoo
        self.pathProject = pathProject
        self.varPredict = varPredict
        self.startHistorical = startHistorical
        self.endHistorical = endHistorical
        self.df = pd.DataFrame()
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.prediction = pd.DataFrame()
        self.model = ""
        self.scaler = ""


    def __checkPath(self):    
        if os.path.exists(self.pathProject) == False:
            os.mkdir(self.pathProject)


    def load_data(self):
        
        '''
        idYahoo: Indicated of company from yahoo finance
        startHistorical: Start date of the history in format 'yyyy-MM-dd'
        endHistorical: End date of the history in format 'yyyy-MM-dd'
        
        return: dataframe with data history
        '''
        
        trainingModel.__checkPath(self)
        yahoo_financials = YahooFinancials(self.idYahoo)
    
        data = yahoo_financials.get_historical_price_data(start_date=self.startHistorical, 
                                                          end_date=self.endHistorical, 
                                                          time_interval='daily')
        
        df = pd.DataFrame(data[self.idYahoo]['prices'])
        df['Date'] = df.formatted_date
        df = df.drop(['date', 'formatted_date'], axis=1)
        # convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')
        df = df.drop_duplicates()
    
        # sort by datetime
        df.sort_values(by='Date', inplace=True, ascending=True)
    
        # df = pd.read_csv('C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Trading/datos_NDAQ.csv', sep=';')
    
        self.df = df
    
    
    def plot_data(self):
        
        '''
        df: Dataset to plot
        '''
        
        plt.figure(figsize = (16,8))
        plt.plot(self.df['Date'], self.df[self.varPredict], label = self.varPredict + ' Price History')
        plt.legend(loc = "upper left")
        plt.xlabel('Year')
        plt.ylabel('Stock Price ($)')
        plt.savefig(self.pathProject + '/plot_data_' + self.varPredict + '.png')
        plt.show()
    
    
    def split_data(self, size):
        
        '''
        df: Dataset for split into train and test
        size: Size to split
        
        return: train and test datasets
        '''
        
        training_size = 1 - size
        
        test_num = int(size * len(self.df))
        train_num = int(training_size * len(self.df))
        
        train = self.df[:train_num][['Date', self.varPredict]]
        test = self.df[train_num:][['Date', self.varPredict]]
    
        self.train = train
        self.test = test
    
    
    def scaler_split_train_test(self):
    
        '''
        train: dataset created with split_data function
        
        return X_train and y_train to training model.
        '''
        
        def _get_x_y(data, N, offset):
            
            """
            Split data into input variable (X) and output variable (y)
            
            INPUT:
            data - dataset to be splitted
            N - time frame to be used
            offset - position to start the split
            
            OUTPUT:
            X - input variable
            y - output variable
            """
            X, y = [], []
            
            for i in range(offset, len(data)):
                X.append(data[i-N:i])
                y.append(data[i])
            
            X = np.array(X)
            y = np.array(y)
            
            return X, y
        
        # scale our dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.df[[self.varPredict]])
        scaled_data_train = scaled_data[:self.train.shape[0]]
    
        # we use past 60 days stock prices for our training to predict 61th day's closing price.
        X_train, y_train = _get_x_y(scaled_data_train, 60, 60)
        
        scaler_filepath = self.pathProject + '/scaler_' + self.idYahoo + '_' + self.varPredict + '.pkl'
        joblib.dump(scaler, scaler_filepath)
        
        self.X_train = X_train
        self.y_train = y_train
        self.scaler = scaler
    
    
    def fit_lstm(self, lstm_units = 50, optimizer = 'adam', epochs = 1, batch_size = 1):
        
        '''
        X_train: Dataset with information on the 60 days previouse to the day to be predicted
        y_train: Value of day to be predicted
        
        return: model
        '''
        
        
        model = Sequential()
        model.add(LSTM(units = lstm_units, return_sequences = True, input_shape = (self.X_train.shape[1],1)))
        model.add(LSTM(units = lstm_units))
        model.add(Dense(1))
    
        # Compile and fit the LSTM network
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        model.fit(self.X_train, self.y_train, epochs = epochs, batch_size = batch_size, verbose = 2)
        
        self.model = model
    
    
    def pred_closing_price(self):
        
        '''
        df: Dataset with all information
        test: Test dataset
        scaler: function to scale
        model: Model of lstm made previously
        
        return: dataset with prediction
        '''
        
        # predict stock prices using past 60 stock prices
        inputs = self.df[self.varPredict][len(self.df) - len(self.test) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs = self.scaler.transform(inputs)
        
        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i-60:i,0])
        X_test = np.array(X_test)
        
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        closing_price = self.model.predict(X_test)
        closing_price = self.scaler.inverse_transform(closing_price)
        
        testNew = self.test.copy()
        varName = 'Predictions_lstm_' + self.varPredict
        testNew[varName] = closing_price
        
        self.prediction = testNew
    
    
    def plot_train_prediction(self):
        
        plt.figure(figsize = (20,10))
        plt.xlabel('Year')
        plt.ylabel('Stock Price ($)')
        plt.plot(self.train['Date'], self.train[self.varPredict], label = self.varPredict + ' Price History [train data]')
        plt.plot(self.prediction['Date'], self.prediction[self.varPredict], label = self.varPredict + ' Price History [test data]')
        plt.plot(self.prediction['Date'], self.prediction['Predictions_lstm_' + self.varPredict], label = self.varPredict + ' Price - Predicted')
        plt.legend(loc = "upper left")
        plt.savefig(self.pathProject + '/plot_data_train_prediction_' + self.varPredict + '.png')
        plt.show()


    def metrics(self):
        
        '''
        Compute Mean Absolute Percentage Error (MAPE)
        Compute Root Mean Squared Error (RMSE)
        INPUT:
        y_true - actual variable
        y_pred - predicted variable
        OUTPUT:
        mape - Mean Absolute Percentage Error (%)
        rmse - Root Mean Squared Error
        '''
    
        y_true = self.prediction[self.varPredict]
        y_pred = self.prediction['Predictions_lstm_' + self.varPredict]
        
        y_true_mape, y_pred_mape = np.array(y_true), np.array(y_pred)
        mape = np.mean(np.abs((y_true_mape - y_pred_mape) / y_true_mape)) * 100
        rmse = np.sqrt(np.mean(np.power((y_true - y_pred), 2)))
    
        print('Root Mean Squared Error: {0} \nMean Absolute Percentage Error (%): {1}'.format(rmse, mape))
        
        return mape, rmse
        

    def saveModel(self):
        
        '''
        Trained model saved in pathProject.
        '''
        
        model_json = self.model.to_json()
        model_filepath = self.pathProject + '/model_' + self.idYahoo + '_' + self.varPredict + '.json'
        with open(model_filepath, 'w') as json_file:
            json_file.write(model_json)
        saveWeights = self.pathProject + '/model_' + self.idYahoo + '_' + self.varPredict + '.h5'
        self.model.save_weights(saveWeights)
            
    
class predictions():
    
    def __init__(self, idYahoo, pathProject, varPredict, datePred):
        self.idYahoo = idYahoo
        self.pathProject = pathProject
        self.varPredict = varPredict
        self.datePred = datePred
        self.model = ''
        self.df = pd.DataFrame()
        self.dfPred = pd.DataFrame()

    def __load_model(self):
        
        '''
        Load the trained model saved in pathProject.
        '''

        nameModel = 'model_' + self.idYahoo + '_' + self.varPredict + '.json'
        fileComplete = self.pathProject + '/' + nameModel
        # load json and create model
        json_file = open(fileComplete, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        fileH5 = self.pathProject + '/model_' + self.idYahoo + '_' + self.varPredict + '.h5'
        loaded_model.load_weights(fileH5)
    
        self.model = loaded_model
    
    def __load_data_pred(self):
        
        '''
        Function to load new data from idYahoo to predict.
        '''
        
        dateInit = str(datetime.datetime.strptime(self.datePred, '%Y-%m-%d') - datetime.timedelta(days=150))[0:10]
        
        yahoo_financials = YahooFinancials(self.idYahoo)
        data = yahoo_financials.get_historical_price_data(start_date=dateInit, 
                                                          end_date=self.datePred, 
                                                          time_interval='daily')
        
        df = pd.DataFrame(data[self.idYahoo]['prices'])
        df['Date'] = df.formatted_date
        df = df.drop(['date', 'formatted_date'], axis=1)
        # convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')
        df = df.drop_duplicates()
        
        # sort by datetime
        df.sort_values(by='Date', inplace=True, ascending=True)
        
        self.df = df

    
    def pred_closing_price(self):
        
        '''
        Develop to prediction.
        '''

        predictions.__load_data_pred(self)
        predictions.__load_model(self)
        
        scaler_name = self.pathProject + '/scaler_' + self.idYahoo + '_' + self.varPredict + '.pkl'
        scaler = joblib.load(scaler_name)
        
        inputs = self.df[self.varPredict][len(self.df) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs = scaler.transform(inputs)
    
        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i-60:i,0])
        X_test.append(inputs[-60:,0])
    
        X_test = np.array(X_test)
    
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        closing_price = self.model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)
        predicted_price = float(closing_price[-1])        
        print('Prediction price date {0}: {1}'.format(self.datePred, predicted_price))

        self.dfPred = predicted_price
              
        fileSummary = self.pathProject + '/predictionsMade_' + self.varPredict + '.csv'
        if os.path.exists(fileSummary):
            dfSummary = pd.read_csv(fileSummary, sep=';')
            dfSummary.loc[len(dfSummary)] = [self.datePred, predicted_price]
            dfSummary.to_csv(fileSummary, index=False, sep=';')
        else:
            dfSummary = pd.DataFrame(columns=['dateToPredict', 'prediction_' + self.varPredict], data=[[self.datePred, predicted_price]])
            dfSummary.to_csv(fileSummary, index=False, sep=';')


class checkMetrics():
    
    def __init__(self, idYahoo, pathProject, varPredict):
        self.idYahoo = idYahoo
        self.pathProject = pathProject
        self.varPredict = varPredict
        self.df = pd.DataFrame()


    def metricsPredictions(self):
        
        '''
        Function for evaluating predictions.
        Only two arguments are required:
            idYahoo: this is the market id in Yahoo finance
            pathProject: path where the model and predictions will be saved            
        This function returns a csv file with the prediction and the actual value of each date.
        The csv file is saved in pathProject.
        '''
        
        filePrediction = self.pathProject + '/predictionsMade_' + self.varPredict + '.csv'
        dfPredictions = pd.read_csv(filePrediction, sep=';').rename(columns={'dateToPredict':'Date'})
        
        startDateCheck = dfPredictions.Date.min()
        endDateCheck = str(datetime.datetime.strptime(dfPredictions.Date.max(), '%Y-%m-%d') + datetime.timedelta(days=1))[0:10]
        
        yahoo_financials = YahooFinancials(self.idYahoo)
        data = yahoo_financials.get_historical_price_data(start_date=startDateCheck, 
                                                          end_date=endDateCheck, 
                                                          time_interval='daily')
        
        dfCheck = pd.DataFrame(data[self.idYahoo]['prices'])
        dfCheck['Date'] = dfCheck.formatted_date
        dfCheck = dfCheck[['Date', self.varPredict]]
        df = pd.merge(dfCheck, dfPredictions, on='Date', how='left')
        
        y_true = df[self.varPredict]
        y_pred = df['prediction_' + self.varPredict]
        
        y_true_mape, y_pred_mape = np.array(y_true), np.array(y_pred)
        mape = np.mean(np.abs((y_true_mape - y_pred_mape) / y_true_mape)) * 100
        rmse = np.sqrt(np.mean(np.power((y_true - y_pred), 2)))
        
        print('Root Mean Squared Error: {0} \nMean Absolute Percentage Error (%): {1}'.format(rmse, mape))
        
        # Saved dataset
        fileName = self.pathProject + '/valueReal_predictions_' + self.varPredict + '.csv'
        df.to_csv(fileName, sep=';', index=False)
        
        # plot
        plt.figure(figsize = (20,10))
        plt.xlabel('Year')
        plt.ylabel('Stock Price ($)')
        plt.plot(df['Date'], df[self.varPredict], label = self.varPredict + ' Price History')
        plt.plot(df['Date'], df['prediction_' + self.varPredict], label = self.varPredict + ' Price - Predicted')
        plt.legend(loc = "upper left")
        plt.savefig(self.pathProject + '/plot_prediction_' + self.varPredict + '.png')
        plt.show()
                
        self.df = df
        