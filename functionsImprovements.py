# -*- coding: utf-8 -*-

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

class improvement():
    
    np.random.seed(1234)

    def __init__(self, df, varPredict, X_train, y_train, test, scaler):
    
        self.df = df
        self.varPredict = varPredict
        self.X_train = X_train 
        self.y_train = y_train 
        self.test = test
        self.scaler = scaler
        
    
    def __get_mape(y_true, y_pred): 
        """
        Compute Mean Absolute Percentage Error (MAPE)
        
        INPUT:
        y_true - actual variable
        y_pred - predicted variable
        
        OUTPUT:
        mape - Mean Absolute Percentage Error (%)
        
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return mape
    
    def __get_rmse(y_true, y_pred):
        """
        Compute Root Mean Squared Error (RMSE)
        
        INPUT:
        y_true - actual variable
        y_pred - predicted variable
        
        OUTPUT:
        rmse - Root Mean Squared Error
        
        """
        rmse = np.sqrt(np.mean(np.power((y_true - y_pred),2)))
                       
        return rmse
    
    # create, compile and fit LSTM netork.
    def __fit_lstm(X_train, y_train, lstm_units = 50, optimizer = 'adam', epochs = 1, 
                 batch_size = 1, loss = 'mean_squared_error'):
        
        """
        INPUT:
        X_train - training input variables (X)
        y_train - training output variable (y)
        
        default(initial) parameters chosen for LSTM
        --------------------------------------------
        lstm_units = 50
        optimizer = 'adam'
        epochs = 1
        batch_size = 1
        loss = 'mean_squared_error'
        
        OUTPUT:
        model - fitted model
        """
        
        model = Sequential()
        model.add(LSTM(units = lstm_units, return_sequences = True, input_shape = (X_train.shape[1],1)))
        model.add(LSTM(units = lstm_units))
        model.add(Dense(1))
        
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 1)
        # verbose changed to 1 to show the animated progress...

        return model
    
    # predict stock price using past 60 stock prices
    def __get_pred_closing_price(self, model):

        """
        INPUT:
        df - dataframe that has been preprocessed
        scaler - instantiated object for MixMaxScaler()
        model - fitted model
        
        OUTPUT:
        closing_price - predicted closing price using fitted model
        """
        
        inputs = self.df['close'][len(self.df) - len(self.test) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs = self.scaler.transform(inputs)
    
        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i-60:i,0])
        X_test = np.array(X_test)
    
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        closing_price = model.predict(X_test)
        closing_price = self.scaler.inverse_transform(closing_price)
        
        return closing_price
    
    
    # evaluate model performance
    def __model_performance(varPredict, test, closing_price):
        
        """
        INPUT:
        test - test dataset that contains only 'Date' & 'Close' columns (i.e.test = df[train_num:][['Date', 'Close']])
        closing_price - predicted closing price using fitted model
        
        OUTPUT:
        rmse_lstm - RMSE for LSTM
        mape_lstm - MAPE(%) for LSTM
        """
        test['Predictions_lstm_tuned'] = closing_price
        rmse_lstm = improvement.__get_rmse(np.array(test[varPredict]), np.array(test['Predictions_lstm_tuned']))
        mape_lstm = improvement.__get_mape(np.array(test[varPredict]), np.array(test['Predictions_lstm_tuned']))
        # print('Root Mean Squared Error: ' + str(rmse_lstm))
        # print('Mean Absolute Percentage Error (%): ' + str(mape_lstm))
        return rmse_lstm, mape_lstm
    

    def train_pred_eval_model(self,
                              lstm_units = 50, optimizer = 'adam', epochs = 1, 
                              batch_size = 1, loss = 'mean_squared_error'):
        
        """
        INPUT:
        X_train - training input variables (X)
        y_train - training output variable (y)
        df - dataframe that has been preprocessed
        scaler - instantiated object for MixMaxScaler()
        test - test dataset that contains only 'Date' & 'Close' columns (i.e.test = df[train_num:][['Date', 'Close']])
        
        default(initial) parameters chosen for LSTM
        --------------------------------------------
        lstm_units = 50
        optimizer = 'adam'
        epochs = 1
        batch_size = 1
        loss = 'mean_squared_error'
        
        OUTPUT:
        rmse_lstm - RMSE for LSTM
        mape_lstm - MAPE(%) for LSTM    
        """
        
        model_tuned = improvement.__fit_lstm(self.X_train, self.y_train, int(lstm_units)
                                             , optimizer, int(epochs), int(batch_size), loss)
        closing_price_tuned = improvement.__get_pred_closing_price(self, model_tuned)
        rmse_lstm, mape_lstm = improvement.__model_performance(self.varPredict, self.test, closing_price_tuned)

        return rmse_lstm, mape_lstm
        
        