# -*- coding: utf-8 -*-

import sys
sys.path.append('~/Documents/SERGIO/Trading/')
import numpy as np
import pandas as pd
from development import trainingModel
from functionsImprovements import improvement
from IPython.display import display


class tuningHyperparameters():
 
    np.random.seed(1234)
    
    def __init__(self, idYahoo, pathProject, varPredict, startDate, endDate):
        self.idYahoo = idYahoo
        self.pathProject = pathProject
        self.varPredict = varPredict
        self.startDate = startDate
        self.endDate = endDate
        self.dfToTrain = pd.DataFrame()
        self.df = pd.DataFrame()
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.model = ""
        self.df_epoch = pd.DataFrame()
        self.df_batch_size = pd.DataFrame()
        self.df_opt = pd.DataFrame()
        
    def training(self):
        
        '''
        This function calls a class called "trainingModel".
        With this class the user can develop a complete traiding model.
        Only two arguments are required:
            idYahoo: this is the market id in Yahoo finance
            pathProject: path where the model and predictions will be saved
            startDate: Start date to train model
            endDate: End date to train model
        In addition, the user can save the final model and graphs if he wished.
        '''

        trainClass = trainingModel(self.idYahoo, self.pathProject, self.varPredict, self.startDate, self.endDate)
        trainClass.load_data()
        trainClass.split_data(size=0.2)
        trainClass.scaler_split_train_test()
        trainClass.fit_lstm()
        trainClass.pred_closing_price()
        mapeNotTuning, rmseNotTuning = trainClass.metrics()
        df = trainClass.df
        test = trainClass.test
        X_train = trainClass.X_train
        y_train = trainClass.y_train
        self.model = trainClass.model
        scaler = trainClass.scaler
        
        # Class for tuning hyperparameters
        develop = improvement(df, self.varPredict, X_train, y_train, test, scaler)
        
        rmse_ls, mape_ls = [], []
        batch_size_ls = [1, 2, 3, 4]
        for i in batch_size_ls:
            rmse_lstm, mape_lstm = develop.train_pred_eval_model(batch_size = i)
            rmse_ls.append(rmse_lstm)
            mape_ls.append(mape_lstm)
        
        df_batch_size = pd.DataFrame(list(zip(batch_size_ls, rmse_ls, mape_ls))
                                     , columns = ['Batch Size', 'RMSE', 'MAPE'])
        
        rmse_ls, mape_ls = [], []
        epoch_ls = [1, 2, 3, 4]
        for i in epoch_ls:
            rmse_lstm, mape_lstm = develop.train_pred_eval_model(epochs = i)           
            rmse_ls.append(rmse_lstm)
            mape_ls.append(mape_lstm)
        
        df_epoch = pd.DataFrame(list(zip(epoch_ls, rmse_ls, mape_ls))
                                , columns = ['Epoch', 'RMSE', 'MAPE'])
        
        rmse_ls, mape_ls = [], []
        opt_ls= ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam']
        for i in opt_ls:
            rmse_lstm, mape_lstm = develop.train_pred_eval_model(optimizer = i)
            rmse_ls.append(rmse_lstm)
            mape_ls.append(mape_lstm)
            
        df_opt = pd.DataFrame(list(zip(opt_ls, rmse_ls, mape_ls))
                              , columns = ['Optimizer', 'RMSE', 'MAPE'])

        self.df_opt = df_opt 
        self.df_epoch = df_epoch 
        self.df_batch_size = df_batch_size
        display(pd.DataFrame({'Metric':['RMSE','MAPE (%)'], 'LSTM': [rmseNotTuning, mapeNotTuning]}))
        display(df_opt)
        display(df_epoch)        
        display(df_batch_size)
        
    def trainingTuning(self, optimizer, epochs, batch_size, saveModel=True, plots=False):
                    
        trainTuning = trainingModel(self.idYahoo, self.pathProject, self.varPredict, self.startDate, self.endDate)
        trainTuning.load_data()
        trainTuning.split_data(size=0.2)
        trainTuning.scaler_split_train_test()
        trainTuning.fit_lstm(optimizer = optimizer, epochs = epochs, batch_size = batch_size)
        trainTuning.pred_closing_price()
        mape, rmse = trainTuning.metrics()
        
        if saveModel == True:
            dfMetricsOld = pd.read_csv(self.pathProject + '/metrics_train_' + self.varPredict + '.csv', sep=';')
            dfMetricsNew = pd.DataFrame({'Metric':['RMSE','MAPE (%)'], 'LSTM_train_new': [rmse, mape]})
            dfMetricsCompare = pd.merge(dfMetricsOld, dfMetricsNew, on='Metric', how='inner')
            dfMetricsCompare['Comparation'] = np.where(dfMetricsCompare['LSTM_train_new'] < dfMetricsCompare['LSTM_train'], 1, 0)

            if dfMetricsCompare.Comparation.sum() > 0:
                print('############################# \nNew model with tuning hyperparameters has better result than model without tuning hyperparameters, so the new model has been saved \n#############################')
                trainTuning.saveModel()
                dfMetrics = pd.DataFrame({'Metric':['RMSE','MAPE (%)'], 'LSTM_train': [rmse, mape]})
                dfMetrics.to_csv(self.pathProject + '/metrics_train_' + self.varPredict + '.csv', sep=';', index=False)

                if plots == True:
                    trainTuning.plot_data()
                    trainTuning.plot_train_prediction()
            else:
                print('############################# \nModel with tuning hyperparameters has worse result than model without tuning hyperparameters, so the new model hasn´t been saved \n#############################')

        self.dfToTrain = trainTuning.df
