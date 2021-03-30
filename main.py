# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('~/Documents/SERGIO/Trading/')
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
from development import trainingModel, predictions, checkMetrics
from tuningHyperparameters import tuningHyperparameters
from plots import plots

class tradingModel():
    
    '''
    Class to develop a complete trading model.
    With this class you can train a new model or you can predict the closing 
    price.
    '''
    
    def __init__(self, idYahoo, pathProject):
        self.idYahoo = idYahoo
        self.pathProject = pathProject
        # self.varPredict = varPredict
        self.dfToTrain = pd.DataFrame()
        self.dfToPred = pd.DataFrame()
        self.dfPrediction = pd.DataFrame()
        self.dfCheck = pd.DataFrame()
        self.df_epoch = pd.DataFrame()
        self.df_batch_size = pd.DataFrame()
        self.df_opt = pd.DataFrame()

        self.mape = ""
        self.rmse = ""
        
    def __takeHyperparameters(self, varPredict, startDate, endDate):
        
        hyperparameters = tuningHyperparameters(self.idYahoo, self.pathProject, varPredict, startDate, endDate)
        hyperparameters.training()
        df_epoch = hyperparameters.df_epoch
        df_batch_size = hyperparameters.df_batch_size
        df_opt = hyperparameters.df_opt
        
        optimizer = df_opt.sort_values('MAPE', ascending=True).reset_index(drop=True).loc[0, 'Optimizer']
        epoch = df_epoch.sort_values('MAPE', ascending=True).reset_index(drop=True).loc[0, 'Epoch']
        batch_size = df_batch_size.sort_values('MAPE', ascending=True).reset_index(drop=True).loc[0, 'Batch Size']

        return optimizer, epoch, batch_size
            
    def __trainingTuning(self, varPredict, startDate, endDate, optimizer, epochs, batch_size, saveModel=True, plots=False):
        
        trainTuning = tuningHyperparameters(self.idYahoo, self.pathProject, varPredict, startDate, endDate)
        trainTuning.trainingTuning(optimizer, epochs, batch_size, saveModel, plots)
        
    def training(self, varPredict, startDate, endDate, saveModel=True, plots=False):
        
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
        
        train = trainingModel(self.idYahoo, self.pathProject, varPredict, startDate, endDate)
        train.load_data()
        train.split_data(size=0.2)
        train.scaler_split_train_test()
        train.fit_lstm()
        train.pred_closing_price()
        mape, rmse = train.metrics()
        self.dfToTrain = train.df
        self.mape = mape
        self.rmse = rmse
        
        if saveModel== True:
            
            if os.path.isfile(self.pathProject + '/metrics_train_' + varPredict + '.csv'):
                dfMetricsOld = pd.read_csv(self.pathProject + '/metrics_train_' + varPredict + '.csv', sep=';')
                dfMetricsNew = pd.DataFrame({'Metric':['RMSE','MAPE (%)'], 'LSTM_train_new': [3.80, 2.30]})
                dfMetricsCompare = pd.merge(dfMetricsOld, dfMetricsNew, on='Metric', how='inner')
                dfMetricsCompare['Comparation'] = np.where(dfMetricsCompare['LSTM_train_new'] < dfMetricsCompare['LSTM_train'], 1, 0)
    
                if dfMetricsCompare.Comparation.sum() > 0:
                    print('############################# \nNew model has better result than model previous, so the new model has been saved \n#############################')
                    train.saveModel()
                    dfMetrics = pd.DataFrame({'Metric':['RMSE','MAPE (%)'], 'LSTM_train': [rmse, mape]})
                    dfMetrics.to_csv(self.pathProject + '/metrics_train_' + varPredict + '.csv', sep=';', index=False)

                    if plots == True:
                        train.plot_data()
                        train.plot_train_prediction()
                else:
                    print('############################# \nNew model has worse result than model previous, so the new model has been saved \n#############################')
            else:
                train.saveModel()
                dfMetrics = pd.DataFrame({'Metric':['RMSE','MAPE (%)'], 'LSTM_train': [rmse, mape]})
                dfMetrics.to_csv(self.pathProject + '/metrics_train_' + varPredict + '.csv', sep=';', index=False)
                
        if plots == True:
            train.plot_data()
            train.plot_train_prediction()

        print('######### Tuning hyperparameters #########')            
        optimizer, epoch, batch_size = tradingModel.__takeHyperparameters(self, varPredict, startDate, endDate)
        print('#########\noptimizer: {0} \nepoch: {1} \nbatch_size: {2} \n#########'.format(optimizer, epoch, batch_size))
        tradingModel.__trainingTuning(self, varPredict, startDate, endDate, optimizer, epoch, batch_size, True, True)       
    
    
    def prediction(self, varPredict, datePred):
        
        '''
        This function calls a class called "predictions".
        Only works if there is a trained model. 
        Use the same arguments as "training" function (idYahoo and pathProject)
        but another argument is required with the date to be predicted. 
        This date has the format "YYYY-MM-DD".
        This function returns a csv file that is saved in pathProject.
        '''
        
        pred = predictions(self.idYahoo, self.pathProject, varPredict, datePred)
        pred.pred_closing_price()
        self.dfToPred = pred.df
        self.dfPrediction = pred.dfPred
        
        
    def checkPredictions(self, varPredict):
        
        '''
        This function calls a class called "checkMetrics".
        Only works if there is a prediction made.
        It doesn´t require new arguments.
        This function returns a csv file that is saved in pathProject.
        '''
        
        check = checkMetrics(self.idYahoo, self.pathProject, varPredict)
        check.metricsPredictions()
        self.dfCheck = check.df
        
        
    def plotProfitability(self, dateRentability=None, dictMarkets=None, allMarkets=True, realClosePredOpen=False):
        
        '''
        This function creates a graph with the profitability market, 
        this can be with a single market or with a list of markets.
        '''
        
        plot = plots(self.pathProject)
        if allMarkets == False:
            plot.profitabilityMarket(self.idYahoo)
        else:
            plot.profitabilityCompletePlot(dateRentability, dictMarkets)

        if realClosePredOpen == True:
            plot.plotProfitabilityRealClosePredOpen(dateRentability, dictMarkets)
            
        
    def plotBollingerBands(self, startDate, endDate, Candlestick=True):
        
        '''
        This function creates a chart using the Bollinger Bands.
        '''
        
        plot = plots(self.pathProject)
        plot.plotBollingerBands(self.idYahoo, startDate, endDate, Candlestick)

