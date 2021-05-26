# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
from development import trainingModel, predictions, checkMetrics
from development_ml import trainingModel_ml, predictions_ml
from tuningHyperparameters import tuningHyperparameters
from plots import plots

class tradingModel():
    
    '''
    Class to develop a complete trading model.
    With this class you can train a new model or you can predict the closing 
    price.
    '''
    
    def __init__(self, idYahoo, path):
        self.idYahoo = idYahoo
        self.path = path
        # self.varPredict = varPredict
        self.dfToTrain = pd.DataFrame()
        self.dfToTrain_ml = pd.DataFrame()
        self.dfToPred = pd.DataFrame()
        self.dfPrediction = pd.DataFrame()
        self.dfToPred_ml = pd.DataFrame()
        self.dfPrediction_ml = pd.DataFrame()
        self.dfCheck = pd.DataFrame()
        self.df_epoch = pd.DataFrame()
        self.df_batch_size = pd.DataFrame()
        self.df_opt = pd.DataFrame()

        self.mape = ""
        self.rmse = ""
        
    def __takeHyperparameters(self, varPredict, startDate, endDate):
        
        hyperparameters = tuningHyperparameters(self.idYahoo, self.path, varPredict, startDate, endDate)
        hyperparameters.training()
        df_epoch = hyperparameters.df_epoch
        df_batch_size = hyperparameters.df_batch_size
        df_opt = hyperparameters.df_opt
        
        optimizer = df_opt.sort_values('MAPE', ascending=True).reset_index(drop=True).loc[0, 'Optimizer']
        epoch = df_epoch.sort_values('MAPE', ascending=True).reset_index(drop=True).loc[0, 'Epoch']
        batch_size = df_batch_size.sort_values('MAPE', ascending=True).reset_index(drop=True).loc[0, 'Batch Size']

        return optimizer, epoch, batch_size
            
    def __trainingTuning(self, varPredict, startDate, endDate, optimizer, epochs, batch_size, saveModel=True, plots=False):
        
        trainTuning = tuningHyperparameters(self.idYahoo, self.path, varPredict, startDate, endDate)
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
        
        train = trainingModel(self.idYahoo, self.path, varPredict, startDate, endDate)
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
            
            if os.path.isfile(self.path + '/' + self.idYahoo + '/output/metricas/metrics_train_' + varPredict + '_lstm.csv'):
                dfMetricsOld = pd.read_csv(self.path + '/' + self.idYahoo + '/output/metricas/metrics_train_' + varPredict + '_lstm.csv', sep=';')
                dfMetricsNew = pd.DataFrame({'Metric':['RMSE','MAPE (%)'], 'LSTM_train_new': [3.80, 2.30]})
                dfMetricsCompare = pd.merge(dfMetricsOld, dfMetricsNew, on='Metric', how='inner')
                dfMetricsCompare['Comparation'] = np.where(dfMetricsCompare['LSTM_train_new'] < dfMetricsCompare['LSTM_train'], 1, 0)
    
                if dfMetricsCompare.Comparation.sum() > 0:
                    print('############################# \nNew model has better result than model previous, so the new model has been saved \n#############################')
                    train.saveModel()
                    dfMetrics = pd.DataFrame({'Metric':['RMSE','MAPE (%)'], 'LSTM_train': [rmse, mape]})
                    dfMetrics.to_csv(self.path + '/' + self.idYahoo + '/output/metricas/metrics_train_' + varPredict + '_lstm.csv', sep=';', index=False)

                    if plots == True:
                        train.plot_data()
                        train.plot_train_prediction()
                else:
                    print('############################# \nNew model has worse result than model previous, so the new model has been saved \n#############################')
            else:
                train.saveModel()
                dfMetrics = pd.DataFrame({'Metric':['RMSE','MAPE (%)'], 'LSTM_train': [rmse, mape]})
                dfMetrics.to_csv(self.path + '/' + self.idYahoo + '/output/metricas/metrics_train_' + varPredict + '_lstm.csv', sep=';', index=False)
                
        if plots == True:
            train.plot_data()
            train.plot_train_prediction()

        print('######### Tuning hyperparameters #########')            
        optimizer, epoch, batch_size = tradingModel.__takeHyperparameters(self, varPredict, startDate, endDate)
        print('#########\noptimizer: {0} \nepoch: {1} \nbatch_size: {2} \n#########'.format(optimizer, epoch, batch_size))
        tradingModel.__trainingTuning(self, varPredict, startDate, endDate, optimizer, epoch, batch_size, True, True)       
    
        # Running machine learning training
        trainClass_ml = trainingModel_ml(self.idYahoo, self.path, varPredict, startDate, endDate)
        trainClass_ml.load_data()
        trainClass_ml.create_datasets(N=3, test_size=0.2, plotSave=True)
        trainClass_ml.development_models_pred_test(plotTest=True)
        self.dfToTrain_ml = trainClass_ml.train
    
    def prediction(self, varPredict, datePred):
        
        '''
        This function calls a class called "predictions".
        Only works if there is a trained model. 
        Use the same arguments as "training" function (idYahoo and pathProject)
        but another argument is required with the date to be predicted. 
        This date has the format "YYYY-MM-DD".
        This function returns a csv file that is saved in pathProject.
        '''
        
        def __predComplete(path, idYahoo, varPredict):
    
            filePath = path + '/' + idYahoo + '/output/predicciones/'
            dfPred_lstm = pd.read_csv(filePath + 'predictionsMade_' + varPredict + '_lstm.csv', sep=';')
            dfPred_ml = pd.read_csv(filePath + 'predictionsMade_' + varPredict + '_ml.csv', sep=';')
        
            dfPredAll = pd.merge(dfPred_ml, dfPred_lstm, on='Date', how='inner')
            varsPred = [elem for elem in dfPredAll.columns if elem not in ['Date', 'pred_ensamble_ml']]
            dfPredAll['pred_ensamble_ml_lstm'] = dfPredAll[varsPred].mean(axis=1)
        
            os.remove(filePath + 'predictionsMade_' + varPredict + '_lstm.csv')
            os.remove(filePath + 'predictionsMade_' + varPredict + '_ml.csv')
            filePred = filePath + 'predictionsMade_' + varPredict + '_' + idYahoo + '.csv'
            
            if os.path.exists(filePred):
                dfAllPreds = pd.read_csv(filePred, sep=';')
                dfAllPreds.loc[len(dfAllPreds)] = dfPredAll.values.tolist()[0]
                dfAllPreds.to_csv(filePred, sep=';', index=False)
            else:
                dfPredAll.to_csv(filePred, sep=';', index=False)
        
        pred = predictions(self.idYahoo, self.path, varPredict, datePred)
        pred.pred_closing_price()
        self.dfToPred = pred.df
        self.dfPrediction = pred.dfPred
        
        # Prediction made with machine learning models
        pred_ml = predictions_ml(self.idYahoo, self.path, varPredict, datePred)
        pred_ml.pred_closing_price(N=3)
        self.dfToPred_ml = pred_ml.dfShift
        self.dfPrediction_ml = pred_ml.dfPred
        
        __predComplete(self.path, self.idYahoo, varPredict)
        
        
    def checkPredictions(self, varPredict):
        
        '''
        This function calls a class called "checkMetrics".
        Only works if there is a prediction made.
        It doesn´t require new arguments.
        This function returns a csv file that is saved in pathProject.
        '''
        
        check = checkMetrics(self.idYahoo, self.path, varPredict)
        check.metricsPredictions()
        self.dfCheck = check.df
        
        
    def plotProfitability(self, dateRentability=None, dictMarkets=None, allMarkets=True, realClosePredOpen=False):
        
        '''
        This function creates a graph with the profitability market, 
        this can be with a single market or with a list of markets.
        '''
        
        plot = plots(self.path)
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
        
        pathProject = self.path + '/' + self.idYahoo
        plot = plots(pathProject)
        plot.plotBollingerBands(self.idYahoo, startDate, endDate, Candlestick)

