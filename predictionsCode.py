# -*- coding: utf-8 -*-

import os
import shutil
os.chdir('OneDrive - Teradata/Documents/SERGIO/Trading/')
from main import tradingModel

# Functions
def predictor(path, markets, varPredict, datePred):

    '''Function to make a one-day prediction date for all markets'''
    
    for market in markets:

        print('Making prediction for the market: {0}'.format(market))    
        # Training model
        trading = tradingModel(idYahoo=market, path=path)
        # Prediction
        trading.prediction(varPredict, datePred)

def checkPred(path, market, varPredict):

    '''Function to check the predictions for all markets'''

    for market in markets:
        
        print('Checking predictions for the market: {0}'.format(market))
        trading = tradingModel(idYahoo=market, path=path)
        # Metrics prediction
        trading.checkPredictions(varPredict)


def createFolderSend(path, varPredict, markets):
    
    '''Function that creates a folder with all files to be sent by email.'''

    pathFinal = path + '/Predicciones/'
    if os.path.exists(pathFinal) == False:
        os.makedirs(pathFinal)
    
    for market in markets:
        
        pathPred = path + '/' + market 
        filePred = pathPred + '/output/predicciones/predictionsMade_' + varPredict + '_' + market + '.csv'
        fileReal = pathPred + '/output/predicciones/valueReal_predictions_' + varPredict + '_' + market + '.csv' 
        filePng = pathPred + '/output/graficas/plot_prediction_' + varPredict + '_' + market + '_value.png'
        fileHtml = pathPred + '/Bollinger_Bands_' + market + '.html'
        
        files = [filePred, fileReal, filePng, fileHtml]
        # Copy files to a new path
        for file in files:
            shutil.copy(file, pathFinal)
            
    # Making zip folder 
    shutil.make_archive('Predicciones', 'zip', pathFinal)


varPredict = 'adjclose'
datePred = '2021-05-28'
path = 'C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Trading'
markets = ['NDAQ', 'TSLA', 'AAPL', 'AMZN', 'GOOG']

predictor(path, markets, varPredict, datePred)
checkPred(path, markets, varPredict)
createFolderSend(path, varPredict, markets)

