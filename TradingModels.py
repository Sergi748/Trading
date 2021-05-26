# -*- coding: utf-8 -*-

import sys
sys.path.append('OneDrive - Teradata/Documents/SERGIO/Trading/')
from main import tradingModel

# Training model
idYahoo = 'NDAQ'
path = 'C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Trading'
varPredict = 'adjclose'
trading = tradingModel(idYahoo=idYahoo, path=path)
trading.training(varPredict, '2019-01-04', '2021-05-14', True, True)

# Prediction
trading.prediction(varPredict, '2021-02-27')

# Metrics prediction
trading.checkPredictions(varPredict)

# Profitability
dictMarkets = {'NDAQ':'pink', 'TSLA':'red', 'AAPL':'blue', 'AMZN':'orange'
               , 'GOOG':'yellow'}
dateRentability = '2021-03-03'
trading.plotProfitability(dateRentability=dateRentability, dictMarkets=dictMarkets)
trading.plotProfitability(allMarkets=False)
trading.plotProfitability(dateRentability=dateRentability, dictMarkets=dictMarkets, allMarkets=False, realClosePredOpen=True)

# BollingerBands
startDate = '2019-01-04'
endDate = '2021-05-26'
trading.plotBollingerBands(startDate, endDate)
