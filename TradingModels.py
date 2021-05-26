# -*- coding: utf-8 -*-

import sys
# import os
sys.path.append('OneDrive - Teradata/Documents/SERGIO/Trading/')
from main import tradingModel
# import pandas as pd
# import datetime
# import matplotlib.pyplot as plt
# from yahoofinancials import YahooFinancials

# Training model
idYahoo = 'NDAQ'
path = 'C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Trading'
varPredict = 'adjclose'
trading = tradingModel(idYahoo=idYahoo, path=path)
trading.training(varPredict, '2019-01-04', '2021-05-14', True, True)

# Prediction
trading.prediction(varPredict, '2021-02-17')
trading.prediction(varPredict, '2021-02-18')
trading.prediction(varPredict, '2021-02-19')
trading.prediction(varPredict, '2021-02-20')
trading.prediction(varPredict, '2021-02-21')
trading.prediction(varPredict, '2021-02-22')
trading.prediction(varPredict, '2021-02-23')
trading.prediction(varPredict, '2021-02-24')
trading.prediction(varPredict, '2021-02-25')
trading.prediction(varPredict, '2021-02-26')
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



# Training model
idYahoo = 'AAPL'
path = 'C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Trading'
varPredict = 'adjclose'
trading = tradingModel(idYahoo=idYahoo, path=path)
trading.training(varPredict, '2019-01-04', '2021-05-14', True, True)

# Prediction
trading.prediction(varPredict, '2021-02-17')
trading.prediction(varPredict, '2021-02-18')
trading.prediction(varPredict, '2021-02-19')
trading.prediction(varPredict, '2021-02-20')
trading.prediction(varPredict, '2021-02-21')
trading.prediction(varPredict, '2021-02-22')
trading.prediction(varPredict, '2021-02-23')
trading.prediction(varPredict, '2021-02-24')
trading.prediction(varPredict, '2021-02-25')
trading.prediction(varPredict, '2021-02-26')
trading.prediction(varPredict, '2021-02-27')

# Metrics prediction
trading.checkPredictions(varPredict)

# BollingerBands
startDate = '2019-01-04'
endDate = '2021-05-26'
trading.plotBollingerBands(startDate, endDate)





# Training model
idYahoo = 'AMZN'
path = 'C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Trading'
varPredict = 'adjclose'
trading = tradingModel(idYahoo=idYahoo, path=path)
trading.training(varPredict, '2019-01-04', '2021-05-14', True, True)

# Prediction
trading.prediction(varPredict, '2021-02-17')
trading.prediction(varPredict, '2021-02-18')
trading.prediction(varPredict, '2021-02-19')
trading.prediction(varPredict, '2021-02-20')
trading.prediction(varPredict, '2021-02-21')
trading.prediction(varPredict, '2021-02-22')
trading.prediction(varPredict, '2021-02-23')
trading.prediction(varPredict, '2021-02-24')
trading.prediction(varPredict, '2021-02-25')
trading.prediction(varPredict, '2021-02-26')
trading.prediction(varPredict, '2021-02-27')


# Metrics prediction
trading.checkPredictions(varPredict)

# BollingerBands
startDate = '2019-01-04'
endDate = '2021-05-26'
trading.plotBollingerBands(startDate, endDate)




# Training model
idYahoo = 'GOOG'
path = 'C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Trading'
varPredict = 'adjclose'
trading = tradingModel(idYahoo=idYahoo, path=path)
trading.training(varPredict, '2019-01-04', '2021-05-14', True, True)

# Prediction
trading.prediction(varPredict, '2021-02-17')
trading.prediction(varPredict, '2021-02-18')
trading.prediction(varPredict, '2021-02-19')
trading.prediction(varPredict, '2021-02-20')
trading.prediction(varPredict, '2021-02-21')
trading.prediction(varPredict, '2021-02-22')
trading.prediction(varPredict, '2021-02-23')
trading.prediction(varPredict, '2021-02-24')
trading.prediction(varPredict, '2021-02-25')
trading.prediction(varPredict, '2021-02-26')
trading.prediction(varPredict, '2021-02-27')


# Metrics prediction
trading.checkPredictions(varPredict)

# BollingerBands
startDate = '2019-01-04'
endDate = '2021-05-26'
trading.plotBollingerBands(startDate, endDate)



# Training model
idYahoo = 'TSLA'
path = 'C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Trading'
varPredict = 'adjclose'
trading = tradingModel(idYahoo=idYahoo, path=path)
trading.training(varPredict, '2019-01-04', '2021-05-14', True, True)

# Prediction
trading.prediction(varPredict, '2021-02-17')
trading.prediction(varPredict, '2021-02-18')
trading.prediction(varPredict, '2021-02-19')
trading.prediction(varPredict, '2021-02-20')
trading.prediction(varPredict, '2021-02-21')
trading.prediction(varPredict, '2021-02-22')
trading.prediction(varPredict, '2021-02-23')
trading.prediction(varPredict, '2021-02-24')
trading.prediction(varPredict, '2021-02-25')
trading.prediction(varPredict, '2021-02-26')
trading.prediction(varPredict, '2021-02-27')


# Metrics prediction
trading.checkPredictions(varPredict)

# BollingerBands
startDate = '2019-01-04'
endDate = '2021-05-26'
trading.plotBollingerBands(startDate, endDate)


