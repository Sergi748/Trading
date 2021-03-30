# -*- coding: utf-8 -*-

import sys
sys.path.append('~/Documents/SERGIO/Trading/')
from main import tradingModel

# Training model
idYahoo = 'NDAQ'
path = 'C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Trading/NDAQ'
varPredict = 'open'
trading = tradingModel(idYahoo=idYahoo, pathProject=path)
trading.training(varPredict, '2019-01-04', '2021-02-03', True, True)

# Prediction
trading.prediction(varPredict, '2021-02-04')
trading.prediction(varPredict, '2021-02-05')
trading.prediction(varPredict, '2021-02-08')
trading.prediction(varPredict, '2021-02-09')
trading.prediction(varPredict, '2021-02-10')
trading.prediction(varPredict, '2021-02-11')
trading.prediction(varPredict, '2021-02-12')
trading.prediction(varPredict, '2021-02-16')
trading.prediction(varPredict, '2021-02-17')
trading.prediction(varPredict, '2021-02-18')
trading.prediction(varPredict, '2021-02-19')
trading.prediction(varPredict, '2021-02-22')
trading.prediction(varPredict, '2021-02-23')
trading.prediction(varPredict, '2021-02-24')
trading.prediction(varPredict, '2021-02-25')
trading.prediction(varPredict, '2021-02-26')
trading.prediction(varPredict, '2021-03-01')
trading.prediction(varPredict, '2021-03-02')
trading.prediction(varPredict, '2021-03-03')
trading.prediction(varPredict, '2021-03-04')

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
endDate = '2021-03-08'
trading.plotBollingerBands(startDate, endDate)



# Training model
idYahoo = 'AAPL'
path = 'C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Trading/AAPL'
varPredict = 'open'
trading = tradingModel(idYahoo=idYahoo, pathProject=path)
trading.training(varPredict, '2019-01-04', '2021-02-03', True, True)

# Prediction
trading.prediction(varPredict, '2021-02-04')
trading.prediction(varPredict, '2021-02-05')
trading.prediction(varPredict, '2021-02-08')
trading.prediction(varPredict, '2021-02-09')
trading.prediction(varPredict, '2021-02-10')
trading.prediction(varPredict, '2021-02-11')
trading.prediction(varPredict, '2021-02-12')
trading.prediction(varPredict, '2021-02-16')
trading.prediction(varPredict, '2021-02-17')
trading.prediction(varPredict, '2021-02-18')
trading.prediction(varPredict, '2021-02-19')
trading.prediction(varPredict, '2021-02-22')
trading.prediction(varPredict, '2021-02-23')
trading.prediction(varPredict, '2021-02-24')
trading.prediction(varPredict, '2021-02-25')
trading.prediction(varPredict, '2021-02-26')
trading.prediction(varPredict, '2021-03-01')
trading.prediction(varPredict, '2021-03-02')
trading.prediction(varPredict, '2021-03-03')
trading.prediction(varPredict, '2021-03-04')

# Metrics prediction
trading.checkPredictions(varPredict)

# BollingerBands
startDate = '2019-01-04'
endDate = '2021-03-08'
trading.plotBollingerBands(startDate, endDate)





# Training model
idYahoo = 'AMZN'
path = 'C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Trading/AMZN'
varPredict = 'open'
trading = tradingModel(idYahoo=idYahoo, pathProject=path)
trading.training(varPredict, '2019-01-04', '2021-02-03', True, True)

# Prediction
trading.prediction(varPredict, '2021-02-04')
trading.prediction(varPredict, '2021-02-05')
trading.prediction(varPredict, '2021-02-08')
trading.prediction(varPredict, '2021-02-09')
trading.prediction(varPredict, '2021-02-10')
trading.prediction(varPredict, '2021-02-11')
trading.prediction(varPredict, '2021-02-12')
trading.prediction(varPredict, '2021-02-16')
trading.prediction(varPredict, '2021-02-17')
trading.prediction(varPredict, '2021-02-18')
trading.prediction(varPredict, '2021-02-19')
trading.prediction(varPredict, '2021-02-22')
trading.prediction(varPredict, '2021-02-23')
trading.prediction(varPredict, '2021-02-24')
trading.prediction(varPredict, '2021-02-25')
trading.prediction(varPredict, '2021-02-26')
trading.prediction(varPredict, '2021-03-01')
trading.prediction(varPredict, '2021-03-02')
trading.prediction(varPredict, '2021-03-03')
trading.prediction(varPredict, '2021-03-04')

# Metrics prediction
trading.checkPredictions(varPredict)

# BollingerBands
startDate = '2019-01-04'
endDate = '2021-03-08'
trading.plotBollingerBands(startDate, endDate)




# Training model
idYahoo = 'GOOG'
path = 'C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Trading/GOOG'
varPredict = 'open'
trading = tradingModel(idYahoo=idYahoo, pathProject=path)
trading.training(varPredict, '2019-01-04', '2021-02-03', True, True)

# Prediction
trading.prediction(varPredict, '2021-02-04')
trading.prediction(varPredict, '2021-02-05')
trading.prediction(varPredict, '2021-02-08')
trading.prediction(varPredict, '2021-02-09')
trading.prediction(varPredict, '2021-02-10')
trading.prediction(varPredict, '2021-02-11')
trading.prediction(varPredict, '2021-02-12')
trading.prediction(varPredict, '2021-02-16')
trading.prediction(varPredict, '2021-02-17')
trading.prediction(varPredict, '2021-02-18')
trading.prediction(varPredict, '2021-02-19')
trading.prediction(varPredict, '2021-02-22')
trading.prediction(varPredict, '2021-02-23')
trading.prediction(varPredict, '2021-02-24')
trading.prediction(varPredict, '2021-02-25')
trading.prediction(varPredict, '2021-02-26')
trading.prediction(varPredict, '2021-03-01')
trading.prediction(varPredict, '2021-03-02')
trading.prediction(varPredict, '2021-03-03')
trading.prediction(varPredict, '2021-03-04')

# Metrics prediction
trading.checkPredictions(varPredict)

# BollingerBands
startDate = '2019-01-04'
endDate = '2021-03-08'
trading.plotBollingerBands(startDate, endDate)



# Training model
idYahoo = 'TSLA'
path = 'C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Trading/TSLA'
varPredict = 'open'
trading = tradingModel(idYahoo=idYahoo, pathProject=path)
trading.training(varPredict, '2019-01-04', '2021-02-03', True, True)

# Prediction
trading.prediction(varPredict, '2021-02-04')
trading.prediction(varPredict, '2021-02-05')
trading.prediction(varPredict, '2021-02-08')
trading.prediction(varPredict, '2021-02-09')
trading.prediction(varPredict, '2021-02-10')
trading.prediction(varPredict, '2021-02-11')
trading.prediction(varPredict, '2021-02-12')
trading.prediction(varPredict, '2021-02-16')
trading.prediction(varPredict, '2021-02-17')
trading.prediction(varPredict, '2021-02-18')
trading.prediction(varPredict, '2021-02-19')
trading.prediction(varPredict, '2021-02-22')
trading.prediction(varPredict, '2021-02-23')
trading.prediction(varPredict, '2021-02-24')
trading.prediction(varPredict, '2021-02-25')
trading.prediction(varPredict, '2021-02-26')
trading.prediction(varPredict, '2021-03-01')
trading.prediction(varPredict, '2021-03-02')
trading.prediction(varPredict, '2021-03-03')
trading.prediction(varPredict, '2021-03-04')

# Metrics prediction
trading.checkPredictions(varPredict)

# BollingerBands
startDate = '2019-01-04'
endDate = '2021-03-08'
trading.plotBollingerBands(startDate, endDate)


