# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib
# # matplotlib.use('SVG')
import datetime
from yahoofinancials import YahooFinancials
import plotly.express as px
import plotly.graph_objs as go


class plots():
    
    '''
    Class to create graph with profitability
    '''
    
    def __init__(self, path):
        self.path = path
    
    def profitabilityCompletePlot(self, dateRentability, dictMarkets):
        
        '''
        This function creates a graph with the profitability of a list of markets for a given date.
        '''
    
        pathNew = os.path.split(self.path)[0]
        markets = list(dictMarkets.keys())
        colors = list(dictMarkets.values())
    
        dfPlot = pd.DataFrame(columns=['datePredict', 'Market', 'profitability'])
        for market in markets:
        
            pathMarket = pathNew + '/' + market    
            dfClose = pd.read_csv(pathMarket + '/predictionsMade_close.csv', sep=';')
            dfOpen = pd.read_csv(pathMarket + '/predictionsMade_open.csv', sep=';')
            
            dfFinal = pd.merge(dfOpen, dfClose, on='dateToPredict', how='inner')
            dfFinal['profitability'] = ((dfFinal.prediction_close - dfFinal.prediction_open) / dfFinal.prediction_open)*100
        
            valueNew = dfFinal.loc[dfFinal.dateToPredict == dateRentability,['dateToPredict', 'profitability']].values.tolist()[0]
            dfPlot.loc[len(dfPlot)] = [valueNew[0], market, valueNew[1]]
        
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        valueRentability = dfPlot.profitability.tolist()
        ax.barh(markets, valueRentability, color=colors)
        ax.invert_yaxis()
        plt.title('Profitability in the markets to date {0}'.format(dateRentability))
        plt.xlabel('Profitability')
        plt.ylabel('Markets')
        plt.savefig(pathNew + '/plot_profitability_markets.png', bbox_inches='tight')
        plt.show()
    
    
    def profitabilityMarket(self, idYahoo):
        
        '''
        This function creates a graph with the profitability of a market for all the historical information.
        '''
    
        dfClose = pd.read_csv(self.path + '/predictionsMade_close.csv', sep=';')
        dfOpen = pd.read_csv(self.path + '/predictionsMade_open.csv', sep=';')
        
        dfFinal = pd.merge(dfOpen, dfClose, on='dateToPredict', how='inner')
        dfFinal['profitability'] = ((dfFinal.prediction_close - dfFinal.prediction_open) / dfFinal.prediction_open)*100
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        dates = dfFinal.dateToPredict.tolist()
        valueProfitability= dfFinal.profitability.tolist()
        ax.barh(dates, valueProfitability, color='cyan')
        ax.invert_yaxis()
        plt.title('Profitability in market {0}'.format(idYahoo))
        plt.xlabel('Profitability')
        plt.ylabel('Dates')
        plt.savefig(self.path + '/plot_profitability_' + idYahoo + '.png', bbox_inches='tight')
        plt.show()
        
        
    def plotProfitabilityRealClosePredOpen(self, dateRentability, dictMarkets):
    
        pathNew = os.path.split(self.path)[0]
    
        def __dataClose(idYahoo, dateRentability):
            
            dateClose = str(datetime.datetime.strptime(dateRentability, '%Y-%m-%d') - datetime.timedelta(days=1))[0:10]
        
            yahoo_financials = YahooFinancials(idYahoo)
            data = yahoo_financials.get_historical_price_data(start_date=dateClose, 
                                                              end_date=dateRentability, 
                                                              time_interval='daily')
            
            dfClose = pd.DataFrame(data[idYahoo]['prices'])
            dfClose['Date'] = dfClose.formatted_date
            dfClose = dfClose.drop(['date', 'formatted_date'], axis=1)
            dfClose = dfClose[['Date', 'close']]
        
            return dfClose
        
        markets = list(dictMarkets.keys())
        dfPlot = pd.DataFrame(columns=['Market', 'dateToPredict', 'prediction_open', 'Date', 'close', 'profitability'])
        for market in markets:
            
            dfPredOpen = pd.read_csv(pathNew + '/' + market + '/predictionsMade_open.csv', sep=';')
            dfPredOpen['dateLess_1'] = dfPredOpen.dateToPredict.apply(lambda x: str(datetime.datetime.strptime(x, '%Y-%m-%d') - datetime.timedelta(days=1))[0:10])
            dfClose = __dataClose(market, dateRentability)
            
            dfNew = pd.merge(dfPredOpen, dfClose, left_on='dateLess_1', right_on='Date', how='inner')
            dfNew = dfNew.drop(['dateLess_1'], axis=1)
            dfNew['profitability'] = ((dfNew.prediction_open - dfNew.close) / dfNew.close)*100
            valueNew = dfNew.loc[dfNew.dateToPredict == dateRentability,:].values.tolist()[0]
            dfPlot.loc[len(dfPlot)] = [[market] + valueNew][0]
        
        fileSummary = pathNew + '/dataProfitabilityRealClosePredOpen_' + dateRentability.replace('-', '') + '.csv'
        dfPlot.to_csv(fileSummary, index=False, sep=';')
    
        colors = list(dictMarkets.values())
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        valueRentability = dfPlot.profitability.tolist()
        ax.barh(markets, valueRentability, color=colors)
        ax.invert_yaxis()
        plt.title('Profitability in the markets to date {0}'.format(dateRentability))
        plt.xlabel('Profitability')
        plt.ylabel('Markets')
        plt.savefig(pathNew + '/plot_profitability_markets_realClose_predOpen_' + dateRentability.replace('-', '') + '.png', bbox_inches='tight')
        plt.show()


    def plotBollingerBands(self, idYahoo, startDate, endDate, Candlestick=True):

        '''
        For this function it´s only necessary the idYahoo and the start and end date to extract information from yahoo.
        You can find the explanation of Bollinger bands on the page: https://www.investopedia.com/articles/technical/102201.asp
        '''
        
        newEndDate = str(datetime.datetime.strptime(endDate, '%Y-%m-%d') + datetime.timedelta(days=1))[0:10]

        yahoo_financials = YahooFinancials(idYahoo)
        data = yahoo_financials.get_historical_price_data(start_date=startDate, 
                                                          end_date=newEndDate, 
                                                          time_interval='daily')
        
        data = pd.DataFrame(data[idYahoo]['prices'])
        data['Date'] = data.formatted_date
        data = data.drop(['date', 'formatted_date'], axis=1)
        data['Middle Band'] = data['close'].rolling(window=21).mean()
        data['Upper Band'] = data['Middle Band'] + 1.96*data['close'].rolling(window=21).std()
        data['Lower Band'] = data['Middle Band'] - 1.96*data['close'].rolling(window=21).std()
        
        fig = px.line(title='Bollinger Bands for stock trading.')
        fig.add_scatter(x=data["Date"], y=data["close"], name='Closing Price - Real')
        fig.add_scatter(x=data['Date'], y=data['Upper Band'], name='Upper Band')
        fig.add_scatter(x=data['Date'], y=data['Middle Band'], name='Middle Band')
        fig.add_scatter(x=data['Date'], y=data['Lower Band'], name='Lower Band')
        
        if Candlestick:
            #Candlestick
            fig.add_trace(go.Candlestick(x=data['Date'],
                            open=data['open'],
                            high=data['high'],
                            low=data['low'],
                            close=data['close'], name = 'market data'))
        fig.write_html(self.path + "/Bollinger_Bands.html")
