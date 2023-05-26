import time
import datetime
import pandas as pd


tickers = ['MSFT', 'GOOG', 'AAPL','^GSPC']
# tickers = ['^GSPC']
interval = '1d'
period1 = int(time.mktime(datetime.datetime(2015, 1, 1, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2020, 6, 30, 23, 59).timetuple()))

# xlwriter = pd.ExcelWriter('GSPC.xlsx', engine='openpyxl')
for ticker in tickers:
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    # df.to_excel(xlwriter, sheet_name=ticker, index=False)
    df.to_csv('data/%s.csv' %ticker)

# xlwriter.save()