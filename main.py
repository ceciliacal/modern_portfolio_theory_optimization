from datetime import datetime

import numpy as np
import pandas as pd
import pandas
from pandas_datareader import data
import matplotlib.pyplot as plt
#matplotlib inline


'''
Next lets look at the portfolios that matter the most.

The minimum variance portfolio
The tangency portfolio (the portfolio with highest sharpe ratio)
'''


def main():

    #step1: pull the stock price data
    print("ciao")
    start = datetime(2017, 12, 31)
    end = datetime.now()

    myData = data.DataReader(['TSLA', 'FB'], 'yahoo', start, end)
    #df = data.DataReader(['AAPL', 'FB', 'GOOGL', 'AMZN', 'MSFT'], 'yahoo', start, end)

    print(myData)

    myData = myData['Adj Close']
    print(myData)

    #Step 2: Calculate percentage change in stock prices
    tesla = myData['TSLA'].pct_change().apply(lambda x: np.log(1 + x))
    print(tesla)
    fb = myData['FB'].pct_change().apply(lambda x: np.log(1 + x))
    print(fb)
    #tesla.head()

    #varianza
    var_tesla = tesla.var()
    print("var_tesla: ",var_tesla)

    var_fb = fb.var()
    print("var_fb: " , var_fb)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()