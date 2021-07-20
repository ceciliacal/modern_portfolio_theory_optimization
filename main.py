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

    #import data
    df = data.DataReader(['AAPL', 'FB', 'GOOGL', 'AMZN', 'MSFT'], 'yahoo', start, end)

    print(df)

    #get closing price
    df = df['Adj Close']
    print(df)

    #It is common practice in portfolio optimization to take log of returns for calculations of covariance and correlation.
    # Percentage change in stock prices (everyday)
    # log of returns is time additive!!
    logChange = df.pct_change().apply(lambda x: np.log(1 + x))
    print("logChange:\n",logChange)


    #The variance in prices of stocks of each asset are an important indicator of how volatile this investment will be (how returns can fluctuate).
    apple = logChange['AAPL']
    fb = logChange['FB']
    google = logChange['GOOGL']
    amazon = logChange['AMZN']
    microsoft = logChange['MSFT']

    var_apple = apple.var()
    var_fb = fb.var()
    var_google = google.var()
    var_amazon = amazon.var()
    var_microsoft = microsoft.var()

    print("\nvar_apple:", var_apple,"var_fb:",var_fb,"var_google:",var_google,"var_amazon:",var_amazon,"var_microsoft:",var_microsoft,"\n" )


    # Volatility is given by the annual standard deviation. We multiply by 250 because there are 250 trading days/year.
    ann_sd = logChange.std().apply(lambda x: x * np.sqrt(250))
    print("annual sd:\n", ann_sd)

    #covariance
    cov_matrix = logChange.cov()
    print("covariance matrix:\n", cov_matrix)

    #correlation
    corr_matrix = logChange.corr()
    print("correlation matrix:\n", corr_matrix)

    #expected return (individual)
    #yearly returns for individual companies (argument 'Y' stands for "yearly")
    #individual_expectedReturn
    ind_er = df.resample('Y').last().pct_change().mean()
    print("individual expected return:\n", ind_er)

    #now, to compute the portfolio expected return we need to multiply each return for its weight





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()