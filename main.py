from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import pandas
from pandas_datareader import data
import matplotlib.pyplot as plt



'''
Next lets look at the portfolios that matter the most.

The minimum variance portfolio
The tangency portfolio (the portfolio with highest sharpe ratio)
'''


def main():

    #pull the stock price data
    start = datetime(2017, 12, 31)
    end = datetime.now()

    #import data
    df = data.DataReader(['AAPL', 'FB', 'GOOGL', 'AMZN', 'MSFT'], 'yahoo', start, end)
    df2 = data.DataReader(['sp500'], 'fred', start, end)

    print("df2:",df2)
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

    # covariance and correlation matrix to understand how different assets behave with respect to each other

    #covariance
    cov_matrix = logChange.cov()
    print("covariance matrix:\n", cov_matrix)

    #The covariance between Apple and Apple, or Nike and Nike is the variance of that asset

    #correlation
    corr_matrix = logChange.corr()
    print("correlation matrix:\n", corr_matrix)

    #expected return (individual)
    #yearly returns for individual companies (argument 'Y' stands for "yearly")
    #individual_expectedReturn
    ind_er = df.resample('Y').last().pct_change().mean()
    print("individual expected return:\n", ind_er)

    #now, to compute the portfolio expected return we need to multiply each return for its weight
    #but we will do it later on once we have got the optimal

    # Creating a table for visualising returns and volatility of individual assets
    assets = pd.concat([ind_er, ann_sd], axis=1)
    assets.columns = ['Returns', 'Volatility']
    print("assets:\n",assets)


    #Next, to plot the graph of efficient frontier, we need run a loop.
    #In each iteration, the loop considers different weights for assets and calculates the return and volatility
    # of that particular portfolio combination.
    # We run this loop a 1000 times.

    p_ret = []  # Define an empty array for portfolio returns
    p_vol = []  # Define an empty array for portfolio volatility
    p_weights = []  # Define an empty array for asset weights

    num_assets = len(df.columns)
    num_portfolios = 10000

    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights, ind_er)  # Returns are the product of individual expected returns of asset and its
        # weights
        p_ret.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()  # Portfolio Variance
        sd = np.sqrt(var)  # Daily standard deviation
        ann_sd = sd * np.sqrt(250)  # Annual standard deviation = volatility
        p_vol.append(ann_sd)

    resultData = {'Returns': p_ret, 'Volatility': p_vol}

    for counter, symbol in enumerate(df.columns.tolist()):
        # print(counter, symbol)
        resultData[symbol + ' weight'] = [w[counter] for w in p_weights]
    portfolios = pd.DataFrame(resultData)
    print("1000 portfolios:\n",portfolios) # Dataframe of the 1000 portfolios created

    #There are a number of portfolios with different weights, returns and volatility
    #Plotting the returns and volatility from this dataframe will show the efficient frontier for our portfolio.

    #plt.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10, 10])
    portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10, 10])
    plt.show()

    #On this graph, we can see the combination of weights that will give all possible combinations:
    #Minimum volatility (left most point), Maximum returns (top most point) and everything in between

    #let's calculate minimum volatility portfolio
    min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
    # idxmin() gives us the minimum value in the column specified.
    print("\nmin_vol_port: \n",min_vol_port)

    #minimum volatility is in this portfolio (min_vol_port)
    #now we'll plot this point on the efficient frontier graph

    # plotting the minimum volatility portfolio
    plt.subplots(figsize=[10, 10])
    plt.scatter(portfolios['Volatility'], portfolios['Returns'], marker='o', s=10, alpha=0.3)
    plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)

    plt.show()

    '''
    It is worthwhile to note that any point to the right of efficient frontier boundary is a sup-optimal portfolio.
    We found the portfolio with minimum volatility, but its return is pretty low
    Now we want to try to maximize our return, even if it is a tradeoff with some level of risk.

    The question is: how do we find this optimal risky portfolio and finally optimize our portfolio to the maximum?
    By using a parameter called the Sharpe Ratio.
    '''

    #The optimal risky portfolio is the one with the highest Sharpe ratio (cfr formula)
    #Now we need to define the risk factor in order to find optimal portfolio

    rf = 0.00  # risk factor
    optimal_risky_port = portfolios.iloc[((portfolios['Returns'] - rf) / portfolios['Volatility']).idxmax()]
    print("\noptimal_risky_port: \n", optimal_risky_port)


    #weighted optimal portfolio's variance
    weightsDictionary = {'AAPL': optimal_risky_port[2], 'FB': optimal_risky_port[3], 'GOOGL': optimal_risky_port[4],
                          'AMZN': optimal_risky_port[5], 'MSFT': optimal_risky_port[6]}
    port_var = cov_matrix.mul(weightsDictionary, axis=0).mul(weightsDictionary, axis=1).sum().sum()
    print("\noptimal portfolio's variance: \n",port_var)

    #optimal portfolio expected returns
    weightsArray = [optimal_risky_port[2], optimal_risky_port[3], optimal_risky_port[4],
                         optimal_risky_port[5], optimal_risky_port[6]]
    port_er = (weightsArray*ind_er).sum()
    print("portfolio expected returns: \n", port_er)



    # Plotting optimal portfolio
    plt.subplots(figsize=(10, 10))
    plt.scatter(portfolios['Volatility'], portfolios['Returns'], marker='o', s=10, alpha=0.3)
    plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
    plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
    plt.show()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()