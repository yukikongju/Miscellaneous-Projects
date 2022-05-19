import os, sys, argparse

# change directory to use scripts
SCRIPTS_PATH = os.getcwd() + '/ML-in-Finance/Regression/StockPrediction/Scripts/'
os.chdir(SCRIPTS_PATH)
sys.path.insert(1, SCRIPTS_PATH)


from STOCKS_FUNCTIONS import get_stock_closing_prices
from OLS_FUNCTIONS import check_OLS_hypothesis


########## Step 1: retrieve stock data

# get stock symbols 
#  stock1 = sys.argv[1]
#  stock2 = sys.argv[2]
#  start_time = sys.argv[3]
#  end_time = sys.argv[4]


stock1 = 'GOOGL'
stock2 = 'MSFT'
start_time = '2015-01-01'
end_time =  '2016-01-01'

X = get_stock_closing_prices(stock1, start_time, end_time)
Y = get_stock_closing_prices(stock2, start_time, end_time)

########## Step 2: Check OLS Hypothesis


check_OLS_hypothesis(X,Y)


