import yfinance as yf


def get_stock_closing_prices(symbol, start, end):
    try: 
        msft = yf.Ticker(symbol)
        hist = msft.history(start = start, end = end)
        prices = hist['Close']
        return prices
    except Exception as e:
        print("The Stock Symbol doesn't exist")
        


