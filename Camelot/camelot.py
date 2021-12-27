#!/bin/env/python 

import pandas as pd
import numpy as np

class Camelot(object):

    def __init__(self,c,p,distr):
        self.c = c
        self.p = p
        self.distr = distr

    def __str__(self):
        return f"journal cost: {self.c}, journal price: {self.p}, probability distribution: {self.distr}"

    def _calculate_profit(self, x):
        """ calculate profit if camelot buys x journals """
        profit = -(self.c * x)
        for _, row in self.distr.iterrows():
            d = row['x']
            perc = row['Percentage']
            profit += self.p * (min(x,d) * perc)
        return profit
        
    def get_optimal_solution(self):
        # calculer le profit si le camelot achète x journal
        self._calculate_all_profit()

        print(self.distr)

        # get max profit and number of journals to buy
        max_profit_index = 0
        for i, val in enumerate(self.distr['Profits']):
            if val > self.distr['Profits'][max_profit_index]:
                max_profit_index = i

        max_profit = self.distr['Profits'][max_profit_index]
        num_journals = self.distr['x'][max_profit_index]

        print(f'Max Profit: {max_profit}')
        print(f'Number of Journals to buy: {num_journals}')

        return num_journals, max_profit

    def _calculate_all_profit(self):
        """ calculate the best number of journal to buy to maximize profit """
        n = len(self.distr)
        profits = [0 for _ in range(n)]
        # calculate profit for each x 
        for i, row in self.distr.iterrows():
            x = row['x']
            profits[i] = self._calculate_profit(x)

        # append profits to distr
        self.distr['Profits'] = profits

def main():
    d = np.transpose([[3,4,5,6,7], [0.1, 0.2, 0.3, 0.3, 0.1]])
    distr = pd.DataFrame(data=d, columns=['x', 'Percentage'])
    camelot = Camelot(2, 2.5, distr)
    num_journals, max_profit = camelot.get_optimal_solution()


if __name__ == "__main__":
    main()

