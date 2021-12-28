#!/bin/env/python

import math


class FileAttente(object):

    def __init__(self, lambdaa, mu, s):
        self.mu = mu                # taux d'entrée
        self.s = s                  # num of serveurs
        self.lambdaa = lambdaa      # taux de sortie
        self.lambda_mu = self.lambdaa / self.mu

        self.rho = self.get_rho()
        self.P0 = self.get_P0()
        self.Lq = self.get_average_people_in_queue()
        self.Wq = self.get_average_waiting_time_in_queue()
        self.W = self.get_average_waiting_time_in_system()
        self.L = self.get_average_people_in_system()

    def __str__(self):
        return f"s:{self.s}, mu: {self.mu}, lambda: {self.lambdaa}, rho: {self.rho},\
                P_0: {self.P0}, L_q: {self.Lq}, L: {self.L}, W_q: {self.Wq}, W: {self.W}"

    def get_rho(self):
        pass

    def get_average_waiting_time_in_system(self): # get W
        return self.Wq + (self.mu)**(-1)

    def get_average_waiting_time_in_queue(self): # get W_q
        return self.Lq / self.lambdaa

    def get_average_people_in_system(self): # get L
        return self.lambdaa * self.W

    def get_average_people_in_queue(self): # get L_q
        return self.lambdaa * self.Wq

        
    def get_P0(self):
        pass

    def get_Pn(self, n):
        pass


class MM1(FileAttente):

    def __init__(self, lambdaa, mu):
        super().__init__(lambdaa, mu, 1)

    def get_rho(self):
        #  return self.lambdaa / self.mu
        return self.lambda_mu

    def get_P0(self):
        return 1 - self.rho

    def get_Pn(self, n):
        return (self.rho ** n)*self.P0

    def get_average_people_in_system(self):
        return (self.lambdaa) / (self.mu - self.lambdaa)

    def get_average_people_in_queue(self):
        return (self.lambdaa**2) / (self.mu * (self.mu - self.lambdaa))

    def get_average_waiting_time_in_system(self):
        return (self.mu - self.lambdaa)**(-1)


    def get_average_waiting_time_in_queue(self):
        return (self.lambdaa) / (self.mu * (self.mu - self.lambdaa))

class MMs(FileAttente):

    def __init__(self, lambdaa, mu, s):
        super().__init__(lambdaa, mu, s)

    def get_rho(self):
        return self.lambdaa / (self.mu * self.s)

    def get_P0(self):
        # calculate p1
        p1 = 0
        for n in range(self.s -1):
            p1 += (self.lambda_mu**n)/math.factorial(n)

        # calculate p2
        p21 = (self.lambda_mu**self.s)/math.factorial(self.s)
        p22 = (1-self.rho)**(-1)
        p2 = p21*p22

        return (p1+p2)**(-1)

    def get_Pn(self, n):
        if n <= self.s:
            cn = (self.lambda_mu**n) / math.factorial(n)
        else: 
            bottom1 = self.s ** (n-self.s)
            bottom2 = math.factorial(self.s)
            cn = (self.lambda_mu**n) / (bottom1*bottom2)
        return cn * self.P0


    def get_average_people_in_queue(self):
        top = self.P0 * (self.lambda_mu**self.s) * self.rho
        bottom = math.factorial(self.s) * ((1-self.rho)**2)
        return top/bottom


def main():
    test_MM1()
    test_MMs()

def test_MM1():
    mm1 = MM1(2,3)
    print(mm1)

def test_MMs():
    mms = MMs(2,3,2)
    print(mms)

if __name__ == "__main__":
    main()


