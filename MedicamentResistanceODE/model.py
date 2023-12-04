import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class ModelCATImproved:
    def __init__(self, l_wt_sens, l_wt_tol, l_M_sens, l_M_tol,
                 a_wt_sens, a_wt_tol, a_M_sens, a_M_tol,
                 Kwt, Km, v_sens, vtol):
        self.l_wt_sens = l_wt_sens
        self.l_wt_tol = l_wt_tol
        self.l_M_sens = l_M_sens
        self.l_M_tol = l_M_tol
        self.a_wt_sens = a_wt_sens
        self.a_wt_tol = a_wt_tol
        self.a_M_sens = a_M_sens
        self.a_M_tol = a_M_tol
        self.Kwt = Kwt
        self.Km = Km
        self.v_sens = v_sens
        self.vtol = vtol

    def model(self, y, t):
        xwt_sens, xwt_tol, xM_sens, xM_tol = y

        dydt = [
            (self.l_wt_sens * xwt_sens + self.a_wt_sens * xwt_sens * (xM_sens + xM_tol))
            * (1 - (xwt_sens + xwt_tol) / self.Kwt) - self.v_sens * xwt_sens,

            (self.l_wt_tol * xwt_tol + self.a_wt_tol * xwt_tol * (xM_sens + xM_tol))
            * (1 - (xwt_sens + xwt_tol) / self.Kwt) + self.vtol * xwt_sens,

            (self.l_M_sens * xM_sens + self.a_M_sens * xM_sens * (xwt_sens + xwt_tol))
            * (1 - (xM_sens + xM_tol) / self.Km) - self.v_sens * xM_sens,

            (self.l_M_tol * xM_tol + self.a_M_tol * xM_tol * (xwt_sens + xwt_tol))
            * (1 - (xM_sens + xM_tol) / self.Km) + self.vtol * xM_sens
        ]

        return dydt

    def solve(self, y0, t_start, t_end, num_points):
        t = np.linspace(t_start, t_end, num_points)
        solution = odeint(self.model, y0, t)
        return t, solution

    def plot_solution(self, t, solution):
        xwt_sens_solution = solution[:, 0]
        xwt_tol_solution = solution[:, 1]
        xM_sens_solution = solution[:, 2]
        xM_tol_solution = solution[:, 3]

        plt.plot(t, xwt_sens_solution, label='xwt_sens')
        plt.plot(t, xwt_tol_solution, label='xwt_tol')
        plt.plot(t, xM_sens_solution, label='xM_sens')
        plt.plot(t, xM_tol_solution, label='xM_tol')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.show()

def get_ode_solution(medicament: str, proportion: str, y0: [float], to_plot: bool = True): 
    """
    Parameters
    ----------
    medicament: str
        > "Docetaxel", "Bortezomib", "Afatinib"
    proportion: str
        > "10_90", "50_50", "90_10"
    y0: 
        > valeurs initiales du systeme ex: [1,1,1,1] => len(y0) == 4

    """
    # read data from csv
    filtered_rows = df[(df['medicament'] == medicament) & (df['proportion'] == proportion)]
    if not filtered_rows.empty:
        # Access the parameter values
        Kwt = filtered_rows['Kwt'].values[0]
        Km = filtered_rows['Km'].values[0]
        l_wt_sens = filtered_rows['l_wt_sens'].values[0]
        l_wt_tol = filtered_rows['l_wt_tol'].values[0]
        l_m_sens = filtered_rows['l_m_sens'].values[0]
        l_m_tol = filtered_rows['l_m_tol'].values[0]
        a_wt_sens = filtered_rows['a_wt_sens'].values[0]
        a_wt_tol = filtered_rows['a_wt_tol'].values[0]
        a_m_sens = filtered_rows['a_m_sens'].values[0]
        a_m_tol = filtered_rows['a_m_tol'].values[0]
        v_sens = filtered_rows['v_sens'].values[0]
        v_tol = filtered_rows['v_tol'].values[0]


    # initialize model
    model_instance = ModelCATImproved(l_wt_sens=l_wt_sens, l_wt_tol=l_wt_tol,
                                      l_M_sens=l_m_sens, l_M_tol=l_m_tol,
                                      a_wt_sens=a_wt_sens, a_wt_tol=a_wt_tol,
                                      a_M_sens=a_m_sens, a_M_tol=a_m_tol,
                                      Kwt=Kwt, Km=Km, v_sens=v_sens, vtol=v_tol)

    # compute solution and plot
    t_start, t_end, num_points = 0, 10, 100
    t, solution = model_instance.solve(y0, t_start, t_end, num_points)
    if to_plot:
        model_instance.plot_solution(t, solution)
    return solution
    

if __name__ == "__main__":
    df = pd.read_csv("MedicamentResistanceODE/data.csv")
    get_ode_solution(medicament="Docetaxel", proportion="50_50", y0=[1,1,1,1], to_plot=True)
    

