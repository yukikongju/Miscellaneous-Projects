import numpy as np
import os
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class ModelCATImproved:
    def __init__(self, medicament: str, proportion: str):
        self.medicament = medicament
        self.proportion = proportion
        self._load_model()

    def _load_model(self):
        """
        Given medicament and proportion, init model from parameters in csv file
        """
        # read data from csv
        filtered_rows = df[(df['medicament'] == self.medicament) & (df['proportion'] == self.proportion)]
        if not filtered_rows.empty:
            # Access the parameter values
            self.Kwt = filtered_rows['Kwt'].values[0]
            self.Km = filtered_rows['Km'].values[0]
            self.l_wt_sens = filtered_rows['l_wt_sens'].values[0]
            self.l_wt_tol = filtered_rows['l_wt_tol'].values[0]
            self.l_M_sens = filtered_rows['l_m_sens'].values[0]
            self.l_M_tol = filtered_rows['l_m_tol'].values[0]
            self.a_wt_sens = filtered_rows['a_wt_sens'].values[0]
            self.a_wt_tol = filtered_rows['a_wt_tol'].values[0]
            self.a_M_sens = filtered_rows['a_m_sens'].values[0]
            self.a_M_tol = filtered_rows['a_m_tol'].values[0]
            self.v_wt = filtered_rows['v_wt'].values[0]
            self.v_m = filtered_rows['v_m'].values[0]


    def model(self, y, t):
        xwt_sens, xwt_tol, xM_sens, xM_tol = y

        dydt = [
            (self.l_wt_sens * xwt_sens + self.a_wt_sens * xwt_sens * (xM_sens + xM_tol))
            * (1 - (xwt_sens + xwt_tol) / self.Kwt) - self.v_wt * xwt_sens,

            (self.l_wt_tol * xwt_tol + self.a_wt_tol * xwt_tol * (xM_sens + xM_tol))
            * (1 - (xwt_sens + xwt_tol) / self.Kwt) + self.v_m * xwt_sens,

            (self.l_M_sens * xM_sens + self.a_M_sens * xM_sens * (xwt_sens + xwt_tol))
            * (1 - (xM_sens + xM_tol) / self.Km) - self.v_wt * xM_sens,

            (self.l_M_tol * xM_tol + self.a_M_tol * xM_tol * (xwt_sens + xwt_tol))
            * (1 - (xM_sens + xM_tol) / self.Km) + self.v_m * xM_sens
        ]

        #  dydt = [
        #      (self.l_wt_sens * xwt_sens + self.a_wt_sens * xwt_sens * (xM_sens + xM_tol))
        #      * (1 - (xwt_sens + xwt_tol) / self.Kwt) - self.v_wt * xwt_sens,

        #      (self.l_wt_tol * xwt_tol + self.a_wt_tol * xwt_tol * (xM_sens + xM_tol))
        #      * (1 - (xwt_sens + xwt_tol) / self.Kwt) + self.v_wt * xwt_sens,

        #      (self.l_M_sens * xM_sens + self.a_M_sens * xM_sens * (xwt_sens + xwt_tol))
        #      * (1 - (xM_sens + xM_tol) / self.Km) - self.v_m * xM_sens,

        #      (self.l_M_tol * xM_tol + self.a_M_tol * xM_tol * (xwt_sens + xwt_tol))
        #      * (1 - (xM_sens + xM_tol) / self.Km) + self.v_m * xM_sens
        #  ]


        return dydt

    def solve(self, y0, t_start, t_end, num_points):
        t = np.linspace(t_start, t_end, num_points)
        solution = odeint(self.model, y0, t)
        return t, solution

    def plot_solution(self, t, solution, to_save: bool = False):
        f = plt.figure(figsize=FIGSIZE)
        axarr = f.add_subplot(1,1,1)

        xwt_sens_solution = solution[:, 0]
        xwt_tol_solution = solution[:, 1]
        xM_sens_solution = solution[:, 2]
        xM_tol_solution = solution[:, 3]

        plt.plot(t, xwt_sens_solution, label='xwt_sens')
        plt.plot(t, xwt_tol_solution, label='xwt_tol')
        plt.plot(t, xM_sens_solution, label='xM_sens')
        plt.plot(t, xM_tol_solution, label='xM_tol')
        plt.legend()
        plt.xlabel('Jours')
        plt.ylabel('Population')
        #  plt.title(r'Croissance des cellules $x_{wt}^{tol}$, $x_{wt}^{sens}$, $x_{M}^{tol}$, $x_{M}^{sens}$ baigné dans le {self.medicament} avec une proportion initiale de {self.proportion}')
        #  plt.title(f'Croissance cellulaire dans le {self.medicament} avec une proportion initiale de {self.proportion}')

        if to_save:
            filename = f"ode_{self.medicament}_{self.proportion}"
            plt.savefig(os.path.join(GRAPHICS_DIR, filename))

        plt.show()

        return f


def get_ode_solution(medicament: str, proportion: str, y0: [float], to_plot: bool = True, to_save: bool = False): 
    """
    Parameters
    ----------
    medicament: str
        > "Docetaxel", "Bortezomib", "Afatinib"
    proportion: str
        > "10_90", "50_50", "90_10"
    y0: 
        > valeurs initiales du systeme ex: [1,1,1,1] => len(y0) == 4
    to_plot: bool
        > True if we want to plot

    """
    # init model
    model_instance = ModelCATImproved(medicament=medicament, proportion=proportion)

    # compute solution and plot
    t_start, t_end, num_points = 0, 10, 100
    t, solution = model_instance.solve(y0, t_start, t_end, num_points)
    if to_plot:
        plot = model_instance.plot_solution(t, solution, to_save=to_save)

    return solution
    

def get_daily_cancer_cells(medicament: str, proportion: str, y0s, to_plot: bool = True, to_save: bool = False):
    """

    """
    # init model
    model_instance = ModelCATImproved(medicament=medicament, proportion=proportion)


    # create figure
    plt.figure(figsize=FIGSIZE)


    # compute solutions for all IVP
    t_start, t_end, num_points = 0, 10, 100
    cancer_solutions = []
    for i, y0 in enumerate(y0s):
        t, solution = model_instance.solve(y0, t_start, t_end, num_points)
        cancer = solution[:, 2] + solution[:, 3]

        # plot cancer cells against days for all IVP
        if to_plot:
            plt.plot(t, cancer, label=f"IVP: {y0}")

    if to_plot:
        plt.legend()
        #  plt.title(f'Nombre de cellules cancérigènes $x_M$ après n jours pour {medicament} {proportion}')
        plt.xlabel('Jours')
        plt.ylabel('Population')

        if to_save:
            filename = f"cancer_{medicament}_{proportion}.png"
            plt.savefig(os.path.join(GRAPHICS_DIR, filename))

        plt.show()

    return cancer_solutions


def get_daily_cancer_cells2(medicament: str, proportion: str, y0, to_plot: bool = True, to_save: bool = False):
    """

    """
    # init model
    model_instance = ModelCATImproved(medicament=medicament, proportion=proportion)


    # create figure
    plt.figure(figsize=FIGSIZE)

    # compute solutions for all IVP
    t_start, t_end, num_points = 0, 10, 100
    t, solution = model_instance.solve(y0, t_start, t_end, num_points)
    healthy = solution[:, 0] + solution[:, 1]
    cancer = solution[:, 2] + solution[:, 3]
        
    # plot cancer cells against days for all IVP
    if to_plot:
        plt.plot(t, cancer, label="Mutant $x_{M}$")
        plt.plot(t, healthy, label='Wild $x_{WT}$')


    if to_plot:
        plt.legend()
        #  plt.title(f'Nombre de cellules cancérigènes $x_M$ après n jours pour {medicament} {proportion}')
        plt.xlabel('Jours')
        plt.ylabel('Population')

        if to_save:
            filename = f"cancer_{medicament}_{proportion}_both.png"
            plt.savefig(os.path.join(GRAPHICS_DIR, filename))

        plt.show()


def get_correlation_cellulaire_ivp(medicament: str, proportion: str, y0s, t1: int, is_exponential = True, to_plot: bool = True): # FIXME
    """
    Parameters
    ----------
    t1: int
        > Correlation au jour t1

    """
    # load model
    model = ModelCATImproved(medicament=medicament, proportion=proportion)

    # compute solutions for all IVP
    t_start, t_end, num_points = 0, 10, 100
    wild_types = []
    mutants = []
    for i, y0 in enumerate(y0s):
        t, solution = model.solve(y0, t_start, t_end, num_points)
        wild_types.append(solution[t1, 0] + solution[t1, 1])
        mutants.append(solution[t1, 2] + solution[t1, 3])

    # make data linear if exponential
    if is_exponential:
        mutants = np.log(mutants)

    # compute correlation
    corr_matrix = np.corrcoef(wild_types, mutants)
    corr_coeff = corr_matrix[0][1]

    if to_plot:
        plt.scatter(wild_types, mutants)
        plt.title('Correlation Cellulaire IVP')
        plt.xlabel('Wild Types')
        plt.ylabel('Mutants')
        #  plt.text(2, 6, f'Correlation Coefficient: {corr_coeff}', bbox=dict(facecolor='white', alpha=0.5))
        #  plt.text(2, 6, f'Correlation Coefficient: {corr_coeff}')
        plt.show()


    return corr_coeff
    
#  ---------------- PARTICULAR CASES -----------------------------------


def test_cases(medicament: str, proportion: str, ode_graph_title: str, cancer_graph_title: str, y0, to_save: bool = True):

    #  [ PART 1 - Plot ODE Solution ]
    solutions = get_ode_solution(medicament=medicament, proportion=proportion, y0=y0, to_plot=True, to_save=to_save)

    #  [ PART 2 - Plot Cancer cells against days for different initial conditions]
    #  y0s = [
    #          [1,1,1,1], 
    #          [2,2,2,2], 
    #          [3,3,3,3], 
    #          [5,5,5,5], 
    #          [8,8,8,8], 
    #      ]
    #  get_daily_cancer_cells(medicament=medicament, proportion=proportion, y0s=y0s, to_plot=True, to_save=to_save)


    y0 = [1,1,1,1]
    get_daily_cancer_cells2(medicament=medicament, proportion=proportion, y0=y0, to_plot=True, to_save=to_save)


    #  [ PART 3 - Correlation entre nombre de cellules en sante vs cancerigenes par jour (IVP)]
    #  get_correlation_cellulaire_ivp(medicament=medicament, proportion=proportion,
    #                                 y0s=y0s, t1=3, is_exponential=True, to_plot=True)

    
    #  [ PART 4 - Correlation entre nombre de cellules en sante vs cancerigenes par jour (proportion)]
    

if __name__ == "__main__":
    #  df = pd.read_csv("MedicamentResistanceODE/data.csv")
    df = pd.read_csv("MedicamentResistanceODE/data2.csv")
    GRAPHICS_DIR = 'MedicamentResistanceODE/graphics/'
    FIGSIZE = (3,3)

    #  test_cases(medicament= "Docetaxel", proportion= "50_50", ode_graph_title= "ode_docetaxel_50_50", cancer_graph_title= "cancer_docetaxel_50_50", to_save=True, 
    #             y0=[1,1,1,1])

    #  test_cases(medicament= "Docetaxel", proportion= "90_10", ode_graph_title= "ode_docetaxel_90_10", cancer_graph_title= "cancer_docetaxel_90_10", to_save=True, 
    #             y0=[2.70, 2.70, 1.5, 1.5])

    #  test_cases(medicament= "Docetaxel", proportion= "10_90", ode_graph_title= "ode_docetaxel_10_90", cancer_graph_title= "cancer_docetaxel_10_90", to_save=True, 
    #             y0=[1,1,1,1])

    #  test_cases(medicament= "Afatinib", proportion= "50_50", ode_graph_title= "ode_afitinib_50_50", cancer_graph_title= "cancer_afitinib_50_50", to_save=True, 
    #             y0=[1,1,1,1])

    #  test_cases(medicament= "Afatinib", proportion= "90_10", ode_graph_title= "ode_afitinib_90_10", cancer_graph_title= "cancer_afitinib_90_10", to_save=True, 
    #             y0=[2.2117, 2.2117, 1.4259, 1.4259])

    #  test_cases(medicament= "Afatinib", proportion= "10_90", ode_graph_title= "ode_afitinib_10_90", cancer_graph_title= "cancer_afitinib_10_90", to_save=True, 
    #             y0=[1,1,1,1])

    #  test_cases(medicament= "Bortezomib", proportion= "50_50", ode_graph_title= "ode_bortezomib_50_50", cancer_graph_title= "cancer_bortezomib_50_50", to_save=True, 
    #             y0=[1,1,1,1])

    #  test_cases(medicament= "Bortezomib", proportion= "10_90", ode_graph_title= "ode_bortezomib_10_90", cancer_graph_title= "cancer_bortezomib_10_90", to_save=True, 
    #             y0=[1,1,1,1])

    #  test_cases(medicament= "Bortezomib", proportion= "90_10", ode_graph_title= "ode_bortezomib_90_10", cancer_graph_title= "cancer_bortezomib_90_10", to_save=True, 
    #             y0=[1,1,1,1])

    test_cases(medicament= "No Drug", proportion= "50_50", ode_graph_title= "tmp", cancer_graph_title= "tmp", to_save=True, 
               y0=[1,0,1,0])


