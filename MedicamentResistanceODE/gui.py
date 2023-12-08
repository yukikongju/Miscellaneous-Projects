import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class ModelCATImproved:
    def __init__(self, medicament: str, proportion: str):
        self.medicament = medicament
        self.proportion = proportion
        self._load_model()

    def _load_model(self):
        filtered_rows = df[(df['medicament'] == self.medicament) & (df['proportion'] == self.proportion)]
        if not filtered_rows.empty:
            self.Kwt = st.sidebar.slider("Kwt", min_value=0.1, max_value=10.0, value=filtered_rows['Kwt'].values[0])
            self.Km = st.sidebar.slider("Km", min_value=0.1, max_value=10.0, value=filtered_rows['Km'].values[0])
            self.l_wt_sens = st.sidebar.slider("l_wt_sens", min_value=0.1, max_value=10.0, value=filtered_rows['l_wt_sens'].values[0])
            self.l_wt_tol = st.sidebar.slider("l_wt_tol", min_value=0.1, max_value=10.0, value=filtered_rows['l_wt_tol'].values[0])
            self.l_M_sens = st.sidebar.slider("l_M_sens", min_value=0.1, max_value=10.0, value=filtered_rows['l_m_sens'].values[0])
            self.l_M_tol = st.sidebar.slider("l_M_tol", min_value=0.1, max_value=10.0, value=filtered_rows['l_m_tol'].values[0])
            self.a_wt_sens = st.sidebar.slider("a_wt_sens", min_value=0.1, max_value=10.0, value=filtered_rows['a_wt_sens'].values[0])
            self.a_wt_tol = st.sidebar.slider("a_wt_tol", min_value=0.1, max_value=10.0, value=filtered_rows['a_wt_tol'].values[0])
            self.a_M_sens = st.sidebar.slider("a_M_sens", min_value=0.1, max_value=10.0, value=filtered_rows['a_m_sens'].values[0])
            self.a_M_tol = st.sidebar.slider("a_M_tol", min_value=0.1, max_value=10.0, value=filtered_rows['a_m_tol'].values[0])
            self.v_wt = st.sidebar.slider("v_wt", min_value=0.1, max_value=10.0, value=filtered_rows['v_wt'].values[0])
            self.v_m = st.sidebar.slider("v_m", min_value=0.1, max_value=10.0, value=filtered_rows['v_m'].values[0])

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
        plt.title(r'Croissance des cellules $x_{wt}^{tol}$, $x_{wt}^{sens}$, $x_{M}^{tol}$, $x_{M}^{sens}$ baigné dans le {self.medicament} avec une proportion initiale de {self.proportion}')
        plt.title(f'Croissance cellulaire dans le {self.medicament} avec une proportion initiale de {self.proportion}')

        if to_save:
            filename = f"ode_{self.medicament}_{self.proportion}"
            plt.savefig(os.path.join(GRAPHICS_DIR, filename))

        plt.show()

        return f

    def plot_cancer(self, y0):
        f = plt.figure(figsize=FIGSIZE)
        axarr = f.add_subplot(1,1,1)

        t_start, t_end, num_points = 0, 10, 100
        t, solution = self.solve(y0, t_start, t_end, num_points)
        healthy = solution[:, 0] + solution[:, 1]
        cancer = solution[:, 2] + solution[:, 3]

        plt.plot(t, cancer, label="Mutant $x_{M}$")
        plt.plot(t, healthy, label='Wild $x_{WT}$')

        plt.legend()
        plt.xlabel('Jours')
        plt.ylabel('Population')
        plt.show()
        return f


# Load the data outside the class to use it in Streamlit
df = pd.read_csv("~/Projects/Miscellaneous-Projects/MedicamentResistanceODE/data2.csv")
GRAPHICS_DIR = '~/Projects/Miscellaneous-Projects/MedicamentResistanceODE/graphics/'
FIGSIZE = (3, 3)

# Streamlit app
def main():
    st.title("Medicament Resistance Model")

    # Sidebar with parameters
    st.sidebar.header("Model Parameters")
    medicament = st.sidebar.selectbox("Select Medicament", ["Docetaxel", "Bortezomib", "Afatinib", "No Drug"])
    proportion = st.sidebar.selectbox("Select Proportion", ["10_90", "50_50", "90_10"])

    # Load and display model parameters
    model_instance = ModelCATImproved(medicament=medicament, proportion=proportion)
    st.sidebar.subheader("Model Parameters:")
    st.sidebar.text(f"Kwt: {model_instance.Kwt}")
    st.sidebar.text(f"Km: {model_instance.Km}")
    # Add other parameters as needed

    y0 = [1, 0, 1, 0]  # Initial values, update as needed
    t_start, t_end, num_points = 0, 10, 100

    # plot cancer
    st.pyplot(model_instance.plot_cancer(y0))


if __name__ == "__main__":
    main()

