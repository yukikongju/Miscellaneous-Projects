import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns


def plot_ode(t, y, labels: [str], title: str):
    # number of ODEs
    n = y.shape[-1]
    for i in range(n):
        plt.plot(t, y[:, i], label=labels[i])
        #  sns.scatterplot(t, y[i:], label=labels[i])
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.legend()
    plt.show()


def model_cat(y, t, K_type_intra: [float], l_type: [float], v_type: [float], a_type: [float], K_type_inter: [float]):
    """
    y = [x wt sens, x wt tol, x m sens, x m tol, x_wt, x_mi]
    """
    # --- Equations intra-specifiques
    # Wild Type WT
    dx_sens_wt = l_type[0] * y[0] * ( 1 - (y[0]+y[1]) / K_type_intra[0] ) - v_type[0] * y[0]
    dx_tol_wt = l_type[1] * y[1] * ( 1 - (y[0]+y[1]) / K_type_intra[0] ) - v_type[0] * y[0]

    # Mutant Type Mi
    dx_sens_mi = l_type[2] * y[2] * ( 1 - (y[2]+y[3]) / K_type_intra[1] ) - v_type[1] * y[2]
    dx_tol_mi = l_type[3] * y[3] * ( 1 - (y[2]+y[3]) / K_type_intra[1] ) - v_type[1] * y[2]

    # --- Equations inter-specifiques
    x_wt, x_mi = y[0] + y[1], y[2] + y[3]
    l1 = np.mean(l_type[0] + l_type[1])
    l2 = np.mean(l_type[2] + l_type[3])
    dx_wt = l1 * x_wt * ( 1 - x_wt / K_type_inter[0] + a_type[0] * x_mi)
    dx_mi = l2 * x_mi * ( 1 - x_mi / K_type_inter[1] + a_type[1] * x_wt)

    return [dx_sens_wt, dx_tol_wt, dx_sens_mi, dx_tol_mi, dx_wt, dx_mi]
    
def main():
    # --- Model CAT with [ NO  DRUGS ]
    n = 2
    l_type = np.array([0.5988, 0, 0.6499, 0])
    K_type_intra = np.array([999, 999])
    K_type_inter = np.array([999, 999])
    v_type = np.array([0, 0])
    a_type = np.array([0.4, 0.5])

    # Initial conditions
    y0_type = np.array([1, 1, 1, 1])
    y0_species = np.array([2, 2])
    y0 = np.concatenate((y0_type, y0_species))

    # Time points
    t = np.linspace(0, 10, 100)

    # Solve the system of ODEs
    y = odeint(model_cat, y0, t, args=(K_type_intra, l_type, v_type, a_type, K_type_inter))

    # plot ODE
    labels = ['dx sens wt', 'dx tol wt', 'dx sens mi', 'dx tol mi', 'dx wt', 'dx mi']
    plt.style.use('seaborn-darkgrid')
    #  sns.set(style='ticks')
    plot_ode(t=t, y=y, labels=labels, title='Basic Model CAT')


if __name__ == "__main__":
    main()
