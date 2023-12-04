import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def plot_ode(t, y):
    # number of ODEs
    n = y.shape[-1]
    for i in range(n):
        plt.plot(t, y[:, i], label='y1')
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.legend()
    plt.show()


def model_cat(y, t, n, l_type, K_type_intra, v_type, K_type_inter, a_type):
    """
    Parameters
    ----------
    > n: number of species (WT, Mi)
    > y = [ x_{sens}^{WT}, x_{tol}^{WT}, ..., x_{sens}^{M}, x_{tol}^{M}] => size: (2*n)
    > l_type => size: (2*n)
    > v_type => size: n
    > K_type => size: n

    #  y = [ x_{sens}^{type_i}, x_{tol}^{type_i}, ..., x_{sens}^{type_n}, x_{tol}^{type_n}]
    """
    # Extract variables from the input array y
    x_type = [t for i, t in enumerate(y[:2*n-2]) if i % 2 == 0]
    x_tol = [t for i, t in enumerate(y[:2*n-2]) if i % 2 == 1]
    x_species = y[2*n-2:]

    # Initialize the array to store the derivatives
    dydt_intras = np.zeros_like(y)
    dydt_inter = np.zeros(n)

    # -- Equations intra-specifiques
    for i in range(n):
        # sens
        dydt_intras[2*i] = l_type[2*i] * x_type[i] * (1 - (np.sum(y) / K_type_intra[i])) - (v_type[i] * x_type[i])

        # tol
        dydt_intras[2*i+1] = l_type[2*i+1] * x_tol[i] * (1 - np.sum(y) / K_type_intra[i]) - (v_type[i] * x_type[i])

    # -- Equations inter-specifiques
    for i in range(n):
        l_species = np.mean(l_type[2*i: 2*(i+1)])
        x_species = np.sum(dydt_intras[2*i: 2*(i+1)])
        x_others = np.sum(x_type) + np.sum(x_tol) - x_type[i] - x_tol[i]

        # wild type WT
        dydt_inter[i] = l_species * x_species * ( 1 - x_species * K_type_inter[0] + a_type[i] * x_others)

        # mutant Mi
        dydt_inter[i] = l_species * x_species * ( 1 - x_species * K_type_inter[0] + a_type[i] * x_others)

    #  return [dydt_intras, dydt_inter]
    return np.vstack((dydt_intras, dydt_inter))

# Example usage: [ No drugs ]
n = 2  # Number of species: WT, Mi
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
y = odeint(model_cat, y0, t, args=(n, l_type, K_type_intra, v_type, K_type_inter, a_type))

# plot ODE
plot_ode(t, y)


