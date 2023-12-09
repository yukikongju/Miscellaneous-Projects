import numpy as np
from scipy.optimize import fsolve
import sympy as sp

# Define the variables and parameters
x, y = sp.symbols('x y')
params = {'a': 1, 'b': 1}

# Define the system of ODEs
ode_system = [
    params['a'] * x - x**3 - y,
    params['b'] * x
]

# Define the equilibrium equation
equilibrium_equation = [equation.subs({x: 0, y: 0}) for equation in ode_system]

# Find equilibrium points using fsolve
equilibrium_points = fsolve(lambda xy: [equation.subs({x: xy[0], y: xy[1]}) for equation in equilibrium_equation], [0, 0])

# Display equilibrium points
print("Equilibrium Points:", equilibrium_points)

# Define the Jacobian matrix
jacobian_matrix = sp.Matrix([[sp.diff(equation, var) for var in [x, y]] for equation in ode_system])

# Evaluate the Jacobian matrix at the equilibrium point
jacobian_at_equilibrium = jacobian_matrix.subs({x: equilibrium_points[0], y: equilibrium_points[1]})

# Display the Jacobian matrix
print("Jacobian Matrix at Equilibrium Point:")
print(jacobian_at_equilibrium)

# Compute eigenvalues and eigenvectors
jacobian_numpy = np.array(jacobian_at_equilibrium.evalf(), dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(jacobian_numpy)

# Display the eigenvalues and eigenvectors
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)


# check if all eigenvalues have real negative parts
if all(np.real(eig) < 0 for eig in eigenvalues):
    print("All eigenvalues have a real negative part.")
else:
    print("Not all eigenvalues have a real negative part.")


