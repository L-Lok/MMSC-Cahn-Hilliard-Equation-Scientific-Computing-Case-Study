import numpy as np
from scipy.integrate import trapz

# initial condition
def init_c(x):
    """
    Initial condition 1-d
    """
    return np.cos(np.pi * x)
        # return np.sin(np.pi * x)

def init_c_2d(x, y):
    """
    Initial condition 2-d
    """
    return np.cos(np.pi*x*y)

# Double well potential
Phi = lambda c: (1 - c**2) ** 2 / 4
DPhi = lambda c: c**3 - c
DDPhi = lambda c: 3 * c**2 - 1


# m(c)
def Mass(c_tn, h):
    """
    c_tn:  the array of c-values at nth time level

    h:      the discretization size along x
    """
    N = len(c_tn)
    return (
        h / 2 * (c_tn[0] + 2 * np.sum([c_tn[i] for i in range(1, N - 1, 1)]) + c_tn[-1])
    )


def mass_plot(c, n, h, dt):
    """
    c:      the c-matrix
    n:      the number of time levels
    dt:     the time step
    h:      the discretization size
    """
    m = []
    T = [i * dt for i in range(n)]
    for n_ in range(n):
        m_ = Mass(c[n_], h)
        m.append(m_)

    return T, m


# J(c)
def J(c_tn, h, eps=0.01):
    """
    c_tn:   the array of c-values at nth time level

    h:      the discretization size along x
    """
    N = len(c_tn)
    part1 = 1 / eps * (Phi(c_tn[0]) + Phi(c_tn[-1]))
    part2 = 2 * np.sum(
        [
            1 / eps * Phi(c_tn[i])
            + eps / 2 * ((c_tn[i + 1] - c_tn[i - 1]) / (2 * h)) ** 2
            for i in range(1, N - 1, 1)
        ]
    )
    return h / 2 * (part1 + part2)


def J_plot(c, n, h, dt):
    """
    c:      the c-matrix
    n:      the number of time levels
    dt:     the time step
    h:      the discretization size
    """
    J_vals = []
    T = [i * dt for i in range(n)]
    for n_ in range(n):
        J_ = J(c[n_], h)
        J_vals.append(J_)

    return T, J_vals


# Newton solver
def Newton_system(F, J, x, tol, h, dt, N, c, tsp, eps=0.01):
    """
    Solve nonlinear system F=0 by Newton's method.
    J is the Jacobian of F.
    Both F and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < tol.
    """
    F_value = F(x, h, dt, N, c, tsp, eps)
    F_norm = np.linalg.norm(F_value, ord=2)  # l2 norm of vector
    iteration_counter = 0
    while abs(F_norm) > tol and iteration_counter < 150:
        delta = np.linalg.solve(J(x, h, dt, N, eps), (-1) * np.array(F_value))
        x = x + delta
        F_value = F(x, h, dt, N, c, tsp, eps)
        F_norm = np.linalg.norm(F_value, ord=2)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > tol:
        iteration_counter = -1
    return x, iteration_counter


# Toeplitz coefficient matrix
def toeplitz(N, main_diag_val, side_val, bc_val):
    diag = np.ones(N + 1) * main_diag_val
    diag_p = np.insert(np.ones(N - 1) * (side_val), 0, bc_val)
    diag_m = np.concatenate((np.ones(N - 1) * (side_val), [bc_val]))
    toep_mat = np.diag(diag, k=0) + np.diag(diag_m, k=-1) + np.diag(diag_p, k=1)
    return toep_mat


def calc_mass(c, h):
        # c is two dimensional
        integral_x = trapz(c,dx=h,axis =0)
        mass = trapz(integral_x,dx=h,axis =0)
        return mass
    
def calc_j(c, h, eps):
    # c is two dimensional
    phi_c = np.power(1-np.power(c[1:,1:],2),2)
    x_derivative = c[1:,1:]-c[:-1,1:]
    y_derivative = c[1:,1:]-c[1:,:-1]
    derivative = np.power(x_derivative,2)+np.power(y_derivative,2)
    integral_x = trapz((1/(4*eps))*phi_c+(eps/2)*derivative,dx=h,axis =0)
    j = trapz(integral_x,dx=h,axis =0)
    return j
