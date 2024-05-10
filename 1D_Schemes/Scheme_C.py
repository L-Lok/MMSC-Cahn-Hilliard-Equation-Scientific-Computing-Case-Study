import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from set_up import init_c, mass_plot, J_plot, toeplitz
import time

def coeff_matrix(h, dt, N, c, tsp, eps):
    """
    h:      discretization size
    dt:     time step
    N + 1:  the number of grid points
    tsp:    the current time level
    c:      the c-matrix
    """
    mu = dt / h**2
    # the coefficient is of a 4-block form
    # part 1 is the top-left block
    part1 = np.eye(N + 1)
    # part 2 is the top-right block
    part2 = mu * toeplitz(N, 2, -1, -2)

    coeff_ = np.concatenate((part1, part2), axis=1)

    # part 3 is the bottom-left matrix
    diag_part3 = []
    for i in range(N + 1):
        diag_part3.append(1 / eps - 3 / eps * (c[tsp][i]) ** 2 - 2 * eps / h**2)
    diag_p = np.insert(np.ones(N - 1) * (eps / h**2), 0, 2 * eps / h**2)
    diag_m = np.concatenate((np.ones(N - 1) * (eps / h**2), [2 * eps / h**2]))
    part3 = np.diag(diag_part3, k=0) + np.diag(diag_m, k=-1) + np.diag(diag_p, k=1)

    # part 4 is the bottom-right matrix
    part4 = part1

    coeff__ = np.concatenate((part3, part4), axis=1)

    coeff = np.concatenate((coeff_, coeff__))

    return coeff


def b_vector(c, tsp, N, eps):
    """
    c:      the c-matrix
    tsp:    the current time level
    N + 1:  the number of grid points
    """
    b1 = c[tsp]
    b2 = []
    for i in range(N + 1):
        b2.append(-2 / eps * (c[tsp][i]) ** 3)
    b = np.concatenate((b1, b2))
    return b


def simulation(lb, ub, N, n, dt = None, power=4, eps=0.01):
    """
    lb:     lower bound
    ub:     upper bound
    N + 1:  the number of grid points
    n:      the number of time levels
    power:  dt = h^4
    """
    x = np.linspace(lb, ub, N + 1)
    h = np.diff(x)[0]
    if dt == None:
        dt = h**power

    # set up the storage arrays
    c = np.zeros((n + 1, N + 1))
    c[0] = init_c(x)


    for tsp in tqdm(range(n)):
        coeff = coeff_matrix(h, dt, N, c, tsp, eps)
        b = b_vector(c, tsp, N, eps).T
        c[tsp + 1] = np.linalg.solve(coeff, b)[0 : N + 1]

    np.savetxt('SC 1D.txt', c)
    return c, dt

def simC(lb, ub, N, n = None, dt = None, T_final = None, power=4, eps=0.01, error = 1e-3):
    """
    lb:     lower bound
    ub:     upper bound
    N + 1:  the number of grid points
    n:      the number of time levels
    power:  dt = h^4
    """
    x = np.linspace(lb, ub, N + 1)
    h = np.diff(x)[0]
    if dt == None:
        dt = h**power

    n = int(T_final/dt + 1)

    # set up the storage arrays
    c = []
    c.append(init_c(x))
    

    iter = 0
    start_time = time.time()

    for tsp in tqdm(range(n)):
        coeff = coeff_matrix(h, dt, N, c, tsp, eps)
        b = b_vector(c, tsp, N, eps).T
        c.append(np.linalg.solve(coeff, b)[0 : N + 1])
        
        if np.sum((c[tsp+ 1] - c[tsp])**2)**(1/2) <= error:
            iter = tsp + 1
            break
    
    end_time = time.time()

    return c[iter], dt, end_time - start_time, iter


def plotting(lb, ub, N, n, dt, power):
    """
    lb:     the lower bound for x
    up:     the upper bound for x
    N:      the number of grid points
    n:      the number of time levels
    dt:     the time step
    c:      the c-matrix
    """
    c = np.loadtxt('SC 1D.txt')
    x = np.linspace(lb, ub, N + 1)
    h = np.diff(x)[0]
    # create a 2x2 sub plots
    gs = gridspec.GridSpec(9, 10)

    # create fig and plots
    fig = plt.figure(figsize=(35, 20))
    fig.suptitle(rf"N = {N} dt = h^{power}", fontsize=20)
    ax = fig.add_subplot(gs[0:5, 0:5])  # row 0, col 0
    ax.plot(x, c[0], label="initial")
    ax.plot(x, c[n], label="final")
    ax.plot(x, c[300], label= f"t = {300*dt:.6f}")
    ax.plot(x, c[1500], label= f"t = {1500*dt:.6f}")
    ax.plot(x, c[2500], label= f"t = {2500*dt:.6f}")
    ax.legend(loc="best", fontsize = 12)
    ax.grid()
    ax.set_xlabel("x", fontsize = 20)
    ax.set_ylabel("c(x, t)", fontsize = 20)
    ax.set_title("Cahn-Hilliard Equation Scheme C", fontsize = 20)

    T_mass, mass = mass_plot(c, n, h, dt)
    ax = fig.add_subplot(gs[0:5, 6:])
    ax.plot(T_mass, mass)
    ax.grid()
    ax.set_xlabel("time t", fontsize = 20)
    ax.set_ylabel("Mass m(c)", fontsize = 20)
    ax.set_title("Mass m(c) over time", fontsize = 20)
    # ax.set_xscale('log')

    T_J, J = J_plot(c, n, h, dt)
    ax = fig.add_subplot(gs[6:, 3:8])
    ax.plot(T_J, J)
    # ax.set_xlim(0, 0.005)
    # ax.set_xscale('log')
    ax.grid()
    ax.set_xlabel("time t", fontsize = 20)
    ax.set_ylabel("Free Energy J(c)", fontsize = 20)
    ax.set_title("Free Energy J(c) over time", fontsize = 20)
    plt.show()
    fig.savefig(f"./Semi Implicit Scheme C Plot dt = h^{power}")


if __name__ == "__main__":
    lb = 0
    ub = 1
    N = 127
    n = 10000
    power = 3
    c, dt = simulation(lb=lb, ub=ub, N=N, n=n, power=power)
    x, h = np.linspace(lb, ub, N + 1, retstep=True)
    plotting(lb=lb, ub=ub, N=N, n=n, dt=h**power, power=power)
