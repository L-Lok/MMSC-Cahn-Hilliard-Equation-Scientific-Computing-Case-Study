import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from set_up import init_c, mass_plot, J_plot, toeplitz
import time

def coeff_matrix(h, dt, N, eps):
    """
    h:      discretization size
    dt:     time step
    N + 1:  the number of grid points
    """
    mu = dt / h**2
    # the coefficient is of a 4-block form
    # part 1 is the top-left block
    part1 = np.eye(N + 1)

    # part 2 is the top-right block
    part2 = mu * toeplitz(N, 2, -1, -2)

    coeff_ = np.concatenate((part1, part2), axis=1)

    # part 3 is the bottom-left matrix
    part3 = toeplitz(N, -(2 / eps + 2 * eps / h**2), eps / h**2, 2 * eps / h**2)

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
        b2.append(1 / eps * (c[tsp][i]) ** 3 - 3 / eps * c[tsp][i])
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
        coeff = coeff_matrix(h, dt, N, eps)
        b = b_vector(c, tsp, N, eps).T
        c[tsp + 1] = np.linalg.solve(coeff, b)[0 : N + 1]
    
    np.savetxt('SB 1D.txt', c)
    return c, dt

def simB(lb, ub, N, n = None, T_final = None, dt = None, power=4, eps=0.01, error = 1e-3):
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
    c = np.zeros((n + 1, N + 1))
    c[0] = init_c(x)

    iter = 0
    start_time = time.time()

    for tsp in tqdm(range(n)):
        coeff = coeff_matrix(h, dt, N, eps)
        b = b_vector(c, tsp, N, eps).T
        c[tsp + 1] = np.linalg.solve(coeff, b)[0 : N + 1]
        
        if np.sum((c[tsp + 1] - c[tsp])**2)**(1/2) <= error:
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
    c = np.loadtxt('SB 1D.txt')
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
    ax.plot(x, c[2], label= f"t = {2*dt:.6f}")
    ax.plot(x, c[6], label= f"t = {6*dt:.6f}")
    ax.plot(x, c[int(15)], label= f"t = {int(15)*dt:.6f}")
    ax.legend(loc="best", fontsize = 18)
    ax.grid()
    ax.set_xlabel("x", fontsize = 20)
    ax.set_ylabel("c(x, t)", fontsize = 20)
    ax.set_title("Cahn-Hilliard Equation Scheme B", fontsize = 20)

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
    # ax.set_xscale('log')
    ax.grid()
    ax.set_xlabel("time t", fontsize = 20)
    ax.set_ylabel("Free Energy J(c)", fontsize = 20)
    ax.set_title("Free Energy J(c) over time", fontsize = 20)
    plt.show()
    fig.savefig(f"./Semi Implicit Scheme B Plot dt = h^{power}")

def make_plotting(lb, ub, N, n, dt):
    """
    lb:     the lower bound for x
    up:     the upper bound for x
    N:      the number of grid points
    n:      the number of time levels
    dt:     the time step
    c:      the c-matrix
    """
    c = np.loadtxt('SB 1D.txt')

    x = np.linspace(lb, ub, N + 1)
    h = np.diff(x)[0]
    # create a 2x2 sub plots
    fig, ax = plt.subplots(2, 2)
    
    # c = c[:(n + 1)]
    # create fig and plots
    # fig = plt.figure(figsize=(18, 10))
    fig.suptitle(fr'N = {N} dt = h^{4}, Time Levels = {n}, Time = {0.001}')
    t1 = int(5000)*dt
    ax[0, 0].plot(x, c[int(5000)], linewidth = 2.0)
    ax[0, 0].set_title(f't = {t1} at {int(5000)}')
    ax[0, 0].axhline(y = 1, linestyle = 'dashed', color = 'black', linewidth = 2.0)
    ax[0, 0].axhline(y = -1, linestyle = 'dashed', color = 'black', linewidth = 2.0)
    ax[0, 0].set_xlim(-0.1, 1.2)
    ax[0, 0].set_ylim(-1.2, 1.2)
    ax[0, 0].axis('off')

    
    t2 = int(n / 2)*dt
    ax[0, 1].plot(x, c[int(n / 2)], linewidth = 2.0)
    ax[0, 1].set_title(f't = {t2} at {int(n/2)}')
    ax[0, 1].axhline(y = 1, linestyle = 'dashed', color = 'black', linewidth = 2.0)
    ax[0, 1].axhline(y = -1, linestyle = 'dashed', color = 'black', linewidth = 2.0)
    ax[0, 1].set_xlim(-0.1, 1.2)
    ax[0, 1].set_ylim(-1.2, 1.2)
    ax[0, 1].axis('off')

    t3 = int(3*n/4)*dt
    ax[1, 0].plot(x, c[int(3 * n / 4)], linewidth = 2.0)
    ax[1, 0].set_title(f't = {t3} at {int(3*n/4)}')
    ax[1, 0].axhline(y = 1, linestyle = 'dashed', color = 'black', linewidth = 2.0)
    ax[1, 0].axhline(y = -1, linestyle = 'dashed', color = 'black', linewidth = 2.0)
    ax[1, 0].set_xlim(-0.1, 1.2)
    ax[1, 0].set_ylim(-1.2, 1.2)
    ax[1, 0].axis('off')

    t4 = int(n)*dt
    ax[1, 1].plot(x, c[n], linewidth = 2.0)
    ax[1, 1].set_title(f't = {t4} at {n}')
    ax[1, 1].axhline(y = 1, linestyle = 'dashed', color = 'black', linewidth = 2.0)
    ax[1, 1].axhline(y = -1, linestyle = 'dashed', color = 'black', linewidth = 2.0)
    ax[1, 1].set_xlim(-0.1, 1.2)
    ax[1, 1].set_ylim(-1.2, 1.2)
    ax[1, 1].axis('off')

    plt.show()
    fig.savefig("./Scheme B Time-Evoluation Plot")

if __name__ == "__main__":
    lb = 0
    ub = 1
    N = 127
    n = 26
    power = 1
    c, dt = simulation(lb=lb, ub=ub, N=N, n=n, power=power)
    x, h = np.linspace(lb, ub, N + 1, retstep=True)
    plotting(lb=lb, ub=ub, N=N, n=n, dt=h, power = power)
