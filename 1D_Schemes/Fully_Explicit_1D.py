import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from set_up import init_c, Phi, DPhi, DDPhi, Mass, J, mass_plot, J_plot
import time

# boundary condition
def bcs_c(
    c,
    w,
    h,
    dt,
    tn,
):
    """
    c:   the matrix of c-values
    w:   the matrix of w-values
    """
    # x = a
    c[tn + 1, 0] = dt * 2 * (w[tn, 1] - w[tn, 0]) / h**2 + c[tn, 0]

    # x = b
    c[tn + 1, -1] = dt * 2 * (w[tn, -2] - w[tn, -1]) / h**2 + c[tn, -1]


def bcs_w(c, w, h, tn, eps=0.01):
    """
    c:   the matrix of c-values
    w:   the matrix of w-values
    """
    # x = a
    w[tn + 1, 0] = (
        1 / eps * DPhi(c[tn + 1, 0]) - 2 * eps * (c[tn + 1, 1] - c[tn + 1, 0]) / h**2
    )

    # x = b
    w[tn + 1, -1] = (
        1 / eps * DPhi(c[tn + 1, -1]) - 2 * eps * (c[tn + 1, -2] - c[tn + 1, -1]) / h**2
    )


# initial conditions
def init(c0, x, h, N, eps=0.01):
    """
    Define the initial conditions

    c0:         the initial c(x) at t = 0
    h:          the discretization size along x
    x:          the discretization
    N + 1:      the number of grid points
    """
    c_0 = [c0(x_) for x_ in x]

    w0 = 1 / eps * DPhi(c_0[0]) - 2 * eps * (c_0[1] - c_0[0]) / h**2
    w_0 = [w0]
    for i in range(1, N, 1):
        w_0.append(
            1 / eps * DPhi(c_0[i]) - eps * (c_0[i + 1] - 2 * c_0[i] + c_0[i - 1]) / h**2
        )
    w_0.append(1 / eps * DPhi(c_0[-1]) - 2 * eps * (c_0[-2] - c_0[-1]) / h**2)

    return c_0, w_0


# perform simulation
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
    w = np.zeros((n + 1, N + 1))

    # initialize the arrays
    c_0, w_0 = init(init_c, x, h, N)
    c[0], w[0] = c_0, w_0

    for n_ in tqdm(range(n)):
        for j in range(1, N, 1):
            c[n_ + 1, j] = (
                w[n_, j + 1] - 2 * w[n_, j] + w[n_, j - 1]
            ) * dt / h**2 + c[n_, j]
        bcs_c(c, w, h, dt, n_)

        for j in range(1, N, 1):
            w[n_ + 1, j] = (
                1 / eps * DPhi(c[n_ + 1, j])
                - eps * (c[n_ + 1, j + 1] - 2 * c[n_ + 1, j] + c[n_ + 1, j - 1]) / h**2
            )

        bcs_w(c, w, h, n_)

    np.savetxt('fully EE 1D.txt', c)

    return c, dt

# perform simulation
def simE(lb, ub, N, n = None, T_final = None, dt = None, power=4, eps=0.01, error = 1e-6):
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
    w = np.zeros((n + 1, N + 1))

    # initialize the arrays
    c_0, w_0 = init(init_c, x, h, N)
    c[0], w[0] = c_0, w_0
    start_time = time.time()
    iter = 0
    for n_ in tqdm(range(n)):
        for j in range(1, N, 1):
            c[n_ + 1, j] = (
                w[n_, j + 1] - 2 * w[n_, j] + w[n_, j - 1]
            ) * dt / h**2 + c[n_, j]
        bcs_c(c, w, h, dt, n_)

        for j in range(1, N, 1):
            w[n_ + 1, j] = (
                1 / eps * DPhi(c[n_ + 1, j])
                - eps * (c[n_ + 1, j + 1] - 2 * c[n_ + 1, j] + c[n_ + 1, j - 1]) / h**2
            )

        bcs_w(c, w, h, n_)

        if np.sum((c[n_ + 1] - c[n_])**2)**(1/2) <= error:
            iter = n_ + 1
            break
    
    end_time = time.time()

    return c, dt, end_time - start_time, iter

def plotting(lb, ub, N, n, dt):
    """
    lb:     the lower bound for x
    up:     the upper bound for x
    N:      the number of grid points
    n:      the number of time levels
    dt:     the time step
    c:      the c-matrix
    """
    c = np.loadtxt('fully EE 1D.txt')

    x = np.linspace(lb, ub, N + 1)
    h = np.diff(x)[0]
    # create a 2x2 sub plots
    gs = gridspec.GridSpec(9, 10)
    
    # c = c[:(n + 1)]
    # create fig and plots
    fig = plt.figure(figsize=(35, 20))
    fig.suptitle(fr'N = {N} dt = h^{4}, Time Levels = {n}', fontsize = 20)
    ax = fig.add_subplot(gs[0:5, 0:5])  # row 0, col 0
    ax.plot(x, c[0], label="initial")
    ax.plot(x, c[n], label="final")
    ax.plot(x, c[int(n / 15)], label= f"t = {int(n/15)*dt:.6f}")
    ax.plot(x, c[int(n / 8)], label= f"t = {int(n/8)*dt:.6f}")
    ax.plot(x, c[int(3 * n / 5)], label= f"t = {int(3*n/5)*dt:.6f}")
    ax.legend(loc="best", fontsize = 18)
    ax.grid()
    ax.set_xlabel("x", fontsize = 20)
    ax.set_ylabel("c(x, t)", fontsize = 20)
    ax.set_title("Cahn-Hilliard Equation Fully Explicit Scheme", fontsize = 20)

    T_mass, mass = mass_plot(c, n, h, dt)
    ax = fig.add_subplot(gs[0:5, 6:])
    ax.plot(T_mass, mass)
    ax.grid()
    ax.set_xlabel("time t", fontsize = 20)
    ax.set_ylabel("Mass m(c)", fontsize = 20)
    ax.set_title("Mass m(c) over time", fontsize = 20)
    ax.set_xscale('log')

    T_J, J = J_plot(c, n, h, dt)
    ax = fig.add_subplot(gs[6:, 3:8])
    ax.plot(T_J, J)
    ax.set_xscale('log')
    ax.grid()
    ax.set_xlabel("time t", fontsize = 20)
    ax.set_ylabel("Free Energy J(c)", fontsize = 20)
    ax.set_title("Free Energy J(c) over time", fontsize = 20)
    plt.show()
    fig.savefig("./Fully Explicit Scheme Plot")


def make_plotting(lb, ub, N, n, dt):
    """
    lb:     the lower bound for x
    up:     the upper bound for x
    N:      the number of grid points
    n:      the number of time levels
    dt:     the time step
    c:      the c-matrix
    """
    c = np.loadtxt('fully EE 1D.txt')

    x = np.linspace(lb, ub, N + 1)
    h = np.diff(x)[0]
    # create a 2x2 sub plots
    fig, ax = plt.subplots(2, 2)
    
    # c = c[:(n + 1)]
    # create fig and plots
    # fig = plt.figure(figsize=(18, 10))
    fig.suptitle(fr'N = {N} dt = h^{4}, Time Levels = {n}, Time = {0.001}')
    t1 = int(1000)*dt
    ax[0, 0].plot(x, c[int(1000)], linewidth = 2.0)
    ax[0, 0].set_title(f't = {t1} at {int(1000)}')
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
    fig.savefig("./Fully Explicit Scheme Time-Evoluation Plot")
    # ax.legend(loc="best", fontsize = 20)
    # ax.grid()
    # ax.set_xlabel("Spatial  = 0 to x = 1", fontsize = 20)
    # ax.set_ylabel("c(x, t)", fontsize = 20)
    # ax.set_title("Cahn-Hilliard Equation Fully Explicit Scheme", fontsize = 20)

if __name__ == "__main__":
    lb = 0
    ub = 1
    N = 79
    n = 75000

    # c, dt= simulation(lb=lb, ub=ub, N=N, n=n)
    # c, dt, _, iter = simE(lb, ub, N, T_final=0.001)
    x, h = np.linspace(lb, ub, N + 1, retstep=True)
    # make_plotting(lb=lb, ub=ub, N=N, n=n, dt=h**4)
    plotting(lb=lb, ub=ub, N=N, n=n, dt=h**4)
