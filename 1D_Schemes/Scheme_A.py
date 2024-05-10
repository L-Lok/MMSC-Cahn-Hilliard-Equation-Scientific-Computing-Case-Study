import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from set_up import init_c, mass_plot, J_plot, Newton_system
import time

def schemeA(x, h, dt, N, c, tsp, eps=0.01):
    """
    x:      the vector variable, optimal is c^{n + 1}
    tsp:    the nth time level
    h:      the discretization size
    dt:     the time step
    N + 1:  the number of grid points
    """

    def W(x, i, c, tsp, h, N, eps):
        if i == 0:
            # BCs at x = 0
            # C_{-1} = C_{1}
            W_i = (
                1 / eps * (x[i]) ** 3
                - 1 / eps * c[tsp, i]
                - eps / h**2 * (x[i + 1] - 2 * x[i] + x[i + 1])
            )

        elif abs(i) == 1:
            # BCs at x = 0
            # W_{-1} = W_{1}
            W_i = (
                1 / eps * (x[abs(i)]) ** 3
                - 1 / eps * c[tsp, abs(i)]
                - eps / h**2 * (x[abs(i) + 1] - 2 * x[abs(i)] + x[abs(i) - 1])
            )

        elif i == N:
            # BCs at x = 1
            # C_{N -1} = C_{N+1}
            W_i = (
                1 / eps * (x[i]) ** 3
                - 1 / eps * c[tsp, i]
                - eps / h**2 * (x[i - 1] - 2 * x[i] + x[i - 1])
            )
        elif i == N - 1 or i == N + 1:
            # BCs at x = 1
            # W_{N + 1} = W_{N - 1}
            j = N - 1
            W_i = (
                1 / eps * (x[j]) ** 3
                - 1 / eps * c[tsp, j]
                - eps / h**2 * (x[j + 1] - 2 * x[j] + x[j - 1])
            )

        else:
            W_i = (
                1 / eps * (x[i]) ** 3
                - 1 / eps * c[tsp, i]
                - eps / h**2 * (x[i + 1] - 2 * x[i] + x[i - 1])
            )
        return W_i

    vec_func = []

    for i in range(N + 1):
        vec_func_ = (x[i] - c[tsp, i]) / dt - (
            W(x, i + 1, c, tsp, h, N, eps)
            - 2 * W(x, i, c, tsp, h, N, eps)
            + W(x, i - 1, c, tsp, h, N, eps)
        ) / h**2
        vec_func.append(vec_func_)
    return vec_func


def schemeA_jac(x, h, dt, N, eps=0.01):
    jac = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        if i == 0:
            # BCs at x = 0
            jac[i, 0:3] = [
                1 / dt + 1 / h**2 * (6 * eps / h**2 + 6 / eps * (x[i]) ** 2),
                -2 / h**2 * (3 / eps * x[i + 1] ** 2 + 4 * eps / h**2),
                2 * eps / h**4,
            ]
        elif i == 1:
            jac[i, 0:4] = [
                -4 * eps / h**4 - 3 * x[0] ** 2 / (h**2 * eps),
                1 / dt + 1 / h**2 * (6 * x[1] ** 2 / eps + 7 * eps / h**2),
                -1 / h**2 * (3 / eps * x[2] ** 2 + 4 * eps / h**2),
                eps / h**4,
            ]
        elif i == N - 1:
            jac[i, -4] = eps / h**4
            jac[i, -3] = -3 / (eps * h**2) * x[i - 1] ** 2 - 4 * eps / h**4
            jac[i, -2] = 1 / dt + 6 * x[i] ** 2 / (eps * h**2) + 7 * eps / h**4
            jac[i, -1] = -3 / (eps * h**2) * x[i + 1] ** 2 - 4 * eps / h**4
        elif i == N:
            jac[i, -3] = 2 * eps / h**4
            jac[i, -2] = -2 / h**2 * (3 / eps * x[i - 1] ** 2 + 4 * eps / h**2)
            jac[i, -1] = 1 / dt + 1 / h**2 * (6 * eps / h**2 + 6 / eps * (x[i]) ** 2)
        else:
            jac[i, (i - 2) : (i + 3)] = [
                eps / h**4,
                -3 / (eps * h**2) * x[i - 1] ** 2 - 4 * eps / h**4,
                1 / dt + 6 * x[i] ** 2 / (eps * h**2) + 7 * eps / h**4,
                -3 / (eps * h**2) * x[i + 1] ** 2 - 4 * eps / h**4,
                eps / h**4,
            ]
    return jac


def simulation(lb, ub, N, n, dt = None, tol=1e-8, power=4, eps=0.01):
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


    for n_ in tqdm(range(n)):
        sol, _ = Newton_system(schemeA, schemeA_jac, c[n_], tol, h, dt, N, c, n_, eps)
        c[n_ + 1] = np.array([sol])

    np.savetxt('SA 1D.txt', c)
    return c, dt

def simA(lb, ub, N, n = None, dt = None, T_final = None, tol=1e-8, power=4, eps=0.01, error = 1e-3):
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

    for n_ in tqdm(range(n)):
        sol, _ = Newton_system(schemeA, schemeA_jac, c[n_], tol, h, dt, N, c, n_, eps)
        c[n_ + 1] = np.array([sol])
        if np.sum((c[n_ + 1] - c[n_])**2)**(1/2) <= error:
            iter = n_ + 1
            break

    end_time = time.time()
    return c, dt, end_time - start_time, iter

def plotting(lb, ub, N, n, dt, c, power):
    """
    lb:     the lower bound for x
    up:     the upper bound for x
    N:      the number of grid points
    n:      the number of time levels
    dt:     the time step
    c:      the c-matrix
    """

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
    ax.set_title("Cahn-Hilliard Equation Scheme A", fontsize = 20)

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

    fig.savefig(f"./Semi Implicit Scheme A Plot dt = h^{power}")

def make_plotting(lb, ub, N, n, dt):
    """
    lb:     the lower bound for x
    up:     the upper bound for x
    N:      the number of grid points
    n:      the number of time levels
    dt:     the time step
    c:      the c-matrix
    """
    c = np.loadtxt('SA 1D.txt')

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
    fig.savefig("./Scheme A Time-Evoluation Plot")

if __name__ == "__main__":
    lb = 0
    ub = 1
    N = 127
    n = 26
    power = 1
    c, dt = simulation(lb=lb, ub=ub, N=N, n=n, tol=1e-7, power=power)
    x, h = np.linspace(lb, ub, N + 1, retstep=True)
    plotting(lb, ub, N, n, dt, c, power)
    # make_plotting(lb=lb, ub=ub, N=N, n=n, dt=h**4)
