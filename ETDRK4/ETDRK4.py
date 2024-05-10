import numpy as np
import matplotlib.pyplot as plt

from copy import copy
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from scipy.linalg import norm
from scipy.fft import fft, ifft, rfft2, irfft2, rfftfreq, fftfreq
from set_up import calc_j, calc_mass


#################################################################################
##################################### 1D Case ###################################
#################################################################################
def etdrk4_1D(u, dt, k, kappa, n, t_num):
    """
    u:  obj
    dt: time step
    k:  wavenumber
    kappa:  eps**2
    n:  the number of spatial grid points
    t_num:  number of time levels
    """

    # Non-linear term
    N = lambda u: k**2 * (u - fft(ifft(u) ** 3))

    # Linear term
    L = -kappa * k**4

    # Pre-computations
    E = np.exp(L * dt)
    E2 = np.exp(L * dt / 2.0)

    m = 32
    Q = np.zeros(len(k))
    f1 = Q
    f2 = Q
    f3 = Q

    sol = np.zeros((int(t_num / 10), n))  # t_num x n
    sol[0] = ifft(u).real  # init

    for i in range(len(k)):
        r = dt * L[i] + np.exp(1j * np.pi * (np.arange(m) - 0.5) / m)
        Q[i] = dt * np.mean(((np.exp(r / 2) - 1) / r)).real
        f1[i] = dt * np.mean(((-4 - r + np.exp(r) * (4 - 3 * r + r**2)) / r**3)).real
        f2[i] = dt * np.mean(((2 + r + np.exp(r) * (-2 + r)) / r**3)).real
        f3[i] = dt * np.mean(((-4 - 3 * r - r**2 + np.exp(r) * (4 - r)) / r**3)).real

    for tsp in range(1, t_num, 1):
        Nu = N(u)
        a = E2 * u + Q * Nu
        Na = N(a)
        b = E2 * u + Q * Na
        Nb = N(b)
        c = E2 * a + Q * (2 * Nb - Nu)
        Nc = N(c)
        u = E * u + f1 * Nu + 2 * f2 * (Na + Nb) + f3 * Nc
        if tsp % 10 == 0:
            sol[int(tsp / 10)] = ifft(u).real

    return sol


def init_1d(x):
    return np.cos(2 * np.pi * x)


#################################################################################
##################################### 2D Case ###################################
#################################################################################
def etdrk4_2D(u, dt, kappa, n, h, t_num):
    """
    u:  obj
    dt: time step
    k:  wavenumber
    kappa:  eps**2
    n:  the number of spatial grid points
    t_num:  number of time levels

    """
    # suppose we have the spatial points n x n
    # in the freq domain, the x-direction will have int(n/2 + 1) points (rfftfreq)
    # the y-direction will have n points (fftfreq)
    k_x, k_y = np.meshgrid(rfftfreq(n, h / (2 * np.pi)), fftfreq(n, h / (2 * np.pi)))
    k = k_x**2 + k_y**2  # k = k**2

    # Non-linear term
    N = lambda u: k * (u - rfft2(irfft2(u) ** 3))

    # Linear term
    L = -kappa * k**2

    # Pre-calculations
    E = np.exp(L * dt)
    E2 = np.exp(L * dt / 2.0)

    m = 32
    Q = np.zeros((n, int(n / 2 + 1)))
    f1 = np.zeros((n, int(n / 2 + 1)))
    f2 = np.zeros((n, int(n / 2 + 1)))
    f3 = np.zeros((n, int(n / 2 + 1)))

    for i in range(n):
        for j in range(int(n / 2 + 1)):
            r = dt * L[i, j] + np.exp(1j * np.pi * (np.arange(m) + 0.5) / m)
            Q[i, j] = dt * np.mean(((np.exp(r / 2) - 1) / r)).real
            f1[i, j] = (
                dt * np.mean(((-4 - r + np.exp(r) * (4 - 3 * r + r**2)) / r**3)).real
            )
            f2[i, j] = dt * np.mean(((2 + r + np.exp(r) * (-2 + r)) / r**3)).real
            f3[i, j] = (
                dt * np.mean(((-4 - 3 * r - r**2 + np.exp(r) * (4 - r)) / r**3)).real
            )

    sol = []
    sol.append(irfft2(u))
    for tsp in range(1, t_num, 1):
        Nu = N(u)
        a = E2 * u + Q * Nu
        Na = N(a)
        b = E2 * u + Q * Na
        Nb = N(b)
        c = E2 * a + Q * (2 * Nb - Nu)
        Nc = N(c)
        u = E * u + f1 * Nu + 2 * f2 * (Na + Nb) + f3 * Nc
        if tsp % 100 == 0:
            sol.append(irfft2(u))
        # sol.append(irfft2(u))
        # if np.sum((sol[tsp] - sol[tsp - 1])**2)**(0.5) <= 1e-2:
        #     break
    return sol


def init_2d(x, y, n):
    z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            z[i, j] = np.cos(2 * np.pi * x[i]) * np.cos(2 * np.pi * y[j])
    return z


################################################################################
################################# Variable Mobility ############################
################################################################################


def non_linear_term(c_hat, B_const, M, k_x, k_y, k, kappa):

    c = irfft2(c_hat)
    c[c > 1] = 1
    c[c < -1] = -1
    f_hat = rfft2(c**3 - c)

    m = 1.0 - B_const * c**2 - M
    # print(np.shape(m));exit()
    vec_x = rfft2(irfft2(1j * k_x * (f_hat + kappa * k * c_hat)) * m)
    vec_y = rfft2(irfft2(1j * k_y * (f_hat + kappa * k * c_hat)) * m)

    return 1j * k_x * vec_x + 1j * k_y * vec_y - M * k * f_hat


def etdrk4_2D_var(u, dt, kappa, n, h, t_num, B_const, M):
    """
    u:  obj
    dt: time step
    k:  wavenumber
    kappa:  eps**2
    n:  the number of spatial grid points
    t_num:  number of time levels

    """
    # suppose we have the spatial points n x n
    # in the freq domain, the x-direction will have int(n/2 + 1) points (rfftfreq)
    # the y-direction will have n points (fftfreq)
    k_x, k_y = np.meshgrid(rfftfreq(n, h / (2 * np.pi)), fftfreq(n, h / (2 * np.pi)))
    k = k_x**2 + k_y**2  # k = k**2

    # Non-linear term
    N = lambda u: non_linear_term(
        u, B_const=B_const, M=M, k_x=k_x, k_y=k_y, k=k, kappa=kappa
    )

    # Linear term
    L = -M * kappa * k**2

    # Pre-calculations
    E = np.exp(L * dt)
    E2 = np.exp(L * dt / 2.0)

    m = 32
    Q = np.zeros((n, int(n / 2 + 1)))
    f1 = np.zeros((n, int(n / 2 + 1)))
    f2 = np.zeros((n, int(n / 2 + 1)))
    f3 = np.zeros((n, int(n / 2 + 1)))

    for i in range(n):
        for j in range(int(n / 2 + 1)):
            r = dt * L[i, j] + np.exp(1j * np.pi * (np.arange(m) + 0.5) / m)
            Q[i, j] = dt * np.mean(((np.exp(r / 2) - 1) / r)).real
            f1[i, j] = (
                dt * np.mean(((-4 - r + np.exp(r) * (4 - 3 * r + r**2)) / r**3)).real
            )
            f2[i, j] = dt * np.mean(((2 + r + np.exp(r) * (-2 + r)) / r**3)).real
            f3[i, j] = (
                dt * np.mean(((-4 - 3 * r - r**2 + np.exp(r) * (4 - r)) / r**3)).real
            )

    sol = []
    sol.append(irfft2(u))
    for tsp in tqdm(range(1, t_num, 1)):
        Nu = N(u)
        a = E2 * u + Q * Nu
        Na = N(a)
        b = E2 * u + Q * Na
        Nb = N(b)
        c = E2 * a + Q * (2 * Nb - Nu)
        Nc = N(c)
        u = E * u + f1 * Nu + 2 * f2 * (Na + Nb) + f3 * Nc
        sol.append(irfft2(u))
        # sol.append(irfft2(u))
        # if np.sum((sol[tsp] - sol[tsp - 1])**2)**(0.5) <= 1e-2:
        #     break
    return sol


def plotting(t_num):
    c = np.load("data_ref_2D_var.npy", allow_pickle=True)
    fig,axes = plt.subplots(2,3)

    time = [0, 1000, 10000, 20000, 30000, -1*100]
    titles = ['Initial', 't = 1000', 't = 10000', 't = 30000', 't = 20000', 'Final']
    for i, ax in enumerate(axes.flat):
        im= ax.imshow(c[int(time[i])], origin = 'lower')
        ax.set_title(titles[i], fontsize = 18)
        ax.axis('off')
    # im =axes[0,0].imshow(c[0],origin = 'lower')
    cbar = fig.colorbar(im, ax = axes.ravel().tolist(), orientation = 'vertical')
    cbar.ax.tick_params(labelsize = 12)
    plt.show()

def mass_J_plot(h, kappa, dt, t_num):
    # c = np.load("data_ref_2D_var.npy", allow_pickle=True)
    # J = [calc_j(c[i], h, kappa) for i in range(len(c))]
    # M = [calc_mass(c[i],h) for i in range(len(c))]
    # np.savetxt('J_var.txt', J)
    # np.savetxt('M_var.txt', M)
    J = np.loadtxt('J_var.txt')
    M = np.loadtxt('M_var.txt')
    t = [dt*i for i in range(int(t_num))]
    _, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(t[100:], 1e-6*(np.diff(M)[99:]/(dt)))
    ax1.set_xlabel('Time t', fontsize = 15)
    ax1.set_ylabel(r'Rate of Change In Mass, $\frac{dm(c)}{dt}$ ', fontsize = 18)
    ax1.yaxis.offsetText.set_fontsize(15)
    ax1.grid()

    ax2.plot(t, J)
    ax2.set_xlabel('Time t', fontsize = 15)
    ax2.set_ylabel(r'Free Energy $J(c)$', fontsize = 18)
    ax2.grid()

    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    plt.show()

if __name__ == "__main__":
    # Problem Set-up
    # kappa = 1e-4
    # t_num = 45000
    # spatial_point_num = 128
    # dt = 1e-6

    # x, h = np.linspace(0, 1, spatial_point_num, endpoint=False, retstep=True)

    # k = fftfreq(spatial_point_num, h/(2*np.pi))

    #######################################################################################
    ####################### Generate Reference Solution Data ##############################
    #######################################################################################
    # c_init = init_1d(x)
    # u_init = fft(c_init)
    # sol = []
    # sol = etdrk4_1D(u_init, dt, k, kappa, spatial_point_num, t_num)

    # for N in tqdm([64, 128, 256, 512, 1024, 2048]):
    #     x, h = np.linspace(0, 1, N, endpoint=False, retstep=True)
    #     k = fftfreq(N, h/(2*np.pi))
    #     c_init = init_1d(x)
    #     u_init = fft(c_init)
    #     sol_ = etdrk4_1D(u_init, dt, k, kappa, N, t_num)
    #     sol.append(sol_[-1])

    # sol_array = np.empty(len(sol), dtype=object)
    # for i, array in enumerate(sol):
    #     sol_array[i] = array

    # np.save('data_ref.npy', sol_array)
    ########################################################################################
    ########################################################################################
    ########################################################################################

    # plt.plot(x, sol[-1])
    # plt.show()

    # c_init_2d = init_2d(x, x, spatial_point_num)
    # u_init_2d = rfft2(c_init_2d)
    # u_init_2d = rfft2(2*np.random.rand(spatial_point_num,spatial_point_num) - 1)

    # sol_2d = etdrk4_2D_var(
    #     u_init_2d, dt, kappa, spatial_point_num, h, t_num, B_const=1, M = 1 / 2
    # )

    ###########################################################################
    ############################# variable Mobility ###########################
    ###########################################################################
    kappa = 1
    t_num = 45000
    spatial_point_num = 128
    dt = 1
    x, h = np.linspace(0, spatial_point_num, spatial_point_num, retstep=True)
    # u_init_2d = rfft2(0.2 * np.random.rand(spatial_point_num, spatial_point_num) - 0.1)

    # sol_2d = etdrk4_2D_var(
    #     u_init_2d, dt, kappa, spatial_point_num, h, t_num, B_const=1, M=1 / 2
    # )

    ############################################################################
    ############################### Saving the Data ############################
    ############################################################################
    # sol_array = np.empty(len(sol_2d), dtype=object)
    # for i, array in enumerate(sol_2d):
    #     sol_array[i] = array

    # np.save("data_ref_2D_var.npy", sol_array)
    ############################################################################
    ############################### Plotting Mass and J ########################
    ############################################################################
    # plotting(t_num)
    mass_J_plot(h=h, kappa=kappa, dt=dt, t_num=t_num)
