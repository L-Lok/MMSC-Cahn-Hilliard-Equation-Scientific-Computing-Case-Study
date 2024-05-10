import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.sparse.linalg import spsolve
from scipy.sparse import vstack, hstack, diags, identity, kron
from scipy.integrate import trapz

# Get the coefficient matrix for the laplacian operator
def laplacian_mat(const, N):
    """
    const:  the constant multiplied before the output
    example:
    const = dt/h**2
        h:      the mesh size
        dt:     the time step
    N + 1:  the number of mesh points
    """
    # A = (
    #     np.diag(np.ones(N + 1) * (-2))
    #     + np.diag(np.ones(N), k=1)
    #     + np.diag(np.ones(N), k=-1)
    # )
    # impletement the boundary conditions
    # A[0, 1] = 2
    # A[-1, -2] = 2

    off_diagonal_m1 = np.ones(N)
    off_diagonal_m1[-1] = 2  # bcs
    off_diagonal_p1 = np.ones(N)
    off_diagonal_p1[0] = 2  # bcs
    A = (
        diags(np.ones(N + 1) * (-2))
        + diags(off_diagonal_p1, 1)
        + diags(off_diagonal_m1, -1)
    )

    lap_2D = kron(A, identity(N + 1)) + kron(identity(N + 1), A)

    return (const * lap_2D).tocsr()


def load_vec(c, eps):
    """
    c:      the vectorized c-matrices at the current time level
    the c-matrix is set up with

    i = 0, ..., N for each row

    j = 0, ..., N for each column

    tsp:    the time step
    N + 1:  the number of time levels
    """

    b1 = c
    b2 = []
    for c_ in c:
        b2.append(1 / eps * (c_) ** 3 - 3 / eps * c_)

    b = np.concatenate((b1, b2))
    return b


def coeff_mat(h, dt, N, eps):
    """
    Set up the 4-block coefficient matrix for the linear system
    h:      the mesh size
    dt:     the time step
    N + 1:  the number of mesh points
    """
    # top-left
    part1 = identity((N + 1) ** 2)

    # top-right
    mu = dt / h**2
    part2 = laplacian_mat((-1) * mu, N)

    A_2d_up = hstack([part1, part2])

    # bottom-left
    part3 = laplacian_mat(eps / h**2, N) - 2 / eps * identity((N + 1) ** 2)

    # bottom-right
    part4 = identity((N + 1) ** 2)

    A_2d_low = hstack([part3, part4])

    A_2d = vstack([A_2d_up, A_2d_low])
    return A_2d.tocsr()


def simulation(init, xlb, xub, ylb, yub, N, n, eps, power=2):
    """
    xlb:    the left end of x

    yub:    the right end of x

    ylb:    the left end of y

    yub:    the right end of y

    N + 1:  the number of mesh points

    power:  control the relation between h and dt

    n:      the number of time levels for simulation

    """

    x = np.linspace(xlb, xub, N + 1)
    y = np.linspace(ylb, yub, N + 1)
    h = np.diff(x)[0]  # assume uniform mesh size
    dt = h**power

    X, Y = np.meshgrid(x, y)
    init_c = init(X, Y)

    # the vectorized c-matrix initially
    c0 = np.concatenate(init_c)

    # create the list to store c each time level
    c = [c0]

    for tsp in tqdm(range(n)):
        A_2d = coeff_mat(h, dt, N, eps)
        b = load_vec(c[tsp], eps)
        # print(np.shape(A_2d), np.shape(b))
        c_next = spsolve(A_2d, b)
        # print(np.shape(c_next))
        c.append(c_next[0 : ((N + 1) ** 2)])

    np.savetxt('2D scheme B_256.txt', c)
    return c


def plotting( xlb, xub, ylb, yub, N, n, power):
    """
    c:      the list that contains the vectorized c-matrix
    xlb:    the left end of x
    yub:    the right end of x
    ylb:    the left end of y
    yub:    the right end of y
    N + 1:  the number of mesh points
    dt:     the time step
    n:      the number of time levels for simulation
    """
    c = np.loadtxt('2D scheme B.txt')
    # create a 2x2 sub plots
    gs = gridspec.GridSpec(3, 2)

    # create fig and plots
    fig = plt.figure(figsize=(15, 10))

    # the first subplot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(
        np.array(c[0]).reshape(N + 1, N + 1),
        origin="lower",
        extent=[xlb, xub, ylb, yub],
    )
    ax1.set_title("Initial condition", fontsize=20)
    ax1.set_xlabel("x", fontsize=15)
    ax1.set_ylabel("y", fontsize=15)
    ax1.axis('off')

    # the second subplot
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        c[127].reshape(N + 1, N + 1), origin="lower", extent=[xlb, xub, ylb, yub]
    )
    ax2.set_title("127", fontsize=20)
    ax2.set_xlabel("x", fontsize=15)
    ax2.set_ylabel("y", fontsize=15)
    ax2.axis('off')

    # the thrid subplot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(
        c[int(15)].reshape(N + 1, N + 1), origin="lower", extent=[xlb, xub, ylb, yub]
    )
    ax3.set_title(f"{15} time level", fontsize=20)
    ax3.set_xlabel("x", fontsize=15)
    ax3.set_ylabel("y", fontsize=15)
    ax3.axis('off')

    # the fourth subplot
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(
        c[int(60)].reshape(N + 1, N + 1),
        origin="lower",
        extent=[xlb, xub, ylb, yub],
    )
    ax4.set_title(f"{60} time level", fontsize=20)
    ax4.set_xlabel("x", fontsize=15)
    ax4.set_ylabel("y", fontsize=15)
    ax4.axis('off')

    # the fifth subplot
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.imshow(
        c[int(200)].reshape(N + 1, N + 1), origin="lower", extent=[xlb, xub, ylb, yub]
    )
    ax5.set_title(f"{200} time level", fontsize=20)
    ax5.set_xlabel("x", fontsize=15)
    ax5.set_ylabel("y", fontsize=15)
    ax5.axis('off')

    # the sixth subplot
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.imshow(
        c[int(127*3)].reshape(N + 1, N + 1), origin="lower", extent=[xlb, xub, ylb, yub]
    )
    ax6.set_title(f"{127*3} time level", fontsize=20)
    ax6.set_xlabel("x", fontsize=15)
    ax6.set_ylabel("y", fontsize=15)
    ax6.axis('off')

    fig.colorbar(im2, ax=(ax1, ax2, ax3, ax4, ax5, ax6), orientation="vertical")
    plt.show()
    # fig.savefig(f"./2D Scheme B Plot dt = h^{power}")

def calc_mass(c, h, N):
    mass = []
    for i in range(3,382, 1):
        c_ = c[i].reshape(N + 1, N+1)
        integral_x = trapz(c_,dx=h,axis =0)
        mass_ = trapz(integral_x,dx=h,axis =0)
        mass.append(mass_)
    # print(mass)
    # exit()
    return mass
    
def calc_j(c, h, N, eps):
   
    j = []
    for i in range(3,382,1):
        c_ = c[i].reshape(N + 1, N + 1)
        phi_c = np.power(1-np.power(c_[1:,1:],2),2)
        x_derivative = (c_[1:,1:]-c_[:-1,1:])
        y_derivative = (c_[1:,1:]-c_[1:,:-1])
        derivative = np.power(x_derivative,2)+np.power(y_derivative,2)
        integral_x = trapz(phi_c+(eps/2)*derivative,dx=h,axis =0)
        j_ = trapz(integral_x,dx=h,axis =0)
        j.append(j_)
    return j

def mass_J_plot(h,n, dt, N):
    c = np.loadtxt('2D scheme B.txt')[:n]
    # print(len(c));exit()
    mass = calc_mass(c, h,N)
    J = calc_j(c, h,N, eps=0.01)
    t = [i*dt for i in range(3, n, 1)]
    fig,(ax1, ax2) = plt.subplots(1,2, figsize = (8, 6))
    ax1.plot(t[1:], np.diff(mass)/dt)
    # ax1.set_title('Smooth Initial Data Mass Plot', fontsize = 20)
    ax1.set_xlabel('Time t', fontsize = 15)
    ax1.set_ylabel(r'Rate of Change In Mass, $\frac{dm(c)}{dt}$ ', fontsize = 18)
    ax1.grid()
    # ax1.set_yticks(fontsize = 15)
    # ax1.set_xticks(fontsize = 12)
    ax2.plot(t, J)
    # ax2.set_title('Smooth Initial Data Free Energy Plot', fontsize = 20)
    ax2.set_xlabel('Time t', fontsize = 15)
    ax2.set_ylabel(r'Free Energy $J(c)$', fontsize = 18)
    ax2.grid()

    # ax2.set_yticks(fontsize = 15)
    # ax2.set_xticks(fontsize = 12)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax1.yaxis.offsetText.set_fontsize(15)
    # ax2.yaxis.offsetText.set_fontsize(20)
    plt.show()


def compare_plot(N):
    soln_ref = np.load(f'data_ref_2D_256.npy', allow_pickle=True)
    soln_B = np.loadtxt(f'2D scheme B_256.txt')
    # error = []
    # for i in range(0, 200, 1):
    #     error.append(np.sum((soln_B[i].reshape(N + 1, N + 1) - soln_ref[359])**2)**0.5)
    # print(np.argmin(error), np.min(error));exit()
    print(np.max(abs(soln_B[100].reshape(N + 1, N + 1) - soln_ref[360])))
    fig = plt.figure()
    im = plt.imshow((soln_B[100].reshape(N + 1, N + 1) - soln_ref[360]), origin= 'lower')
    plt.xlabel('x', fontsize = 20)
    plt.ylabel('y', fontsize =20)
    # plt.xlabel('The Absolute Difference with 512 Grid Points', fontsize = 20)
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=13)
    plt.show()


if __name__ == "__main__":
    xlb, xub = 0, 1
    ylb, yub = 0, 1
    N = 255
    n = 101
    eps = 0.01
    power = 1

    def init(x, y):
        # return np.random.rand(N + 1,N + 1) 
        return np.cos(2*np.pi * x) * np.cos(2*np.pi * y)

    # c = simulation(init, xlb, xub, ylb, yub, N, n, eps, power=power)
    # plotting( xlb, xub, ylb, yub, N, n = 635, power=power)
    # x, h = np.linspace(xlb, xub, N + 1, retstep=True)
    # mass_J_plot(h, 382, h**power, N)
    compare_plot(N)


    