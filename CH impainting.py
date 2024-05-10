import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.sparse.linalg import spsolve
from scipy.sparse import vstack, hstack, diags, identity, kron
from imagesetup import cross_line


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


def aux_para(x, y, D, delta_0, type="and"):
    """
    x, y:       the points in the domain \Omega
    D:          the damaged area
    delta_0:    a given constant
    """
    # [Dx, Dy] = D
    if type == "and":
        delta = delta_0
        for D_ in D:
            [Dx_, Dy_] = D_
            if x in Dx_ and y in Dy_:
                delta = 0
                break
        
        return delta
            
    # elif type == "or":
    #     try:
    #         if x in Dx or y in Dy:
    #             return 0
    #         else:
    #             return delta_0
    #     except Exception as e:
    #         raise KeyError


def load_vec(c, x, y, D, delta_0, init_image_arr, N, dt, eps):
    """
    c:              the vectorized c-matrix at the current time level
    the c-matrix is set up with

    i = 0, ..., N for each row

    j = 0, ..., N for each column

    x, y:           the grid points

    delta_0:        the parameter in the modified CH

    D:              the damaged area

    init_image_arr: stay at the same and not updated

    N + 1:          the number of time levels
    dt:             the timestep
    """

    b1 = []
    for j in range(N + 1):
        for i in range(N + 1):
            b1.append(
                np.array(c).reshape(N + 1, N + 1)[j, i]
                + dt
                * aux_para(x[i], y[j], D, delta_0)
                * (init_image_arr[j, i] - np.array(c).reshape(N + 1, N + 1)[j, i])
            )
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


def simulation(init_c, x, y, N, n, eps, D, delta_0, dt):
    """

    init_image_arr: the initial image

    xlb:    the left end of x

    yub:    the right end of x

    ylb:    the left end of y

    yub:    the right end of y

    N + 1:  the number of mesh points

    power:  control the relation between h and dt

    n:      the number of time levels for simulation

    """
    h = np.diff(x)[0]  # assume uniform mesh size
    # dt = h**power

    # the vectorized c-matrix initially
    c0 = np.concatenate(init_c)

    # create the list to store c each time level
    c = [c0]

    for tsp in tqdm(range(n)):
        A_2d = coeff_mat(h, dt, N, eps)
        b = load_vec(c[tsp], x, y, D, delta_0, init_c, N, dt, eps)
        # print(np.shape(A_2d), np.shape(b))
        c_next = spsolve(A_2d, b)
        # print(np.shape(c_next))
        c.append(c_next[0 : ((N + 1) ** 2)])

    return c


def plotting(good_image, damaged_image, c_blur, c_sharpen, xlb, xub, ylb, yub, N, dt):
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
    # create a 2x2 sub plots
    gs = gridspec.GridSpec(2, 4)

    # create fig and plots
    fig = plt.figure(figsize=(15, 10))

    # the first subplot
    ax1 = fig.add_subplot(gs[0, :2])
    # ax1.imshow(
    #     good_image,
    #     cmap="gray_r",
    #     origin="lower",
    #     extent=[xlb, xub, ylb, yub],
    # )
    ax1.imshow(
        good_image,
        cmap="gray",
        extent=[xlb, xub, ylb, yub],
    )
    ax1.set_title("Initial Image", fontsize=20)
    ax1.set_xlabel("x", fontsize=15)
    ax1.set_ylabel("y", fontsize=15)
    ax1.axis('off')

    # the second subplot
    ax2 = fig.add_subplot(gs[0, 2:])
    # ax2.imshow(
    #     damaged_image, cmap="gray_r", origin="lower", extent=[xlb, xub, ylb, yub]
    # )
    ax2.imshow(damaged_image, cmap="gray", extent=[xlb, xub, ylb, yub])
    ax2.set_title("Damaged Initial Image", fontsize=20)
    ax2.set_xlabel("x", fontsize=15)
    ax2.set_ylabel("y", fontsize=15)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, :2])
    # ax3.imshow(
    #     np.array(c[-1]).reshape(N + 1, N + 1),
    #     cmap="gray_r",
    #     origin="lower",
    #     extent=[xlb, xub, ylb, yub],
    # )
    ax3.imshow(
        np.array(c_blur[-1]).reshape(N + 1, N + 1),
        cmap="gray",
        extent=[xlb, xub, ylb, yub],
    )
    ax3.set_title("Inpainted Image Blur", fontsize=20)
    ax3.set_xlabel("x", fontsize=15)
    ax3.set_ylabel("y", fontsize=15)
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 2:])
    # ax3.imshow(
    #     np.array(c[-1]).reshape(N + 1, N + 1),
    #     cmap="gray_r",
    #     origin="lower",
    #     extent=[xlb, xub, ylb, yub],
    # )
    ax4.imshow(
        np.array(c_sharpen[-1]).reshape(N + 1, N + 1),
        cmap="gray",
        extent=[xlb, xub, ylb, yub],
    )
    ax4.set_title("Inpainted Image Sharpen", fontsize=20)
    ax4.set_xlabel("x", fontsize=15)
    ax4.set_ylabel("y", fontsize=15)
    ax4.axis('off')
    fig.savefig(f"./CH Inpainting dt = {dt}.png")


if __name__ == "__main__":
    # loading the dataset
    # (train_X, train_y), (test_X, test_y) = mnist.load_data()

    xlb, xub = 0, 1
    ylb, yub = 0, 1

    N = 127  # no. of pixels = 28 = N + 1

    # step 1
    n1 = 10
    eps1 = 0.01

    # step 2
    n2 = 6
    eps2 = 0.001

    # power = 1
    dt = 0.0001

    x = np.linspace(xlb, xub, N + 1)
    y = np.linspace(ylb, yub, N + 1)

    # D = [x, np.concatenate((y[5:6],y[17:18]))]

    # linearmap =lambda x: 2*x/255 - 1

    # # good_image
    # good_image = np.zeros((N + 1,N + 1))

    # for i in range(N+1):
    #     for j in range(N+1):
    #         good_image[j,i] = linearmap(train_X[18][j,i]) # pick a digit
    #         if good_image[j,i] + 1 > 0.01:
    #             good_image[j,i] = 1
    # # introduce damage;
    # damaged_image = np.copy(good_image)

    # # damaged_image[:, 10:12] = 0
    # damaged_image[5:6, :] = 0
    # damaged_image[17:18,:] = 0

    mid = int((N + 1) / 2)

    good_image, damaged_image, D = cross_line(
        N=N,
        x=x,
        y=y,
        Dxlb=2,
        Dxub=8,
        Dylb=2,
        Dyub=8,
        random="noise",
        white_width=5,
        gray_width=6
    )

    c_blur = simulation(
        init_c=damaged_image, x=x, y=y, N=N, n=n1, eps=eps1, D=D, delta_0=1 / dt, dt=dt
    )
    c_sharpen = simulation(
        init_c=c_blur[-1].reshape(N + 1, N + 1),
        x=x,
        y=y,
        N=N,
        n=n2,
        eps=eps2,
        D=D,
        delta_0=0,
        dt=dt,
    )

    plotting(
        good_image=good_image,
        damaged_image=damaged_image,
        c_blur=c_blur,
        c_sharpen=c_sharpen,
        xlb=xlb,
        xub=xub,
        ylb=ylb,
        yub=yub,
        N=N,
        dt=dt,
    )
