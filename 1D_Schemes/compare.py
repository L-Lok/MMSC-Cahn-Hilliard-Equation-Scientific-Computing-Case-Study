import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from Fully_Explicit_1D import simE
from Fully_Implicit_1D import simI
from Scheme_A import simA
from Scheme_B import simB
from Scheme_C import simC
from Scheme_D import simD

# from Fully_Explicit_1D import simulation as simEE
from Fully_Implicit_1D import simulation as simII

# from Scheme_A import simulation as simAA
from Scheme_B import simulation as simBB
from Scheme_C import simulation as simCC
from Scheme_D import simulation as simDD


def data_generator(lb, ub, N_list, T_final=0.5):
    time_list = np.zeros((6, len(N_list)))

    # sol_A = []
    sol_B = []
    # sol_C = []
    sol_D = []

    for i, N in enumerate(N_list):
        print(f"N = {N}")
        # time_list[0][i]= simE(lb, ub, N, T_final=T_final, error = 1e-15)[2]
        # sol_IE_, _, time_list[1][i],_ = simI(lb, ub, N, T_final=T_final, power = 3, error= 1e-5)
        # sol_A_, _, time_list[2][i], _ = simA(lb, ub, N, T_final=T_final, power=1, error= 1e-5)
        sol_B_, _, time_list[3][i], _ = simB(
            lb, ub, N, T_final=T_final, power=1, error=1e-5
        )
        # sol_C_, _, time_list[4][i],_ = simC(lb, ub, N, T_final=T_final, power=3, error= 1e-5)
        sol_D_, _, time_list[5][i], _ = simD(
            lb, ub, N, T_final=T_final, power=1, error=1e-5, i=i
        )

        # sol_IE_, _ = simII(lb, ub, N, n, dt)
        # sol_A.append(sol_A_)

        sol_B.append(sol_B_)

        # sol_C.append(sol_C_)

        sol_D.append(sol_D_)

    ########################################################
    # save data
    ########################################################
    # sol_array_IE = np.empty(len(sol_A), dtype=object)
    # for i, array in enumerate(sol_A):
    #     sol_array_IE[i] = array
    # np.save('data_IE.npy', sol_array_IE)

    sol_array_B = np.empty(len(sol_B), dtype=object)
    for i, array in enumerate(sol_B):
        sol_array_B[i] = array
    np.save("data_B.npy", sol_array_B)

    # sol_array_C = np.empty(len(sol_C), dtype=object)
    # for i, array in enumerate(sol_C):
    #     sol_array_C[i] = array
    # np.save('data_C.npy', sol_array_C)

    sol_array_D = np.empty(len(sol_D), dtype=object)
    for i, array in enumerate(sol_D):
        sol_array_D[i] = array
    np.save("data_D.npy", sol_array_D)

    # time data
    np.savetxt("time.txt", time_list)

    return time_list


def compare_rate(lb, ub, N, n, dt_list):
    l2_error = np.zeros((4, len(dt_list)))
    sol_ref = np.loadtxt("data_ref.txt")[-1]  # steady state
    # error = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    for i, dt_ in enumerate(dt_list):
        # sol_IE,_,_, iter_IE = simI(lb, ub, N, n, dt = dt_, error = error[i])
        sol_IE, _ = simII(lb, ub, N, n, dt_)
        l2_error[0][i] = np.sqrt(np.sum((sol_ref - sol_IE[-1]) ** 2))

        (
            sol_B,
            _,
        ) = simBB(lb, ub, N, n, dt_)
        l2_error[1][i] = np.sqrt(np.sum((sol_ref - sol_B[-1]) ** 2))

        sol_C, _ = simCC(lb, ub, N, n, dt_)
        l2_error[2][i] = np.sqrt(np.sum((sol_ref - sol_C[-1]) ** 2))

        (
            sol_D,
            _,
        ) = simDD(lb, ub, N, n, dt_)
        l2_error[3][i] = np.sqrt(np.sum((sol_ref - sol_D[-1]) ** 2))

    np.savetxt("compare_l2.txt", l2_error)
    return l2_error


def time_plot(N_list):
    time_list = np.loadtxt("time.txt")

    name_list = ["EE", "IE", "SA", "SB", "SC", "SD"]
    style = ["s--", "o--", "v--", "*--", "x--", "^--"]
    plt.figure(figsize=(10, 8))
    for i in range(6):
        plt.plot([i + 1 for i in N_list], np.log10(time_list[i]), style[i], label=name_list[i])

    # plt.yscale("log")
    plt.xticks([i + 1 for i in N_list], labels=[f"$2^{i}$" for i in range(5, 10, 1)])

    plt.xlabel("Number of Grid Points", fontsize=20)
    plt.ylabel(r"$\log_{10}$( Execution Time [s] )", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid()
    plt.show()

def sol_plot(N_list):
    x = np.linspace(0, 1, N_list[-1] + 1)
    sol_ref = np.load("data_ref.npy", allow_pickle=True)[-1]
    sol_B = np.load("data_B.npy", allow_pickle=True)[-1]
    plt.plot(np.linspace(0, 1, 2048), sol_ref, label = 'ETDRK4',zorder = -1, linewidth = 3.5)
    plt.plot(x, sol_B, label = 'SB', linestyle = 'dashed', linewidth = 2.0)
    plt.xlabel('x', fontsize = 20)
    plt.ylabel('c(x, t)', fontsize = 20)
    plt.title('SB and ETDRK4 Solutions with 1024 Grid Points',fontsize = 18)
    plt.grid()
    plt.legend(fontsize = 20)
    plt.show()

def rate_plot(N_list):
    # err_list = np.loadtxt('compare_l2.txt')
    sol_ref = np.load("data_ref.npy", allow_pickle=True)
    sol_B = np.load("data_B.npy", allow_pickle=True)
    sol_D = np.load("data_D.npy", allow_pickle=True)

    err_list = np.zeros((1, len(N_list)))
    err_max_list = np.zeros((2, len(N_list)))
    for i, N_ in enumerate(N_list):
        err_list[0][i] = np.sqrt(np.sum((sol_ref[i] - sol_B[i]) ** 2))
        # err_list[1][i] = np.sqrt(np.sum((sol_ref[i] - sol_D[i]) ** 2))
        err_max_list[0][i] = np.max(abs(sol_ref[i] - sol_B[i]))
        # err_max_list[1][i] = np.max(abs(sol_ref[i] - sol_D[i]))

    # err_list[1][1] *= 0.5
    # err_list[1][2] *=0.1
    name_list = ["SB", "SD"]
    style = ["s--", "x--"]
    plt.figure(figsize=(12, 8))
    for i in range(1):
        # if i == 2:
        #     continue
        plt.plot(
            [i + 1 for i in N_list], np.log10(err_list[i]), style[i], label=name_list[i] + " - l2"
        )
        plt.plot(
            [i + 1 for i in N_list],
            np.log10(err_max_list[i]),
            style[i],
            label=name_list[i] + " - max",
        )

    # plt.yscale("log")
    plt.xticks(
        [i + 1 for i in N_list],
        labels=[rf"$2^{6}$", rf"$2^{7}$", rf"$2^{8}$", rf"$2^{9}$", r"$2^{10}$"],
        fontsize=20,
    )
    plt.yticks(fontsize = 22)
    # plt.yticks(
    #     np.concatenate((err_list[0][:-2], [err_list[0][-1]], [err_max_list[0][0]], err_max_list[0][2:])),
    #     labels=np.concatenate(
    #         (
    #             [f"{np.log10(i):.3f}" for i in err_list[0][:-2]],
    #             [f"{np.log10(err_list[0][-1]):.3f}"],
    #             [f"{np.log10(err_max_list[0][0]):.3f}"],
    #             [f"{np.log10(i):.3f}" for i in err_max_list[0][2:]],
    #         )
    #     ), fontsize = 15
    # )

    plt.xlabel("Number of Grid Points", fontsize=20)
    plt.ylabel(r"$\log_{10} \ || c - c_{ETDRK4} ||_{L_2, \ max}$", fontsize=20)
    plt.legend(fontsize=25)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    N_list = [63, 127, 255, 511, 1023]  # N + 1
    # compare_time(0, 1, N_list, 600000) # arbitray as long as large enough
    # time_plot(N_list)

    # dt_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    # compare_rate(0, 1, 127, 30000, dt_list)
    rate_plot(N_list)
    sol_plot(N_list)
    # data_generator(0, 1, N_list)
