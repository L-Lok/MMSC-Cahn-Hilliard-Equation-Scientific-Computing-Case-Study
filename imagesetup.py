import numpy as np
# from keras.datasets import mnist
import matplotlib.pyplot as plt

def cross_line(N, x, y, Dxlb, Dxub, Dylb, Dyub, random=None, white_width=2, gray_width=2):
    """
    N + 1: the number of grid points
    Dx:  the range of x
    Dy:  the range of y
    random:  on or off; introducing the damage area
    white_size  white strip
    noise: bool add random noise or not
    """

    mid = int((N + 1) / 2)

    # create a cross-line

    good_image = (-1) * np.ones((N + 1, N + 1))
    good_image[(mid - white_width) : (mid + white_width), :] = 1
    good_image[:, (mid - white_width) : (mid + white_width)] = 1
    damaged_image = np.copy(good_image)

    if random == "off":
        # introduce the damage area
        damaged_image[ (mid -white_width*2):(mid + white_width*2),(mid - white_width*2):(mid + white_width*2) ] = 0
        D = [x[(mid -white_width*2):(mid + white_width*2)], y[(mid - white_width*2):(mid + white_width*2)]]
        
    elif random == "on":
        randnum_x = np.random.randint(gray_width, N- gray_width)
        randnum_y = np.random.randint(gray_width, N- gray_width)
        damaged_image[(randnum_x - gray_width) : (randnum_x + gray_width), (randnum_y - gray_width) : (randnum_y + gray_width)] = 0
        D = [x[(randnum_x - gray_width) : (randnum_x + gray_width)], y[(randnum_y - gray_width) : (randnum_y + gray_width)]]

    elif random == "noise":
        D = []
        # for _ in range(np.random.randint(11)):
        for _ in range(8):
            randnum_x = np.random.randint(gray_width, N- gray_width)
            randnum_y = np.random.randint(gray_width, N- gray_width)
            damaged_image[(randnum_x - gray_width) : (randnum_x + gray_width), (randnum_y - gray_width) : (randnum_y + gray_width)] = 0
            D.append([x[(randnum_x - gray_width) : (randnum_x + gray_width)], y[(randnum_y - gray_width) : (randnum_y + gray_width)]])

    return good_image, damaged_image, D

if __name__ == '__main__':
    N = 127
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    Dxlb, Dxub, Dylb, Dyub = 2, 8, 2, 8
    good_image, damaged_image, D = cross_line(N, x, y, Dxlb, Dxub, Dylb, Dyub, random="noise", white_width= 5, gray_width=5)

    plt.imshow(good_image,cmap="gray")
    plt.show()
    plt.imshow(damaged_image,cmap="gray")
    plt.show()