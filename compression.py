import numpy as np


def pixel_equal(p0, p1, colour):
    return (p0 == p1).all() if colour else (p0 == p1)


def line_equal(l0, l1, colour):
    len_0 = len(l0)
    if not len_0 == len(l1):
        return False
    else:
        for i in range(len_0):
            l0_i = l0[i]
            l1_i = l1[i]
            if colour and not len(l0_i) == len(l1_i):
                return False
            if (type(l0_i) is np.ndarray and type(l1_i) is np.ndarray) and len(l0_i) > 1:
                if not (l0_i == l1_i).all():
                    return False
            else:
                if not l0_i == l1_i:
                    return False
    return True


def run_length_c(img, colour):
    y_l = len(img)
    x_l = len(img[0])
    out = []
    for y in range(y_l):
        x = 0
        out_x = []
        while x in range(x_l):
            start = img[y][x]
            out_x.append(start)
            length = 0
            for j in range(1, x_l - x):
                if pixel_equal(start, img[y][x + j], colour):
                    length += 1
                else:
                    break
            if length > (2 if colour else 4):
                out_x.append([length])
                x += length
            x += 1
        out.append(out_x)
    out2 = []
    y = 0
    while y in range(y_l):
        start = out[y]
        out2.append(start)
        length = 0
        for j in range(1, y_l - y):
            if line_equal(start, out[y + j], colour):
                length += 1
            else:
                break
        if length > 0:
            out2.append([length])
            y += length
        y += 1
    return out2


def run_length_d(img):
    out2 = []
    for y in range(len(img)):
        value = img[y]
        if len(value) == 1:
            length = value[0]
            for j in range(length):
                out2.append(img[y - 1])
        else:
            out2.append(value)

    out = []
    for y in range(len(out2)):
        out_x = []
        for x in range(len(out2[y])):
            value = out2[y][x]
            if isinstance(value, list) and type(value) is not np.ndarray:
                for j in range(value[0]):
                    out_x.append(out2[y][x - 1])
            else:
                out_x.append(value)
        out.append(out_x)
    return np.array(out)
