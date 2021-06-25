import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def centralize_profline(prof_line):
    length = len(prof_line)
    center = length // 2
    argmini = np.argmin(prof_line[length // 3:2 * length // 3]) + length // 3
    shift = center - argmini

    # centralize
    prof_line = np.roll(prof_line, shift)

    # padding 'same'
    if shift < 0:
        prof_line[shift:] = prof_line[shift - 1]
    elif shift > 0:
        prof_line[:shift] = prof_line[shift]

    return prof_line


def rectangle_transform(prof_line, base=20, height=0.9):
    # determine relevant measures
    medi = np.median(prof_line)
    mini = np.min(prof_line[10:20])
    hori = (medi - mini) * height + mini
    base = min(base, mini)

    # determine width
    x = np.arange(0, 30.1, 0.1)
    gx = np.interp(x, np.arange(0, 30), prof_line)
    a = np.full((len(x)), hori)
    b = np.full((len(x)), base)

    # apply equation from paper
    nom = a[0] * 30 - np.trapz(gx - np.abs(gx - a), x)
    w = nom / (2 * (a[0] - b[0]))

    return w


def intersection_approach(prof_line, height=0.5):
    sampling = 100
    prof_line = np.interp(np.arange(0, 30, 1 / sampling), np.arange(0, 30), prof_line)

    # determine relevant measures
    center = len(prof_line) // 2
    medi = np.median(prof_line)
    mini = np.min(prof_line[10 * sampling:20 * sampling])
    hori = (medi - mini) * height + mini

    # determine intersection points
    diff = prof_line - hori
    signs = np.sign(diff)
    inter = signs[1:] + signs[:-1]

    try:
        inter_left = np.where(inter[:center] == 0)[0][-1]
        inter_right = np.where(inter[center:] == 0)[0][0] + center

    except:
        center = np.argmin(inter[10 * sampling:20 * sampling]) + 10 * sampling
        try:
            inter_left = np.where(inter[:center] == 0)[0][-1]
        except:
            return np.nan
        try:
            inter_right = np.where(inter[center:] == 0)[0][0] + center
        except:
            inter_right = np.nan

    return (inter_right - inter_left) / sampling


def fit_parabola(prof_line):
    x = np.arange(0, 30, 1.0)
    y = np.interp(np.arange(0, 30, 1.0), np.arange(0, 30), prof_line)

    first = True

    # infinite loop (break upon width being found)
    while True:
        if first:
            z = np.polyfit(np.append(x[:10], x[20:]), np.append(y[:10], y[20:]), 2)
            first = False
        else:
            z = np.polyfit(x, y, 2)

        p = np.poly1d(z)

        # low-pass filtering
        mean_y = gaussian_filter(np.full((30 * 10), np.mean(prof_line) / 2), sigma=2)

        # find the roots
        roots = mean_y - p(np.arange(0, 30, 0.1))
        roots = np.array(np.nonzero(np.diff(np.sign(roots)))) / 10

        diff = np.abs(y - p(x))
        std = np.std(diff)

        if len(x) > 20:
            keep = np.where(diff < 2 * std, False, True)
        else:
            keep = np.where(diff > 2 * std, False, True)

        # breaking conditions
        if len(x[keep]) < 3 or np.array_equal(x, x[keep]):
            if z[0] < 0 or len(roots[0]) < 2:
                return np.nan

            return roots[0][1] - roots[0][0]

        x = x[keep]
        y = y[keep]
