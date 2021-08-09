import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def place_nodes(G, gap=0.01):
    """ Interpolate and place graph nodes with a specific gap in-between. """
    # place nodes a gap points
    GG = nx.Graph()

    # place nodes
    for i in G.edges:
        pts = np.array(G.edges[i]['points'])
        nrmls = np.array(G.edges[i]['normals'])

        # determine x
        diff = np.sqrt(np.sum(np.power(np.diff(pts, axis=0), 2), axis=1))
        xp = np.append(0, np.cumsum(diff))
        x = np.arange(0 + xp[-1] % gap / 2, xp[-1], gap)
        x = np.hstack((0, x, xp[-1]))

        # interpolate points
        positions = np.stack([np.interp(x, xp, pts[:, 0]),
                              np.interp(x, xp, pts[:, 1]),
                              np.interp(x, xp, pts[:, 2])], axis=1)
        # interpolate normals
        normals = np.stack([np.interp(x, xp, nrmls[:, 0]),
                            np.interp(x, xp, nrmls[:, 1]),
                            np.interp(x, xp, nrmls[:, 2])], axis=1)

        # construct keys
        keys = ["_".join(item) for item in positions.astype(str)]

        # set nodes and edges
        nodes = [(keys[i], {"pos": positions[i, ...], "normal": normals[i, ...]}) for i in range(0, len(positions))]
        edges = [(keys[i - 1], keys[i]) for i in range(1, len(positions))]

        GG.add_nodes_from(nodes)
        GG.add_edges_from(edges)

    return GG


def profline_variance(prof_line):
    """ Compares the variance of the inner and outer variance of the profile line. """
    outer_var = np.var(np.hstack([prof_line[:len(prof_line) // 3], prof_line[2 * len(prof_line) // 3:]]))
    inner_var = np.var(prof_line[len(prof_line) // 3:2 * len(prof_line) // 3:])

    return inner_var - outer_var


def centralize_profline(prof_line):
    """ Move profile minimum to center. """
    length = len(prof_line)
    center = length // 2
    argmini = np.argmin(prof_line)
    shift = center - argmini

    # centralize
    prof_line = np.roll(prof_line, shift)

    # padding 'same'
    if shift < 0:
        prof_line[shift:] = prof_line[shift - 1]
    elif shift > 0:
        prof_line[:shift] = prof_line[shift]

    return prof_line, shift


def rectangle_transform(prof_line, base=20, height=0.9):
    """ Apply rectangel transform for width estimation. """
    # determine relevant measures
    length = len(prof_line)
    medi = np.median(prof_line)
    mini = np.min(prof_line[length // 3:2 * length // 3])
    hori = (medi - mini) * height + mini
    base = min(base, mini)

    # set parameters
    x = np.arange(0, length, 0.1)
    gx = np.interp(x, np.arange(0, length), prof_line)
    a = np.full((len(x)), hori)
    b = np.full((len(x)), base)

    # cancel irrelevant regions
    intersects = np.nonzero(np.gradient(np.sign(gx - a)))[0]
    inter_idx = np.where(np.sign(intersects - len(gx) // 2) == 1)[0][0]
    inter_idxs = intersects[inter_idx-1:inter_idx+1]
    gx[:inter_idxs[0]] = a[:inter_idxs[0]]
    gx[inter_idxs[1]:] = a[inter_idxs[1]:]

    # apply equation from paper
    nom = a[0] * length - np.trapz(gx - np.abs(gx - a), x)
    w = nom / (2 * (a[0] - b[0]))

    return w, a[0], b[0], inter_idxs/10


def intersection_approach(prof_line, height=0.5):
    """ Apply naive intersection approach for width estimation. """
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
    """ Apply parabola fitting for width estimation. """
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
