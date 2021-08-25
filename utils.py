import os
import time
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from skimage.measure import profile_line
from scipy.ndimage.filters import gaussian_filter


def compute_area(positions):
    """ Computes area of 3D polygon using the cross product. """
    center_gravity = np.mean(positions, axis=0)
    positions = positions - center_gravity
    norms = np.linalg.norm(np.cross(positions, np.roll(positions, 1, axis=0)), axis=1)
    area = 0.5 * np.sum(norms)
    return area


def compute_length(positions):
    """ Computes the length of the polygonal chain. """
    return np.sum(np.linalg.norm(positions - np.roll(positions, 1, axis=0), axis=1))


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
    try:
        inter_idx = np.where(np.sign(intersects - len(gx) // 2) == 1)[0][0]
    except:
        return -1, a[0], b[0], np.array([2, 3])

    inter_idx = max(inter_idx, 1)
    inter_idxs = intersects[inter_idx - 1:inter_idx + 1]
    gx[:inter_idxs[0]] = a[:inter_idxs[0]]
    gx[inter_idxs[1]:] = a[inter_idxs[1]:]

    # apply equation from paper
    nom = a[0] * length - np.trapz(gx - np.abs(gx - a), x)
    w = nom / (2 * (a[0] - b[0]))

    return w, a[0], b[0], inter_idxs / 10


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


def graph2widths(G, scene, plot_dir=""):
    # set parameters
    pixel_size = (scene.cameras[0].pixel_height + scene.cameras[0].pixel_width) / 2
    f = scene.cameras[0].focal_length
    length = 25

    for node in G.nodes:
        neighbors = list(G.neighbors(node))

        if len(neighbors) == 2:
            # get positions
            pos0 = G.nodes[node]["pos"]
            pos1 = G.nodes[neighbors[0]]["pos"]
            pos2 = G.nodes[neighbors[1]]["pos"]

            # get norms
            norm0 = G.nodes[node]["normal"]
            norm1 = G.nodes[node]["normal"]
            norm2 = G.nodes[node]["normal"]

            # project points
            uv0, uv_mask, distances, angles = scene.point2uvs(pos0, norm0)
            uv1, uv_mask, distances, angles = scene.point2uvs(pos1, norm1)
            uv2, uv_mask, distances, angles = scene.point2uvs(pos2, norm2)

            # sort distances (ascending)
            argsort_distances = np.argsort(distances[uv_mask])

            # get closed image
            for i in range(len(distances[uv_mask])):
                idx = argsort_distances[i]

                # prepare orthogonal line
                angle = np.arctan2(uv2[idx, 1] - uv1[idx, 1], uv2[idx, 0] - uv1[idx, 0])
                yd = np.cos(angle) * length
                xd = np.sin(angle) * length
                p1 = (int(uv0[idx, 1]) + yd, int(uv0[idx, 0]) - xd)
                p2 = (int(uv0[idx, 1]) - yd, int(uv0[idx, 0]) + xd)

                # check crossing image border
                if 0 < p1[0] - 2 * length and p1[0] + 2 * length < scene.cameras[0].h and \
                        0 < p2[0] - 2 * length and p2[0] + 2 * length < scene.cameras[0].h and \
                        0 < p1[1] - 2 * length and p1[1] + 2 * length < scene.cameras[0].w and \
                        0 < p2[1] - 2 * length and p2[1] + 2 * length < scene.cameras[0].w:
                    break

            # extract line
            img = scene.images[idx, ..., 0]
            prof_line = profile_line(img, p1, p2, linewidth=3, mode='constant')
            prof_line2, shift = centralize_profline(prof_line)
            width, a, b, idxs = rectangle_transform(prof_line2, base=40)

            # translate px to mm
            width_px = width
            d = distances[idx] * 1000  # to mm
            width = width * d * pixel_size / f

            # check if width estimation is viable
            if np.isnan(width):
                G.nodes[node]["width"] = -1
                width = -1
            else:
                G.nodes[node]["width"] = width

            if plot_dir is not "":
                # vertex id
                id = str(time.time())
                G.nodes[node]["id"] = id

                plt.clf()
                fig = plt.figure()
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

                # subplot 1
                ax = fig.add_subplot(121)  # plt.subplot(121)
                plt.xlabel("Position [px]")
                plt.ylabel("Grauwert")
                plt.title("Grauwertprofil")

                # plot profile, median, and rectangle
                ax.plot(prof_line, label="Profil")
                ax.plot(np.full((len(prof_line)), a), 'gray', label="Median", linestyle='dashed')
                half = np.sum(idxs) / 2 - shift
                ax.plot([half - width_px / 2 - 0.0001, half - width_px / 2, half + width_px / 2,
                         half + width_px / 2 + 0.0001, half - width_px / 2 - 0.0001],
                        [a, b, b, a, a], label="Rechteck")

                ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
                ax.legend(loc="lower right", fontsize=6)
                plt.xticks([0, 10, 20, 30, 40, 50])
                # subplot 2
                plt.subplot(122)
                plt.imshow(img[uv0[idx, 1] - 2 * length:uv0[idx, 1] + 2 * length,
                           uv0[idx, 0] - 2 * length:uv0[idx, 0] + 2 * length], 'gray')
                plt.plot([p1[1] - uv0[idx, 0] + 2 * length, p2[1] - uv0[idx, 0] + 2 * length],
                         [p1[0] - uv0[idx, 1] + 2 * length, p2[0] - uv0[idx, 1] + 2 * length])
                # plt.plot(2 * length, 2 * length, 'x')
                plt.xlabel("Bildkoordinate horizontal [px]")
                plt.ylabel("Bildkoordinate vertikal [px]")
                plt.title("Bildausschnitt")
                plt.xticks([0, 25, 50, 75, 100])
                plt.yticks([0, 25, 50, 75, 100])
                plt.subplots_adjust(top=0.85)

                # save figure
                fig.suptitle(
                    "Breite [px]: " + str(np.round(width_px, 2)) +
                    "            Breite [mm]: " + str(np.round(width, 2)), y=0.8)
                #plt.show()
                plt.savefig(
                    os.path.join(plot_dir, id + ".png"),
                    dpi=300,
                    bbox_inches='tight')
                plt.close(fig)
    return G
