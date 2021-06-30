import os
import time
import numpy as np
import open3d as o3d
import networkx as nx
from projection.scene import Scene
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from skimage.measure import profile_line
from utils import centralize_profline, profline_variance, rectangle_transform, place_nodes
from sdiff_utils import create_empty_SDIFF, append_feature, save_SDIFF


def graph2sdiff(scene, graph_path, sdiff_path, file_name, width_plots=False):
    """ Converts the networkx graph to a SDIFF. """
    # load graph
    G = nx.read_gpickle(graph_path)

    # interpolate nodes
    G = place_nodes(G, gap=0.01)

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

            # get closed image
            idx = np.argmin(distances[uv_mask])

            # prepare orthogonal line
            angle = np.arctan2(uv2[idx, 1] - uv1[idx, 1], uv2[idx, 0] - uv1[idx, 0])
            yd = np.cos(angle) * length
            xd = np.sin(angle) * length
            p1 = (int(uv0[idx, 1]) + yd, int(uv0[idx, 0]) - xd)
            p2 = (int(uv0[idx, 1]) - yd, int(uv0[idx, 0]) + xd)

            # extract line
            img = scene.images[idx, ..., 0]
            prof_line = profile_line(img, p1, p2, linewidth=3)
            prof_line2 = centralize_profline(prof_line)
            width = rectangle_transform(prof_line2)

            # translate px to mm
            width_px = width
            d = distances[idx] * 1000  # to mm
            width = width * d * pixel_size / f

            # check if width estimation is viable
            if profline_variance(prof_line) < 0:
                G.nodes[node]["width"] = -1
                width = -1
            else:
                G.nodes[node]["width"] = width

            if width_plots:
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
                ax.plot(prof_line)
                ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

                # subplot 2
                plt.subplot(122)
                plt.imshow(img[uv0[idx, 1] - 2 * length:uv0[idx, 1] + 2 * length,
                           uv0[idx, 0] - 2 * length:uv0[idx, 0] + 2 * length], 'gray')
                plt.plot([p1[1] - uv0[idx, 0] + 2 * length, p2[1] - uv0[idx, 0] + 2 * length],
                         [p1[0] - uv0[idx, 1] + 2 * length, p2[0] - uv0[idx, 1] + 2 * length])
                plt.plot(2 * length, 2 * length, 'x')
                plt.xlabel("Bildkoordinate horizontal [px]")
                plt.ylabel("Bildkoordinate vertikal [px]")
                plt.title("Bildausschnitt")
                plt.subplots_adjust(top=0.85)

                # save figure
                fig.suptitle(
                    "Breite [px]: " + str(np.round(width_px, 2)) +
                    "            Breite [mm]: " + str(np.round(width, 2)), y=0.8)
                plt.savefig("/home/******/repos/demonstrator_pcd_lines/resources/data/live_demo/widths/" + id + ".png",
                            dpi=300,
                            bbox_inches='tight')
                plt.close(fig)

    sdiff = create_empty_SDIFF()
    sdiff = append_feature(sdiff, G, file_name)
    save_SDIFF(sdiff, os.path.join(sdiff_path, file_name + ".sdiff"), None)

if __name__ == "__main__":
    # paths
    xml_path = "/media/******/******/data/referenzobjekte/******bruecke/points/cameras_all.xml"
    path_list = [
        "/media/******/******/data/referenzobjekte/******bruecke/Christian/VSued_Abplatzung_20210428/0_jpg"]
    imgs_path = "images.npy"
    graph_path = "/home/******/repos/iterative-contraction/graphs/graph_complete.pickle"
    sdiff_path = "/home/******/repos/demonstrator_pcd_lines/resources/data/******/sdiff/insensitive/"
    file_name = "any_3D.jpg"

    # load and prepare scene
    scene = Scene.from_xml(xml_path)
    scene.prepare_matrices()
    scene.load_images(path_list=None, npy_path=imgs_path, scale=1.0)

    # convert graph to sdiff
    graph2sdiff(scene, graph_path, sdiff_path, file_name, width_plots=True)


