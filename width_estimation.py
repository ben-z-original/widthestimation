import numpy as np
import open3d as o3d
import networkx as nx
from projection.scene import Scene
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from skimage.measure import profile_line
from utils import centralize_profline, rectangle_transform, place_nodes

# paths
xml_path = "/media/******/******/data/referenzobjekte/******bruecke/points/cameras_all.xml"
path_list = [
    "/media/******/******/data/referenzobjekte/******bruecke/Christian/VSued_Abplatzung_20210428/0_jpg"]

# load point clouds
pcd = o3d.io.read_point_cloud("points_tmp.pcd")
pcd_orig = o3d.io.read_point_cloud(
    "/media/******/******/data/referenzobjekte/******bruecke/points/crack_1_4M.pcd")

# o3d.visualization.draw_geometries([pcd_orig, pcd])

# load and prepare scene
scene = Scene.from_xml(xml_path)
scene.prepare_matrices()
scene.load_images(path_list=None, npy_path="images.npy", scale=1.0)

# load graph
G = nx.read_gpickle("../iterative-contraction/graphs/graph_0.pickle")

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
        d = distances[idx] * 1000  # to mm
        width = width * d * pixel_size / f

        # visualize
        if True:
            figure(figsize=(80, 60))
            plt.subplot(121)
            plt.title(str(width))
            plt.imshow(img, 'gray')
            plt.plot(uv0[idx, 0], uv0[idx, 1], 'o')
            plt.plot(uv1[idx, 0], uv1[idx, 1], 'x')
            plt.plot(uv2[idx, 0], uv2[idx, 1], 'x')
            plt.plot(p1[1], p1[0], '+')
            plt.plot(p2[1], p2[0], '+')

            plt.subplot(122)
            plt.plot(prof_line)
            plt.plot(prof_line2)
            plt.show()
