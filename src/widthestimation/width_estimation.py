import os
import numpy as np
from tqdm import tqdm
import networkx as nx
from matplotlib import pyplot as plt

try:
    from scene import Scene
    from width_utils import place_nodes, graph2widths, compute_area
    from sdiff_utils import create_empty_SDIFF, append_feature, save_SDIFF
except:
    from .width_utils import place_nodes, graph2widths, compute_area
    from .sdiff_utils import create_empty_SDIFF, append_feature, save_SDIFF
    from ..cloudutils.markers import refine_markers, compute_linepoints

categories = ["background", "control_point", "vegetation", "efflorescence", "corrosion", "spalling", "crack",
              "exposed_rebar"]


def graph2sdiff(scene, graph_path, sdiff_path, plot_dir=""):
    """ Converts the networkx graph to a SDIFF. """
    # load graph and split to subgraphs
    G = nx.read_gpickle(graph_path)
    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    sdiff = create_empty_SDIFF()

    for SG in tqdm(subgraphs):

        category = list(SG.nodes(data='category'))[0][1]

        # crack case
        if category == 6:
            # interpolate nodes
            SG = place_nodes(SG, gap=0.01, min_len=0.05)

            # if graph is empty
            if len(SG.nodes) == 0:
                continue

            SG = graph2widths(SG, scene, os.path.join(plot_dir, "11_widths"))

            sdiff = append_feature(sdiff, SG, category=categories[category])

        # control point case
        elif category == 1:
            # get centroid
            node = list(SG.nodes)[0]
            pos = SG.nodes[node]["pos"]
            norm = SG.nodes[node]["normal"]

            # project point
            res = scene.compute_uvs(pos.tolist(), norm.tolist())
            uv = np.array([res[0], res[1]], np.int32).T
            mask_uv = np.array(res[3], np.int32)
            angles = np.array(scene.compute_angles(norm.tolist()))
            mask_angles = np.where((120 < angles) * (angles < 240), 1, 0)

            # view-based center refinement
            SG.nodes[node]["locations"] = {}
            for idx in np.nonzero(mask_uv * mask_angles)[0]:
                patch_size = 20
                offset = [uv[idx, 0] - patch_size // 2, uv[idx, 1] - patch_size // 2]

                # get image and refine marker
                img = np.array(scene.get_image(idx)).reshape(scene.height, scene.width)
                patch = np.uint8(img[offset[1]:offset[1] + patch_size,
                                 offset[0]:offset[0] + patch_size])
                inter, line1, line2 = refine_markers(patch, offset)

                # produce plot
                if plot_dir != "":
                    pts1 = compute_linepoints(line1, img.shape)
                    pts2 = compute_linepoints(line2, img.shape)

                    plt.clf()
                    fig = plt.figure()
                    plt.imshow(img, 'gray')
                    plt.plot(pts1[:, 0], pts1[:, 1], '#7570b3', linewidth=3)
                    plt.plot(pts2[:, 0], pts2[:, 1], '#7570b3', linewidth=3)
                    plt.plot(inter[0], inter[1], 'x', color='#d95f02', markersize=10)
                    plt.xlim(inter[0] - 2 * patch_size, inter[0] + 2 * patch_size)
                    plt.ylim(inter[1] + 2 * patch_size, inter[1] - 2 * patch_size)
                    plt.title("View: " + scene.get_label(idx) + f"\nCoordinates: [{inter[0]:.2f}, {inter[1]:.2f}]")
                    plt.xlabel("Horizontal Coordinate [px]")
                    plt.ylabel("Vertical Coordinate [px]")
                    plt.savefig(
                        os.path.join(plot_dir, "14_control_points",
                                     f"{scene.get_label(idx)}_{inter[0]:.0f}_{inter[1]:.0f}.png"), dpi=300,
                        bbox_inches='tight')
                    plt.close(fig)

                # add result to graph
                SG.nodes[node]["locations"][scene.get_label(idx)] = inter

            sdiff = append_feature(sdiff, SG, category=categories[category])

        # areal defects
        else:
            if len(SG.edges) == 0:
                continue

            positions = np.array(list(nx.get_edge_attributes(SG, "points").values()))[0, ...]
            area = compute_area(positions)

            if area * 10000 < 0.5:
                continue

            SG = place_nodes(SG, gap=0.02)  # 0.005)
            sdiff = append_feature(sdiff, SG, category=categories[category])

    save_SDIFF(sdiff, sdiff_path, None)


if __name__ == "__main__":
    # paths
    xml_path = os.path.join("/home/******/repos/defect-demonstration/static/uploads/mtb/cameras.xml")
    graph_path = "/home/******/repos/defect-demonstration/static/uploads/mtb/exprebar_clustered.pickle"
    sdiff_path = "/home/******/repos/defect-demonstration/static/uploads/mtb/extracted_defects.sdiff"
    plot_dir = "/home/******/repos/defect-demonstration/static/uploads/mtb"

    for f in os.listdir(plot_dir):
        os.remove(os.path.join(plot_dir, f))

    scene = Scene(xml_path)
    scene.cache_images(
        [
            "/home/******/repos/defect-demonstration/static/uploads/mtb/0_images"
        ],
        "/home/******/repos/defect-demonstration/static/uploads/mtb/9_sharpness/",
        "/home/******/repos/defect-demonstration/static/uploads/mtb/10_depth/",
        1.0
    )

    graph2sdiff(scene, graph_path, sdiff_path, plot_dir=plot_dir)
