import os
import numpy as np
from tqdm import tqdm
import networkx as nx

try:
    from scene import Scene
    from utils import place_nodes, graph2widths, compute_area
    from sdiff_utils import create_empty_SDIFF, append_feature, save_SDIFF
except:
    from .utils import place_nodes, graph2widths, compute_area
    from .sdiff_utils import create_empty_SDIFF, append_feature, save_SDIFF

categories = ["background", "control_point", "vegetation", "efflorescence", "corrosion", "spalling", "crack", "exposed_rebar"]


def graph2sdiff(scene, graph_path, sdiff_path, plot_dir=""):
    """ Converts the networkx graph to a SDIFF. """
    # load graph and split to subgraphs
    G = nx.read_gpickle(graph_path)
    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    sdiff = create_empty_SDIFF()

    for SG in tqdm(subgraphs):

        category = list(SG.nodes(data='category'))[0][1]

        if category == 6:
            # interpolate nodes
            SG = place_nodes(SG, gap=0.01)
            SG = graph2widths(SG, scene, plot_dir)

            sdiff = append_feature(sdiff, SG, category=categories[category])
        else:
            if len(SG.edges) == 0:
                continue

            positions = np.array(list(nx.get_edge_attributes(SG, "points").values()))[0, ...]
            area = compute_area(positions)

            if area * 10000 < 1:
                continue

            SG = place_nodes(SG, gap=0.02)  # 0.005)
            sdiff = append_feature(sdiff, SG, category=categories[category])

    save_SDIFF(sdiff, sdiff_path, None)


if __name__ == "__main__":
    # paths
    xml_path = os.path.join("/home/******/repos/defect-demonstration/static/uploads/mtb/cameras.xml")
    graph_path = "/home/******/repos/defect-demonstration/static/uploads/mtb/exprebar_clustered.pickle"
    sdiff_path = "/home/******/repos/defect-demonstration/static/uploads/mtb/extracted_defects.sdiff"
    width_path = "/home/******/repos/defect-demonstration/static/uploads/mtb/11_widths"

    for f in os.listdir(width_path):
        os.remove(os.path.join(width_path, f))

    scene = Scene(xml_path)
    scene.cache_images(
        [
            "/home/******/repos/defect-demonstration/static/uploads/mtb/0_images"
        ],
        "/home/******/repos/defect-demonstration/static/uploads/mtb/9_sharpness/",
        "/home/******/repos/defect-demonstration/static/uploads/mtb/10_depth/",
        1.0
    )

    graph2sdiff(scene, graph_path, sdiff_path, plot_dir=width_path)
