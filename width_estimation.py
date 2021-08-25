import os
import numpy as np
from tqdm import tqdm
import networkx as nx
try:
    from projection.scene import Scene
    from utils import place_nodes, graph2widths, compute_area
    from sdiff_utils import create_empty_SDIFF, append_feature, save_SDIFF
except:
    from .projection.scene import Scene
    from .utils import place_nodes, graph2widths, compute_area
    from .sdiff_utils import create_empty_SDIFF, append_feature, save_SDIFF


categories = ["background", "control_point", "vegetation", "efflorescence", "corrosion", "spalling", "crack"]


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

            if area * 10000 < 5:
                continue

            SG = place_nodes(SG, gap=0.02)  # 0.005)
            sdiff = append_feature(sdiff, SG, category=categories[category])

    save_SDIFF(sdiff, sdiff_path, None)


if __name__ == "__main__":

    # paths
    xml_path = os.path.join("/home/******/repos/defect-demonstration/static/uploads/bbv/cameras.xml")
    path_list = [
        "/home/******/repos/defect-demonstration/static/uploads/bbv/0_images"]
    imgs_path = "/home/******/repos/defect-demonstration/static/uploads/bbv/images.npy"
    graph_path = "/home/******/repos/defect-demonstration/static/uploads/bbv/balken_3_9M.pickle"
    sdiff_path = "/home/******/repos/defect-demonstration/static/uploads/bbv/extracted_defects.sdiff"
    width_path = "/home/******/repos/defect-demonstration/static/uploads/bbv/10_widths"

    # load and prepare scene
    scene = Scene.from_xml(xml_path)
    scene.prepare_matrices()
    scene.load_images(path_list=None, npy_path=imgs_path, scale=1.0)

    # convert graph to sdiff
    graph2sdiff(scene, graph_path, sdiff_path, plot_dir=width_path)

