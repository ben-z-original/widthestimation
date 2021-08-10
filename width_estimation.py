import os
from tqdm import tqdm
import networkx as nx
from projection.scene import Scene
from utils import place_nodes, graph2widths
from sdiff_utils import create_empty_SDIFF, append_feature, save_SDIFF

categories = ["background", "control_point", "vegetation", "efflorescence", "corrosion", "spalling", "crack"]


def graph2sdiff(scene, graph_path, sdiff_path, file_name, width_plots=False):
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
            SG = graph2widths(SG, scene, width_plots)

            sdiff = append_feature(sdiff, SG, file_name, category=categories[category])
        else:
            if len(SG.edges) == 0:
                continue
            SG = place_nodes(SG, gap=0.005)
            try:
                sdiff = append_feature(sdiff, SG, file_name, category=categories[category])
            except:
                print()

    save_SDIFF(sdiff, os.path.join(sdiff_path, file_name + ".sdiff"), None)


if __name__ == "__main__":
    # paths
    xml_path = "/home/******/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/cameras.xml"  # "/media/******/******/data/referenzobjekte/******bruecke/points/cameras_all.xml"
    path_list = [
        "/media/******/******/data/referenzobjekte/******bruecke/Christian/VSued_Abplatzung_20210428/0_jpg"]
    imgs_path = "images.npy"
    graph_path = "/home/******/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/graph_complete.pickle"  # "/home/******/repos/iterative-contraction/graphs/graph_complete.pickle"
    sdiff_path = "/home/******/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/"  # "/home/******/repos/demonstrator_pcd_lines/resources/data/******/sdiff/insensitive/"
    file_name = "aaaany_3D.jpg"

    # load and prepare scene
    scene = Scene.from_xml(xml_path)
    scene.prepare_matrices()
    scene.load_images(path_list=None, npy_path=imgs_path, scale=1.0)

    # convert graph to sdiff
    graph2sdiff(scene, graph_path, sdiff_path, file_name, width_plots=True)
