import json
import uuid
import numpy as np
import networkx as nx

try:
    from utils import compute_area, compute_length
except:
    from .utils import compute_area, compute_length


def create_empty_SDIFF():
    """ Create empty sdiff. """
    sdiff = {}
    sdiff['features'] = []
    sdiff['features_groups'] = []

    return sdiff


def construct_polygon(sdiff, G):
    """ Construct a polygon from G. """
    # get positions (assumption: positions are already in right order!)
    positions = np.array(list(nx.get_node_attributes(G, "pos").values()))

    # create stub for line
    sdiff['features'][-1]['reconstruction']['boundary'] = []
    sdiff['features'][-1]['reconstruction']['boundary'].append({
        'polygon': {
            'length': compute_length(positions),
            'area': compute_area(positions),
            'vertices': []
        }
    })

    # starting node
    start_node = list(G.nodes)[0]
    sdiff['features'][-1]['reconstruction']['boundary'][-1]['polygon']['vertices'].append({
        'coordinates': list(G.nodes[start_node]['pos'])
    })

    # next node
    curr_node = list(G.neighbors(start_node))[0]
    G.remove_edge(start_node, curr_node)

    # intermediate nodes
    while len(list(G.neighbors(curr_node))) == 1:
        # add vertex
        sdiff['features'][-1]['reconstruction']['boundary'][-1]['polygon']['vertices'].append({
            'coordinates': list(G.nodes[curr_node]['pos'])
        })

        # update node
        last_node = curr_node
        curr_node = list(G.neighbors(curr_node))[0]
        G.remove_edge(last_node, curr_node)

    # pre-end node
    sdiff['features'][-1]['reconstruction']['boundary'][-1]['polygon']['vertices'].append({
        'coordinates': list(G.nodes[curr_node]['pos'])
    })

    # end node: equals start node
    sdiff['features'][-1]['reconstruction']['boundary'][-1]['polygon']['vertices'].append({
        'coordinates': list(G.nodes[start_node]['pos'])
    })
    return sdiff, G


def construct_line(sdiff, G, node, succ_node):
    """ Construct a line from G and starting node. """

    curr_node = succ_node

    # create stub for line
    sdiff['features'][-1]['reconstruction']['skeleton'].append({
        'polyline': {
            'length': -1,
            'width': {
                'max': -1,
                'mean': -1,
                'median': -1
            },
            'vertices': []
        }
    })

    # starting node
    sdiff['features'][-1]['reconstruction']['skeleton'][-1]['polyline']['vertices'].append({
        'coordinates': list(G.nodes[node]['pos'])
    })
    G.remove_edge(node, succ_node)

    widths = []
    positions = []

    # intermediate nodes
    while len(list(G.neighbors(succ_node))) == 1:
        # TODO: why -9?
        try:
            width = G.nodes[succ_node]['width']
            widths.append(width)
        except:
            width = -9

        positions.append(list(G.nodes[succ_node]['pos']))

        # add vertex
        sdiff['features'][-1]['reconstruction']['skeleton'][-1]['polyline']['vertices'].append({
            'coordinates': list(G.nodes[succ_node]['pos']),
            'width': width
        })

        # in case vertex has id for width plot
        try:
            sdiff['features'][-1]['reconstruction']['skeleton'][-1]['polyline']['vertices'][-1]['id'] = \
                G.nodes[succ_node]['id']
        except:
            pass

        # update node
        last_node = succ_node
        succ_node = list(G.neighbors(succ_node))[0]
        G.remove_edge(last_node, succ_node)

        # end node
        sdiff['features'][-1]['reconstruction']['skeleton'][-1]['polyline']['vertices'].append({
            'coordinates': list(G.nodes[succ_node]['pos'])
        })

    # update widths and length
    if positions != []:
        positions = np.array(positions)
        tmp = positions[:-1, ...] - np.roll(positions, 1, axis=0)[:-1, ...]
        tmp = np.sum(np.linalg.norm(tmp, axis=1))
        sdiff['features'][-1]['reconstruction']['skeleton'][-1]['polyline']['length'] = tmp

        widths = np.array(widths)

        if len(widths) > 0 and not np.all(np.isnan(widths)):
            sdiff['features'][-1]['reconstruction']['skeleton'][-1]['polyline']['width']['max'] = np.nanmax(widths)
            sdiff['features'][-1]['reconstruction']['skeleton'][-1]['polyline']['width']['mean'] = np.nanmean(widths)
            sdiff['features'][-1]['reconstruction']['skeleton'][-1]['polyline']['width']['median'] = np.nanmedian(
                widths)

    return sdiff, G


def append_feature(sdiff, G, category="crack"):
    """ Append a feature to the sdiff. """
    # loop over connected components

    # create new feature (one feature per edge/line)
    sdiff['features'].append({
        'id': str(uuid.uuid4()),  # img_name + "|" + time.asctime().replace(' ', '_'),  # str(uuid.uuid4()),
        'category': category,
        'reconstruction': {},
    })

    # fill view
    sdiff['features'][-1]['reconstruction'] = {
        'annotation': 'automatic',
        'metric_unit': 'm',
        'skeleton': []
    }

    # determine intermediate and end nodes
    deg = G.degree(G.nodes)
    end_nodes = np.array([elem for elem in deg if elem[1] == 1])
    inter_nodes = np.array([elem for elem in deg if elem[1] > 2])

    if category == "crack":
        # subgraph without furcations
        if len(inter_nodes) == 0:
            node = end_nodes[0]
            sdiff, G = construct_line(sdiff, G, node[0], succ_node=list(G.neighbors(node[0]))[0])

        # subgraph with furcations
        else:
            for node in inter_nodes:
                for neighbor in list(G.neighbors(node[0])):
                    pass
                    sdiff, G = construct_line(sdiff, G, node[0], succ_node=neighbor)

    # category != "crack":
    else:
        # subgraph with cycle
        sdiff, G = construct_polygon(sdiff, G)

    return sdiff


def append_feature_noncrack(sdiff, category):
    sdiff['features'].append({
        'id': str(uuid.uuid4()),  # img_name + "|" + time.asctime().replace(' ', '_'),  # str(uuid.uuid4()),

        'reconstruction': {},
    })


def save_SDIFF(sdiff, path, schema):
    """ Save sdiff as json to drive. """
    with open(path, 'w') as outfile:
        json.dump(sdiff, outfile, indent=2)
