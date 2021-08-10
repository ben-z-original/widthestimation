import json
import uuid
import numpy as np


def create_empty_SDIFF():
    """ Create empty sdiff. """
    sdiff = {}
    sdiff['features'] = []
    sdiff['features_groups'] = []

    return sdiff


def construct_line(sdiff, G, node):
    """ Construct a line from G and starting node. """

    neighbors = list(G.neighbors(node))

    # cycle case
    if len(neighbors) == 2:
        neighbors = [neighbors[0]]

    # loop over node's neighbors
    for neighbor in neighbors:
        curr_node = neighbor

        # create stub for line
        sdiff['features'][-1]['reconstruction']['skeleton'].append({
            'polyline': {
                'length': -1,
                'width': {
                    'max': -1,
                    'mean': -1,
                    'median': -1
                },
                'probability': {
                    'max': -1,
                    'mean': -1,
                    'median': -1,
                    'std': -1
                },
                'vertices': []
            }
        })

        # starting node
        sdiff['features'][-1]['reconstruction']['skeleton'][-1]['polyline']['vertices'].append({
            'coordinates': list(G.nodes[node]['pos'])
        })
        G.remove_edge(node, curr_node)
        print()
        # intermediate nodes
        while len(list(G.neighbors(curr_node))) == 1:
            # TODO: why -9?
            try:
                width = G.nodes[curr_node]['width']
            except:
                width = -9

            # add vertex
            sdiff['features'][-1]['reconstruction']['skeleton'][-1]['polyline']['vertices'].append({
                'coordinates': list(G.nodes[curr_node]['pos']),
                'width': width
            })

            # in case vertex has id for width plot
            try:
                sdiff['features'][-1]['reconstruction']['skeleton'][-1]['polyline']['vertices'][-1]['id'] = \
                    G.nodes[curr_node]['id']
            except:
                pass

            # update node
            last_node = curr_node
            curr_node = list(G.neighbors(curr_node))[0]
            G.remove_edge(last_node, curr_node)

        # end node
        sdiff['features'][-1]['reconstruction']['skeleton'][-1]['polyline']['vertices'].append({
            'coordinates': list(G.nodes[curr_node]['pos'])
        })
    return sdiff, G


def append_feature(sdiff, G, img_name, category="crack"):
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
    tmpp = list(deg)

    # subgraph with cycle
    if len(end_nodes) == 0:
        node = list(list(deg)[0])
        sdiff, G = construct_line(sdiff, G, node[0])

    # subgraph without furcations
    elif len(inter_nodes) == 0:
        node = end_nodes[0]
        sdiff, G = construct_line(sdiff, G, node[0])

    # subgraph with furcations
    else:
        for node in inter_nodes:
            for neighbor in list(G.neighbors(node[0])):
                sdiff, G = construct_line(sdiff, G, neighbor)

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
