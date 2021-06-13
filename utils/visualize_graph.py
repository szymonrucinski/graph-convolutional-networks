import os.path as osp

import numpy as np
import pandas as pd
from PIL import ImageColor
import seaborn as sns
import networkx as nx
from matplotlib import pyplot as plt

from torchvision.datasets import MNIST
from torch_geometric.datasets import MNISTSuperpixels

def visualize_MNISTSuperpixels(image, data):
    plt.figure(figsize=(17, 8))

    # plot the mnist image
    plt.subplot(1, 2, 1)
    plt.title("MNIST")
    np_image = np.array(image)
    plt.imshow(np_image)

    # plot the super-pixel graph
    plt.subplot(1, 2, 2)
    x, edge_index = data.x, data.edge_index

    # construct networkx graph
    df = pd.DataFrame({'from': edge_index[0], 'to': edge_index[1]})
    G = nx.from_pandas_edgelist(df, 'from', 'to')    

    # flip over the axis of pos, this is because the default axis direction of networkx is different
    pos = {i: np.array([data.pos[i][0], 27 - data.pos[i][1]]) for i in range(data.num_nodes)}

    # get the current node index of G
    idx = list(G.nodes())

    # set the node sizes using node features
    size = x[idx] * 500 + 200

    # set the node colors using node features
    color = []
    for i in idx:
        grey = x[i]
        if grey == 0:
            color.append('skyblue')
        else:
            color.append('red')

    nx.draw(G, with_labels=True, node_size=size, node_color=color, pos=pos)
    plt.title("MNIST Superpixel")