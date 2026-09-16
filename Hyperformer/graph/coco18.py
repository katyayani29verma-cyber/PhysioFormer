import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 18

self_link = [(i, i) for i in range(num_node)]

inward = [
    (1,0),    # neck -> nose

    (2,1),    # r shoulder -> neck
    (3,2),    # r elbow -> r shoulder
    (4,3),    # r wrist -> r elbow

    (5,1),    # l shoulder -> neck
    (6,5),    # l elbow -> l shoulder
    (7,6),    # l wrist -> l elbow

    (8,1),    # r hip -> neck
    (9,8),    # r knee -> r hip
    (10,9),   # r ankle -> r knee

    (11,1),   # l hip -> neck
    (12,11),  # l knee -> l hip
    (13,12),  # l ankle -> l knee

    (14,0),   # r eye -> nose
    (15,0),   # l eye -> nose

    (16,14),  # r ear -> r eye
    (17,15)   # l ear -> l eye
]

outward = [(j,i) for (i,j) in inward]

neighbor = inward + outward

class Graph:

    def __init__(
        self,
        labeling_mode='spatial',
        scale=1
    ):

        self.num_node = num_node

        self.self_link = self_link

        self.inward = inward

        self.outward = outward

        self.neighbor = neighbor

        self.A = self.get_adjacency_matrix(
            labeling_mode
        )

        self.A_binary = tools.edge2mat(
            neighbor,
            num_node
        )

        self.A_norm = tools.normalize_adjacency_matrix(
            self.A_binary + 2*np.eye(num_node)
        )

        self.A_binary_K = tools.get_k_scale_graph(
            scale,
            self.A_binary
        )

    def get_adjacency_matrix(
        self,
        labeling_mode=None
    ):

        if labeling_mode is None:
            return self.A

        if labeling_mode == 'spatial':

            A = tools.get_spatial_graph(
                num_node,
                self_link,
                inward,
                outward
            )

        else:
            raise ValueError()

        return A