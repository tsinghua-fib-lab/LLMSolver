"""MIS (Maximal Independent Set) dataset."""

import glob
import os
import pickle

import numpy as np
import torch

from torch_geometric.data import Data as GraphData


class MISDataset(torch.utils.data.Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def get_example(self, idx):

        graph = self.graph_list[idx]
        num_nodes = graph.number_of_nodes()

        node_labels = [_[1] for _ in graph.nodes(data='label')]
        if node_labels is not None and node_labels[0] is not None:
            node_labels = np.array(node_labels, dtype=np.int64)
        else:
            node_labels = np.zeros(num_nodes, dtype=np.int64)

        edges = np.array(graph.edges, dtype=np.int64)
        edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
        # add self loop
        self_loop = np.arange(num_nodes).reshape(-1, 1).repeat(2, axis=1)
        edges = np.concatenate([edges, self_loop], axis=0)
        edges = edges.T

        return num_nodes, node_labels, edges

    def __getitem__(self, idx):
        # import time
        # b = time.time()
        num_nodes, node_labels, edge_index = self.get_example(idx)
        graph_data = GraphData(x=torch.from_numpy(node_labels),
                               edge_index=torch.from_numpy(edge_index))

        point_indicator = np.array([num_nodes], dtype=np.int64)
        # print(time.time() - b)
        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            graph_data,
            torch.from_numpy(point_indicator).long(),
        )
