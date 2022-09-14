import json
from six.moves import urllib
from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader, ChickenpoxDatasetLoader
import networkx as nx
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal, StaticGraphTemporalSignal
import pytest
import torch
import numpy as np
import pandas as pd


class GraphDatasetLoader(object):
    def __init__(self, G):
        self.G = G

    def _get_edges(self):
        self._edges = []

    def _get_edge_weights(self):
        self._edge_weights = []

    def _get_targets_and_features(self):
        self.features = []
        self.targets = []

    def get_dataset(self, lags: int = 8) -> DynamicGraphTemporalSignal:
        """Returning the England COVID19 data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The England Covid dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = DynamicGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset


class CustomChickenpoxDatasetLoader(ChickenpoxDatasetLoader):

    def _read_web_data(self):
        with open('./Checkenpox.json', 'r') as file:
            self._dataset = json.load(file)


@pytest.mark.skip
def test_download_dataset():
    url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/england_covid.json"
    dataset = json.loads(urllib.request.urlopen(url).read())

    with open('./EnglandCovid.json', 'w') as outfile:
        json.dump(dataset, outfile)

    assert dataset is not None


# DynamicGraphTemporalSignal
def test_EnglandCovidDatasetLoader():
    print('\n')

    loader = EnglandCovidDatasetLoader()
    dataset = loader.get_dataset(lags=1)

    dataset_org = loader._dataset
    print(dataset_org["time_periods"])

    """
    영국 코로나 관련 데이터(EnglandCovidDatasetLoader)
    - 기간 : from 3 March to 12 of May (61일)
    - 노드 : 129개 (우리: keyword)
    - 노드 feature : 1개 (우리: centrality...)
    - 노드 target : 1개 (우리: word count)
    - 엣지 : 2158개 (시간에따라 바뀜, 우리: co-occurrence 여부)
    - 엣지 weight : 2158개 (시간에따라 바뀜, 우리: co-occurrence 값)
    """
    # time_period   : 61 → 60 = 61 - 1(lag)

    # parameters
    # edge indices  : 60 x (2, 2158)
    # edge weight   : 60 x (2158,)
    # node features : 60 x (129, 1)
    # node target    : 60 x (129,)

    edge = np.array(dataset_org["edge_mapping"]["edge_index"][str(50)]).T
    # print(dataset_org["edge_mapping"]["edge_index"])
    print(edge.shape)
    print(edge)
    #
    # edge_weight = np.array(dataset_org["edge_mapping"]["edge_weight"][str(1)])
    # print(dataset_org["edge_mapping"]["edge_weight"])
    # print(edge_weight.shape)
    # print(edge_weight)

    # print(len(dataset.features), dataset.features[0].shape)
    #
    # print(len(dataset.targets), dataset.targets[0].shape)
    # print(dataset.targets)
    #
    # # np.save('./targets.npy', dataset.targets)
    #
    # for features in dataset.features:
    #     print(features.shape)
    #
    # for targets in dataset.targets:
    #     print(targets.shape)


def test_edge_npy():
    edge_indices = np.load('./edge_indices.npy')
    edge_weights = np.load('./edge_weights.npy')

    print(edge_indices)
    print(edge_weights)


def test_tsv():
    node_indices = pd.read_csv('node_indices.tsv', sep='\t', names=["node_index", "node_name"])

    result = node_indices[node_indices['node_name'] == '-th']['node_index']
    print(type(result))
    print(result.values[0])

def test_DynamicGraphTemporalSignal():

    edge_indices = np.random.rand(60, 2, 2158)
    edge_weights = np.random.rand(60, 2158)

    node_target = np.random.rand(60, 129)
    node_features = np.random.rand(60, 129, 1)

    dgts = DynamicGraphTemporalSignal(edge_indices, edge_weights, node_features, node_target)

    print(dgts)

