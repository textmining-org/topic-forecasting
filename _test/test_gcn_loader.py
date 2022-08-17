import json
from six.moves import urllib
from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader, ChickenpoxDatasetLoader
import networkx as nx
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal, StaticGraphTemporalSignal
import pytest
import torch
import numpy as np

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

    # time_period   : 61 → 60 = 61 - 1(lag)

    # parameters
    # edge indices  : 60 x (2, 2158)
    # edge weight   : 60 x (2158,)
    # node features : 60 x (129, 1)
    # node label    : 60 x (129,)

    edge = np.array(dataset_org["edge_mapping"]["edge_index"][str(1)]).T
    print(edge.shape)

    edge_weight = np.array(dataset_org["edge_mapping"]["edge_weight"][str(1)])
    print(len(dataset.edge_weights))
    print(edge_weight.shape)

    print(len(dataset.features), dataset.features[0].shape)

    print(len(dataset.targets), dataset.targets[0].shape)

    for features in dataset.features:
        print(features.shape)

    for targets in dataset.targets:
        print(targets.shape)

