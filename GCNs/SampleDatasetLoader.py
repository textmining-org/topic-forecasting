from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
import json

class CustomChickenpoxDatasetLoader(ChickenpoxDatasetLoader):

    def _read_web_data(self):
        with open('./checkenpox.json', 'r') as file:
            self._dataset = json.load(file)
