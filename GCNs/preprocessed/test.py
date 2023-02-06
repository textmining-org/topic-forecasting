import numpy as np

from torch_geometric_temporal.dataset.encovid import EnglandCovidDatasetLoader
from torch_geometric_temporal.dataset.pems_bay import PemsBayDatasetLoader

ec_loader = EnglandCovidDatasetLoader()

print('##### lags is 3')
ec_dataset1 = ec_loader.get_dataset(lags=3)
for time, snapshot in enumerate(ec_dataset1):
    print(f'edge_index: {snapshot.edge_index.shape}')
    print(f'edge_attr: {snapshot.edge_attr.shape}')
    print(f'x: {snapshot.x.shape}')
    print(f'y: {snapshot.y.shape}')
    if time == 0:
        break

print('##### lags is 8')
ec_dataset2 = ec_loader.get_dataset(lags=8)
for time, snapshot in enumerate(ec_dataset2):
    print(f'edge_index: {snapshot.edge_index.shape}')
    print(f'edge_attr: {snapshot.edge_attr.shape}')
    print(f'x: {snapshot.x.shape}')
    print(f'y: {snapshot.y.shape}')
    if time == 0:
        break

pb_loader = PemsBayDatasetLoader()

print('##### num_timesteps_in is 12, num_timesteps_out is 3')
pb_dataset1 = pb_loader.get_dataset(num_timesteps_in=12, num_timesteps_out=3)
for time, snapshot in enumerate(pb_dataset1):
    print(f'edge_index: {snapshot.edge_index.shape}')
    print(f'edge_weight: {type(snapshot.edge_weight)}')
    print(f'x: {snapshot.x.shape}')
    print(f'y: {snapshot.y.shape}')
    if time == 0:
        break

print('##### num_timesteps_in is 12, num_timesteps_out is 6')
pb_dataset2 = pb_loader.get_dataset(num_timesteps_in=12, num_timesteps_out=6)
for time, snapshot in enumerate(pb_dataset2):
    print(f'edge_index: {snapshot.edge_index.shape}')
    print(f'edge_weight: {type(snapshot.edge_weight)}')
    print(f'x: {snapshot.x.shape}')
    print(f'y: {snapshot.y.shape}')
    if time == 0:
        break