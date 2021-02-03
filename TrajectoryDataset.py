import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class TrajectoryDataset(Dataset):

    @staticmethod
    def history(step):
        if step == 0:
            # for t_k history = [signal during previous chunk, signal during
            # current chunk]
            history = [0, 0]
        elif step == 200:
            history = [0, 1]
        elif step == 400:
            history = [1, 1]
        else:
            raise NotImplementedError
        return history

    @staticmethod
    def diagnosis(experiment):
        if experiment.split('/')[4] == 'Control':
            label = 0
        elif experiment.split('/')[4] == 'AVH':
            label = 1
        elif experiment.split('/')[4] == 'nonAVH':
            label = 2
        else:
            raise NotImplementedError
        return label

    def __init__(self, experiments, edge_index):

        self.pairs = []

        for experiment in experiments:

            ascii_grid = np.loadtxt(experiment)
            ascii_grid = ascii_grid.T

            if ascii_grid.shape[0] != 61:
                continue

            label = self.diagnosis(experiment)

            for current, next_ in list(zip([0, 200], [200, 400])):

                pair = []

                x_current = torch.tensor(ascii_grid[:, current:next_]).double()
                x_next = torch.tensor(
                    ascii_grid[:, next_:next_ + 200]).double()

                u_current = torch.tensor(
                    self.history(current)).double().reshape(
                    1, -1)
                u_next = torch.tensor(
                    self.history(next_)).double().reshape(
                    1, -1)

                y = torch.tensor([label])

                batch = torch.zeros(61).long()

                pair = [
                    Data(
                        x=x_current,
                        y=y,
                        u=u_current,
                        batch=batch,
                        edge_index=edge_index),
                    Data(
                        x=x_next,
                        y=y,
                        u=u_next,
                        batch=batch,
                        edge_index=edge_index)]
                self.pairs.append(pair)

    def __len__(self):
        """
        Length of the dataset
        """
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]
