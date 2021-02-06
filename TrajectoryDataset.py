import numpy as np
import torch
import random
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
    def side(experiment):
        if experiment.split('/')[4] == 'Left':
            side = 0
        elif experiment.split('/')[4] == 'Right':
            side = 1
        else:
            raise NotImplementedError
        return side

    @staticmethod
    def diagnosis(experiment):
        if experiment.split('/')[3] == 'Control':
            label = 0
        elif experiment.split('/')[3] == 'AVH':
            label = 1
        elif experiment.split('/')[3] == 'nonAVH':
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
            side = self.side(experiment)

            if label == 0 and random.randint(0, 3) == 0:
                continue

            for current, next_ in list(zip([0, 200], [200, 400])):

                pair = []

                x_current = torch.tensor(ascii_grid[:, current:next_]).double()
                x_next = torch.tensor(
                    ascii_grid[:, next_:next_ + 200]).double()

                u_current = self.history(current)
                u_current.append(side)
                u_current = torch.tensor(u_current).double().reshape(
                    1, -1)
                u_next = self.history(next_)
                u_next.append(side)
                u_next = torch.tensor(u_next).double().reshape(
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

        random.shuffle(self.pairs)

    def __len__(self):
        """
        Length of the dataset
        """
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]
