import random
import torch
import numpy as np
from torch_geometric.data import Data


class TrajectoryDataset():
    """Dataset with trajectories to train the Euler model on.
    It creates a list of lists contating consequent states given asc files of EEG recordings.
    Each state is defined by:
        1) 61 channel each having 200 ms EEG signal;
        2) Global features of the state: history of listening + ear a signal was given to;
        3) Label reflecting a diagnosis of the patient: 0-None, 1-AVH, 2-Schizophrenia;
        4) Connectivity information with shape [2, number of edges];
        5) Batch information - torch.long tensor with zeros. Used by graph operators.
    To make things work one should provide absolute paths to experiments that have to be accounted.
    Each path must contain:
        1) Folder of an ear the signal was given to ('Left' or 'Right');
        2) Diagnosis of the patient ('Control', 'AVH', 'nonAVH').
        For example: 'home/data/Control/Left/01337/experiment.asc'
    """

    @staticmethod
    def history(t_state):
        """Method to provide history of events for a state.
        Args:
            t_state (int): time when the state started.
        Returns:
            List containing information whether there was sound produced
                during the state or during the previous state.
            Example: [0, 1] - no signal during the previous state, signal during the state given.
        Raises:
            AttributeError: if start time of the state is not a multiple of 200 or out of the scope.
        """
        if t_state == 0:
            history = [0, 0]
        elif t_state == 200:
            history = [0, 1]
        elif t_state == 400:
            history = [1, 1]
        else:
            raise NotImplementedError
        return history

    @staticmethod
    def side(experiment):
        """Method to provide ear a signal was given to.
        Args:
            experiment (str): absolute path to the experiment (asc file). See class doc for details.
        Returns:
            0 if a signal was given to the left ear;
            1 if a signal was given to the right ear.
        Raises:
            AttributeError: if no folder reflecting ear was presented in the path.
        """
        if 'Left' in experiment.split('/'):
            side = 0
        elif 'Right' in experiment.split('/'):
            side = 1
        else:
            raise NotImplementedError
        return side

    @staticmethod
    def diagnosis(experiment):
        """Method to provide diagnosis of a patient.
        Args:
            experiment (str): path to the experiment (asc file). See class doc for details.
        Returns:
            0 if the patient is from control group;
            1 if the patient is from AVH group;
            2 if the patient is from nonAVH group.
        Raises:
            AttributeError: if no folder reflecting diagnosis was presented in the path.
        """
        if 'Control' in experiment.split('/'):
            label = 0
        elif 'AVH' in experiment.split('/'):
            label = 1
        elif 'nonAVH' in experiment.split('/'):
            label = 2
        else:
            raise NotImplementedError
        return label

    def __init__(self, experiments, edge_index):
        """Initialization of a dataset.
        Args:
            expariments (list): paths to experiments (asc files). See class doc for details.
            edge_index (torch.long tensor): connectivity information with shape [2, number_edges].
        """

        self.pairs = []

        for experiment in experiments:
            ascii_grid = np.loadtxt(experiment)
            ascii_grid = ascii_grid.T
            # Some experiments miss channels so neglect them
            if ascii_grid.shape[0] != 61:
                continue
            diagnosis = self.diagnosis(experiment)
            side = self.side(experiment)
            # There is abundant amount of control patient so take only each 4th
            if diagnosis == 0 and random.randint(0, 3) == 0:
                continue
            # We only consider pairs 0-200/200-400 and 200-400/400-600
            for current, next_ in list(zip([0, 200], [200, 400])):
                pair = []
                # Node features - EEG signals
                x_current = torch.Tensor(ascii_grid[:, current:next_]).double()
                x_next = torch.Tensor(
                    ascii_grid[:, next_:next_ + 200]).double()
                # Global features - history of events, ear
                u_current = self.history(current)
                u_current.append(side)
                u_current = torch.Tensor(u_current).double().reshape(1, -1)
                u_next = self.history(next_)
                u_next.append(side)
                u_next = torch.Tensor(u_next).double().reshape(1, -1)

                diagnosis = torch.Tensor([diagnosis])
                batch = torch.zeros(61).long()

                pair = [
                    Data(
                        x=x_current,
                        y=diagnosis,
                        u=u_current,
                        batch=batch,
                        edge_index=edge_index),
                    Data(
                        x=x_next,
                        y=diagnosis,
                        u=u_next,
                        batch=batch,
                        edge_index=edge_index)]
                self.pairs.append(pair)

        random.shuffle(self.pairs)

    def __len__(self):
        """Length of the dataset."""
        return len(self.pairs)

    def __getitem__(self, index):
        """Return a pair of consequent states."""
        return self.pairs[index]
