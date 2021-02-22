import random
import torch
import numpy as np
from torch_geometric.data import Data


class TrajectoryDataset():
    """Dataset with trajectories to train the Euler model on.
    It creates a list of lists contating consequent states given asc files of EEG recordings.
    Each state is defined by:
        1) 61 channel each having <duration> ms EEG signal;
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
    def is_event(t_state, duration):
        """ Indicates if a signal was given during the state
        Args:
            t_state (int): time when the state started.
            duration (int): duration of a state's signal [ms].
        Returns:
            1 if signal was given during the state
            0 otherwise
        """
        return 1 if t_state >= 0 and t_state + duration <= 500 else 0

    @staticmethod
    def history(t_state, duration, n_events):
        """Method to provide history of events for a state.
        Args:
            t_state (int): time when the state started.
            duration (int): duration of a state's signal [ms].
            n_events (int): number of events to take into account.
        Returns:
            List containing information whether there was sound produced
                during the state or during the previous state.
            Example (n_events = 2): [0, 1] - no signal during the previous state,
                signal during the state given.
        """
        history = []
        t = t_state
        while t > t_state - n_events * duration:
            history.append(TrajectoryDataset.is_event(t))
            t = t - duration
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

    def __init__(self, experiments, edge_index,
                 duration, n_events, t_max, useGPU=True):
        """Initialization of a dataset.
        Args:
            expariments (list): paths to experiments (asc files). See class doc for details.
            edge_index (torch.long tensor): connectivity information with shape [2, number_edges].
            duration (int): duration of a state's signal [ms].
            n_events (int): number of events to take into account.
            t_max (int):  duration of the whole trajectory [ms].
            useGPU (boolean): sends dataset to GPU if True
        """
        self.positions = np.genfromtxt(
            '../EEG_data/Easycap_Koordinaten_61CH.txt')[:, 0:3]
        self.positions = torch.Tensor(self.positions).double()
        self.pairs = []
        self.node_signal_max = 0
        self.node_signal_min = 0
        for experiment in experiments:
            ascii_grid = np.loadtxt(experiment)
            ascii_grid = ascii_grid.T
            # Some experiments miss channels so neglect them
            if ascii_grid.shape[0] != 61:
                continue
            # Update min and max value of EEG signal if needed
            if ascii_grid[:, :t_max].min() < self.node_signal_min:
                self.node_signal_min = ascii_grid[:, :(t_max + duration)].min()
            if ascii_grid[:, :t_max].max() > self.node_signal_max:
                self.node_signal_max = ascii_grid[:, :(t_max + duration)].max()
            for current in range(0, (t_max - duration), duration):
                pair = []
                next_ = current + duration
                # Node features - EEG signals
                x_current = ascii_grid[:, current:next_]
                x_next = ascii_grid[:, next_:next_ + duration]
                # Global features - history of events
                u_current = self.history(current, duration, n_events)
                u_current = torch.Tensor(u_current).double().reshape(1, -1)
                u_next = self.history(next_, duration, n_events)
                u_next = torch.Tensor(u_next).double().reshape(1, -1)
                batch = torch.zeros(61).long()
                pair = [
                    Data(
                        x=x_current,
                        u=u_current,
                        batch=batch,
                        edge_index=edge_index,
                        positions=self.positions),
                    Data(
                        x=x_next,
                        u=u_next,
                        batch=batch,
                        edge_index=edge_index,
                        positions=self.positions)]
                self.pairs.append(pair)
        self.shuffle()
        # Min-max normalization for each state of the dataset
        for current, next_ in self.pairs:
            current.x = (current.x - self.node_signal_min) / \
                (self.node_signal_max - self.node_signal_min)
            next_.x = (next_.x - self.node_signal_min) / \
                (self.node_signal_max - self.node_signal_min)
            current.x = torch.Tensor(current.x).double()
            next_.x = torch.Tensor(next_.x).double()
            # Send to GPU if requested
            if useGPU:
                current.x = current.x.cuda()
                next_.x = next_.x.cuda()

    def __len__(self):
        """Length of the dataset."""
        return len(self.pairs)

    def __getitem__(self, index):
        """Return a pair of consequent states."""
        return self.pairs[index]

    def shuffle(self):
        """Shuffle pairs in the dataset"""
        random.shuffle(self.pairs)

    def train_val_split(self, train_size=0.8):
        """Split dataset into train and validation subsets"""
        self.shuffle()
        train_pairs = self.pairs[:int(self.__len__() * train_size)]
        val_pairs = self.pairs[int(self.__len__() * train_size):]
        return train_pairs, val_pairs
