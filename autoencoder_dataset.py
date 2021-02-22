import random
import torch
import numpy as np
from torch.utils.data import Dataset


class AutoencoderDataset(Dataset):
    """Dataset with trajectories to train the autoencoder on.
    It creates a matrix of shape num_states x duration contating states given asc files of EEG recordings.
    Each state is defined by:
        1) 61 channel each having duration [ms] EEG signal;
    To make things work one should provide absolute paths to experiments that have to be accounted.
    Each path must contain:
        1) Folder of an ear the signal was given to ('Left' or 'Right');
        For example: 'home/data/Control/Left/01337/experiment.asc'
    """

    def __init__(self, experiments, duration, t_max, useGPU=True):
        """Initialization of a dataset.
        Args:
            expariments (list): paths to experiments (asc files). See class doc for details.
            duration (int): duration of a state's signal [ms].
            t_max (int):  duration of the whole trajectory [ms].
            useGPU (boolean): sends dataset to GPU if True
        """
        self.states = np.zeros(0)
        self.node_signal_max = 0
        self.node_signal_min = 0
        for experiment in experiments:
            ascii_grid = np.loadtxt(experiment)
            ascii_grid = ascii_grid.T
            # Some experiments miss channels so neglect them
            if ascii_grid.shape[0] != 61:
                continue
            for i in range(0, t_max, duration):
                grid = ascii_grid[:, i:i + duration]
                self.states = np.append(self.states, grid.reshape(-1))
                # Update min and max value of EEG signal if needed
                if grid.min() < self.node_signal_min:
                    self.node_signal_min = grid.min()
                if grid.max() > self.node_signal_max:
                    self.node_signal_max = grid.max()
        # Min-max normalization for each state of the dataset
        self.states = (self.states - self.node_signal_min) / \
            (self.node_signal_max - self.node_signal_min)
        self.states = self.states.reshape(-1, duration)
        self.states = torch.Tensor(self.states).double()
        # Send to GPU if requested
        if useGPU:
            self.states = self.states.cuda()
        self.shuffle()

    def __len__(self):
        """Length of the dataset."""
        return (self.states).shape[0]

    def __getitem__(self, index):
        """Return a node signal."""
        return self.states[index]

    def shuffle(self):
        """Shuffle node signals in the dataset"""
        np.random.shuffle(self.states)

    def train_val_split(self, train_size=0.8):
        """Split dataset into train and validation subsets"""
        self.shuffle()
        train_states = self.states[:int(self.__len__() * train_size)]
        val_states = self.states[int(self.__len__() * train_size):]
        return train_states, val_states
