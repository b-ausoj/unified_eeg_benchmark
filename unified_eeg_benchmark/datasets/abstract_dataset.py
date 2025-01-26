from abc import ABC, abstractmethod
import os
from joblib import Memory
import json
from ..enums.split import Split
import numpy as np

base_path = "/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/"


class AbstractDataset(ABC):
    def __init__(
        self,
        interval,
        name,
        task,
        tasks,
        split: Split,
        sampling_frequency=None,
        channel_names=None,
        target_channels=None,
        target_frequency=None,
        preload=False,
    ):
        self.data = None
        self._interval = interval
        self._name = name
        self._task = task
        self._tasks = tasks
        self._split = split
        self._channel_names = channel_names
        # Default value, to be overridden by subclasses
        self._sampling_frequency = sampling_frequency
        # Default value, to be overridden by subclasses
        if target_channels is not None:
            assert all(
                [channel in self._channel_names for channel in target_channels]
            ), "Target channels must be a subset of the available channels"
        self._target_channels = target_channels
        self._target_frequency = target_frequency
        self._preload = False
        self._cache = Memory(location=os.path.join(base_path, "cache"), verbose=0)
        # TODO make this more generic with a config file and parameters

    def __getitem__(self, index):
        if self.data is None:
            self.load_data()
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return self._name

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def _download(self, subject: int):
        pass

    def _load_task_split(self):
        task_split_path = os.path.join(
            base_path, "unified_eeg_benchmark", "task_split", f"{self._name}.json"
        )
        if os.path.exists(task_split_path):
            with open(task_split_path, "r") as f:
                self.task_split = json.load(f)
        else:
            raise FileNotFoundError(f"Task split file not found at {task_split_path}")

    # Getters for channel_names and sampling_frequency
    @property
    def channel_names(self):
        """Return the names of the channels."""
        return self._channel_names

    @property
    def sampling_frequency(self):
        """Return the sampling frequency."""
        return self._sampling_frequency

    # Getters and setters for target_channels
    @property
    def target_channels(self):
        """Return the target channels."""
        return self._target_channels

    @target_channels.setter
    def target_channels(self, channels):
        """Set the target channels."""
        self._target_channels = channels

    # Getters and setters for target_frequency
    @property
    def target_frequency(self):
        """Return the target frequency."""
        return self._target_frequency

    @target_frequency.setter
    def target_frequency(self, frequency):
        """Set the target frequency."""
        self._target_frequency = frequency

    # Getters and setters for task
    @property
    def task(self):
        """Return the task."""
        return self._task

    @task.setter
    def task(self, task):
        """Set the task."""
        self._task = task

    # Getters and setters for split
    @property
    def split(self) -> Split:
        """Return the split."""
        return self._split

    @split.setter
    def split(self, split: Split):
        """Set the split."""
        self._split = split
