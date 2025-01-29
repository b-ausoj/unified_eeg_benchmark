from .abstract_dataset import AbstractDataset
from ..tasks.abstract_task import AbstractTask
import numpy as np
import logging
import os
import shutil
from pooch import Unzip, retrieve
from scipy.io import loadmat
from ..enums.split import Split
from resampy import resample
from mne.filter import filter_data


"""
Weibo2013 dataset aka MI Limb
Simple and compound motor imagery
Paper 1:    https://doi.org/10.1186/1743-0003-10-106
Paper 2:    https://doi.org/10.1371/journal.pone.0114853
Data:       http://dx.doi.org/10.7910/DVN/27306
"""

log = logging.getLogger(__name__)
base_path = "/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/"
data_path = os.path.join(base_path, "data", "weibo2013")


def _load_data_weibo2013(
    task_name: str,  # needed for the cache
    split_value: str,  # needed for the cache
    subjects: list[int],
    target_labels: list[str],
    interval: list[int],
    filter_band: list[int],
    sampling_frequency: int,
    target_frequency: int,
) -> tuple[np.ndarray, np.ndarray]:
    print("Weibo2013Dataset._load_data_weibo2013")

    data = []
    labels = []

    events_dict = {
        "left_hand": 1,
        "right_hand": 2,
        "hands": 3,
        "feet": 4,
        "left_hand_right_foot": 5,
        "right_hand_left_foot": 6,
        "rest": 7,
    }
    target_event_ids = [events_dict[label] for label in target_labels]

    for subject in subjects:
        file_name = os.path.join(data_path, f"subject_{subject}.mat")
        mat = loadmat(
            file_name,
            squeeze_me=True,
            struct_as_record=False,
            verify_compressed_data_integrity=False,
        )

        # until we get the channel names montage is None
        events = mat["label"].ravel()
        mask = np.isin(events, target_event_ids)
        events = events[mask]

        raw_data = np.transpose(mat["data"], axes=[2, 0, 1])
        channels_to_keep = np.setdiff1d(
            np.arange(raw_data.shape[1]),
            [57, 61, 62, 63],  # remove CB1, CB2 and EOG channels
        )
        raw_data = raw_data[:, channels_to_keep, :]
        raw_data = raw_data[mask]

        # de-mean each trial
        raw_data = raw_data - np.mean(raw_data, axis=2, keepdims=True)
        log.warning("Trial data de-meaned")
        # raw_data = 1e-6 * raw_data  # scale

        data.append(raw_data)
        labels.append(events)

    # save the data in the cache
    # intervall isn't applied yet
    # nor resampled to target frequency

    # first apply bandpass filter, for each run individually wihtout clipping
    if filter_band is not None:
        [
            filter_data(
                run,
                sampling_frequency,
                filter_band[0],
                filter_band[1],
                method="iir",
                verbose=False,
            )
            for run in data
        ]

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    if interval is not None:
        start = interval[0] * sampling_frequency
        end = interval[1] * sampling_frequency
        data = data[:, :, start:end]

    if target_frequency is not None:
        if sampling_frequency != target_frequency:
            data = resample(
                data,
                sampling_frequency,
                target_frequency,
                axis=-1,
                filter="kaiser_best",
            )
    return data, labels


class Weibo2013Dataset(AbstractDataset):
    def __init__(
        self,
        task: AbstractTask,
        split: Split,
        target_channels=None,
        target_frequency=None,
        preload=False,
    ):
        # fmt: off
        super().__init__(
            interval=[3, 7],
            name="weibo2013", # MI Limb
            target_classes=task,
            target_classes=["left_right", "right_feet"],
            split=split,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=200,
            channel_names=["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5","C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "O1", "Oz", "O2"],
            preload=preload,
        )
        # fmt: on
        print("Weibo2013Dataset.__init__")
        self.data = None  # has the raw data in a array of shape (n_samples, n_channels, n_times) already for the specific task and split
        self.labels = None  # has the labels in a array of shape (n_samples,) already for the specific task
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {
                "left_hand": 1,
                "right_hand": 2,
                "hands": 3,
                "feet": 4,
                "left_hand_right_foot": 5,
                "right_hand_left_foot": 6,
                "rest": 7,
            },
            "name": self.name,
        }
        self.task_split = None  # defines which subject, session, run is relevant for the specific task and split
        # annotations has the structure:
        # {
        #     "task": {         i.e. "left_right" or "right_feet"
        #         "split": {    i.e. "train" or "test"
        #             "subjects": [i.e. "cl", "cyy", "kyf", "lnn"],
        #             "labels": [i.e. "left_hand", "right_hand", "hands", "feet", "left_hand_right_foot", "right_hand_left_foot", "rest"],

        self._load_task_split()
        if preload:
            self.load_data()

    def _download(self, subject: int):
        print("Weibo2013Dataset._download")
        if os.path.isfile(os.path.join(data_path, f"subject_{subject}.mat")):
            return

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        if subject in list(range(1, 5)):
            download_url = "https://dataverse.harvard.edu/api/access/datafile/2499178"
            data_name = "files1"
            sub_map = {
                1: "cl",
                2: "cyy",
                3: "kyf",
                4: "lnn",
            }
        elif subject in list(range(5, 8)):
            download_url = "https://dataverse.harvard.edu/api/access/datafile/2499182"
            data_name = "files2"
            sub_map = {5: "ls", 6: "ry", 7: "wcf"}
        elif subject in list(range(8, 11)):
            download_url = "https://dataverse.harvard.edu/api/access/datafile/2499179"
            data_name = "files3"
            sub_map = {8: "wx", 9: "yyx", 10: "zd"}
        else:
            raise ValueError(f"Subject {subject} not found in the dataset Weibo2013")

        if not os.path.isfile(os.path.join(data_path, data_name + ".zip")):
            retrieve(
                url=download_url,
                known_hash=None,  # TODO known_hash
                fname=data_name + ".zip",
                path=data_path,
                processor=Unzip(),
                progressbar=True,
            )
        for file_name in os.listdir(os.path.join(data_path, data_name + ".zip.unzip")):
            for sub_index, file_prefix in sub_map.items():
                if file_name.startswith(file_prefix):
                    os.rename(
                        os.path.join(data_path, data_name + ".zip.unzip", file_name),
                        os.path.join(data_path, "subject_{}.mat".format(sub_index)),
                    )
        os.remove(os.path.join(data_path, data_name + ".zip"))
        shutil.rmtree(os.path.join(data_path, data_name + ".zip.unzip"))

    def load_data(self):
        subjects = self.task_split[self._target_classes.name][self._split.value][
            "subjects"
        ]
        for subject in subjects:
            if not os.path.isfile(os.path.join(data_path, f"subject_{subject}.mat")):
                self._download(subject)
        # now the data is downloaded and unpacked
        # load the data and labels and cache them
        self.data, self.labels = self._cache.cache(_load_data_weibo2013)(
            self._target_classes.name,
            self._split.value,
            subjects,
            self.task_split[self._target_classes.name]["labels"],
            self._interval,
            [8, 32],
            self._sampling_frequency,
            self._target_frequency,
        )

        if self._target_channels is not None:
            target_indices = [
                self._channel_names.index(ch) for ch in self._target_channels
            ]
            self.data = self.data[:, target_indices, :]
