from .abstract_dataset import AbstractDataset
import numpy as np
import logging
import os
from pooch import Unzip, retrieve, file_hash
from pooch.downloaders import choose_downloader
from scipy.io import loadmat
from ..enums.split import Split
from ..tasks.abstract_task import AbstractTask
from resampy import resample

"""
BCI IV 2a dataset aka BNCI2014_001
Four class motor imagery
Paper 1:    https://www.bbci.de/competition/iv/desc_2a.pdf
Paper 2:    https://lampx.tugraz.at/~bci/database/001-2014/description.pdf
Data 1:     https://www.bbci.de/competition/iv/download/index.html?agree=yes&submit=Submit
Data 2:     https://bnci-horizon-2020.eu/database/data-sets/
"""

log = logging.getLogger(__name__)
base_path = "/itet-stor/jbuerki/net_scratch/unified_eeg_benchmark/"
data_path = os.path.join(base_path, "data", "bcicomp_iv_2a")


def _load_data(
    task_name: str,  # needed for the cache
    split_value: str,  # needed for the cache
    subjects: list[int],
    target_labels: list[str],
    interval: list[int],
    sampling_frequency: int,
    target_frequency: int,
) -> tuple[np.ndarray, np.ndarray]:
    print("BCICompIV2aDataset._load_data")

    data = []
    labels = []

    events_dict = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
    target_event_ids = [events_dict[label] for label in target_labels]

    for subject in subjects:
        for s in ["T", "E"]:
            file_name = os.path.join(data_path, f"A{subject:02d}{s}.mat")
            mat = loadmat(file_name, struct_as_record=False, squeeze_me=True)
            raw_data = mat["data"]
            for run in raw_data:
                run_data = []
                if len(run.trial) == 0:
                    continue
                eeg_data = 1e-6 * run.X  # scale
                eeg_data = eeg_data[:, :22]  # only use EEG channels

                for i, start_idx in enumerate(run.trial):
                    if run.y[i] not in target_event_ids:
                        continue
                    run_data.append(
                        eeg_data[start_idx : start_idx + 7 * sampling_frequency].T
                    )
                    labels.append(run.y[i])
                data.append(run_data)

    data = np.concatenate(data, axis=0)
    labels = np.array(labels)

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


class BCICompIV2aDataset(AbstractDataset):
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
            interval=[2, 6],
            name="bcicomp_iv_2a", # MI Limb
            task=task,
            tasks=["left_right", "right_feet", "left_right_feet_tongue"],
            split=split,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=250,
            channel_names=["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"],
            preload=preload,
        )
        # fmt: on
        print("BCICompIV2aDataset")
        self.data = None  # has the raw data in a array of shape (n_samples, n_channels, n_times) already for the specific task and split
        self.labels = None  # has the labels in a array of shape (n_samples,) already for the specific task
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
        print("BCICompIV2aDataset._download")
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        if subject < 1 or subject > 9:
            raise ValueError(
                f"For {self} Subject must be between 1 and 9, got {subject}"
            )

        download_url = "https://bnci-horizon-2020.eu/database/data-sets/"
        downloader = choose_downloader(download_url, progressbar=True)
        if type(downloader).__name__ in ["HTTPDownloader", "DOIDownloader"]:
            downloader.kwargs.setdefault("verify", False)

        for s in ["T", "E"]:
            file_name = f"A{subject:02d}{s}.mat"
            if os.path.isfile(os.path.join(data_path, file_name)):
                continue

            retrieve(
                url=f"{download_url}001-2014/{file_name}",
                known_hash=None,  # TODO known_hash
                fname=file_name,
                path=data_path,
                progressbar=True,
                downloader=downloader,
            )

    def load_data(self):
        # check if the data is downloaded
        subjects = self.task_split[self._task.name][self._split.value]["subjects"]
        for subject in subjects:
            for s in ["T", "E"]:
                if not os.path.isfile(
                    os.path.join(data_path, f"A{subject:02d}{s}.mat")
                ):
                    self._download(subject)
        # now the data is downloaded and unpacked
        # load the data
        self.data, self.labels = self._cache.cache(_load_data)(
            self._task.name,
            self._split.value,
            subjects,
            self.task_split[self._task.name]["labels"],
            self._interval,
            self._sampling_frequency,
            self._target_frequency,
        )

        if self._target_channels is not None:
            target_indices = [
                self._channel_names.index(ch) for ch in self._target_channels
            ]
            self.data = self.data[:, target_indices, :]
