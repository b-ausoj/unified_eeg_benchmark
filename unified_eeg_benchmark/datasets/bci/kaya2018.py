from .base_bci_dataset import BaseBCIDataset
import warnings
from ...enums.classes import Classes
import moabb
from typing import Optional, Sequence
import logging
import numpy as np
from glob import glob
import os
from scipy.io import loadmat
from resampy import resample
from mne.filter import filter_data
from mne.filter import notch_filter

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

data_path = "/itet-stor/jbuerki/net_scratch/data/MI_SCP/"

def translate_5F_Event(event):
    if event == 1:
        return "thumb"
    elif event == 2:
        return "index finger"
    elif event == 3:
        return "middle finger"
    elif event == 4:
        return "ring finger"
    elif event == 5:
        return "little finger"
    else:
        return "unknown"

def translate_CLA_HaLT_Event(event):
    if event == 1:
        return "left_hand"
    elif event == 2:
        return "right_hand"
    elif event == 4:
        return "feet" # "left_leg"
    elif event == 5:
        return "tongue"
    elif event == 6:
        return "feet" # "right_leg"
    else:
        return "unknown"

def _load_data_kaya2018(task: str, subjects: Sequence[str]):
    neu = "neu20001"
    if task == "leftright":
        print("Loading left vs right hand data...")
        paradigms = ["CLA"] 
        relevant_events = [1, 2]
        translate = translate_CLA_HaLT_Event
    elif task == "rightfeet":
        paradigms = ["HaLT"]
        relevant_events = [2, 4, 6]
        translate = translate_CLA_HaLT_Event
    elif task == "leftrightfeettongue":
        paradigms = ["HaLT"]
        relevant_events = [1, 2, 4, 5] # 6 is right leg
        translate = translate_CLA_HaLT_Event
    elif task == "5F":
        paradigms = ["5F"]
        relevant_events = [1, 2, 3, 4, 5]
        translate = translate_5F_Event
    else:
        raise ValueError("Invalid task: ", task)
    
    all_data = []
    all_labels = []
    for subject in subjects:
        # TODO load data
        for paradigm in paradigms:
            print(f"Loading {paradigm} data for subject {subject}...")
            file_path = os.path.join(data_path, f"{paradigm}-Subject{subject}-*.mat")
            files = glob(file_path)
            if len(files) == 0:
                logging.warning(f"No files found for {paradigm} and subject {subject}")
                continue
            for file in files:
                file_name = os.path.basename(file)
                print(f"Loading {file_name}")
                sfreq = 200
                if "HFREQ" in file_name:
                    sfreq = 1000

                mat = loadmat(file)
                data = mat["o"]
                if "Inter" in file_name:
                    continue
                    eeg = data[0][0][6] # shape (n_timepoints, n_channels)
                else:
                    eeg = data[0][0][5] # shape (n_timepoints, n_channels)

                mark = data[0][0][4][:, 0] # shape (n_timepoints, )
                # every time the mark changes, a new event starts
                events = np.where(mark[:-1] != mark[1:])[0] + 1

                trials = []
                labels = []
                print(relevant_events)
                for event in events:
                    if mark[event] in relevant_events:
                        trials.append(eeg[event:(event + sfreq), :21]) # TODO actual event is only 1 second long
                        labels.append(translate(mark[event]))

                trials = np.array(trials).transpose(0, 2, 1)  # (n_trials, n_channels, n_timepoints)
                labels = np.array(labels)

                print(f"Trials shape: {trials.shape}")  # (n_trials, n_channels, n_timepoints)
                print(f"Labels shape: {labels.shape}")  # (n_trials,)

                if sfreq == 1000:
                    # resample to 200 Hz
                    trials = resample(trials, 1000, 200, axis=-1, filter='kaiser_best')

                all_data.append(trials)
                all_labels.append(labels)

    all_data = np.concatenate(all_data, axis=0)
    all_data = all_data / 10
    #all_data = notch_filter(all_data, 200, 50, verbose='WARNING')
    #all_data = filter_data(all_data, 200, 8, 32, method='iir', verbose='WARNING')

    all_labels = np.concatenate(all_labels, axis=0)
    return all_data, all_labels


class Kaya2018Dataset(BaseBCIDataset):
    """
    - data is a np.ndarray of shape (n_trials, n_channels, n_timepoints)
    - labels is a np.ndarray of shape (n_trials,)
    - doi: https://doi.org/10.6084/m9.figshare.c.3917698.v1
    - subjects: 13 
    - 5F (8): A, B, C, E, F, G, H, I
    - CLA (7): A, B, C, D, E, F, J
    - HaLT (12): A, B, C, E, F, G, H, I, J, K, L, M
    - hardware: EEG-1200
    - so far only CLA (left vs right hand) and HaLT (left hand vs right hand vs legs vs tongue) paradigms
    - but could be extended to also include 5F (five fingers) paradigm
    """
    def __init__(
        self,
        target_classes: Sequence[Classes],
        subjects: Sequence[str],
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = None,
        preload: bool = True,
    ):
        # fmt: off
        super().__init__(
            name="kaya_2018", # MI SCP
            interval=(0, 1),
            target_classes=target_classes,
            available_classes=[Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI, Classes.FEET_MI, Classes.TONGUE_MI, Classes.FIVE_FINGERS_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=200,
            channel_names=["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "A1", "A2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"],
            preload=preload,
        )
        # fmt: on
        logging.info("in Kaya2018Dataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
            "name": "Kaya2018",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        print(f"Please download the data from https://doi.org/10.6084/m9.figshare.c.3917698.v1 and place it in the data folder: {data_path}")

    def load_data(self) -> None:
        if self.target_classes is None:
            logging.warning("target_classes is None, loading CLA...")
            task = "leftright"
        elif set(self.target_classes) == set(
            [
                Classes.LEFT_HAND_MI,
                Classes.RIGHT_HAND_MI,
                Classes.FEET_MI,
                Classes.TONGUE_MI,
            ]
        ):
            task = "leftrightfeettongue"
        elif set(self.target_classes) == set(
            [Classes.LEFT_HAND_MI, Classes.RIGHT_HAND_MI]
        ):
            task = "leftright"
        elif set(self.target_classes) == set([Classes.RIGHT_HAND_MI, Classes.FEET_MI]):
            task = "rightfeet"
        elif set(self.target_classes) == set([Classes.FIVE_FINGERS_MI]):
            task = "5F"
        else:
            raise ValueError("Invalid target classes")

        if (self.subjects is None) or (len(self.subjects) == 0):
            self.data = np.array([])
            self.labels = np.array([])
        else:
            self.data, self.labels = self.cache.cache(_load_data_kaya2018)(
                task, self.subjects
            )  # type: ignore
