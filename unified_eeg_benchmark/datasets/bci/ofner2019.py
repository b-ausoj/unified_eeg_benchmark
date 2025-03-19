from .base_bci_dataset import BaseBCIDataset
import warnings
from ...enums.classes import Classes
import moabb
from typing import Optional, Sequence
import logging
import numpy as np
from mne.io import read_raw_gdf
import mne

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

data_path = "/itet-stor/jbuerki/net_scratch/data/MI_SCI/"


def _load_data_ofner2019(classes: Sequence[str], subjects: Sequence[int]):
    
    all_data = []
    all_labels = []
    for subject in subjects:
        if subject > 6:
            subject = subject + 1
        if subject == 10:
            file_prefix = f"{data_path}P{subject} Run "
        else:
            file_prefix = f"{data_path}P0{subject} Run "

        subject_data = []
        subject_labels = []
        for j in [3, 4, 5, 6, 7, 10, 11, 12, 13]:
            file_path = file_prefix + str(j) + '.gdf'
            raw = read_raw_gdf(file_path, preload=True)
            mne.set_eeg_reference(raw, ref_channels='average')
            raw.plot().savefig(f"raw_plot_{subject}_{j}.png")
            events, event_dict = mne.events_from_annotations(raw)
            new_event_dict = {v:k for k,v in event_dict.items()}
            data = raw.get_data()[:61, :] / 10 # shape (61, 75520)
            
            k = 0
            while k < len(events)-8:
                start = events[k+3][0]
                end = events[k+4][0]
                if (np.amax(data[:, start:end]) > 1000) or (np.amin(data[:, start:end]) < -1000):
                    k += 7
                    print("skipping due to high amplitude")
                    print(np.amax(data[:, start:end]), np.amin(data[:, start:end]))
                    max_channel = np.argmax(np.amax(data[:, start:end], axis=1))
                    min_channel = np.argmin(np.amin(data[:, start:end], axis=1))
                    print(f"Max value from channel: {raw.ch_names[max_channel]}")
                    print(f"Min value from channel: {raw.ch_names[min_channel]}")
                    continue

                code = new_event_dict.get(events[k+3][2])
                if code == "776" and Classes.RIGHT_SUPINATION_MI in classes:
                    subject_data.append(data[:, start:end])
                    subject_labels.append("supination")
                elif code == "777" and Classes.RIGHT_PRONATION_MI in classes:
                    subject_data.append(data[:, start:end])
                    subject_labels.append("pronation")
                elif code == "779" and Classes.RIGHT_HAND_OPEN_MI in classes:
                    subject_data.append(data[:, start:end])
                    subject_labels.append("hand_open")
                elif code == "925" and Classes.RIGHT_PALMAR_GRASP_MI in classes:
                    subject_data.append(data[:, start:end])
                    subject_labels.append("palmar_grasp")
                elif code == "926" and Classes.RIGHT_LATERAL_GRASP_MI in classes:
                    subject_data.append(data[:, start:end])
                    subject_labels.append("lateral_grasp")
                k += 7
        
        all_data.append(np.array(subject_data))
        all_labels.append(np.array(subject_labels))
    
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_data.shape)
    print(all_labels.shape)
    
    return all_data, all_labels


class Ofner2019Dataset(BaseBCIDataset):
    """
    - data is a 
    - labels is 
    - doi: https://doi.org/10.1038/s41598-019-43594-9
    - subjects: total 9 subjects
    - supination, pronation, hand open, palmar grasp, lateral grasp,
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
            name="ofner_2019", # MI SCI
            interval=(2, 5),
            target_classes=target_classes,
            available_classes=[Classes.RIGHT_PRONATION_MI, Classes.RIGHT_SUPINATION_MI, Classes.RIGHT_HAND_OPEN_MI, Classes.RIGHT_PALMAR_GRASP_MI, Classes.RIGHT_LATERAL_GRASP_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=256,
            channel_names=['AFZ', 'F3', 'F1', 'FZ', 'F2', 'F4', 'FFC5h', 'FFC3h', 'FFC1h', 'FFC2h', 'FFC4h', 'FFC6h', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FCC5h', 'FCC3h', 'FCC1h', 'FCC2h', 'FCC4h', 'FCC6h', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CCP5h', 'CCP3h', 'CCP1h', 'CCP2h', 'CCP4h', 'CCP6h', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'CPP5h', 'CPP3h', 'CPP1h', 'CPP2h', 'CPP4h', 'CPP6h', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'PPO1h', 'PPO2h',  'POz'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Ofner2019Dataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {"supination": 0, "pronation": 1, "hand_open": 2, "palmar_grasp": 3, "lateral_grasp": 4},
            "name": "Kaya2018",
        }

        if preload:
            self.load_data()

    def _download(self, subject: int):
        print(f"Please download the data from https://bnci-horizon-2020.eu/database/data-sets and place it in the data folder: {data_path}")

    def load_data(self) -> None:
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all...")
            classes = self.available_classes
        elif set(self.target_classes) == set(
            [
                Classes.RIGHT_PRONATION_MI,
                Classes.RIGHT_SUPINATION_MI,
                Classes.RIGHT_HAND_OPEN_MI,
                Classes.RIGHT_PALMAR_GRASP_MI,
                Classes.RIGHT_LATERAL_GRASP_MI,
            ]
        ):
            classes = self.target_classes
        elif set(self.target_classes) == set(
            [Classes.RIGHT_PRONATION_MI, Classes.RIGHT_SUPINATION_MI]
        ):
            classes = self.target_classes
        elif set(self.target_classes) == set([Classes.RIGHT_PALMAR_GRASP_MI, Classes.RIGHT_LATERAL_GRASP_MI]):
            classes = self.target_classes
        else:
            raise ValueError("Invalid target classes")

        if (self.subjects is None) or (len(self.subjects) == 0):
            self.data = np.array([])
            self.labels = np.array([])
        else:
            self.data, self.labels = self.cache.cache(_load_data_ofner2019)(
                classes, self.subjects
            )  # type: ignore
