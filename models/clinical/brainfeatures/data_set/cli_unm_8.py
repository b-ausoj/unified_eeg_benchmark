import pandas as pd
import numpy as np
from glob import glob
from brainfeatures.data_set.abstract_data_set import DataSet
from scipy.io import loadmat
import mne
from resampy import resample


class CliUnm8DataSet(DataSet):
    """
    Dataset class for CLI_UNM_D008 EEG data.
    """
    def __init__(self, data_path, extension=".cnt", subset=None,
                 channels=sorted(['FP1', 'FPZ',   'FP2', 'AF3',   'AF4', 'F7',    'F5', 'F3',     'F1', 'FZ', 
            'F2', 'F4',     'F6', 'F8',     'FT7', 'FC5',   'FC3', 'FC1',   'FCZ', 'FC2', 
            'FC4', 'FC6',   'FT8', 'T7',    'C5', 'C3',     'C1', 'CZ',     'C2', 'C4', 
            'C6', 'T8',     'M1', 'TP7',    'CP5', 'CP3',   'CP1', 'CPZ',   'CP2', 'CP4', 
            'CP6', 'TP8',   'M2', 'P7',     'P5', 'P3',     'P1', 'PZ',     'P2', 'P4', 
            'P6', 'P8',     'PO7', 'PO5',   'PO3', 'POZ',   'PO4', 'PO6',   'PO8', 'CB1', 
            'O1', 'OZ', 'O2', 'CB2']), key="natural", n_recordings=None,
                 target="label", max_recording_mins=None, tfreq=100):
        self.max_recording_mins = max_recording_mins
        self.n_recordings = n_recordings
        self.extension = extension
        self.data_path = data_path
        self.channels = channels
        assert target in ["label", "age", "oci", "sex"], "target must be 'label', 'age', 'oci' or 'sex'"
        self.target = target # "label" (bool) or "age", "oci", "sex" (int)
        self.subset = subset
        self.key = key
        self.tfreq = tfreq

        self.subject_ids = []
        self.file_names = []
        self.targets = []


        assert data_path.endswith("/"), "data path must end with '/'"
        assert extension.startswith("."), "file extension must start with '.'"

    def load(self):
        print("\nLoading data...")
        """
        Load metadata and file paths for the dataset.
        """
        self.file_names = glob(self.data_path + "**/*" + self.extension, recursive=True)
        if len(self.file_names) == 0:
            raise ValueError(f"No files with extension {self.extension} found in {self.data_path}.")

        # Filter the number of recordings if n_recordings is specified
        if self.n_recordings:
            self.file_names = self.file_names[:self.n_recordings]

        toDelete = []

        for file_name in self.file_names:
            # Use mne to read the .cnt file and extract necessary data
            try:
                if self.extension == ".cnt":
                    # Extract clinical label based on parsing logic
                    subject_id = int(file_name.split('/')[-1].split('_')[0][:3])
                    print(f"loading Subject ID: {subject_id}")
                    if (subject_id >= 945):
                        toDelete.append(file_name)
                        continue
                    self.subject_ids.append(subject_id)
                    self.targets.append(self._get_targets(subject_id))

                elif self.extension == ".h5":
                    targets_df = pd.read_hdf(file_name, key="targets")
                    assert len(targets_df) == 1, "too many rows in targets df"
                    targets = targets_df.iloc[-1].to_dict()
                    self.targets.append(targets)

                    info_df = pd.read_hdf(file_name, key="info")
                    assert len(info_df) == 1, "too many rows in info df"
                    info = info_df.iloc[-1].to_dict()
                    sfreq = info["sfreq"]
                    assert sfreq == self.tfreq, f"sfreq {sfreq} does not match target frequency {self.tfreq}"
                
                else:
                    raise ValueError(f"Unsupported file extension: {self.extension}")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

        for file in toDelete:
            self.file_names.remove(file)
        
        assert len(self.file_names) == len(self.targets), "lengths differ"

    def __getitem__(self, index):
        """
        Returns a single data sample.
        """
        file_name = self.file_names[index]
        target = self.targets[index][self.target]

        try:
            if self.extension == ".h5":
                signals = pd.read_hdf(file_name, key="data")
            elif self.extension == ".cnt":
                raw = mne.io.read_raw_cnt(file_name, preload=True, data_format='int16')
                sfreq = raw.info["sfreq"]
                raw = raw.reorder_channels(self.channels)
                signals = raw.get_data()

                # preprocess signals
                signals = signals * 1e6
                signals = signals[:, int(30 * sfreq):]
                signals = self._resampl(signals, sfreq, self.tfreq)
                signals = self._clip_values(signals, 800)

                signals = pd.DataFrame(signals, index=self.channels)
            else:
                raise ValueError(f"Unsupported file extension: {self.extension}")
        except Exception as e:
            raise ValueError(f"Error reading file {file_name}: {e}")

        return signals, self.tfreq, target

    def __len__(self):
        """
        Returns the total number of examples in the dataset.
        """
        return len(self.file_names)

    def _get_clinical_label(self, subject_id):
        """
        Extract the clinical label based on subject ID and `pd_list` logic.
        """
        pd_list, ctr_list = self._load_clinical_groups()
        if subject_id in pd_list:
            return True
        elif subject_id in ctr_list:
            return False
        else:
            print(f"Unknown subject ID for clinical group identification: {subject_id}")
            return False
        

    def _get_targets(self, subject_id):
        pd_list, ctr_list, values_df = self._load_clinical_data()

        targets_df = values_df.loc[values_df['ID'] == subject_id]

        assert len(targets_df) <= 1, f"Multiple rows found for subject ID {subject_id}"
        assert not targets_df.empty, f"No rows found for subject ID {subject_id}"

        targets = {key.lower(): value for key, value in targets_df.iloc[0].to_dict().items() if key != 'ID'}

        if subject_id in pd_list:
            targets["label"] = True
        elif subject_id in ctr_list:
            targets["label"] = False
        else:
            print(f"Unknown subject ID for clinical group identification: {subject_id}")
        
        return targets
        
    def _load_clinical_data(self):
        """
        Load clinical data based on parsing logic.
        """
        df_vars = pd.read_excel('/itet-stor/jbuerki/deepeye_storage/foundation_clinical/CLI_UNM_D008/data/OCI Flankers/Info.xlsx',
                                sheet_name='SELECT', skiprows=[47, 48, 49, 50])
        oci_ids = df_vars.loc[df_vars['OCI'] >= 21.0, ['ID']].values.flatten().astype(int)
        ctr_ids = df_vars.loc[df_vars['OCI'] < 21.0, ['ID']].values.flatten().astype(int)
        selected_columns = ['ID', 'OCI', 'Sex', 'Age']
        values_df = df_vars[selected_columns]
        return oci_ids, ctr_ids, values_df
    
    def _resampl(self, signals: np.ndarray, fs: int, resample_fs: int) -> np.ndarray:
                return resample(x=signals, sr_orig=fs, sr_new=resample_fs, axis=1, filter='kaiser_fast')

    def _clip_values(self, signals: np.ndarray, max_abs_value: float) -> np.ndarray:
        return np.clip(signals, -max_abs_value, max_abs_value)
