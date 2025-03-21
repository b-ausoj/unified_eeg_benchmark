import pandas as pd
import numpy as np
import re
from glob import glob
from brainfeatures.data_set.abstract_data_set import DataSet
from scipy.io import loadmat
import mne
from resampy import resample
import fnmatch as fm
import warnings
import mat73 # type: ignore


intersection = ['C4', 'FC3', 'P6', 'O1', 'CP4', 'C5', 'PO7', 'TP7', 'F4', 'P3', 'CP6', 'C3', 'FC4', 'F5', 'FC5', 'CP2', 'F2', 'P2', 'P5', 'F8', 'CP1', 'FC1', 'C6', 'F7', 'C2', 'T7', 'FCZ', 'CZ', 'AF3', 'FC6', 'F6', 'TP8', 'CP5', 'P7', 'O2', 'F1', 'FC2', 'FZ', 'F3', 'P8', 'C1', 'P4', 'POZ', 'T8', 'PO8', 'AF4', 'P1', 'OZ', 'CP3'] # len:  49
union = ['FT8', 'EKG', 'HEOG', 'FT7', 'CP4', 'C5', 'PO7', 'M2', 'F4', 'VEOGL', 'P5', 'FC1', 'C2', 'T7', 'F6', 'CB2', 'O2', 'F1', 'FP2', 'FC2', 'FZ', 'F3', 'C1', 'PZ', 'PO8', 'POZ', 'AFZ', 'AF4', 'OZ', 'CP3', 'C4', 'FC3', 'M1', 'CPZ', 'AF8', 'P6', 'CB1', 'O1', 'TP7', 'PO5', 'P3', 'CP6', 'C3', 'FC4', 'F5', 'FC5', 'CP2', 'F2', 'P2', 'F8', 'CP1', 'C6', 'FP1', 'F7', 'FCZ', 'CZ', 'AF3', 'FC6', 'AF7', 'TP8', 'CP5', 'FPZ', 'TP10', 'VEOG', 'P7', 'AFP10', 'AFP9', 'P8', 'PO3', 'P4', 'T8', 'TP9', 'VEOGU', 'P1', 'PO6', 'PO4'] # len:  76
filtered_union = ['AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ', 'CZ', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCZ', 'FP1', 'FP2', 'FT7', 'FT8', 'FZ', 'O1', 'O2', 'OZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'PZ', 'T7', 'T8', 'TP7', 'TP8'] # len:  60

class CliUnmDataSet(DataSet):
    """
    Dataset class for all CLI_UNM EEG datasets.
    """
    def __init__(self, data_path, extension=".cnt", subset=None,
                 key="natural", n_recordings=None,
                 target="label", max_recording_mins=None, tfreq=100):
        self.max_recording_mins = max_recording_mins
        self.extension = extension
        self.data_path = data_path
        assert target in ["pd", "sc", "dep", "ocd", "mtbi", "bdi", "oci", "age", "sex"], "target was " + target # TODO
        self.target = target
        self.subset = subset
        self.key = key
        self.tfreq = tfreq
        
        self.subject_ids = []
        self.file_paths = []
        self.targets = []

        self.dataset = -1
        if len(data_path.split("/")) >= 6:
            tmp = data_path.split("/")[5]
            if (tmp.startswith("CLI_UNM_D")):
                tmp = tmp.split("_")[2]
                self.dataset = int(re.search(r'\d+', tmp).group())
        if self.dataset == -1 and not extension == ".h5":
            raise ValueError("Dataset could not be identified from data path.")

        assert data_path.endswith("/"), "data path must end with '/'"
        assert extension.startswith("."), "file extension must start with '.'"

    def load(self):
        """
        Load metadata and file paths for the dataset.
        """
        print("\nLoading data...")
        self.file_paths = glob(self.data_path + "/*" + self.extension, recursive=False)
        if len(self.file_paths) == 0:
            raise ValueError(f"No files with extension {self.extension} found in {self.data_path}.")
        # file_names contains all file paths to either a cli unm dataset or some train/eval data

        toDelete = []

        for file_path in self.file_paths:
            if self.dataset == 1:
                file_name = file_path.split('/')[-1]
                self.subject_ids.append(file_name.split('_')[0])
                self.targets.append(self._get_targets_1(file_name))
            elif self.dataset == 2:
                if (not file_path.endswith('T.mat')):
                    toDelete.append(file_path)
                    continue
                file_name = file_path.split('/')[-1]
                self.subject_ids.append(file_name.split('_')[0])
                self.targets.append(self._get_targets_pd(file_name, self.dataset))
            elif self.dataset == 3:
                file_name = file_path.split('/')[-1]
                file_name = file_name.replace('.', '_')
                self.subject_ids.append(file_name.split('_')[0])
                self.targets.append(self._get_targets_dep(file_name, self.dataset))
            elif self.dataset == 4:
                if not fm.fnmatch(file_path, '*EEG*'):
                    toDelete.append(file_path)
                    continue
                file_name = file_path.split('/')[-1]
                subj_int = int(file_name.split('_')[2][1:])
                if subj_int in [148, 106]:
                    toDelete.append(file_path)
                    continue
                file_name = file_name.split('.')[0]
                self.subject_ids.append(file_name)
                self.targets.append(self._get_targets_4(file_name))
            elif self.dataset == 5:
                file_name = file_path.split('/')[-1]
                self.subject_ids.append(file_name.split('_')[0])
                self.targets.append(self._get_targets_pd(file_name, self.dataset))
            elif self.dataset == 6:
                file_name = file_path.split('/')[-1]
                file_name = file_name.replace('.', '_')
                self.subject_ids.append(file_name.split('_')[0])
                self.targets.append(self._get_targets_dep(file_name, self.dataset))
            elif self.dataset == 7:
                file_name = file_path.split('/')[-1]
                self.subject_ids.append(file_name.split('_')[0])
                self.targets.append(self._get_targets_7(file_name))
            elif self.dataset == 8:
                subject_id = int(file_path.split('/')[-1].split('_')[0][:3])
                if (subject_id >= 945):
                    toDelete.append(file_path)
                    continue
                self.subject_ids.append(str(subject_id))
                self.targets.append(self._get_targets_8(subject_id))
            elif self.dataset == 9:
                file_name = file_path.split('/')[-1]
                self.subject_ids.append(file_name.split('_')[0])
                self.targets.append(self._get_targets_9(file_name))
            elif self.dataset == 11:
                if (not file_path.endswith('_Ped_Processed.mat')):
                    toDelete.append(file_path)
                    continue
                file_name = file_path.split('/')[-1]
                self.subject_ids.append(file_name.split('_')[0])
                self.targets.append(self._get_targets_11(file_name))
            elif self.dataset == 12:
                file_name = file_path.split('/')[-1]
                self.subject_ids.append(file_name.split('_')[0])
                self.targets.append(self._get_targets_12(file_name))
            elif self.dataset == 13:
                file_name = file_path.split('/')[-1]
                self.subject_ids.append(file_name.split('.')[0])
                self.targets.append(self._get_targets_13(file_name))
            elif self.dataset == 14:
                file_name = file_path.split('/')[-1]
                subj = file_name.split('.')[0]
                subj_int = int(subj[len(subj)-4:])
                if subj_int > 1800:
                    toDelete.append(file_path)
                    continue
                self.subject_ids.append(subj)
                self.targets.append(self._get_targets_14(file_name))
            elif self.dataset == -1: # self.extension == ".h5"
                targets_df = pd.read_hdf(file_path, key="targets")
                assert len(targets_df) == 1, "too many rows in targets df"
                targets = targets_df.iloc[-1].to_dict()
                self.targets.append(targets)

                info_df = pd.read_hdf(file_path, key="info")
                assert len(info_df) == 1, "too many rows in info df"
                info = info_df.iloc[-1].to_dict()
                sfreq = info["sfreq"]
                self.subject_ids.append(info["subject_id"])
                assert sfreq == self.tfreq, f"sfreq {sfreq} does not match target frequency {self.tfreq}"
            else:
                raise ValueError(f"Unsupported file from dataset number: {self.dataset}")

        for file in toDelete:
            self.file_paths.remove(file)
        
        assert len(self.file_paths) == len(self.targets), "lengths differ"

    def __getitem__(self, index):
        """
        Returns a single data sample.
        """
        file_name = self.file_paths[index]
        target = self.targets[index][self.target]

        if self.dataset == 1 or self.dataset == 2 or self.dataset == 3 or self.dataset == 4 or self.dataset == 5 or self.dataset == 6 or self.dataset == 7 or self.dataset == 9 or self.dataset == 12:
            mat = loadmat(file_name, simplify_cells=True)
            mdata = mat['EEG']['data']
            mchan = mat['EEG']['chanlocs']
            sfreq = mat['EEG']['srate']  # Extract sampling frequency

            signals = pd.DataFrame(data=np.transpose(mdata).reshape(-1, np.transpose(mdata).shape[-1]))
            signals = signals.T.values
                        
            signals = signals[:, int(30 * sfreq):] # Remove first 30 seconds
            signals = signals[:, :-int(30 * sfreq)] # Remove last 30 seconds
            if (self.dataset == 2):
                signals = signals * 1e-6
            signals = self._resampl(signals, sfreq, self.tfreq)  # Resample
            signals = self._clip_values(signals, 800)  # Clip values
            
            cols = pd.DataFrame.from_dict(mchan)['labels'].str.upper().values
            signals = pd.DataFrame(signals, index=cols)
        elif self.dataset == 8:
            raw = mne.io.read_raw_cnt(file_name, preload=True, data_format='int16')
            sfreq = raw.info["sfreq"]
            cols = [ch['ch_name'] for ch in raw.info['chs'] if ch['ch_name'].upper() in intersection]
            raw = raw.reorder_channels(cols)
            signals = raw.get_data()

            # preprocess signals
            signals = signals * 1e6
            signals = signals[:, int(30 * sfreq):] # Remove first 30 seconds
            signals = signals[:, :-int(30 * sfreq)] # Remove last 30 seconds
            signals = self._resampl(signals, sfreq, self.tfreq)
            signals = self._clip_values(signals, 800)

            signals = pd.DataFrame(signals, index=cols)
        elif self.dataset == 11:
            mat = mat73.loadmat(file_name)
            mdata = mat['EEG']['data']
            mchan = mat['EEG']['chanlocs']
            sfreq = mat['EEG']['srate']  # Extract sampling frequency

            signals = pd.DataFrame(data=np.transpose(mdata).reshape(-1, np.transpose(mdata).shape[-1]))
            signals = signals.T.values
                        
            signals = signals[:, int(30 * sfreq):] # Remove first 30 seconds
            signals = signals[:, :-int(30 * sfreq)] # Remove last 30 seconds
            signals = self._resampl(signals, sfreq, self.tfreq)  # Resample
            signals = self._clip_values(signals, 800)  # Clip values
            
            cols = pd.DataFrame.from_dict(mchan)['labels'].str.upper().values          
            signals = pd.DataFrame(signals, index=cols)
        elif self.dataset == 13 or self.dataset == 14:
            raw = mne.io.read_raw_brainvision(file_name, preload=True)
            sfreq = raw.info["sfreq"]
            cols = [ch['ch_name'] for ch in raw.info['chs'] if ch['ch_name'].upper() in intersection]
            raw = raw.reorder_channels(cols)
            signals = raw.get_data()

            # preprocess signals idk if correct TODO
            signals = signals * 1e6
            signals = signals[:, int(30 * sfreq):] # Remove first 30 seconds
            signals = signals[:, :-int(30 * sfreq)] # Remove last 30 seconds
            signals = self._resampl(signals, sfreq, self.tfreq)
            signals = self._clip_values(signals, 800)

            signals = pd.DataFrame(signals, index=cols)
        elif self.dataset == -1: # self.extension == ".h5"
            signals = pd.read_hdf(file_name, key="data")
        else:
            raise ValueError(f"Unsupported file from dataset number: {self.dataset}")

        if not self.dataset == -1:
            signals.index = signals.index.str.upper()
            signals = signals.loc[intersection]

        return signals, self.tfreq, target

    def __len__(self):
        """
        Returns the total number of examples in the dataset.
        """
        return len(self.file_paths)

    
    def _resampl(self, signals: np.ndarray, fs: int, resample_fs: int) -> np.ndarray:
            return resample(x=signals, sr_orig=fs, sr_new=resample_fs, axis=1, filter='kaiser_fast')

    def _clip_values(self, signals: np.ndarray, max_abs_value: float) -> np.ndarray:
        return np.clip(signals, -max_abs_value, max_abs_value)

    def _get_targets_1(self, file_name):
        isParkinson = file_name.split('_')[1] == '2'
        return {"pd": isParkinson}
    
    def _get_targets_4(self, file_name):
        df_vars = pd.read_csv('/itet-stor/jbuerki/deepeye_storage/foundation_clinical/CLI_UNM_D004/data/Cost Conflict in Schizophrenia/DeID_Dems.csv')
        sz_ids = df_vars.loc[df_vars['group'] == 'SZ', ['subno']].values
        subj_int = int(file_name.split('_')[2][1:])
        isSchizophrenia = subj_int in sz_ids
        age = df_vars.loc[df_vars['subno']==subj_int, ['Age']].values[0][0]
        sex = df_vars.loc[df_vars['subno']==subj_int, ['Sex']].values[0][0] # TODO check if 0 is M and 1 is F
        return {"sc": isSchizophrenia, "age": age, "sex": sex}
    
    def _get_targets_pd(self, file_name, set_nr):
        if set_nr == 5:
            df_vars = pd.read_excel('/itet-stor/jbuerki/deepeye_storage/foundation_clinical/CLI_UNM_D005/data/PD Conflict Task/Scripts used in manuscript/classificaiton scripts/PD_CONFLICT_VARS.xlsx')
        elif set_nr == 2:
            df_vars = pd.read_excel('/itet-stor/jbuerki/deepeye_storage/foundation_clinical/CLI_UNM_D002/data/PDREST/IMPORT_ME_REST.xlsx')
        else:
            raise ValueError(f"Unsupported dataset number: {set_nr}")
        pd_ids = df_vars['PD_ID'].values
        subj_int = int(file_name.split('_')[0])
        isParkinson = subj_int in pd_ids
        if isParkinson:
            bdi = df_vars.loc[df_vars['PD_ID']==subj_int, ['BDI']].values[0][0]
            age = df_vars.loc[df_vars['PD_ID']==subj_int, ['PD_Age']].values[0][0]
            sex = df_vars.loc[df_vars['PD_ID']==subj_int, ['PD_Sex']].values[0][0] # TODO check if 0 is M and 1 is F
        else:
            bdi = -1
            age = df_vars.loc[df_vars['MATCH CTL_ID']==subj_int, ['MATCH CTL_Age']].values[0][0]
            sex = df_vars.loc[df_vars['MATCH CTL_ID']==subj_int, ['MATCH CTL_Sex']].values[0][0] # TODO check if 0 is M and 1 is F
        return {"pd": isParkinson, "bdi": bdi, "age": age, "sex": sex}
    
    def _get_targets_dep(self, file_name, set_nr):
        if set_nr == 3: 
            df_vars = pd.read_excel('/itet-stor/jbuerki/deepeye_storage/foundation_clinical/CLI_UNM_D003/data/Depression Rest/Data_4_Import_REST.xlsx')
        elif set_nr == 6:
            df_vars = pd.read_excel('/itet-stor/jbuerki/deepeye_storage/foundation_clinical/CLI_UNM_D006/data/Depression PS Task/Scripts from Manuscript/Data_4_Import.xlsx')
        else:
            raise ValueError(f"Unsupported dataset number: {set_nr}")
        dep_ids = df_vars.loc[df_vars['BDI']>=7.0, ['id']].values
        subj_int = int(file_name.split('_')[0])
        isDepression = subj_int in dep_ids
        bdi = df_vars.loc[df_vars['id']==subj_int, ['BDI']].values[0][0]
        age = df_vars.loc[df_vars['id']==subj_int, ['age']].values[0][0]
        sex = df_vars.loc[df_vars['id']==subj_int, ['sex']].values[0][0] # TODO check if 0 is M and 1 is F
        return {"dep": isDepression, "bdi": bdi, "age": age, "sex": sex}

    def _get_targets_7(self, file_name):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            df_vars = pd.read_excel('/itet-stor/jbuerki/deepeye_storage/foundation_clinical/CLI_UNM_D007/data/PD RewP/MEASURES.xlsx', sheet_name='Sheet1', header=None)
        subj_int = int(file_name.split('_')[0])
        isParkinson = file_name.split('_')[2] == '2'
        if isParkinson:
            bdi = df_vars.loc[df_vars[0]==subj_int, [2]].values[0][0]
        else:
            bdi = -1
        return {"pd": isParkinson, "bdi": bdi}

    def _get_targets_8(self, subject_id):
        df_vars = pd.read_excel('/itet-stor/jbuerki/deepeye_storage/foundation_clinical/CLI_UNM_D008/data/OCI Flankers/Info.xlsx',
                                sheet_name='SELECT', skiprows=[47, 48, 49, 50])
        pd_list = df_vars.loc[df_vars['OCI'] >= 21.0, ['ID']].values.flatten().astype(int)
        ctr_list = df_vars.loc[df_vars['OCI'] < 21.0, ['ID']].values.flatten().astype(int)
        selected_columns = ['ID', 'OCI', 'Sex', 'Age']
        values_df = df_vars[selected_columns]

        # TODO add bdi

        targets_df = values_df.loc[values_df['ID'] == subject_id]

        assert len(targets_df) <= 1, f"Multiple rows found for subject ID {subject_id}"
        assert not targets_df.empty, f"No rows found for subject ID {subject_id}"

        targets = {key.lower(): value for key, value in targets_df.iloc[0].to_dict().items() if key != 'ID'}

        if subject_id in pd_list:
            targets["ocd"] = True
        elif subject_id in ctr_list:
            targets["ocd"] = False
        else:
            print(f"Unknown subject ID for clinical group identification: {subject_id}")
        
        return targets

    def _get_targets_9(self, file_name):
        mat_raw = loadmat('/itet-stor/jbuerki/deepeye_storage/foundation_clinical/CLI_UNM_D009/data/Scripts/BigAgg_Data.mat', simplify_cells=True)
        
        df_vars = pd.DataFrame()
        df_vars_q = pd.DataFrame()

        df_vars['SubID'] = mat_raw['DEMO']['ID'][:, 0]
        df_vars['ControlEquals1'] = mat_raw['DEMO']['Group_CTL1'][:, 0]
        df_vars['FemaleEquals1'] = mat_raw['DEMO']['Sex_F1']
        df_vars['Age'] = mat_raw['DEMO']['Age']
        df_vars['BDItotal'] = mat_raw['QUEX']['BDI'][:, 0]

        df_vars_q['SubID'] = mat_raw['Q_DEMO']['URSI']
        df_vars_q['ControlEquals1'] = 0
        df_vars_q['FemaleEquals1'] = mat_raw['Q_DEMO']['Sex_F1']
        df_vars_q['Age'] = mat_raw['Q_DEMO']['Age']
        df_vars_q['BDItotal'] = mat_raw['Q_QUEX']['BDI']

        df_vars = pd.concat([df_vars, df_vars_q], ignore_index=True)

        #df_vars.loc[df_vars['FemaleEquals1'] == 1, ['FemaleEquals1']] = 'F'
        #df_vars.loc[df_vars['FemaleEquals1'] == 0, ['FemaleEquals1']] = 'M'

        mtbi_list = df_vars.loc[df_vars['ControlEquals1'] != 1, ['SubID']].values.astype(int).flatten()

        subj_int = int(file_name.split('_')[0])
        isMTBI = subj_int in mtbi_list

        bdi = df_vars.loc[df_vars['SubID'] == subj_int, 'BDItotal'].values[0]
        age = df_vars.loc[df_vars['SubID'] == subj_int, 'Age'].values[0]
        sex = df_vars.loc[df_vars['SubID'] == subj_int, 'FemaleEquals1'].values[0]

        return {"mtbi": isMTBI, "bdi": bdi, "age": age, "sex": sex}

    def _get_targets_11(self, file_name):
        df_vars = pd.read_csv('/itet-stor/jbuerki/deepeye_storage/foundation_clinical/CLI_UNM_D011/data/PD Gait/ALL_data_Modeling.csv', sep='\t')
        df_vars['id_unique'] = 'PD' + df_vars['Pedal_ID'].astype(str)
        df_vars.loc[df_vars['Group']=='Control', ['id_unique']] = 'Control' + df_vars['Pedal_ID'].astype(str)
        subj = file_name.split('_')[0]
        isParkinson = 'PD' in file_name
        age = df_vars.loc[df_vars['id_unique']==subj, ['Age']].values[0][0]
        return {"pd": isParkinson, "age": age}

    def _get_targets_12(self, file_name):
        df_vars = pd.read_excel('/itet-stor/jbuerki/deepeye_storage/foundation_clinical/CLI_UNM_D012/data/mTBI Rest/Quex.xlsx', sheet_name='S1', skiprows=0)
        # df_vars.loc[df_vars['FemaleEquals1']==1,['FemaleEquals1']] = 'F'
        # df_vars.loc[df_vars['FemaleEquals1']==0,['FemaleEquals1']] = 'M'
        mtbi_list = df_vars.loc[df_vars['ControlEquals1']!=1, ['SubID']].values.astype(int).flatten()
        
        subj_int = int(file_name.split('_')[0])
        isMTBI = subj_int in mtbi_list
        bdi = df_vars.loc[df_vars['SubID']==subj_int, ['BDItotal']].values[0][0]
        age = df_vars.loc[df_vars['SubID']==subj_int, ['Age']].values[0][0]
        sex = df_vars.loc[df_vars['SubID']==subj_int, ['FemaleEquals1']].values[0][0]
        return {"mtbi": isMTBI, "bdi": bdi, "age": age, "sex": sex}

    def _get_targets_13(self, file_name):
        df_vars = pd.read_excel('/itet-stor/jbuerki/deepeye_storage/foundation_clinical/CLI_UNM_D013/data/Data and Code/Dataset/IowaDataset/DataIowa.xlsx', sheet_name='ALL')
        # df_vars.loc[df_vars['Gender']==0,['Gender']] = 'F'
        # df_vars.loc[df_vars['Gender']==1,['Gender']] = 'M'
        sz_ids = df_vars.loc[df_vars['Rest'].str.contains('PD'), ['Rest']].values.astype(str).flatten()
        subj = file_name.split('.')[0]
        isParkinson = subj in sz_ids
        age = df_vars.loc[df_vars['Rest']==subj, ['Age']].values[0][0]
        sex = df_vars.loc[df_vars['Rest']==subj, ['Gender']].values[0][0]
        return {"pd": isParkinson, "age": age, "sex": sex}

    def _get_targets_14(self, file_name):
        df_vars = pd.read_excel('/itet-stor/jbuerki/deepeye_storage/foundation_clinical/CLI_UNM_D014/data/PD Interval Timing NPJ/Copy of IntervalTiming_Subj_Info_AIE.xlsx', sheet_name='MAIN')
        # df_vars.loc[df_vars['Gender']=='Female',['Sex']] = 'F'
        # df_vars.loc[df_vars['Sex']=='Male',['Sex']] = 'M'
        sz_ids = df_vars.loc[df_vars['ITT'].str.contains('PD'), ['ITT']].values.astype(str).flatten()
        subj = file_name.split('.')[0]
        isParkinson = subj in sz_ids
        age = df_vars.loc[df_vars['ITT']==subj, ['Age']].values[0][0]
        sex = df_vars.loc[df_vars['ITT']==subj, ['Gender']].values[0][0]
        return {"pd": isParkinson, "age": age, "sex": sex}

