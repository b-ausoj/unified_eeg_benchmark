# 🧠 Unified EEG Benchmark
**A standardized and extensible benchmark for evaluating classical and foundation models across clinical and BCI EEG decoding tasks.**

This benchmark supports rigorous cross-subject and cross-dataset evaluation across 23 datasets. It includes clinical classification tasks (e.g. epilepsy, Parkinson’s, schizophrenia) and motor imagery paradigms (e.g. left vs. right hand, 5-finger decoding), and provides baselines from CSP to foundation models like BENDR, LaBraM, and NeuroGPT.

## 📦 Installation

### Clone and Setup Environment

```bash
git clone https://github.com/b-ausoj/unified_eeg_benchmark
cd unified_eeg_benchmark
conda env create -f environment.yml
conda activate unified_eeg_benchmark
```

### Configure Paths
I would quickly search the whole project (Shift + CMD + F) for occurences of `jbuerki` or and adjust it for your personal username. Furthermore the cache path is defined in `utils/config.json`. The download paths for MOABB probably also need to be adjusted (in MNE config).
- Update hardcoded paths (jbuerki) using global search (e.g. Shift+Cmd+F).
- Adjust cache and download paths:
    - utils/config.json (benchmark cache)
    - MNE config for MOABB dataset paths

## 📁 Project Structure

```bash
unified_eeg_benchmark/
├── datasets/         # EEG dataset loaders (BCI & clinical)
├── models/           # All model implementations (CSP, LaBraM, etc.)
├── tasks/            # Benchmark tasks (MI, clinical diagnosis)
├── slurm/            # SLURM batch scripts (optional)
├── results/          # Stores experiment outputs
├── benchmark_console.py  # CLI interface to run experiments
└── utils/            # Helpers, enums, configs
```


## 🚀 Running a Benchmark
The benchmark can be run via the script `benchmark_console.py` with the arguments `--model` and `--task`. 

```bash
python benchmark_console.py --model labram --task lr
```

### Available Tasks
| Task Code | Task Class                     |
|-----------|--------------------------------|
| pd        | ParkinsonsClinicalTask         |
| sz        | SchizophreniaClinicalTask      |
| mtbi      | MTBIClinicalTask               |
| ocd       | OCDClinicalTask                |
| ep        | EpilepsyClinicalTask           |
| ab        | AbnormalClinicalTask           |
| lr        | LeftHandvRightHandMITask       |
| rf        | RightHandvFeetMITask           |
| lrft      | LeftHandvRightHandvFeetvTongueMITask |
| 5f        | FiveFingersMITask              |

### Available Models
| Model Code | Model Class                   |
|------------|-------------------------------|
| lda        | CSP or Brainfeatures with LDA |
| svm        | CSP or Brainfeatures with SVM |
| labram     | LaBra                         |
| bendr      | BENDR                         |
| neurogpt   | NeuroGPT                      |


## ➕ Adding Your Own Dataset
So far this benchmark supports two paradigms: Clinical and BCI (Motor Imagery). In Clinical one has to classify an entire recording whereas in BCI, one classifies a short sequence (trial). To add your dataset:
1. Place your class in `datasets/bci/` or `datasets/clinical/`
2. Inherit from `BaseBCIDataset` or `BaseClinicalDataset`
3. Implement:
    1. `_download` by either downloading the data or give the user the instruction how to do so
    2. `load_data` to populate:
        - `self.data` with type `np.ndarray | List [BaseRaw]` and dim `(n_samples, n_channels, n_sample_length)`
        - `self.labels` with type `np.ndarray | List[str]` and dim `(n_samples, )`
        - `self.meta`: contains at least `sampling_frequency`, `channel_names` and `name`
    4. if your dataset contains classes not yet part of the enum `enums.Classes` or `enums.ClinicalClasses` please add it
    5. All EEG signals should be standardized to the microvolt (µV ) scale. To reduce memory usage and computational overhead, signals with sampling rate more than 300 Hz typically resampled to 250 Hz.

## 🧪 Adding Your Own Task
Tasks constitute the central organizing principle of the benchmark, encapsulating paradigms, datasets, prediction classes, subject splits (i.e., training and test sets), and evaluation metrics. Each task class implements a get_data() method that returns training or testing data, along with the corresponding labels and metadata. These predefined splits ensure evaluation consistency and facilitate reproducibility. The tasks are split into Clinical and BCI aswell.

Tasks define:
- Datasets to use
- Train/test subject splits
- Target classes
- Evaluation metrics

Add your task to:
- `tasks/bci/` → inherit from `AbstractBCITask`
- `tasks/clinical/` → inherit from `AbstractClinicalTask`

Implement `get_data()` to return training/testing splits with data, labels, and metadata.

## 📊 Evaluation & Reproducibility
All experiments:
- Use fixed subject-level splits
- Support held-out dataset generalization
- Report balanced accuracy and weighted F1-score
- Use a fixed random seed for NumPy/PyTorch/random

##
Thesis available **upon request**.
