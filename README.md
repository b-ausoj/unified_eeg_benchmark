# Unified EEG Benchmark

## Installation

### Installing packages

The necessary dependencies can be found in the `environment.yml` file.

Run 

```
conda env create -f environment.yml
```

to install the environment on your machine.

### Adjusting paths
I would quickly search the whole project (Shift + CMD + F) for occurences of `jbuerki` or and adjust it for your personal username. Furthermore the cache path is defined in `utils/config.json`. The download paths for MOABB probably also need to be adjusted (in MNE config).

## Usage

The important parts of the project structure are:
- unified_eeg_benchmark/datasets/: all the datasets contained
- unified_eeg_benchmark/models/: all the implemented models
- unified_eeg_benchmark/tasks/: all tasks part of the benchmark
- benchmark_*.py: python script to run the benchmarks

Supporting folders are:
- slurm/: scripts to submit with `sbatch`
- results/: results of the executed experiments

## Running
The benchmark can be run via the script `benchmark_console.py` with the arguments `--model` and `--task`. The options are 

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

| Model Code | Model Class                   |
|------------|-------------------------------|
| lda        | CSP or Brainfeatures with LDA |
| svm        | CSP or Brainfeatures with SVM |
| labram     | LaBra                         |
| bendr      | BENDR                         |
| neurogpt   | NeuroGPT                      |

for example
```
python benchmark_console.py --model labram --task lr
```