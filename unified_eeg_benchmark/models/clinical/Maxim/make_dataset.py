from tqdm import tqdm
import os
from multiprocessing import Pool, Process, Manager
from functools import partial
from .make_dataset_utils import writer_task, process_one_abnormal, process_one_epilepsy
from .fine_tune_dataset import FinetuneDataset
from ....utils.config import get_config_value
import h5py
import logging

def make_dataset(X, y, meta, task_name, model_name, is_test, use_cache):
    # Create or override the HDF5 file.
    h5_path = os.path.join(get_config_value("data"), "make_dataset", f"{task_name}_{model_name}_{meta[0]['name'].replace(' ', '_')}_{is_test}_{sum(len(obj) for obj in X)}.h5")
    
    if os.path.exists(h5_path) and use_cache:
        print(f"[Info] Dataset already exists at {h5_path}. Loading existing dataset.")
        return FinetuneDataset(h5_path, is_test)

    with h5py.File(h5_path, 'w') as hf:
        hf.create_group('/recordings')

    manager = Manager()
    output_queue = manager.Queue()
    writer = Process(target=writer_task, args=(output_queue, h5_path))
    writer.start()
    n_jobs = os.cpu_count() - 1
    if n_jobs < 1:
        n_jobs = 1
    
    if "abnormal" in task_name:
        X = X[0]
        if y is None:
            y = [None] * len(X)
        else:
            y = y[0]
        parameters = [(i, raw, label, model_name) for i, (raw, label) in enumerate(zip(X, y))]
        worker_func = partial(process_one_abnormal, output_queue=output_queue)
        with Pool(n_jobs) as pool:
            list(tqdm(pool.imap(worker_func, parameters), total=len(parameters),
                      desc="Processing abnormal data"))
        logging.info("--------- All recordings have been processed.")

    elif "epilepsy" in task_name:
        X, montage_types = X[0], meta[0]["montage_type"]
        if y is None:
            y = [None] * len(X)
        else:
            y = y[0]
        parameters = [(i, raw, label, montage, model_name) for i, (raw, label, montage) in enumerate(zip(X, y, montage_types))]
        worker_func = partial(process_one_epilepsy, output_queue=output_queue)
        with Pool(n_jobs) as pool:
            list(tqdm(pool.imap(worker_func, parameters), total=len(parameters),
                      desc="Processing epilepsy data"))
        logging.info("--------- All recordings have been processed.")
    else:
        raise ValueError(f"Unknown task name: {task_name}. Supported tasks are 'abnormal' and 'epilepsy'.")

    # Signal the writer process that all workers are done.
    print("[Main] Signaling writer process that all workers are done.")
    output_queue.put(None)
    writer.join()

    return FinetuneDataset(h5_path, is_test)
