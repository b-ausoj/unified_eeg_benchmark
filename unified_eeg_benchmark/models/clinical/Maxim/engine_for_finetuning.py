from tqdm import tqdm
import gc
import torch
from ..LaBraM import utils
from torch.utils.data import DataLoader
from torch import device
from torch import nn
from torch import optim

def move_to_device(obj, device: device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    # Add other types as necessary (e.g., sets)
    return obj

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, device: device):
    model.train()
    running_loss, running_corrects, total_samples = 0.0, 0, 0

    for batch in tqdm(dataloader, desc="Training", leave=True):
        inputs, targets = batch
        
        inputs = move_to_device(inputs, device)  # Move inputs to the GPU
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = model.loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        preds = torch.argmax(logits, dim=0)
        running_corrects += torch.sum(preds == targets).item()
        total_samples += 1

        # del inputs, targets, logits, loss  # Delete tensors no longer needed
        # torch.cuda.empty_cache()  # Clear cached memory on GPU
        # gc.collect()  # Invoke garbage collection

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

def validate_epoch(model: nn.Module, dataloader: DataLoader, device: device):
    model.eval()
    running_loss, running_corrects, total_samples = 0.0, 0, 0
    all_targets = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=True):
            inputs, targets = batch
            
            inputs = move_to_device(inputs, device)
            targets = targets.to(device)
            
            logits = model(inputs)
            loss = model.loss_fn(logits, targets)
            
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=0)
            running_corrects += torch.sum(preds == targets).item()
            total_samples += 1
            
            all_targets.append(targets.cpu())
            all_logits.append(logits.cpu())

            # del inputs, targets, logits  # Delete tensors no longer needed
            # torch.cuda.empty_cache()  # Clear cached memory on GPU
            # gc.collect()  # Invoke garbage collection
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    
    # Concatenate predictions and labels
    all_targets = torch.stack(all_targets).numpy() # all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.stack(all_logits).numpy() # all_logits = torch.cat(all_logits, dim=0)
    
    # Compute additional metrics using get_metrics
    metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
    results = utils.get_metrics(all_logits, all_targets, metrics, False)
    
    return epoch_loss, epoch_acc, results

def inference(model: nn.Module, dataloader: DataLoader, device: device):
    """
    Perform inference using the given model and dataloader.

    Args:
        model (torch.nn.Module): The trained model to use for inference.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the test data.
        device (torch.device): The device (CPU or GPU) to perform inference on.

    Returns:
        np.ndarray: Predictions for the test data as a NumPy array.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=True):
            x, _  = batch
            x = move_to_device(x, device)
            logits = model(x)
            preds = torch.argmax(logits, dim=0)
            predictions.append(preds.cpu())
    predictions = torch.stack(predictions).numpy()
    return predictions
