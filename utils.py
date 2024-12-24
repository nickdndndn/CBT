from pathlib import Path
import torch
import datetime
import numpy as np

def get_checkpoint_path():
    """
    Returns a Path object pointing to the "configs" directory.
    """
    current_dir = Path(__file__).resolve().parent
    checkpoints_path = current_dir / "checkpoints"
    return checkpoints_path


def get_config_path():
    """
    Returns a Path object pointing to the "configs" directory.
    """
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir / "configs"
    return config_path


def get_data_loader_path():
    """
    Returns a Path object pointing to the "configs" directory.
    """
    current_dir = Path(__file__).resolve().parent
    data_loader_path = current_dir / "data_loader"
    return data_loader_path


def get_model_path():
    """
    Returns a Path object pointing to the "configs" directory.
    """
    current_dir = Path(__file__).resolve().parent
    model_path = current_dir / "model"
    return model_path


def save_checkpoint(state, filename: str, current_daytime: str):
    # print("=> Saving checkpoint")
    checkpoint_path = get_checkpoint_path()

    (checkpoint_path/filename).mkdir(parents=True, exist_ok=True)
    torch.save(state,  checkpoint_path / filename /
               f'{filename}_{current_daytime}.pth.tar')


def load_checkpoint(checkpoint, model, optimizer, tr_metrics, val_metrics):
    # print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    tr_metrics = checkpoint['tr_metrics']
    #val_metrics = checkpoint['val_metrics']

    return (tr_metrics)

def ergas_batch(reference_batch, synthesized_batch, scale_ratio):
    reference_batch = reference_batch.cpu().numpy()
    synthesized_batch = synthesized_batch.cpu().numpy()

    n, h, w, c = reference_batch.shape
    rmse = np.sqrt(np.mean((reference_batch - synthesized_batch) ** 2, axis=(1, 2)))
    mean_ref = np.mean(reference_batch, axis=(1, 2))
    ergas_values = 100 * scale_ratio * np.sqrt(np.mean((rmse / mean_ref) ** 2, axis=1))
    return ergas_values


def sam_batch(reference_batch, synthesized_batch):
    reference_batch = reference_batch.cpu().numpy()
    synthesized_batch = synthesized_batch.cpu().numpy()

    product = np.sum(reference_batch * synthesized_batch, axis=3)
    norm_ref = np.linalg.norm(reference_batch, axis=3)
    norm_syn = np.linalg.norm(synthesized_batch, axis=3)
    cos_theta = product / (norm_ref * norm_syn)
    cos_theta = np.clip(cos_theta, -1, 1)  # Ensure the values are within [-1, 1]
    sam_values = np.mean(np.arccos(cos_theta), axis=(1, 2))
    return sam_values

def q2n_batch(reference_batch, synthesized_batch):
    reference_batch = reference_batch.cpu().numpy()
    synthesized_batch = synthesized_batch.cpu().numpy()

    mean_ref = np.mean(reference_batch, axis=(1, 2, 3))
    mean_syn = np.mean(synthesized_batch, axis=(1, 2, 3))
    var_ref = np.var(reference_batch, axis=(1, 2, 3))
    var_syn = np.var(synthesized_batch, axis=(1, 2, 3))
    covariance = np.mean((reference_batch - mean_ref[:, None, None, None]) * (synthesized_batch - mean_syn[:, None, None, None]), axis=(1, 2, 3))
    
    q2n_values = (4 * covariance * mean_ref * mean_syn) / ((var_ref + var_syn) * (mean_ref**2 + mean_syn**2))
    return q2n_values
