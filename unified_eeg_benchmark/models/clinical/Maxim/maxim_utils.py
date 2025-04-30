import torch
import torchaudio
import json
from .transforms import (
    crop_spg,
    normalize_spg,
)

self_win_shifts = [0.25, 0.5, 1, 2, 4, 8]
self_patch_size = 16
self_win_shift_factor = 0.25
self_max_win_shift = self_win_shifts[-1]
self_max_y_datapoints = 4_000
max_nr_patches = 8_500

def self_get_generic_channel_name(channel_name):
    channel_name = channel_name.lower()
    # Remove "eeg " prefix if present
    if channel_name.startswith("eeg "):
        channel_name = channel_name[4:]
    # Simplify names with a dash and check if it ends with "-"
    if "-" in channel_name:
        if channel_name.endswith("-"):
            return "None"
        return channel_name.split("-")[0]
    return channel_name

def get_nr_y_patches(win_size, sr):
    return int((sr / 2 * win_size + 1) / self_patch_size)

def get_nr_x_patches(win_size, dur):
    win_shift = win_size * self_win_shift_factor
    x_datapoints_per_second = 1 / win_shift
    x_datapoints = dur * x_datapoints_per_second + 1
    return int(x_datapoints // self_patch_size)

def self_encode_mean(mean, win_size):
    y_datapoints = mean.shape[0]
    encoded_mean = torch.zeros(self_max_y_datapoints)
    step_size = int(self_max_win_shift // win_size)
    end_idx = step_size * y_datapoints
    indices = torch.arange(0, end_idx, step_size)
    encoded_mean[indices] = mean.squeeze_().float()
    encoded_mean.unsqueeze_(1)
    return encoded_mean


def sample_collate_fn(batch):
    signals, target, sr, dur, channels, dataset = (
        batch[0]["signals"],
        batch[0]["output"],
        batch[0]["sr"],
        batch[0]["dur"],
        batch[0]["channels"],
        batch[0]["dataset"],
    )
    max_dur = 600
    if dur > max_dur:
        dur = max_dur
        signals = signals[:, : int(max_dur * sr)]
    
    # TODO: compute spectrograms for each win_size
    # gives a new dimension (S) in batch
    # need another extra transformer after the encoder
    # (B, 1, H, W) -> (S*B, 1, H, W)
    valid_win_shifts = [
        win_shift
        for win_shift in self_win_shifts
        if get_nr_y_patches(win_shift, sr) >= 1
        and get_nr_x_patches(win_shift, dur) >= 1
    ]

    channel_name_map_path = (
        "/itet-stor/jbuerki/home/unified_eeg_benchmark/unified_eeg_benchmark/models/clinical/Maxim/channels_to_id.json"
    )
    with open(channel_name_map_path, "r") as file:
        self_channel_name_map = json.load(file)

    # list holding assembled tensors for varying window shifts
    full_batch = {}

    for win_size in valid_win_shifts:
        if win_size != 4:
            continue

        fft = torchaudio.transforms.Spectrogram(
            n_fft=int(sr * win_size),
            win_length=int(sr * win_size),
            hop_length=int(sr * win_size * self_win_shift_factor),
            normalized=True,
        )

        spg_list = []
        chn_list = []
        mean_list = []
        std_list = []

        for signal, channel in zip(signals, channels):

            # Channel information
            channel_name = self_get_generic_channel_name(channel)
            channel = (
                self_channel_name_map[channel_name]
                if channel_name in self_channel_name_map
                else self_channel_name_map["None"]
            )

            # Spectrogram Computation & Cropping
            spg = fft(signal)
            spg = spg**2
            spg = crop_spg(spg, self_patch_size)

            H_new, W_new = spg.shape[0], spg.shape[1]
            h_new, w_new = H_new // self_patch_size, W_new // self_patch_size

            # Prepare channel information (per-patch)
            channel = torch.full((h_new, w_new), channel, dtype=torch.float16)

            spg, mean, std = normalize_spg(spg)
            mean = self_encode_mean(mean, win_size)
            std = self_encode_mean(std, win_size)

            spg_list.append(spg)
            chn_list.append(channel)
            mean_list.append(mean)
            std_list.append(std)

        win_batch = torch.stack(spg_list)
        win_channels = torch.stack(chn_list)
        win_means = torch.stack(mean_list)
        win_stds = torch.stack(std_list)

        win_batch.unsqueeze_(1)
        win_channels = win_channels.flatten(1)
        win_means = win_means.transpose(1, 2)
        win_stds = win_stds.transpose(1, 2)

        full_batch[win_size] = {
            "batch": win_batch,
            "channels": win_channels,
            "means": win_means,
            "stds": win_stds,
        }
        # print(f"[collate_fn] win_size={win_size}: {win_batch.shape}")

    # == Finished iterating over all possible window shifts

    return full_batch, target