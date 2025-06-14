from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class UrbanSoundDataset(Dataset):
    def __init__(self, annotations, audio_dir, device, transform, target_sample_rate, num_samples, num_channels=1):
        self.annotations = pd.read_csv(annotations)
        self.audio_dir = Path(audio_dir)
        self.device = device
        self.transformation = transform
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.num_channels = num_channels

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        get index of item
        ex: my_list[1] -> my_list.__getitem__(1)
        :param idx:
        :return:
        """
        audio_sample_path = self._audio_sample_path(idx)
        label = self._get_audio_sample_label(idx)
        # Load in the audio file as signal
        signal, sr = torchaudio.load(audio_sample_path)
        # Register signal to the device
        signal = signal.to(self.device)
        # Resample the signal if needed
        signal = self._resample_if_needed(signal, sr)
        # Transform the signal to Mono for Spectrogram if needed
        signal = self._mono_if_needed(signal)
        # Cut the signal length if necessary (only handle signals with length >= num_samples)
        signal = self._cut_if_needed(signal)
        # Add padding when needed
        signal = self._add_padding_if_needed(signal)
        # Pass the signal to the transformation (mel spectrogram)
        signal = self.transformation(signal)

        return signal, label

    def _resample_if_needed(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mono_if_needed(self, signal):
        if signal.shape[0] > self.num_channels:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_needed(self, signal):
        # signal -> Tensor -> (1, num_samples) -> (1, 50000) -> (1, 22050) # First 22050 samples of audio
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _add_padding_if_needed(self, signal):
        signal_length = signal.shape[1]
        if signal_length < self.num_samples:
            num_missing = self.num_samples - signal_length
            last_dim_padding = (0, num_missing)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _audio_sample_path(self, idx):
        # indexes below refer to the columns in the annotations file (.csv)
        fold = f"fold{self.annotations.iloc[idx, 5]}"
        path = self.audio_dir / fold / self.annotations.iloc[idx, 0]
        return path

    def _get_audio_sample_label(self, idx):
        return self.annotations.iloc[idx, 6]
