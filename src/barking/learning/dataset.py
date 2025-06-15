from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class UrbanSound8KDataset(Dataset):
    metadata: pd.DataFrame
    audio_dir: Path

    def __init__(self, audio_dir, annotations: str | pd.DataFrame = None, sample_rate=22050, transform=None):
        """
        Args:
            audio_dir (str): Path to the directory containing audio files.
            annotations (str): Path to the CSV file containing metadata (with fsID, classID, etc.).
            sample_rate (int): The sample rate to resample audio to (default is 22050).
            transform (callable, optional): A function/transform to apply to the audio (e.g., MFCC).
        """
        self.audio_dir = Path(str(audio_dir))
        if isinstance(annotations, str):
            self.metadata = pd.read_csv(annotations)
        else:
            self.metadata = annotations
        self.sample_rate = sample_rate
        self.transform = transform

        # Label encoding for classID to numerical labels
        self.label_encoder = LabelEncoder()
        self.metadata["classID"] = self.label_encoder.fit_transform(self.metadata["class"])

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.metadata)

    def extract_features(self, X):
        result = np.array([])

        # MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=self.sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))

        # Chroma_STFT
        stft = np.abs(librosa.stft(X))
        chroma = np.mean(
            librosa.feature.chroma_stft(S=stft, sr=self.sample_rate, n_chroma=32, window="hamming", n_fft=1024).T,
            axis=0,
        )
        result = np.hstack((result, chroma))

        # Mel Spectrogram
        mel = np.mean(
            librosa.feature.melspectrogram(
                y=X, sr=self.sample_rate, n_mels=128, fmax=8000, n_fft=1024, hop_length=512, window="hamming"
            ).T,
            axis=0,
        )
        result = np.hstack((result, mel))

        # Zero Crossing Rate
        Z = np.mean(librosa.feature.zero_crossing_rate(y=X), axis=1)
        result = np.hstack((result, Z))

        # Root Mean Square Energy
        rms = np.mean(librosa.feature.rms(y=X).T, axis=0)
        result = np.hstack((result, rms))

        return result

    def __getitem__(self, idx):
        """Return the sample (audio, label, metadata) at index `idx`."""
        # Get the metadata for the current sample
        row = self.metadata.iloc[idx]
        start_time = row["start"]
        end_time = row["end"]
        fold = row["fold"]
        file_name = row["slice_file_name"]
        label = row["classID"]

        # Load the audio file using librosa
        audio_path = self.audio_dir / f"fold{fold}" / file_name
        waveform, sample_rate = librosa.load(audio_path, sr=self.sample_rate)

        # Resample if the sample rate does not match the desired rate
        if sample_rate != self.sample_rate:
            waveform = librosa.resample(waveform, sample_rate, self.sample_rate)

        # Extract features
        features = self.extract_features(waveform)

        # Convert features to tensor and label to long
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)  # Ensure label is of type long

        sample = {
            "features": features_tensor,
            "start": start_time,
            "end": end_time,
            "fold": fold,
            "file_name": file_name,
            "label": label_tensor,
        }

        return sample
