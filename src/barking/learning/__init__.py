from pathlib import Path

import librosa
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from barking.learning.ann import AudioClassifierANN
from barking.learning.cnn import CNNNetwork
from barking.learning.dataset import UrbanSound8KDataset

__all__ = ["AudioClassifierANN", "CNNNetwork", "UrbanSound8KDataset", "run_inference", "run_training"]


def run_training(model_path, audio_dir, annotations, num_epochs, learn_rate) -> None:
    train_dataset, val_dataset, test_dataset = create_datasets(audio_dir, annotations)

    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize best validation accuracy and model checkpoint
    best_val_acc = 0.0

    # Initialize model, loss function, and optimizer
    input_size = 40 + 32 + 128 + 1 + 1  # Features for MFCC, Chroma, Mel, ZCR, RMS (202)
    num_classes = len(train_dataset.label_encoder.classes_)  # Number of sound classes (10)

    model = load_model(model_path, input_size, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = _train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _test(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%"
        )

        # Save the model if it has better validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with Val Accuracy: {val_acc:.2f}%")

    # Load the best model for final evaluation
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Evaluate on the test set
    test_loss, test_acc = _test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")


def run_inference(model_path, audio_file, sample_rate, annotations, class_labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    waveform, sr = librosa.load(audio_file, sr=sample_rate)

    dataset = UrbanSound8KDataset(audio_dir="", annotations=annotations)
    features = dataset.extract_features(waveform)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    features_tensor = features_tensor.to(device)

    model = load_model(model_path, input_size=202, num_classes=10, device=device)

    with torch.no_grad():
        outputs = model(features_tensor)

    _, predicted_class = torch.max(outputs, 1)
    confidence = torch.softmax(outputs, dim=1)[0][predicted_class].item()

    idx = ([predicted_class.item()])[0]
    label_name = class_labels[int(idx)]

    return label_name, confidence


def _predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


def _train(model, train_loader, criterion, optimiser, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with tqdm(train_loader, desc="Training", unit="batch", ncols=100) as pbar:
        for batch in pbar:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            # Flatten features for input to the ANN
            features = features.view(features.size(0), -1)

            # Zero the parameter gradients
            optimiser.zero_grad()

            # forward pass
            output = model(features)

            # calculate loss
            loss = criterion(output, labels)
            running_loss += loss.item()

            # backpass and optimization
            loss.backward()
            optimiser.step()

            # calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total_predictions += predicted.size(0)
            correct_predictions += (predicted == labels).sum().item()

            avg_loss = running_loss / len(train_loader)
            accuracy = (correct_predictions / total_predictions) * 100
            pbar.set_postfix(loss=f"{avg_loss:0.4f}", accuracy=f"{accuracy:0.2f}")

    avg_loss = running_loss / len(train_loader)
    accuracy = (correct_predictions / total_predictions) * 100
    return avg_loss, accuracy


def _test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            features = features.view(features.size(0), -1)

            # forward pass
            outputs = model(features)

            # calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # calc accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += predicted.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions * 100
    return avg_loss, accuracy


def create_datasets(audio_dir, annotations, test_size=0.2, val_size=0.2):
    dataset = UrbanSound8KDataset(audio_dir, annotations)

    # Splitting the data into train, validation, and test sets
    train_metadata, temp_metadata = train_test_split(
        dataset.metadata, test_size=test_size + val_size, stratify=dataset.metadata["classID"]
    )
    val_metadata, test_metadata = train_test_split(
        temp_metadata, test_size=test_size / (test_size + val_size), stratify=temp_metadata["classID"]
    )

    # Create Dataset instances for train, validation, and test
    train_dataset = UrbanSound8KDataset(audio_dir=audio_dir, annotations=train_metadata)
    val_dataset = UrbanSound8KDataset(audio_dir=audio_dir, annotations=val_metadata)
    test_dataset = UrbanSound8KDataset(audio_dir=audio_dir, annotations=test_metadata)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_model(model_path: str, input_size, num_classes, device):
    """Load the trained model from the saved checkpoint."""
    model_path = Path(model_path)
    model = AudioClassifierANN(input_size=input_size, num_classes=num_classes)
    if model_path.exists() and model_path.is_file():
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
