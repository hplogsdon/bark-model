"""Console script for barking."""

from pathlib import Path

import click
import librosa
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from barking.ann import AudioClassifierANN
from barking.dataset import UrbanSound8KDataset


def predict(model, input, target, class_mapping):
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
            pbar.set_postfix(loss=avg_loss, accuracy=accuracy)

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


@click.group()
@click.version_option()
@click.pass_context
def cli(ctx):
    if ctx.obj is None:
        ctx.obj = {}

    if not Path("data/UrbanSound8K").exists():
        click.echo("Downloading UrbanSound8K...")
        # remote_url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1"
        # Do download. I dont care to do it in code. just wget and untar


@cli.command()
@click.option("-e", "--num-epochs", default=10, help="Number of epochs to train.")
@click.option("-l", "--learn-rate", default=0.001, help="Learning rate.")
@click.pass_context
def train(ctx: click.Context, num_epochs, learn_rate):
    annotations = "data/UrbanSound8K/metadata/UrbanSound8K.csv"
    audio_dir = "data/UrbanSound8K/audio"

    train_dataset, val_dataset, test_dataset = create_datasets(audio_dir, annotations)

    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize best validation accuracy and model checkpoint
    best_val_acc = 0.0
    best_model_path = "models/best_model.pth"  # Path where the best model will be saved

    # Initialize model, loss function, and optimizer
    input_size = 40 + 32 + 128 + 1 + 1  # Features for MFCC, Chroma, Mel, ZCR, RMS
    num_classes = len(train_dataset.label_encoder.classes_)  # Number of sound classes

    model = load_model(best_model_path, input_size, num_classes, device)
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
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with Val Accuracy: {val_acc:.2f}%")

    # Load the best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    # Evaluate on the test set
    test_loss, test_acc = _test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")


@cli.command()
@click.option("-f", "--audio-file", help="Audio file to process.")
@click.option("-sr", "--sample-rate", type=int, help="Sample rate.")
@click.pass_context
def infer(ctx: click.Context, audio_file, sample_rate):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    waveform, sr = librosa.load(audio_file, sr=sample_rate)

    dataset = UrbanSound8KDataset(audio_dir="", annotations="data/UrbanSound8K/metadata/UrbanSound8K.csv")
    features = dataset.extract_features(waveform)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    features_tensor = features_tensor.to(device)

    model_path = "models/best_model.pth"
    model = load_model(model_path, input_size=202, num_classes=10, device=device)

    with torch.no_grad():
        outputs = model(features_tensor)

    _, predicted_class = torch.max(outputs, 1)
    confidence = torch.softmax(outputs, dim=1)[0][predicted_class].item()

    labels = [
        "air_conditioner",
        "car_horn",
        "children_playing",
        "dog_bark",
        "drilling",
        "engine_idling",
        "gun_shot",
        "jackhammer",
        "siren",
        "street_music",
    ]
    id = ([predicted_class.item()])[0]
    label_name = labels[int(id)]

    print(label_name, confidence)


if __name__ == "__main__":
    cli()
