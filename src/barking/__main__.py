"""Console script for barking."""

from pathlib import Path

import click
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

import barking.utils as util
from barking.cnn import CNNNetwork
from barking.dataset import UrbanSoundDataset


@click.group()
@click.version_option()
@click.option("-m", "--model", required=True, type=click.Path(exists=True))
@click.option("-b", "--batch-size", default=128, type=int)
@click.option("-e", "--epochs", default=10, type=int)
@click.pass_context
def cli(ctx, model, batch_size, epochs):
    if ctx.obj is None:
        ctx.obj = {}

    if not Path("data/UrbanSound8K").exists():
        click.echo("Downloading UrbanSound8K...")
        # remote_url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1"
        # Do download. I dont care to do it in code. just wget and untar

    annotations = "data/UrbanSound8K/metadata/UrbanSound8K.csv"
    audio_dir = "data/UrbanSound8K/audio"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}")
    cnn = CNNNetwork().to(device)

    state_dict = torch.load(model, map_location=device)
    cnn.load_state_dict(state_dict)

    mel_spect = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    ).to(device)

    ctx.obj["model_path"] = model
    ctx.obj["lr"] = 0.001
    ctx.obj["epochs"] = epochs
    ctx.obj["batch_size"] = batch_size
    ctx.obj["annotations"] = annotations
    ctx.obj["audio_dir"] = audio_dir
    ctx.obj["class_mapping"] = [
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
    ctx.obj["sample_rate"] = 22050
    ctx.obj["num_samples"] = 22050
    ctx.obj["cnn"] = cnn
    ctx.obj["mel_spect"] = mel_spect
    ctx.obj["device"] = device
    ctx.obj["dataset"] = UrbanSoundDataset(
        annotations,
        audio_dir,
        device,
        mel_spect,
        22050,
        22050,
    )


@cli.command()
@click.pass_context
def inference(ctx: click.Context):
    class_mapping = ctx.obj["class_mapping"]
    device = ctx.obj["device"]
    cnn = ctx.obj["cnn"]
    dataset = ctx.obj["dataset"]

    # get a sample from the dataset
    sample, target = dataset[0][0], dataset[0][1]
    # add extra dimension to input tensor
    sample = sample.unsqueeze(0).to(device)

    # make inference
    predicted, expected = util.predict(cnn, sample, target, class_mapping)
    # print the results
    click.echo(f"Predicted: '{predicted}', Expected: '{expected}'")


@cli.command()
@click.pass_context
def train(ctx: click.Context):
    learning_rate = ctx.obj["lr"]
    epochs = ctx.obj["epochs"]
    batch_size = ctx.obj["batch_size"]
    device = ctx.obj["device"]

    dataset = ctx.obj["dataset"]

    # Initiate Dataset from Urban Sound 8k
    train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cnn = ctx.obj["cnn"]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    util.train(cnn, train_data_loader, loss_fn, optimizer, device, epochs)

    torch.save(cnn.state_dict(), ctx.obj["model_path"])
    click.echo("Model saved")


if __name__ == "__main__":
    cli()
