"""Console script for barking."""

from pathlib import Path

import click

from barking.learning import run_inference, run_training
from barking.service import app as bark_app


@click.group()
@click.version_option()
@click.pass_context
def cli(ctx):
    if ctx.obj is None:
        ctx.obj = {}

    if not Path("data/UrbanSound8K").exists():
        # remote_url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1"
        # Do download. I dont care to do it in code. just wget and untar
        raise ValueError("UrbanSound8K dataset not found.")

    ctx.obj["annotations"] = "data/UrbanSound8K/metadata/UrbanSound8K.csv"
    ctx.obj["audio_dir"] = "data/UrbanSound8K/audio"
    ctx.obj["labels"] = {
        0: "air_conditioner",
        1: "car_horn",
        2: "children_playing",
        3: "dog_bark",
        4: "drilling",
        5: "engine_idling",
        6: "gun_shot",
        7: "jackhammer",
        8: "siren",
        9: "street_music",
    }


@cli.command()
@click.option("-m", "--model-path", default="models/UrbanSound8K.pth", help="Path to model to use")
@click.option("-e", "--num-epochs", default=10, help="Number of epochs to train.")
@click.option("-l", "--learn-rate", default=0.001, help="Learning rate.")
@click.pass_context
def train(ctx: click.Context, model_path, num_epochs, learn_rate):
    """Run Training on the given model

    Args:
        ctx (click.Context): Click Context object.
        model_path (str): Path to the saved model.
        num_epochs (int): Number of epochs to train.
        learn_rate (float): Learning rate.
    """
    annotations = ctx.obj["annotations"]
    audio_dir = ctx.obj["audio_dir"]
    run_training(model_path, annotations, audio_dir, num_epochs, learn_rate)


@cli.command()
@click.option("-m", "--model-path", default="models/UrbanSound8K.pth", help="Path to model to use")
@click.option("-f", "--audio-file", help="Audio file to process.")
@click.option("-r", "--sample-rate", type=int, default=22050, help="Sample rate.")
@click.pass_context
def inference(ctx: click.Context, model_path, audio_file, sample_rate):
    """Run inference on a single audio file with the given model.

    Args:
        ctx: Click context object.
        model_path: Path to model to use.
        audio_file: Audio file to process.
        sample_rate: Sample rate.
    """
    inferred_class, confidence_score = run_inference(
        model_path, audio_file, sample_rate, annotations=ctx.obj["annotations"], class_labels=ctx.obj["labels"]
    )
    print(f"Identified class: {inferred_class}, confidence: {confidence_score}")


@cli.command()
@click.option("-m", "--model-path", default="models/UrbanSound8K.pth", help="Path to model to use")
@click.option("-f", "--audio-file", help="Audio file to process.")
@click.option("-r", "--sample-rate", type=int, default=22050, help="Sample rate.")
@click.pass_context
def run(ctx: click.Context, model_path):
    """Run the application.

    Args:
        ctx (click.Context): Click Context object.
        model_path: Path to model to use.
    """
    svc = bark_app


if __name__ == "__main__":
    cli()
