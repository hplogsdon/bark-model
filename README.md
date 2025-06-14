# Bark Detection

Quick example using trained UrbanSound8K model via pytorch to detect content various audio samples, intended to
be used for dog barking detection (but can be used for any of the audio classifications within that dataset).

## Usage

Training is done in the console, not a notebook. A click cli has been provided to perform basic tasks. This was
written in WSL2 on Win11. No idea if it'll run on native windows. Best not to hurt yourself though. Model was
trained on an Nvidia 4070Ti

1. Install Dependencies via uv:
  * `uv venv --python 3.13`
  * `source .venv/bin/activate`
  * `uv sync --all-extras`

2. Run training. model will be saved to models/ directory
  * This is on the CLI only.
    * `uv run -- barking train --num-epochs 20 --learn-rate 0.002`
    * Parameters are:
      * `--model-path`: Path to the model file. Defaults to `models/UrbanSound8K.pth`
      * `--num-epochs`: Number of epoch iterations. Defaults to 10
      * `--learn-rate`: Learning rate. Defaults to 0.001

3. Run inference.
  * Can be done via the cli:
    * ex: `uv run -- barking infer --audio-file data/bark.wav --sample-rate 22050`
    * Parameters are
      * `--model-path`: Path to the model file. Defaults to `models/UrbanSound8K.pth`
      * `--audio-file`: Audio file that we want to run inference on. Needs to be a Wav file.
      * `--sample-rate`: Sample rate (bitrate) of the audio file. Defaults to 22050.

  * Or can be done in the included notebook. Instructions are included.
