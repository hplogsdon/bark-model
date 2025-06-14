# Bark Detection

Quick example using trained UrbanSound8K model via pytorch to detect content various audio samples, intended to
be used for dog barking detection (but can be used for any of the audio classifications within that dataset).

## Usage

Training is done in the console, not a notebook. A click cli has been written to perform basic tasks. This was
written using WSL2 on Win11. No idea if it'll run on native windows. Best not to hurt yourself though. Model was
trained on an Nvidia 4070Ti

1. Install Dependencies via uv:
  * `uv venv --python 3.13`
  * `source .venv/bin/activate`
  * `uv sync --all-extras`

2. Run training. model will be saved to models/ directory
  * ex: `uv run -- barking train --num-epochs 20 --learn-rate 0.002`

3. Run inference.
  * Can be done via the cli: `uv run -- barking infer --audio-file data/bark.wav --sample-rate 22050`
  * Or can be done in the included notebook.
