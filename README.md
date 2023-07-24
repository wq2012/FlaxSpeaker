# FlaxSpeaker

 [![Python application](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml/badge.svg)](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml) [![PyPI Version](https://img.shields.io/pypi/v/flaxspeaker.svg)](https://pypi.python.org/pypi/flaxspeaker) [![Python Versions](https://img.shields.io/pypi/pyversions/flaxspeaker.svg)](https://pypi.org/project/flaxspeaker) [![Downloads](https://pepy.tech/badge/flaxspeaker)](https://pepy.tech/project/flaxspeaker)


## Overview

 A simple speaker recognition library in [JAX](https://jax.readthedocs.io) and [Flax](https://flax.readthedocs.io).

 For the PyTorch version, see: [SpeakerRecognitionFromScratch](https://github.com/wq2012/SpeakerRecognitionFromScratch)

## Installation

```
pip install flaxspeaker
```

## Tutorial

### Experiment config

All your experiment configurations are represented in a single YAML file.

See [myconfig.yml](myconfig.yml) as an example.

### Hook up with data

In the configuration file, you need to correctly hook it up with your
downloaded dataset.

For example, if you have downloaded LibriSpeech on your own machine, you need
to set these two fields correctly:

```
data:
  train_librispeech_dir: "YOUR LIBRISPEECH TRAINING SET PATH"
  test_librispeech_dir: "YOUR LIBRISPEECH TESTING SET PATH"
```

If you are using a different dataset than LibriSpeech, you need to represent
your dataset as CSV files. Then set these two fields correctly:

```
data:
  train_csv: "YOUR TRAINING SET CSV"
  test_csv: "YOUR TESTING SET CSV"
```

### Generate dataset CSV

To represent your downloaded datasets by CSV files, you can use the
`generate_csv` mode for the `flaxspeaker` command.

For example, you can use a command like below to
generate a CSV file `CN-Celeb.csv` to represent your downloaded CN-Celeb
dataset located at `"${HOME}/Downloads/CN-Celeb_flac/data"`:

```
python -m flaxspeaker \
--mode generate_csv \
--path_to_dataset "${HOME}/Downloads/CN-Celeb_flac/data" \
--audio_format ".flac" \
--speaker_label_index -2 \
--output_csv "CN-Celeb.csv"
```

You can use `--help` to understand the meaning of each flag:

```
python -m flaxspeaker --help
```

### Training

Once you have the config file ready (e.g. `your_config.yml`), you can launch
your training with this command:

```
python -m flaxspeaker --mode train --config your_config.yml
```

### Evaluation

After you finished training, you can evaluate the Equal Error Rate (EER) of
the model you just trained with:

```
python -m flaxspeaker --mode eval --config your_config.yml
```
