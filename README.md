# ML Experiments

This repository contains many small ML experiments that mess around with model
architecture. The main focus lies on text processing and all (or most) training
experiments are done on text sequence-to-sequence tasks.

## Running

The code currently only correctly works under `linux`, due to limiations coming
from `jax`. To run it, simply run `run_ml_py.sh`, which will open up a selection
menu.

## Structure

The code is structured around the concept of an `experiment`, which is defined
as an independent program performing some sort of test. The experiments are
defined in their own files and registered in `experiments/__init__.py`. The rest
of the code is a minimal framework to make creating new experiments easier.
