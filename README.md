# thinkingsquishy

![workflow](https://github.com/Alex-Lee-chang/thinkingsquishy/actions/workflows/ci.yml/badge.svg)

A reimplementation of DittoGym for use in soft body deformation robotics.

**Current work in progress!**

## Instructions
1. Clone the repo.
2. Run `conda create -n <env name> python=3.11.9` with the desired environment name and activate it.
3. Run `pip install -e “.[develop]”`.
4. Install the correct version of PyTorch.

## Basic testing
`python train.py --env_name RUN-coarse-v0 --config_file_path  ./models/RUN/coarse.json --visualize True --gui True`