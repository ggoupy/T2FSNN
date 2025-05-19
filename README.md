# T2FSNN

PyTorch-based framework for **ultra-fast training** of Time-to-First-Spike (TTFS)-based deep Spiking Neural Networks (SNNs) with event-driven BP. 

This framework uses an analytical expression for spike times, enabling GPU-based training of the SNN in a manner analogous to an ANN, **without requiring simulation over multiple time steps**. It is limited to a single SNN model with state-of-the-art performance, described in [1]. The SNN model features TTFS coding, single-spike Rel-PSP neurons, non-overlapping spike time windows, and event-driven BP algorithm.

**Note:** PyTorch is only used for low-level operations. All computations, including the forward pass and gradient calculations, are implemented manually for educational purposes.

## Getting started

The code is written with Python3 (3.9.2) and **runs exclusively on GPUs**.

### Requirements

- numpy (1.19.5)
- tqdm (4.66.4)
- torch (2.0.1)
- torchvision (0.15.2)
- setuptools (45.2.0)

### Install
  
The `T2FSNN` package and its dependencies can be installed with:
```
python3 -m pip install -e .
```

This package provides all the core classes and functions (see `ttfsnn/`). 


## Usage

The `SNNTrainer` class (`app/run.py`) provides an example of how to build and train a flexible SNN for classification using the `T2FSNN` package.  

```
python3 app/run.py <dataset> <network> /config/file [--output output/dir/] [--seed 0] [--gpu_id 0]
```

- `<dataset>`: `{"mnist", fmnist", "cifar10", "cifar100"}` (see `load_dataset` function in `run.py`)
- `<network>`: `{"vgg7", "vgg11"}` (see `ARCHITECTURES` variable in `run.py`)
- Configuration files use the JSON format (see examples in `config/`)


## Limitations

- Limited to a single SNN model
- Requires a GPU; training on CPU would take too much time
- Supports only the following types of layers: convolutional, max-pooling, fully-connected


## Acknowledgments

This implementation is a simplified version of my following code: https://gitlab.univ-lille.fr/fox/fbp.


## References 

[1] Wei et al. Temporal-Coded Spiking Neural Networks with Dynamic Firing Threshold, ICCV 2023.
