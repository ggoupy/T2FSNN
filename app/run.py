import os
import sys
import argparse
import json
import random
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from augment import CIFAR10Policy, Cutout
from ttfsnn.snn import Fc, Conv, MaxPool
from ttfsnn.train import EventBP
from ttfsnn.utils import Logger, EarlyStopper

np.set_printoptions(suppress=True) # Remove scientific notation


# NOTE: Output layer is added later, based on the number of classes
ARCHITECTURES = {
    "vgg7": "64C3 - 128C3 - P2 - 256C3 - 256C3 - P2 - 512C3 - 512C3 - P2",
    "vgg11": "128C3 - 128C3 - 128C3 - P2 - 256C3 - 256C3 - 256C3 - P2 - 512C3 - 512C3 - 512C3 - 512C3 - P2",
}


# Parse a string describing the network architecture to a list of dictionaries
def parse_network_str(str):
    layers = []
    tokens = str.split(" - ")
    for token in tokens:
        if "C" in token:  # Convolutional layer
            parts = token.split("C")
            n_channels = int(parts[0])
            kernel_size = int(parts[1])
            layers.append({
                "layer": "conv",
                "n_channels": n_channels,
                "kernel_size": kernel_size,
                "stride_size": 1,
                "padding_size": kernel_size // 2
            })
        elif "P" in token:  # Pooling layer
            kernel_size = int(token[1:])
            layers.append({
                "layer": "maxpool",
                "kernel_size": kernel_size,
                "stride_size": kernel_size
            })
        elif "F" in token:  # Fully connected layer
            n_neurons = int(token.replace("F", ""))
            layers.append({
                "layer": "fc",
                "n_neurons": n_neurons
            })
    return layers


# For data loading with multiple workers
def worker_init_fn(worker_id):
    # Use torch.initial_seed to generate a worker-specific seed
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# Load dataset and split into train/val/test
def load_dataset(dataset, batch_size, load_n_workers=8):

    if dataset == "cifar100":
        input_shape = (3, 32, 32)
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32,padding=4,fill=128),
            torchvision.transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            torchvision.transforms.ToTensor(),
            Cutout(n_holes=1,length=16),
            torchvision.transforms.Normalize(
                (0.4914,    0.4822,   0.4465),
                (0.2023,    0.1994,   0.2010)),
        ])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914,    0.4822,    0.4465),
                (0.2023,    0.1994,    0.2010))
        ])
        train_dataset = torchvision.datasets.CIFAR100(root='../datasets/cifar100/', 
                                                      train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR100(root='../datasets/cifar100/', 
                                                    train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root='../datasets/cifar100/', 
                                                     train=False, download=True, transform=transform)
        n_classes = 100
            
    elif dataset == "cifar10":
        input_shape = (3, 32, 32)
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32,padding=4,fill=128),
            torchvision.transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            torchvision.transforms.ToTensor(),
            Cutout(n_holes=1,length=16),
            torchvision.transforms.Normalize(
                (0.4914,    0.4822,   0.4465),
                (0.2023,    0.1994,   0.2010)),
        ])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914,    0.4822,    0.4465),
                (0.2023,    0.1994,    0.2010))
        ]) 
        train_dataset = torchvision.datasets.CIFAR10(root='../datasets/cifar10/', 
                                                     train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(root='../datasets/cifar10/', 
                                                   train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='../datasets/cifar10/', 
                                                    train=False, download=True, transform=transform)
        n_classes = 10

    elif dataset == "fmnist":
        input_shape = (1, 28, 28)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = torchvision.datasets.FashionMNIST(root='../datasets/fmnist/', 
                                                          train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.FashionMNIST(root='../datasets/fmnist/', 
                                                        train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='../datasets/fmnist/', 
                                                         train=False, download=True, transform=transform)
        n_classes = 10

    elif dataset == "mnist":
        input_shape = (1, 28, 28)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='../datasets/mnist/', 
                                                   train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.MNIST(root='../datasets/mnist/', 
                                                 train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='../datasets/mnist/', 
                                                  train=False, download=True, transform=transform)
        n_classes = 10
    
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    
    # Split training dataset into training and validation datasets
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_indices = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=load_n_workers,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=load_n_workers,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=load_n_workers,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    return input_shape, n_classes, train_loader, val_loader, test_loader


# Class for training and inference of the SNN
class SNNtrainer:

    '''
    dataset: str {"cifar10", "cifar100", "mnist", "fmnist"}, see load_dataset
    network: str {"vgg7", "vgg11"}, see ARCHITECTURES
    config: dict
    device: torch.device
    output_dir: str
    '''
    def __init__(self, dataset, network, config, device, output_dir=None):
        
        # Training parameters
        self.device = device # GPU device to use
        self.epochs = config["epochs"]
        self.batch_size = config['batch_size'] 
        self.early_stopper = EarlyStopper(config.get("early_stopping", 25))
        self.logger = Logger(output_dir, output_dir is not None)
        
        # Get data loaders
        input_shape, n_classes, self.train_loader, self.val_loader, self.test_loader = load_dataset(dataset, self.batch_size)
        
        # Convert network string to a config dict
        try:
            archi_str = ARCHITECTURES.get(network, None)
            # Assume network has the format 'aCb - Pc - ... - dF"
            if archi_str is None: archi_str = network
            archi_str += f" - {n_classes}F" # Output layer
            config_network = parse_network_str(archi_str)
        except KeyError:
            raise ValueError(f"Error: Architecture '{network}' can not be parsed")
        self.logger.log((archi_str, config))
        
        # Create the network
        self.network = []
        previous_shape = input_shape
        for i,layer in enumerate(config_network):
            layername = layer.get("layer", "fc")
            
            if layername == "conv":
                conv = Conv(
                    input_shape=previous_shape,
                    n_channels=layer["n_channels"],
                    kernel_size=layer["kernel_size"],
                    stride_size=layer["stride_size"],
                    padding_size=layer["padding_size"],
                    t_win=layer.get("t_win", 1), # Length of the spike time window
                    w_ei=config.get("w_ei", False), # True to enable excitatory/inhibitory weights
                    w_init_mean=layer.get("w_init_mean", None),
                    w_init_std=layer.get("w_init_std", None),
                    device=device # GPU device
                )
                previous_shape = conv.output_shape
                self.network.append(conv)
                
            elif layername == "maxpool":
                maxpool = MaxPool(
                    input_shape=previous_shape,
                    kernel_size=layer["kernel_size"],
                    stride_size=layer.get("stride_size", 1)
                )
                previous_shape = maxpool.output_shape
                self.network.append(maxpool)
                
            else: # Fully-connected layer
                if i == len(config_network)-1: no_spk = True
                else: no_spk = False
                fc = Fc(
                    input_size=np.prod(previous_shape),
                    n_neurons=layer["n_neurons"],
                    t_win=layer.get("t_win", 1), # Length of the spike time window
                    w_ei=config.get("w_ei", False), # True to enable excitatory/inhibitory weights
                    w_init_mean=layer.get("w_init_mean", None),
                    w_init_std=layer.get("w_init_std", None),
                    no_spk=no_spk,
                    device=device # GPU device
                )
                previous_shape = fc.output_shape
                self.network.append(fc)
        
        # Create the trainer
        self.trainer = EventBP(
            network=self.network, # List of snn.SpikingLayer
            lr=config.get("lr", 1e-4), # Learning rate
            grad_clip=config.get('grad_clip', 1), # To avoid gradient explosion 
            w_decay=config.get('w_decay', 0), # > 0 for weight decay
            use_adam=config.get('use_adam', True), # True for Adam, False for Mini-Batch SGD
            annealing=config.get("annealing", 1), # Annealing on the learning rate (after every epoch)
        )


    def train(self):

        early_stop = False
        
        train_acc, val_acc, test_acc = 0, 0, 0
        
        for epoch in range(self.epochs):

            ####################################
            ############# TRAINING #############

            # Stats
            train_acc = 0
            layer_act = [0 for _ in range(len(self.network))]
            n_iters = len(self.train_loader.dataset)//self.batch_size  
            for x, y in tqdm(self.train_loader, total=n_iters, disable=not sys.stdout.isatty()):
                
                # Spike encoding (TTFS)
                x = x.to(self.device)
                y = y.to(self.device)
                x = (1 - x) * self.network[0].t_win
                x[x == self.network[0].t_win] = torch.inf
                
                outputs = []
                for i,layer in enumerate(self.network):
                    # Forward pass
                    x = layer(x)                    
                    # Save output for backward pass
                    outputs.append(x)
                    # Stats
                    layer_act[i] += (x != torch.inf).float().sum() / x.numel()

                # Prediction 
                # Based on membrane potential
                if self.network[-1].no_spk: predicted = x.argmax(1) 
                # Based on spike times
                else: predicted = x.argmin(1) 
                train_acc += torch.sum(predicted == y)
            
                # Backward pass
                self.trainer(outputs, y) 
            
            # Training logs
            train_acc = (train_acc / len(self.train_loader.dataset)).item()
            for i, layer in enumerate(self.network):
                self.logger.log(f"=== Layer {i} ===")
                if layer.trainable:
                    self.logger.log(f"Mean weights: {layer.weights.mean()} +- {layer.weights.std()} (min:{layer.weights.min()} ; max:{layer.weights.max()})")
                self.logger.log(f"Mean activity: {layer_act[i]/n_iters}")
            self.logger.log(f"Accuracy on training set after epoch {epoch}: {round(train_acc,4)}")

            # Annealing on the learning rates
            self.trainer.anneal()
            

            ####################################
            ############ VALIDATION ############

            val_acc = 0
            for x, y in self.val_loader:
                
                # Spike encoding (TTFS)
                x = x.to(self.device)
                y = y.to(self.device)
                x = (1 - x) * self.network[0].t_win
                x[x == self.network[0].t_win] = torch.inf
                
                for layer in self.network:
                    # Forward pass
                    x = layer(x)
                        
                # Prediction 
                # Based on membrane potential
                if self.network[-1].no_spk: predicted = x.argmax(1) 
                # Based on spike times
                else: predicted = x.argmin(1) 
                val_acc += torch.sum(predicted == y)
            
            # Validation stats
            val_acc = (val_acc / len(self.val_loader.dataset)).item()
            self.logger.log(f"Accuracy on validation set after epoch {epoch}: {round(val_acc,4)}")
            
            # Early stopping
            early_stop = self.early_stopper.early_stop(val_acc)
            if early_stop: 
                self.logger.log(f"[LOG] Early stopping triggered (max:{self.early_stopper.max_acc})")
                break
            
            
            ####################################
            ############### TEST ###############

            test_acc = 0
            for x, y in self.test_loader:
                
                # Spike encoding (TTFS)
                x = x.to(self.device)
                y = y.to(self.device)
                x = (1 - x) * self.network[0].t_win
                x[x == self.network[0].t_win] = torch.inf
                
                for layer in self.network:
                    # Forward pass
                    x = layer(x)
                        
                # Prediction 
                # Based on membrane potential
                if self.network[-1].no_spk: predicted = x.argmax(1) 
                # Based on spike times
                else: predicted = x.argmin(1) 
                test_acc += torch.sum(predicted == y)
            
            # Test stats
            test_acc = (test_acc / len(self.test_loader.dataset)).item()
            self.logger.log(f"Accuracy on test set after epoch {epoch}: {round(test_acc,4)}")
            

        return train_acc, val_acc, test_acc



###################################################################################################
###################################################################################################

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, default="cifar10", help="Name of the dataset")
    parser.add_argument("network", type=str, default="vgg11", help="Architecture of the network")
    parser.add_argument("config", type=str, help="Path to a JSON config for hyperparameters")
    parser.add_argument("--output", type=str, default=None, help="Output data directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU")
    args = parser.parse_args()

    # Seeding
    random.seed(args.seed)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # GPU device
    device = torch.device(f"cuda:{args.gpu_id}")
    
    # Read the JSON config
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Output dir
    if args.output is not None: os.makedirs(args.output, exist_ok=True)
    
    # Run
    st = SNNtrainer(args.dataset, args.network, config, device, args.output)
    st.train()