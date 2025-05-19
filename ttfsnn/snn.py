import torch

"""
Implementation of the SNN described in:
    Wei et al. Temporal-Coded Spiking Neural Networks with Dynamic Firing Threshold. ICCV 2023.
"""


# Base spiking layer with Rel-PSP neurons and Dynamic Firing thresholds (DFT)
class SpikingLayer:

    def __init__(self, trainable, output_shape, shape, w_init_mean, w_init_std, w_clip, w_ei, w_min, w_max, t_win, device):
        
        self.trainable = trainable # True if the layer is trainable (eg. fully-connected, convolutional), False otherwise (eg. pooling)
        self.output_shape = output_shape # Output shape of the layer
        self.shape = shape # Shape of the weights (N, M, ..., K), where N is the number of output neurons (Fc) or channels (Conv)
        self.w_clip = w_clip # True for weight clipping
        self.w_min = w_min # Min weight value with weight clipping
        self.w_max = w_max # Max weight value with weight clipping
        self.t_win = t_win # Length of the time window where neurons can fire
        self.train_cache = {} # Cache for training -- to init in subclasses
        
        # Compute spike time window of the layer
        # Layers have non-overlapping spike time windows
        # Neurons in layer l can not fire at the same time as neurons in layer l-1
        # The spike time window of neurons in layer l is [t_win*l, t_win*(l+1)]
        # For simplicity of implementation, we always consider the current layer as layer 1 and the input layer as layer 0
        layer_ind = 1 # Just to clarify computation
        self.beg_spk_time = self.t_win * layer_ind # Begin allowed spike time
        self.end_spk_time = self.t_win * (layer_ind + 1) # End allowed spike time

        # Weights initialization with a normal distribution
        if w_init_mean is None: # Kaiming initialization
            w_init_mean = 0
        if w_init_std is None: # Kaiming initialization
            w_init_std = torch.sqrt(2 / torch.prod(torch.tensor(self.shape[1:], dtype=torch.float32)))
        self.weights = torch.normal(
            mean=w_init_mean, std=w_init_std,
            size=self.shape).to(device)
        
        # Mask to fix excitatory (positive) & inhibitory (negative) weights
        self.w_ei = self.weights >= 0 if w_ei else None

        # Clip weights if needed
        self.clip_weights()


    # Clip weights in the range [w_min, w_max]
    # Additionally, weights can have a fixed sign if w_ei is used
    def clip_weights(self):
        if self.w_ei is not None:
            self.weights[self.w_ei] = torch.maximum(self.weights[self.w_ei], torch.tensor(0.))
            self.weights[~self.w_ei] = torch.minimum(self.weights[~self.w_ei], torch.tensor(0.))
        if self.w_clip:
            self.weights = torch.clamp(self.weights, self.w_min, self.w_max)


    # Save forward computations for training
    # NOTE: values must be passed in the exact same order as they are defined in train_cache 
    def _save_for_training(self, *values):
        if len(values) != len(self.train_cache):
            raise ValueError(f"Expected {len(self.train_cache)} values, but got {len(values)}.")
        for key, value in zip(self.train_cache.keys(), values):
            self.train_cache[key] = value
        
    
    # Compute spike times according to Rel-PSP model
    def compute_spike_times(self, weighted_input, valid_w_sum):
        out_spks = (self.end_spk_time + weighted_input) / (1 + valid_w_sum) # Compute spike time
        out_spks[out_spks > self.end_spk_time] = torch.inf # Outside allowed spike time window
        out_spks[out_spks < 0] = torch.inf # Outside allowed spike time window
        out_spks[valid_w_sum >= self.beg_spk_time + weighted_input] = self.beg_spk_time # Before allowed spike time window
        out_spks = out_spks - self.beg_spk_time # To scale in [0, t_win]
        out_spks[out_spks < 0] = torch.inf # Outside allowed spike time window
        return out_spks
        
        
    # Forward pass
    # Must be define in sub classes
    # Sample must contain value in [0,t_win] or torch.inf (no spike)
    def __call__(self, sample):
        pass



# Fully-connected spiking layer
class Fc(SpikingLayer):

    def __init__(self, input_size, n_neurons, no_spk=False, w_init_mean=None, w_init_std=None, 
                 w_clip=False, w_ei=False, w_min=-1, w_max=1, t_win=1, device=torch.device("cuda")):
        
        super().__init__(True, (n_neurons,), (n_neurons, input_size), 
                         w_init_mean, w_init_std, w_clip, w_ei, w_min, w_max, t_win, device)
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.no_spk = no_spk # True to prevent neurons from firing, return accumulated membrane potentials instead
        self.train_cache = {"in_spk_mask": None, "valid_in_spk": None, "valid_w_sum": None, "weighted_input": None}


    def __call__(self, sample):
        # Flatten
        sample = sample.view(sample.shape[0], -1)
        # Identify input neurons that fired
        in_spk_mask = sample != torch.inf # (batch, n_in)
        # Sum of weights of neurons that fired
        valid_w_sum = torch.nn.functional.linear(in_spk_mask.float(), self.weights)
        # Compute accumulated membrane potentials at the end of the spike time window
        if self.no_spk:
            valid_in_spk = self.t_win - sample
            valid_in_spk[valid_in_spk == -torch.inf] = 0.
            weighted_input = torch.nn.functional.linear(valid_in_spk, self.weights)
            mem_pots = valid_w_sum + weighted_input # Membrane potential at end_spk_time
            # Cache for backward pass
            self._save_for_training(in_spk_mask, valid_in_spk, valid_w_sum, weighted_input)
            return mem_pots
        # Compute spike times 
        else:
            # Contribution of input neurons to membrane potentials
            valid_in_spk = sample.clone()
            valid_in_spk[valid_in_spk == torch.inf] = 0
            weighted_input = torch.nn.functional.linear(valid_in_spk, self.weights)
            # Compute output spikes based on closed-form response
            out_spks = self.compute_spike_times(weighted_input, valid_w_sum)
            # Cache for backward pass
            self._save_for_training(in_spk_mask, valid_in_spk, valid_w_sum, weighted_input)
            return out_spks


    
# Convolutional spiking layer
class Conv(SpikingLayer):

    def __init__(self, input_shape, n_channels, kernel_size=3, stride_size=1, padding_size=0, 
                 w_init_mean=None, w_init_std=None, w_clip=False, w_ei=False, w_min=-1, w_max=1, t_win=1, device=torch.device("cuda")):
        
        # Conv parameters
        self.n_c = n_channels
        self.k_h = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.k_w = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size
        self.s_h = stride_size[0] if isinstance(stride_size, tuple) else stride_size
        self.s_w = stride_size[1] if isinstance(stride_size, tuple) else stride_size
        self.p_h = padding_size[0] if isinstance(padding_size, tuple) else padding_size
        self.p_w = padding_size[1] if isinstance(padding_size, tuple) else padding_size
        self.input_shape = input_shape # Shape (n_c, n_h, n_w)
        in_c, in_height, in_width = input_shape[0], input_shape[1], input_shape[2] 
        w_shape = (self.n_c, in_c, self.k_h, self.k_w)
        
        # Compute output shape
        out_h = int(((in_height + 2 * self.p_h - self.k_h) / self.s_h) + 1)
        out_w = int(((in_width + 2 * self.p_w - self.k_w) / self.s_w) + 1)
        output_shape = (self.n_c, out_h, out_w)
        
        # Init base layer
        super().__init__(True, output_shape, w_shape, w_init_mean, w_init_std, w_clip, w_ei, w_min, w_max, t_win, device)
        self.train_cache = {"in_spk_mask": None, "valid_in_spk": None, "valid_w_sum": None, "weighted_input": None}


    def __call__(self, sample):
        # Identify input neurons that fired
        in_spk_mask = sample != torch.inf # Shape (batch, n_in_c, n_in_h, n_in_w) 
        # Sum of weights of neurons that fired
        valid_w_sum = torch.nn.functional.conv2d(
            in_spk_mask.float(), self.weights, 
            stride=(self.s_h, self.s_w), padding=(self.p_h, self.p_w)
        )
        # Contribution of input neurons to membrane potentials
        valid_in_spk = sample.clone()
        valid_in_spk[valid_in_spk == torch.inf] = 0
        weighted_input = torch.nn.functional.conv2d(
            valid_in_spk, self.weights, 
            stride=(self.s_h, self.s_w), padding=(self.p_h, self.p_w)
        )
        # Compute output spikes based on closed-form response
        out_spks = self.compute_spike_times(weighted_input, valid_w_sum)
        # Cache for backward pass
        self._save_for_training(in_spk_mask, valid_in_spk, valid_w_sum, weighted_input)
        return out_spks



# Max-pooling spiking layer
# TODO: to improve -- inherits from unused arguments in SpikingLayer 
class MaxPool(SpikingLayer):
    
    def __init__(self, input_shape, kernel_size=2, stride_size=2):

        # Pooling layers have no weights so they are not trainable
        self.trainable = False
        self.weights = None
        
        # Pooling parameters
        self.k_h = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.k_w = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size
        self.s_h = stride_size[0] if isinstance(stride_size, tuple) else stride_size if stride_size else kernel_size
        self.s_w = stride_size[1] if isinstance(stride_size, tuple) else stride_size if stride_size else kernel_size
        self.p_h = 0 # No padding in this implementation
        self.p_w = 0 # No padding in this implementation
        self.input_shape = input_shape
        
        # Compute output shape
        in_c, in_h, in_w = input_shape
        out_h = int(((in_h + 2 * self.p_h - self.k_h) / self.s_h) + 1)
        out_w = int(((in_w + 2 * self.p_w - self.k_w) / self.s_w) + 1)
        self.n_c = in_c
        self.output_shape = (in_c, out_h, out_w)
        
        self.train_cache = {"pool_idx": None}


    def __call__(self, sample):
        # Sample is a batch of spike times (batch_size, n_c, n_h, n_w)
        # We use negated spike times to apply first-spike-based pooling
        pooled_sample, pool_idx = torch.nn.functional.max_pool2d(-sample, 
            kernel_size=(self.k_h, self.k_w), stride=(self.s_h, self.s_w), return_indices=True
        )
        # Convert back to positive spike times
        pooled_sample = -pooled_sample 
        # Cache for backward pass
        self._save_for_training(pool_idx)
        return pooled_sample