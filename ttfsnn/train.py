import torch
from ttfsnn.snn import Fc, Conv, MaxPool

"""
Based on the event-driven BP algorithm described in:
    Wei et al. Temporal-Coded Spiking Neural Networks with Dynamic Firing Threshold. ICCV 2023.
"""


# Trainer with event-driven backpropagation
class EventBP:
    
    def __init__(self, network, lr=1e-4, grad_clip=1, w_decay=0, annealing=1,
                 use_adam=True, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8):

        self.network = network # List of SpikingLayer
        self.lr = lr # learning rate
        self.annealing = annealing # Epoch-wise annealing
        self.w_decay = w_decay # L2 regularizattion, should be > 0
        self.grad_clip = grad_clip # To avoid gradient explosion on custom gradients 
                
        # Adam optimizer
        # True to use Adam optimizer, False for mini-batch SGD
        self.use_adam = use_adam
        self.adam_beta1 = adam_beta1 
        self.adam_beta2 = adam_beta2 
        self.adam_epsilon = adam_epsilon 
        if use_adam: # First moment vector
            self.adam_m = [torch.zeros_like(layer.weights) if layer.trainable else None for layer in network] 
        else: self.adam_m = None
        if use_adam: # Second moment vector
            self.adam_v = [torch.zeros_like(layer.weights) if layer.trainable else None for layer in network] 
        else: self.adam_v = None
        self.it_cnt = 0 # Iteration counter
            
        
    def anneal(self):
        self.lr = self.lr * self.annealing
        
    
    def _apply_weight_change(self, layer_ind, dW):
        # Adam optimization
        if self.use_adam:
            self.adam_m[layer_ind] = self.adam_beta1 * self.adam_m[layer_ind] + (1 - self.adam_beta1) * dW
            self.adam_v[layer_ind] = self.adam_beta2 * self.adam_v[layer_ind] + (1 - self.adam_beta2) * (dW ** 2)
            m_hat = self.adam_m[layer_ind] / (1 - self.adam_beta1 ** (self.it_cnt)) # Assume it_cnt starts at 1
            v_hat = self.adam_v[layer_ind] / (1 - self.adam_beta2 ** (self.it_cnt)) # Assume it_cnt starts at 1
            dW = m_hat / (torch.sqrt(v_hat) + self.adam_epsilon)
        # Update weights
        self.network[layer_ind].weights -= self.lr * dW 
        # L2 regularization
        if self.w_decay > 0:
            self.network[layer_ind].weights -= self.lr * self.w_decay * self.network[layer_ind].weights
        # Clip weights if desired
        self.network[layer_ind].clip_weights()
            

    def _backward_fc_no_spk(self, layer_ind, output, dY): # gradients computed based on membrane potentials
        # Forward cache
        in_spk_mask = self.network[layer_ind].train_cache["in_spk_mask"]
        valid_in_spk = self.network[layer_ind].train_cache["valid_in_spk"]

        # Gradient wrt. weights
        dW = ((valid_in_spk[:, :, None] + in_spk_mask[:, :, None]) * dY[:, None, :]).sum(axis=0).T
        
        # Gradient wrt. inputs
        dX = torch.sum(
            (valid_in_spk > 0)[:, :, None] # (batch, n_in, 1)
            * dY[:, None, :] # (batch, 1, n_out)
            * (self.network[layer_ind].weights.T[None, :, :]), # (1, n_in, n_out)
        axis=2) # (batch_size, n_in)
        
        return dX, dW

    
    def _backward_fc(self, layer_ind, output, dY): # gradients computed based on spike times
        # Forward cache
        valid_weight_sum = self.network[layer_ind].train_cache["valid_w_sum"]
        I = self.network[layer_ind].train_cache["weighted_input"]
        in_spk_mask = self.network[layer_ind].train_cache["in_spk_mask"]
        valid_in_spk = self.network[layer_ind].train_cache["valid_in_spk"]
        
        # Spike mask
        t_win = self.network[layer_ind].t_win
        out_spk_mask = output.lt(t_win) * output.gt(0)      
        
        end_spk_time = self.network[layer_ind].end_spk_time
        grad_weight_sum = -(end_spk_time + I)/((1 + valid_weight_sum)**2)
        grad_I = 1 / (1 + valid_weight_sum)
        
        # Gradient clipping
        grad_weight_sum[grad_weight_sum > self.grad_clip] = self.grad_clip
        grad_weight_sum[grad_weight_sum < -self.grad_clip] = -self.grad_clip
        grad_I[grad_I > self.grad_clip] = self.grad_clip
        grad_I[grad_I < -self.grad_clip] = -self.grad_clip
        
        # Gradient wrt. weights
        delta = dY * out_spk_mask.float() 
        tmp1 = delta * grad_I  # (batch, n_out)
        dW_1 = tmp1.T @ valid_in_spk  # (n_out x n_in)
        tmp2 = delta * grad_weight_sum  # (batch, n_out)
        dW_2 = tmp2.T @ in_spk_mask.float()  # (n_out x n_in)
        dW = -(dW_1 + dW_2)

        # Gradient wrt. inputs
        dX = tmp1 @ self.network[layer_ind].weights  # (batch, n_in)

        return dX, dW
        
            
    def _backward_conv(self, layer_ind, output, dY):
        # Forward cache
        valid_weight_sum = self.network[layer_ind].train_cache["valid_w_sum"]
        I = self.network[layer_ind].train_cache["weighted_input"]
        in_spk_mask = self.network[layer_ind].train_cache["in_spk_mask"]
        valid_in_spk = self.network[layer_ind].train_cache["valid_in_spk"]
        
        # Spike mask
        t_win = self.network[layer_ind].t_win
        out_spk_mask = output.lt(t_win) * output.gt(0)         

        end_spk_time = self.network[layer_ind].end_spk_time
        grad_weight_sum = -(end_spk_time + I)/((1 + valid_weight_sum)**2)
        grad_I = 1 / (1 + valid_weight_sum)
        
        # Gradient clipping
        grad_weight_sum[grad_weight_sum > self.grad_clip] = self.grad_clip
        grad_weight_sum[grad_weight_sum < -self.grad_clip] = -self.grad_clip
        grad_I[grad_I > self.grad_clip] = self.grad_clip
        grad_I[grad_I < -self.grad_clip] = -self.grad_clip

        # Gradient wrt. weights
        delta = dY * out_spk_mask.float() 
        tmp1 = delta * grad_I
        tmp2 = delta * grad_weight_sum
        w_shape = self.network[layer_ind].weights.shape
        dW_1 = torch.nn.grad.conv2d_weight(
            valid_in_spk, w_shape, tmp1,
            stride=(self.network[layer_ind].s_h, self.network[layer_ind].s_w),
            padding=(self.network[layer_ind].p_h, self.network[layer_ind].p_w)
        )
        dW_2 = torch.nn.grad.conv2d_weight(
            in_spk_mask.float(), w_shape, tmp2,
            stride=(self.network[layer_ind].s_h, self.network[layer_ind].s_w),
            padding=(self.network[layer_ind].p_h, self.network[layer_ind].p_w)
        )
        dW = -(dW_1 + dW_2)

        # Gradient wrt. inputs
        dX = torch.nn.grad.conv2d_input(
            in_spk_mask.shape, self.network[layer_ind].weights, tmp1,
            stride=(self.network[layer_ind].s_h, self.network[layer_ind].s_w),
            padding=(self.network[layer_ind].p_h, self.network[layer_ind].p_w)
        )
        
        return dX, dW
    
    
    def _backward_maxpool(self, layer_ind, output, dY):
        # Forward cache
        pool_idx = self.network[layer_ind].train_cache["pool_idx"]

        # Gradient wrt. weights
        B, C, H_out, W_out = output.shape
        dY = dY.view(B, C, H_out, W_out)
        dX = torch.nn.functional.max_unpool2d(
            dY, pool_idx, 
            kernel_size=(self.network[layer_ind].k_h, self.network[layer_ind].k_w),
            stride=(self.network[layer_ind].s_h, self.network[layer_ind].s_w),
            padding=(self.network[layer_ind].p_h, self.network[layer_ind].p_w),
            output_size=self.network[layer_ind].input_shape[1:]
        )
        
        return dX, None
    
    
    def __call__(self, outputs, y):
        # Update iteration counter
        self.it_cnt += 1

        # Compute error in the output layer
        # NOTE: assume a fully connected layer
        # Based on membrane potentials
        if self.network[-1].no_spk:
            mem_pots = outputs[-1]
            p = torch.nn.functional.softmax(mem_pots, dim=1) # Softmax probabilities
        # Based on spike times
        else:
            spk_times = outputs[-1]
            p = torch.nn.functional.softmax(-spk_times, dim=1) # Softmax probabilities
        batch_size = len(y)
        # One hot encoded labels
        y_oh = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=self.network[-1].n_neurons).to(torch.bool)
        dY = p - y_oh.float()
        dY = dY / batch_size # Mean reduction 

        # Backpropagation
        for layer_ind in range(len(self.network)-1, -1, -1): # Reversed loop
            
            # Compute dW and dX
            if isinstance(self.network[layer_ind], MaxPool):
                dY, dW = self._backward_maxpool(layer_ind, outputs[layer_ind], dY)
            elif isinstance(self.network[layer_ind], Conv):
                dY, dW = self._backward_conv(layer_ind, outputs[layer_ind], dY)
            elif isinstance(self.network[layer_ind], Fc):
                if self.network[layer_ind].no_spk:
                    dY, dW = self._backward_fc_no_spk(layer_ind, outputs[layer_ind], dY)
                else:
                    dY, dW = self._backward_fc(layer_ind, outputs[layer_ind], dY)
            
            # Apply weight changes
            if self.network[layer_ind].trainable: 
                self._apply_weight_change(layer_ind, dW)