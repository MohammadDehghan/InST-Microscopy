import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Compute LoRA output
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * (self.alpha / self.rank)
        return lora_output

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=1.0):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )
        
    def forward(self, x):
        return self.linear(x) + self.lora(x)

def apply_lora_to_attention(attention_layer, rank=4, alpha=1.0):
    """Apply LoRA to the query, key, and value projections in an attention layer"""
    attention_layer.to_q = LoRALinear(attention_layer.to_q, rank, alpha)
    attention_layer.to_k = LoRALinear(attention_layer.to_k, rank, alpha)
    attention_layer.to_v = LoRALinear(attention_layer.to_v, rank, alpha)
    return attention_layer 