import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.attention import CrossAttention
import numpy as np

def create_dummy_inputs(batch_size=1, channels=4, height=64, width=64, context_dim=768):
    # Create dummy latent input
    latent = torch.randn(batch_size, channels, height, width)
    
    # Create dummy timestep
    timesteps = torch.tensor([500], dtype=torch.long)
    
    # Create dummy context (text embedding)
    context = torch.randn(batch_size, 77, context_dim)
    
    return latent, timesteps, context

def print_lora_params(model):
    print("\nLoRA Parameters:")
    total_params = 0
    for name, param in model.named_parameters():
        if 'lora' in name:
            print(f"{name}: {param.shape}")
            total_params += param.numel()
    print(f"\nTotal LoRA parameters: {total_params:,}")

def verify_gradients(model, freeze_base=True):
    print("\nVerifying gradients:")
    if freeze_base:
        # Freeze all non-LoRA parameters
        for name, param in model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
    
    # Forward pass
    latent, timesteps, context = create_dummy_inputs()
    output = model(latent, timesteps, context)
    
    # Backward pass
    loss = output.mean()
    loss.backward()
    
    # Check gradients
    has_grad = False
    no_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'lora' in name:
                has_grad = True
                print(f"✓ Gradient exists for LoRA parameter: {name}")
            else:
                no_grad = True
                print(f"✗ Gradient exists for non-LoRA parameter: {name}")
    
    if has_grad and not no_grad:
        print("✓ All gradients are correctly flowing only through LoRA layers")
    else:
        print("✗ Gradient flow verification failed")

def main():
    # Model configuration
    config = {
        "image_size": 32,
        "in_channels": 4,
        "model_channels": 320,
        "out_channels": 4,
        "num_res_blocks": 2,
        "attention_resolutions": [4, 2, 1],
        "dropout": 0.0,
        "channel_mult": [1, 2, 4, 4],
        "num_heads": 8,
        "use_spatial_transformer": True,
        "transformer_depth": 1,
        "context_dim": 768,
    }
    
    # Create dummy inputs
    latent, timesteps, context = create_dummy_inputs()
    
    # 1. Load model without LoRA
    print("1. Testing model without LoRA...")
    model_no_lora = UNetModel(**config, use_lora=False)
    with torch.no_grad():
        output_no_lora = model_no_lora(latent, timesteps, context)
    
    # 2. Load model with LoRA
    print("\n2. Testing model with LoRA...")
    model_lora = UNetModel(**config, use_lora=True, lora_rank=4, lora_alpha=1.0)
    with torch.no_grad():
        output_lora = model_lora(latent, timesteps, context)
    
    # Calculate L2 distance between outputs
    l2_distance = F.mse_loss(output_no_lora, output_lora)
    print(f"L2 distance between outputs: {l2_distance.item():.6f}")
    
    # 3. Print LoRA parameters
    print_lora_params(model_lora)
    
    # 4. Verify gradients
    verify_gradients(model_lora, freeze_base=True)
    
    # 5. Verify trainable parameters
    trainable_params = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_lora.parameters())
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {(trainable_params/total_params*100):.2f}%")

if __name__ == "__main__":
    main() 