import os
import random
import torch
import warnings
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_absolute_error
from data.regression.data import cutmix
import copy

def init_env(gpu_id='0', seed=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')
    

def visualize_dataloader(data_loader, num_samples=4):
    """
    Visualizes outputs from the data loader.

    Args:
        data_loader (DataLoader): The DataLoader to fetch data from.
        num_samples (int): Number of samples to visualize from a batch.
    """
    # Get a single batch of data
    for img, num_cells, density_map_img, pixel_pos_img in data_loader:
        # Display the first `num_samples` images
        for i in range(min(num_samples, len(img))):
            plt.figure(figsize=(12, 6))

            # Convert tensor to numpy for visualization
            img_np = img[i].permute(1, 2, 0).numpy() if img[i].ndim == 3 else img[i].numpy()
            density_map_np = np.squeeze(density_map_img[i].numpy())  # Fix here
            pixel_pos_np = np.squeeze(pixel_pos_img[i].numpy())  # Fix for consistency

            # Calculate the sum of the density map
            density_sum = np.sum(density_map_np)

            # Plot the original image
            plt.subplot(1, 3, 1)
            plt.imshow(img_np, cmap='gray')
            plt.title(f"Image - Cells: {num_cells[i].item()}")
            plt.axis('off')

            # Plot the density map with sum in the title
            plt.subplot(1, 3, 2)
            plt.imshow(density_map_np, cmap='jet')
            plt.title(f"Density Map - Sum: {density_sum:.2f}")
            plt.axis('off')

            # Plot the pixel position map
            plt.subplot(1, 3, 3)
            plt.imshow(pixel_pos_np, cmap='gray')
            plt.title("Pixel Position Map")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        break  # Visualize only the first batch
        
    



def train_val(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
    device=None,
    exp_name="experiment",
    apply_cutmix=True,
    mix_prob=0.5
):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = float('inf')

    # TensorBoard Writer
    writer = SummaryWriter(log_dir=f"runs/{exp_name}")

    # Get synthetic dataloader if available
    synthetic_dataloader = dataloaders.get('train_syn', None)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 100)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                real_dataloader = dataloaders['train']
                # Initialize synthetic data iterator if available
                synthetic_iter = iter(synthetic_dataloader) if synthetic_dataloader else None
            else:
                model.eval()   # Set model to evaluate mode
                real_dataloader = dataloaders['val']

            running_loss_real = 0.0
            running_loss_syn = 0.0
            all_preds_real = []
            all_targets_real = []
            all_preds_syn = []
            all_targets_syn = []
            accurate_predictions_real = 0
            accurate_predictions_syn = 0

            # Iterate over real data
            real_iter = iter(real_dataloader)
            while True:
                try:
                    real_batch = next(real_iter)
                except StopIteration:
                    break  # End of real data

                inputs_real, num_cells_real, _, _ = real_batch
                inputs_real = inputs_real.to(device)
                num_cells_real = num_cells_real.to(device)

                if phase == 'train' and synthetic_iter:
                    try:
                        synthetic_batch = next(synthetic_iter)
                    except StopIteration:
                        synthetic_iter = iter(synthetic_dataloader)
                        synthetic_batch = next(synthetic_iter)

                    inputs_syn, num_cells_syn, _, _ = synthetic_batch
                    inputs_syn = inputs_syn.to(device)
                    num_cells_syn = num_cells_syn.to(device)

                    if apply_cutmix:
                        # Apply CutMix to combine real and synthetic batches
                        inputs_mixed, num_cells_mixed, _, _ = cutmix(
                            real_batch,
                            synthetic_batch,
                            mix_prob=mix_prob
                        )
                        inputs = inputs_mixed.to(device)
                        num_cells = num_cells_mixed.to(device)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            outputs = outputs.view(-1)  # Flatten outputs to match target

                            loss = criterion(outputs, num_cells)

                            loss.backward()
                            optimizer.step()

                        # Since data is mixed, we cannot separate metrics
                        # So, we'll record metrics on the mixed data
                        preds = outputs.detach().cpu().numpy()
                        targets = num_cells.cpu().numpy()

                        running_loss_real += loss.item() * inputs.size(0)
                        all_preds_real.extend(preds)
                        all_targets_real.extend(targets)

                        # ACP calculation for mixed data
                        accurate_predictions_real += sum(
                            abs(pred - target) <= 0.05 * target
                            for pred, target in zip(preds, targets)
                        )
                    else:
                        # Process real data
                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs_real = model(inputs_real)
                            outputs_real = outputs_real.view(-1)

                            loss_real = criterion(outputs_real, num_cells_real)

                        # Process synthetic data
                        outputs_syn = model(inputs_syn)
                        outputs_syn = outputs_syn.view(-1)

                        loss_syn = criterion(outputs_syn, num_cells_syn)

                        # Combine losses
                        total_loss = loss_real + loss_syn

                        if phase == 'train':
                            total_loss.backward()
                            optimizer.step()

                        # Aggregate metrics for real data
                        running_loss_real += loss_real.item() * inputs_real.size(0)
                        preds_real = outputs_real.detach().cpu().numpy()
                        targets_real = num_cells_real.cpu().numpy()

                        all_preds_real.extend(preds_real)
                        all_targets_real.extend(targets_real)

                        accurate_predictions_real += sum(
                            abs(pred - target) <= 0.05 * target
                            for pred, target in zip(preds_real, targets_real)
                        )

                        # Aggregate metrics for synthetic data
                        running_loss_syn += loss_syn.item() * inputs_syn.size(0)
                        preds_syn = outputs_syn.detach().cpu().numpy()
                        targets_syn = num_cells_syn.cpu().numpy()

                        all_preds_syn.extend(preds_syn)
                        all_targets_syn.extend(targets_syn)

                        accurate_predictions_syn += sum(
                            abs(pred - target) <= 0.05 * target
                            for pred, target in zip(preds_syn, targets_syn)
                        )
                else:
                    # Validation phase or no synthetic data
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs_real = model(inputs_real)
                        outputs_real = outputs_real.view(-1)

                        loss_real = criterion(outputs_real, num_cells_real)

                        if phase == 'train':
                            loss_real.backward()
                            optimizer.step()

                    # Aggregate metrics for real data
                    running_loss_real += loss_real.item() * inputs_real.size(0)
                    preds_real = outputs_real.detach().cpu().numpy()
                    targets_real = num_cells_real.cpu().numpy()

                    all_preds_real.extend(preds_real)
                    all_targets_real.extend(targets_real)

                    accurate_predictions_real += sum(
                        abs(pred - target) <= 0.05 * target
                        for pred, target in zip(preds_real, targets_real)
                    )

            # Compute metrics for real data
            epoch_loss_real = running_loss_real / len(real_dataloader.dataset)
            epoch_mae_real = float(mean_absolute_error(all_targets_real, all_preds_real))
            epoch_acp_real = float((accurate_predictions_real / len(all_targets_real)) * 100)

            # Print metrics for real data
            print(f'{phase} Real Data - Loss: {epoch_loss_real:.4f} | MAE: {epoch_mae_real:.4f} | ACP: {epoch_acp_real:.2f}%')

            # TensorBoard Logging for real data
            writer.add_scalar(f"{phase}_loss_real", epoch_loss_real, epoch)
            writer.add_scalar(f"{phase}_mae_real", epoch_mae_real, epoch)
            writer.add_scalar(f"{phase}_acp_real", epoch_acp_real, epoch)

            if phase == 'train' and not apply_cutmix and synthetic_iter:
                # Compute metrics for synthetic data
                epoch_loss_syn = running_loss_syn / len(synthetic_dataloader.dataset)
                epoch_mae_syn = float(mean_absolute_error(all_targets_syn, all_preds_syn))
                epoch_acp_syn = float((accurate_predictions_syn / len(all_targets_syn)) * 100)

                # Print metrics for synthetic data
                print(f'{phase} Synthetic Data - Loss: {epoch_loss_syn:.4f} | MAE: {epoch_mae_syn:.4f} | ACP: {epoch_acp_syn:.2f}%')

                # TensorBoard Logging for synthetic data
                writer.add_scalar(f"{phase}_loss_syn", epoch_loss_syn, epoch)
                writer.add_scalar(f"{phase}_mae_syn", epoch_mae_syn, epoch)
                writer.add_scalar(f"{phase}_acp_syn", epoch_acp_syn, epoch)

            # Deep copy the model if validation improves
            if phase == 'val' and epoch_loss_real < min_loss:
                min_loss = epoch_loss_real
                best_model_wts = copy.deepcopy(model.state_dict())

                # Ensure the checkpoints directory exists
                os.makedirs("checkpoints", exist_ok=True)

                # Save model checkpoint
                checkpoint_path = os.path.join("checkpoints", f"{exp_name}_best_model.pth")
                torch.save(best_model_wts, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        scheduler.step()

    writer.close()

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best val Loss: {min_loss:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def predict(model, dataloader, device=None):
    """
    Makes predictions on the test data using the trained model.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): Device to run the inference on (CPU or GPU).

    Returns:
        predictions (list): Predicted values.
        targets (list): Ground truth values.
        test_mae (float): Mean Absolute Error of the predictions.
        test_acp (float): Accuracy within a certain threshold (e.g., 5% of target).
    """
    model.eval()  # Set model to evaluation mode
    predictions = []
    targets = []
    running_loss = 0.0
    accurate_predictions = 0
    
    # Ensure the model is on the correct device
    model = model.to(device)
    
    with torch.no_grad():
        for inputs, num_cells, density_maps, _ in dataloader:
            inputs = inputs.to(device)
            num_cells = num_cells.to(device)
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.view(-1)  # Flatten outputs to match target
            
            # Store predictions and targets
            preds = outputs.detach().cpu().numpy()
            target = num_cells.cpu().numpy()
            
            predictions.extend(preds)
            targets.extend(target)
            
            # Compute Accuracy Within 5% Threshold
            accurate_predictions += sum(
                abs(pred - target) <= 0.05 * target
                for pred, target in zip(preds, target)
            )
    
    # Compute Metrics
    test_mae = float(mean_absolute_error(targets, predictions))
    test_acp = float((accurate_predictions / len(targets)) * 100)  # Ensure scalar
    
    print(f"Test MAE: {test_mae:.4f} | Test ACP: {test_acp:.2f}%")
    
    return predictions, targets, test_mae, test_acp