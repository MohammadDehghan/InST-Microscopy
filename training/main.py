import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models import build_model
from data.regression.data import make_data_loaders, train_transform, val_transform, test_transform
from utils import train_val, init_env, predict
from sklearn.model_selection import StratifiedKFold



def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def create_folds(image_paths, cell_counts, n_splits=5):
    """
    Create stratified folds for cross-validation.

    Args:
        image_paths (np.array): Numpy array of paths to the images.
        cell_counts (np.array): Numpy array of cell counts corresponding to the images.
        n_splits (int): Number of splits for cross-validation.

    Returns:
        folds (list): A list of (train_index, val_index) tuples for each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(image_paths, cell_counts))
    return folds

def prepare_data(config):
    """Prepare image paths and cell counts for cross-validation."""
    image_paths = []
    cell_counts = []
    for filename in os.listdir(config["csv_directory"]):
        if filename.endswith('.csv'):
            file_path = os.path.join(config["csv_directory"], filename)
            data = pd.read_csv(file_path)
            num_cells = len(data)
            cell_counts.append(num_cells)
            image_name = os.path.splitext(filename)[0] + '.tiff'  # Adjust as needed
            image_path = os.path.join(config["image_directory"], image_name)
            image_paths.append(image_path)
    return np.array(image_paths), np.array(cell_counts)


def train_and_evaluate(config, fold, model_name, dataloaders):
    """
    Train and evaluate the model for a single fold.

    Args:
        config (dict): Configuration dictionary.
        fold (int): Fold number.
        model_name (str): Name of the model.
        dataloaders (dict): Dictionary containing train, val, and test data loaders.

    Returns:
        None
    """
    exp_name = f"{model_name}_fold_{fold}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, loss, optimizer, and scheduler setup
    criterion = nn.MSELoss().to(device)
    model = build_model(model_name=model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-5)))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(config.get("step_size", 10)), gamma=float(config.get("gamma", 0.1)))

    # Train the model
    train_val(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        num_epochs=int(config.get("num_epochs", 25)),
        device=device,
        exp_name=exp_name
    )

    # Evaluate on the test set
    predictions, targets, test_mae, test_acp = predict(model, dataloaders['test'], device=device)
    results = pd.DataFrame({"Predictions": predictions, "Targets": targets})
    results.to_csv(f"{exp_name}_test_results.csv", index=False)
    print(f"Predictions saved to {exp_name}_test_results.csv")


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Train and evaluate a regression model using cross-validation.")
    parser.add_argument('--config', type=str, default='./config.yml', help='Path to the configuration file.')
    parser.add_argument('--model', type=str, default='resnet50', help='Model name (e.g., resnet50, vgg16).')
    parser.add_argument('--folds', type=int, default=5, help='Number of cross-validation folds.')
    args = parser.parse_args()

    # Initialize environment
    init_env()

    # Load configuration
    config = load_config(args.config)

    # Prepare data
    print("Preparing data...")
    image_paths, cell_counts = prepare_data(config)

    # Create stratified folds
    folds = create_folds(image_paths, cell_counts, n_splits=args.folds)

    # Train and evaluate for each fold
    for fold, (train_index, val_index) in enumerate(folds):
        print("*" * 100)
        print(f"Starting fold {fold + 1}/{args.folds}")
        
        # Create data loaders for the current fold
        train_loader, val_loader, test_loader = make_data_loaders(
            image_paths, cell_counts, train_index, val_index,
            batch_size=config["batch_size_tr"],
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform
        )

        # Train and evaluate
        dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
        train_and_evaluate(config, fold, args.model, dataloaders)



if __name__ == "__main__":
    main()