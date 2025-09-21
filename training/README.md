
# **A Responsible AI Approach for Cell Counting in Microscopic Images**

This project focuses on **cell counting** in microscopic images using machine learning models. It includes scripts for training, testing, and evaluating the model. For refactoring some parts of code, the GPT4 model was used.

---

## **Project Overview**

The pipeline leverages:
- **Convolutional Neural Networks (CNNs)** for regression-based cell counting.
- **Density Map Estimation** using CSRNet.
- **Data Augmentation** techniques such as CutMix to improve model generalization.

---

## **Table of Contents**

1. [Project Structure](#project-structure)
2. [Training the Model](#training-the-model)
3. [Testing the Model](#testing-the-model)
4. [Configuration](#configuration)

---


## **Project Structure**

```plaintext
.
├── data/               # Directory for handling the dataset
├── models/             # Model architectures (e.g., EfficientNet, CSRNet)
├── config.yml          # Configuration file for model parameters and paths
├── main.py             # Training and testing pipeline
├── utils.py            # Utility functions for preprocessing and augmentation
└── test_predict.py     # Testing notebook
```

---


## **Training the Model**

To train the model, use the `main.py` script.

### **Command to Run Training**
```bash
python main.py --config config.yml
```

### **Training Configuration**
Edit the `config.yml` file to set your desired parameters:
```yaml
data_path: "./data/"
model: "EfficientNet"       # Choose between EfficientNet, CSRNet etc.
epochs: 50
batch_size: 8
learning_rate: 0.001
```

---

## **Testing the Model**

To test the trained model on new data, use the `test_predict.ipynb` notebook.

---

## **Configuration**

The entire pipeline is configurable using the `config.yml` file. Modify it as needed to customize:
- **Model architecture**
- **Dataset paths**
- **Training hyperparameters**



