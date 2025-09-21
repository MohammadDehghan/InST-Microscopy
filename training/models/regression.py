import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



class ModelFactory:
    def __init__(self, model_name, output_size=1):
        """
        Initialize a pretrained model.

        Args:
            model_name (str): Name of the model (e.g., "resnet50", "vgg16").
            output_size (int): Size of the output layer (e.g., 1 for regression).
        """
        self.model_name = model_name.lower()
        self.output_size = output_size
        self.model = self._create_model()

    def _create_model(self):
        """Creates and returns the specified model."""
        if self.model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.output_size)

        elif self.model_name == "vgg16":
            model = models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.output_size)

        elif self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.output_size)

        elif self.model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=True)
            model.last_channel = nn.Linear(model.last_channel, self.output_size)

        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

        return model

    def get_model(self):
        """
        Returns the initialized model.

        Returns:
            nn.Module: Pretrained model with modified head.
        """
        return self.model