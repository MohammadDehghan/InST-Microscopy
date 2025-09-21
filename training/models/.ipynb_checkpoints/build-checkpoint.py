from .regression import ModelFactory

def build_model(model_name, output_size=1):
    """
    Build a model using ModelFactory.

    Args:
        model_name (str): Name of the model (e.g., "vgg16", "resnet50").
        output_size (int): Size of the output layer.

    Returns:
        nn.Module: Pretrained model with a custom output layer.
    """
    factory = ModelFactory(model_name=model_name, output_size=output_size)
    model = factory.get_model()
    return model


