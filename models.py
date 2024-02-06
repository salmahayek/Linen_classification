import torch.nn as nn
from torchvision import models


def create_mobilenetv2_model(num_classes=2):
    """
    Create a MobileNetV2 model with a modified classifier for the specified number of classes.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: MobileNetV2 model.
    """
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model.features[0][0] = nn.Conv2d(
        3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    return model


def create_shufflenetv2_model(num_classes=2):
    """
    Create a ShuffleNetV2 model with a modified classifier for the specified number of classes.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: ShuffleNetV2 model.
    """
    model = models.shufflenet_v2_x1_0(pretrained=True)

    # Modify the classifier to match the number of output classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Modify the first convolution layer to accept 3 channels
    model.conv1[0] = nn.Conv2d(
        3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
    )
    return model


def create_shufflenet_model(num_classes=2):
    """
    Create a ShuffleNet model with a modified classifier for the specified number of classes.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: ShuffleNet model.
    """
    model = models.shufflenet_v2_x0_5(pretrained=True)

    # Modify the classifier to match the number of output classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def create_squeezenet_model(num_classes=2):
    """
    Create a SqueezeNet model with a modified classifier for the specified number of classes.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: SqueezeNet model.
    """
    model = models.squeezenet1_0(pretrained=True)

    # Modify the classifier to match the number of output classes
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    return model
