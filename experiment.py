import torch, os
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from train_func import train_model
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from image_processing import convert_to_jpg, split_dataset
from models import (
    create_mobilenetv2_model,
    create_shufflenetv2_model,
    create_shufflenet_model,
    create_squeezenet_model,
)


def experiment(
    lr, batch_size, model_name, epochs=10, data_dir="/content/Linen_Dataset"
):
    """
    Train a model using the specified learning rate, batch size, and model name.

    Args:
        lr (float): The learning rate for the optimizer.
        batch_size (int): The batch size for training and validation.
        model_name (str): The name of the model to use (e.g., 'mobilenetv2', 'shufflenetv2', 'shufflenet', 'squeezenet').
        epochs (int): The number of training epochs. Default is 10.
        data_dir (str): The directory containing the dataset. Default is '/content/Linen_Dataset'.
    """
    # Convert model name to lowercase and check if it's a valid model
    model_name = model_name.lower()
    if model_name not in ["mobilenetv2", "shufflenetv2", "shufflenet", "squeezenet"]:
        raise ValueError(f"Invalid model name: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define data transforms for training and testing
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dirs = os.listdir(data_dir)

    # Convert images in the data directory to JPEG format
    for class_label in dirs:
        conversion_dir = os.path.join(data_dir, class_label)
        convert_to_jpg(conversion_dir, conversion_dir)

    if not "train" in dirs or not "valid" in dirs:
        split_dataset(data_dir, train_ratio=0.8, random_seed=42)

    data_dir_train = os.path.join(data_dir, "train")
    data_dir_valid = os.path.join(data_dir, "valid")

    # Load the dataset and split into train and validation
    train_data = datasets.ImageFolder(data_dir_train, transform=train_transform)
    valid_data = datasets.ImageFolder(data_dir_valid, transform=valid_transform)

    # Count occurrences of each class
    class_counts = Counter(train_data.targets)
    total_samples = sum(class_counts.values())

    class_weight = [(1.0 - count / total_samples) for count in class_counts.values()]
    class_weight = torch.Tensor(class_weight).to(device)

    print("Class Counts:", class_counts)
    print("Class Weight:", class_weight)

    # Define the data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    class_labels = {value: key for key, value in train_data.class_to_idx.items()}

    print("class_labels = ", class_labels)

    # Create the model based on the specified model name
    if model_name == "mobilenetv2":
        model = create_mobilenetv2_model(num_classes=len(class_labels))
    elif model_name == "shufflenetv2":
        model = create_shufflenetv2_model(num_classes=len(class_labels))
    elif model_name == "shufflenet":
        model = create_shufflenet_model(num_classes=len(class_labels))
    elif model_name == "squeezenet":
        model = create_squeezenet_model(num_classes=len(class_labels))
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_model_path = train_model(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        device,
        epochs,
        model_name,
    )

    return class_labels, best_model_path
