import torch, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from models import (
    create_mobilenetv2_model,
    create_shufflenetv2_model,
    create_shufflenet_model,
    create_squeezenet_model,
)


def perform_inference(image_path, model_name, model_checkpoint, class_labels, save_image=False):
    """
    Perform inference on an input image using a trained model and display the predicted class.

    Args:
        image_path (str): Path to the input image file.
        model (torch.nn.Module): The trained model for inference.
        model_checkpoint (str): Path to the model checkpoint file.
        class_labels (list): List of class labels.
        save_image (bool): Whether to save the predicted image with the class label in its filename (default: False).
    """
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
    # Load the model checkpoint
    model = model.to("cpu")
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    # Define the transformation for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image using PyTorch
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class index
    probabilities = torch.softmax(output, dim=1)
    predicted_class_index = torch.argmax(probabilities).item()

    # Plot the image and print the predicted class
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.title(f"Predicted Class: {class_labels[predicted_class_index]}")
    plt.show()

    # Save the predicted image with the class label in its filename
    if save_image:
        predicted_class = class_labels[predicted_class_index]
        image_filename = os.path.basename(image_path)
        image_name, image_ext = os.path.splitext(image_filename)
        predicted_image_filename = f"{image_name}_{predicted_class}{image_ext}"
        predicted_image_path = predicted_image_filename
        image.save(predicted_image_path)
        print(f"Predicted image saved as: {predicted_image_path}")