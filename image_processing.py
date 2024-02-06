import os
import shutil
import pillow_avif
from PIL import Image
from pillow_heif import register_heif_opener
from sklearn.model_selection import train_test_split

register_heif_opener()


def convert_to_jpg(input_dir, output_dir):
    """
    Convert image files in the input directory to JPEG format and save them in the output directory.
    Supported input file formats: JPEG, HEIC, AVIF.

    Args:
        input_dir (str): Path to the input directory containing the image files.
        output_dir (str): Path to the output directory where the converted JPEG files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all files in the input directory
    files = os.listdir(input_dir)

    for file in files:
        file_path = os.path.join(input_dir, file)

        try:
            # Check if it's an image file
            if os.path.isfile(file_path) and file.lower().endswith(
                (".jpeg", ".heic", ".avif")
            ):
                # Open the image
                img = Image.open(file_path)

                # Convert to JPEG and save
                jpg_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".jpg")
                img.convert("RGB").save(jpg_path, "JPEG", quality=95)

                print(f"Converted {file} to {jpg_path}")

                # Remove the original image
                os.remove(file_path)
                print(f"Removed original image: {file}\n")
        except Exception as e:
            print(f"Error processing {file}: {e}")


def split_dataset(data_dir, train_ratio=0.8, random_seed=42):
    # Create directories for train and valid sets
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Get the list of subdirectories (classes) in the dataset
    classes = os.listdir(data_dir)

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)

        # Skip non-directory entries
        if not os.path.isdir(class_dir):
            continue

        # Get the list of files in the class directory
        files = os.listdir(class_dir)

        # Check if there are enough samples to split
        if len(files) < 2:
            print(f"Skipping class '{class_name}' due to insufficient samples.")
            continue

        # Split the files into train and valid sets
        train_files, valid_files = train_test_split(
            files, test_size=1 - train_ratio, random_state=random_seed
        )

        # Create directories for each class in train and valid sets
        train_class_dir = os.path.join(train_dir, class_name)
        valid_class_dir = os.path.join(valid_dir, class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(valid_class_dir, exist_ok=True)

        # Copy files to the respective directories
        for file in train_files:
            src_path = os.path.join(class_dir, file)
            dest_path = train_class_dir
            shutil.move(src_path, dest_path)

        for file in valid_files:
            src_path = os.path.join(class_dir, file)
            dest_path = valid_class_dir
            shutil.move(src_path, dest_path)

    print("Dataset split into train and valid sets successfully.")
