import os
import pickle
import numpy as np
from torchvision import datasets, transforms
import shutil

def pickleFileExists(pickle_file):
    return pickle_file and os.path.exists(pickle_file)

def load_or_download_mnist(data_dir="mnist", pickle_file=None):
    """
    Load the MNIST dataset from pickle files or download it.

    Parameters:
    - data_dir: str
        Directory to save or look for the MNIST dataset.
    - pickle_file: str or None
        Path to the pickle file to create/load. If None, data is downloaded from the Internet.

    Returns:
    - images: ndarray of shape (n_samples, 784)
        The MNIST images reshaped into 1D arrays.
    - labels: ndarray of shape (n_samples,)
        The corresponding labels.
    """
    images, labels = None, None

    if pickleFileExists(pickle_file):
        print("Loading MNIST dataset from pickle files...")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            images, labels = data["images"], data["labels"]
    else:
        print("Downloading MNIST dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        images = mnist.data
        images = images.reshape(images.shape[0], -1).numpy()
        labels = mnist.targets.numpy()

        if pickle_file:
            with open(pickle_file, 'wb') as f:
                data = {"data": images, "labels": labels}
                pickle.dump(data, f)
                print(f"Data saved to {pickle_file}")

            shutil.rmtree(os.path.join(data_dir, "MNIST"))
    return images, labels


# Example usage
if __name__ == "__main__":
    data_dir = "./mnist"
    os.makedirs(data_dir, exist_ok=True)
    pickle_file = os.path.join(data_dir, "mnist.pkl")

    # Load or download MNIST dataset
    images, labels = load_or_download_mnist(data_dir=data_dir, pickle_file=pickle_file)