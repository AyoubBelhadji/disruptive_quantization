import os
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def load_or_download_mnist(data_dir="data", pickle_file=None):
    """
    Load the MNIST dataset from a pickle file or download it.

    Parameters:
    - data_dir: str
        Directory to save or look for the MNIST dataset.
    - pickle_file: str or None
        Path to the pickle file. If None, data is downloaded from the Internet.

    Returns:
    - data_dict: dict
        A dictionary with keys 'data' (images) and 'labels'.
    """
    data_dict = None

    if pickle_file and os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            data_dict = pickle.load(f)
            print(f"Loaded data from {pickle_file}")
    else:
        print("Downloading MNIST dataset...")
        transform = transforms.Compose([transforms.ToTensor()])
        mnist = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        images = np.array([img.numpy().squeeze().reshape(-1) for img, _ in mnist])
        labels = np.array([label for _, label in mnist])

        data_dict = {'data': images, 'labels': labels, 'params': None}

        if pickle_file:
            with open(pickle_file, 'wb') as f:
                pickle.dump(data_dict, f)
                print(f"Data saved to {pickle_file}")

        # Clean up the raw MNIST folder
        shutil.rmtree(os.path.join(data_dir, "MNIST"))

    return data_dict

# def plot_mnist_images(images, labels, n_rows=4, n_cols=4):
#     """
#     Plot a grid of MNIST images with their labels.

#     Parameters:
#     - images: ndarray of shape (n_samples, 784)
#         The MNIST images reshaped into 1D arrays.
#     - labels: ndarray of shape (n_samples,)
#         The corresponding labels.
#     - n_rows: int
#         Number of rows in the grid.
#     - n_cols: int
#         Number of columns in the grid.
#     """
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
#     indices = np.random.choice(len(images), n_rows * n_cols, replace=False)

#     for ax, idx in zip(axes.flatten(), indices):
#         ax.imshow(images[idx].reshape(28, 28), cmap='gray')
#         ax.set_title(f"Label: {labels[idx]}")
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()

# Example usage
if __name__ == "__main__":
    data_dir = "./mnist"
    os.makedirs(data_dir, exist_ok=True)
    pickle_file = os.path.join(data_dir, "mnist_data.pkl")

    # Load or download MNIST dataset
    data_dict = load_or_download_mnist(data_dir=data_dir, pickle_file=pickle_file)

    # Plot some sample images
    #plot_mnist_images(data_dict['data'], data_dict['labels'])

# # Example usage
# if __name__ == "__main__":
#     data_dir = "./mnist"
#     os.makedirs(data_dir, exist_ok=True)
#     x_pickle_file = os.path.join(data_dir, "mnist_images.pkl")
#     y_pickle_file = os.path.join(data_dir, "mnist_labels.pkl")

#     # Load or download MNIST dataset
#     images, labels = load_or_download_mnist(data_dir=data_dir, x_pickle_file=x_pickle_file, y_pickle_file=y_pickle_file)

