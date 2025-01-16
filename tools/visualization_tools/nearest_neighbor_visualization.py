import numpy as np
import matplotlib.pyplot as plt
import os

def find_nearest_neighbors(nodes: np.ndarray, data: np.ndarray, L: int) -> np.ndarray:
    """ Find the L nearest neighbors of each node in the data """
    M = len(nodes)
    all_neighbor_idxs = np.empty((M, L), dtype=int)
    for (i,node) in enumerate(nodes):
        distances = np.sum((data - node)**2, axis=-1)
        all_neighbor_idxs[i] = np.argsort(distances)[:L]
    return all_neighbor_idxs

def to_BW_images(array):
    # Assume image will be square, BW
    side_len = int(np.sqrt(array.shape[1]))
    if side_len**2 != array.shape[1]:
        raise ValueError("Array must be square")
    return array.reshape(-1, side_len, side_len)

def get_assigned_label(neighbors, labels):
    return np.argmax(np.bincount(labels[neighbors]))

def style_nearest_neighbor_axis(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

def image_nearest_neighbors(nodes, data, labels, neighbor_idxs, node_labels, title):
    use_labels = labels is not None
    N, num_neighbors = neighbor_idxs.shape
    node_imgs = to_BW_images(nodes)
    fig, axes = plt.subplots(N, num_neighbors+1, figsize=(num_neighbors+1, N+2))
    fig.suptitle(title)
    for i in range(N):
        neighbors = data[neighbor_idxs[i]]
        node_img, neighbor_imgs = node_imgs[i], to_BW_images(neighbors)
        axes[i, 0].imshow(node_img, cmap='gray')
        title = f"Node {i}" + (f" ({node_labels[i]})" if use_labels else "")
        style_nearest_neighbor_axis(axes[i, 0])
        axes[i, 0].set_title(title)
        for j in range(num_neighbors):
            axes[i, j+1].imshow(neighbor_imgs[j], cmap='gray')
            style_nearest_neighbor_axis(axes[i, j+1])
            title = f"({labels[neighbor_idxs[i, j]]})" if use_labels else ""
            axes[i, j+1].set_title(title)
    return node_labels

def labelled_nearest_neighbors(nodes, data, labels, num_neighbors):
    neighbor_idxs = find_nearest_neighbors(nodes, data, num_neighbors)
    if labels is None:
        return neighbor_idxs, None
    node_labels = np.empty(len(nodes), dtype=labels.dtype)
    for i in range(len(nodes)):
        node_labels[i] = get_assigned_label(neighbor_idxs[i], labels)
    return neighbor_idxs, node_labels

def plot_nearest_neighbors(nodes_0, weights_0, nodes_T, weights_T, data, labels_data, r, alg_name, plot_path, num_neighbors = 5, use_images = None):
    if use_images is None: # Rudimentary check: see if the data is square and larger than 10x10
        use_images = data.shape[1] > 100 and int(np.sqrt(data.shape[1]))**2 == data.shape[1]
    neighbor_idxs, labels_T = labelled_nearest_neighbors(nodes_T, data, labels_data, num_neighbors)
    # Plot nearest neighbors as images if the data is in image format
    if use_images:
        print("Plotting nearest neighbors as images")
        plot_title = f"Nearest neighbors ({num_neighbors}) of nodes, r={r}, {alg_name}"
        # Plot nearest neighbors
        image_nearest_neighbors(nodes_T, data, labels_data, neighbor_idxs, labels_T, plot_title)
        plt.savefig(os.path.join(plot_path, f"nearest_neighbors_r_{r}.png"))
        plt.show()

    # Plot distribution of assigned labels compared to true distribution of assigned labels
    if labels_T is not None:
        _, labels_0 = labelled_nearest_neighbors(nodes_0, data, labels_data, num_neighbors)
        # Find offset to align the histograms according to the number of unique labels
        visualize_label_frequencies(labels_0, weights_0, labels_T, weights_T, labels_data, r, alg_name, plot_path)
    return labels_T

def visualize_label_frequencies(labels_0, weights_0, labels_T, weights_T, labels_data, r, alg_name, plot_path):
    freqs_data = np.bincount(labels_data)/len(labels_data)
    K = len(freqs_data)
    label_locations = np.arange(K)
    freqs_T = np.bincount(labels_T, weights=weights_T/weights_T.sum(), minlength=K)
    freqs_0 = np.bincount(labels_0, weights=weights_0/weights_0.sum(), minlength=K)
    legend_labels = {"True labels": freqs_data, "Assigned labels": freqs_T, "Initial labels": freqs_0}
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")
    offsets = []
    for attr, freq in legend_labels.items():
        offset = width*multiplier
        ax.bar(label_locations + offset, freq, width, label=attr)
        multiplier += 1
        offsets.append(offset)
    ax.set_title(f"Assigned labels distribution, r={r}, {alg_name}")
    ax.legend()
    ax.set_xticks(label_locations + offsets[len(offsets)//2], labels=label_locations)
    plt.savefig(os.path.join(plot_path, f"assigned_labels_r_{r}.png"))
    plt.show()

def nearest_neighbors(centroid_trajectory, weight_trajectory, data_array, labels, alg_name, plot_path, **kwargs):
    R = len(centroid_trajectory)
    for r in range(R):
        c_r_0, w_r_0 = centroid_trajectory[r, 0], weight_trajectory[r, 0]
        c_r_T, w_r_T = centroid_trajectory[r, -1], weight_trajectory[r, -1]
        plot_nearest_neighbors(c_r_0, w_r_0, c_r_T, w_r_T, data_array, labels, r, alg_name, plot_path, **kwargs)