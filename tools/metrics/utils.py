import os
import pickle
from tools import metrics
import numpy as np
import numba as nb


def calculate_mmd_and_logdets(y_array, w_array, data_array, kernel, mmd_self, subpath):
    mmd_folder_serial = os.path.join("experiments", "sandbox", subpath)
    os.makedirs(mmd_folder_serial, exist_ok=True)

    mmd_values = metrics.mmd_array(y_array, w_array, data_array, kernel, mmd_self)
    logdets = metrics.logdet_array(y_array, kernel)

    # Save the mmd_values and logdets
    np.save(os.path.join(mmd_folder_serial, "mmd_values.npy"), mmd_values)
    np.save(os.path.join(mmd_folder_serial, "logdets.npy"), logdets)
    print("MMD values and logdets saved to ", mmd_folder_serial)
    return mmd_values, logdets


def get_data_mmd(data, kernel, mmd_self, kernel_eval):
    data_mmd = mmd_self[kernel]
    if data_mmd is None:
        data_mmd = metrics.mmd_calc.compute_mmd_entropy_large_unweighted(
            data, kernel_eval
        )
        mmd_self[kernel] = data_mmd
    return data_mmd


@nb.jit(parallel=True)
def calcluate_all_metrics_impl(all_nodes, all_node_weights, data, kernel, data_mmd):
    mmds = np.zeros(len(all_nodes))
    logdets = np.zeros(len(all_nodes))
    haussdorffs = np.zeros(len(all_nodes))
    voronoi_mses = np.zeros(len(all_nodes))
    for i in nb.prange(len(all_nodes)):
        Y, weights_Y = all_nodes[i], all_node_weights[i]
        mmd_sq = data_mmd
        mmd_sq += metrics.mmd_calc.compute_cross_mmd_large_weighted_Y(
            data, Y, kernel, weights_Y=weights_Y
        )
        mmds[i] = np.sqrt(mmd_sq)
        logdets[i] = metrics.logdet(Y, kernel)
        haussdorffs[i] = metrics.hausdorff_distance(Y, data)
        voronoi_mses[i] = metrics.voronoi_mse(Y, data)
    return {
        "mmd": mmds,
        "logdet": logdets,
        "haussdorff": haussdorffs,
        "voronoi_mse": voronoi_mses,
    }


def calculate_all_metrics(alg_name, y_array, w_array, data, kernel, mmd_self, subpath):
    metric_serialize_path = os.path.join("experiments", "sandbox", subpath)
    os.makedirs(metric_serialize_path, exist_ok=True)
    all_nodes = y_array.reshape(-1, y_array.shape[-2], y_array.shape[-1])
    all_node_weights = w_array.reshape(-1, w_array.shape[-1])
    kernel_eval = kernel.kernel
    data_mmd = get_data_mmd(data, kernel, mmd_self, kernel_eval)

    all_metrics = calcluate_all_metrics_impl(
        all_nodes, all_node_weights, data, kernel_eval, data_mmd
    )
    all_metrics_shaped = {
        k: v.reshape(y_array.shape[:-2]) for (k, v) in all_metrics.items()
    }
    all_metrics_shaped["alg_name"] = alg_name
    with open(os.path.join(metric_serialize_path, "all_metrics.pkl"), "wb") as f:
        pickle.dump(all_metrics_shaped, f)
