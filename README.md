# DisruptiveQuantization
## A Python package for quantizing data
This code is to accompany the work (TODO).

### Dependencies
This code runs using Python using `numpy, scipy, matplotlib, numba`.

_TODO_: create an environment file

### Running the code
Create a directory in `experiment_configs` and incorporate a config file of your choosing. See the `experiment_configs/examples` for examples. Then, use `main.py` with this directory. For example, if you use `experiment_configs/mnist`, you would do

```
python main.py -d mnist
```

Use the `-h` option to see the visualization options. Note that some postproc steps may take longer than others. We try to serialize and cache wherever possible.

### Loading the data
You should see some output after running `main.py` regarding where the data is located. For example, if you used a config at `experiment_configs/mnist/msip.json`, you should look for output like

```
experiments/sandbox/mnist/msip_2_20250116_164850/
```

Then, if you want all of the possible experiment data, make sure to stay in this top-level directory `disruptive_quantization` and load via pickle like so:
```python
import pickle
path = "experiments/sandbox/mnist/msip_2_20250116_164850/"
with open(path + "experiment_data_with_metadata.pkl", "rb") as f:
    data = pickle.load(f)

# ... do things with data ...
```

> Remark: It is necessary to stay in this directory, because this will load virtually all objects used during the experiment, which include classes that pickle will only be able to reconstruct by looking for the files from the top-level directory.

If you just want to load the centroid and weight trajectory outputs, you can do as so:

```python
import numpy as np
path = "experiments/sandbox/mnist/msip_2_20250116_164850/"
with np.load(path + "experiment_data.npz") as f:
    centroids, weights = np_file["centroids"], np_file["weights"]

# ... do things with the centroids and weights
```