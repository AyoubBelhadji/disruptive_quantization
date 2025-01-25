# disruptive_quantization
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

### Configuration matrices
The script `config_generator.py` in the `experiment_configs` directory is intended to create a matrix of configs. It takes into account a template config to base on and a destination directory for the generated files, then creates configs with every combination of keyword arguments provided. For example,
```
./config_generator.py mnist/msip.json mnist_gen K=20,30 bandwidth=0.1,1
```
This example will generate four configuration files: all of them will be based on `mnist/msip.json`, and only changing the number of centroids `K` and the kernel bandwidth `bandwidth`. They will be created in `mnist_gen` and have all combinations of the two lists of parameters.

### SLURM workflow
The `slurm` directory is intended to help with use on slurm systems by providing a very basic script. Suppose know that you can max out `N` compute nodes on slurm. Then, you might generate `N` folders of configs, `mnist_gen1` `mnist_gen2`, ... `mnist_genN` using `config_generator`. If your conda environment is named `idp`, you would submit a job for `N` nodes in slurm accordingly. For MIT's supercloud, the starter allocation is 2 nodes with 48 cores each, so you can do
```
LLsub ./slurm_submit.sh [2,1,48]
```

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
with np.load(path + "experiment_data.npz") as np_file:
    centroids, weights = np_file["centroids"], np_file["weights"]

# ... do things with the centroids and weights
```
