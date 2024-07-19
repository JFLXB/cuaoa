# CUAOA: A Novel CUDA-Accelerated Simulation Framework for the QAOA

[![arXiv](https://img.shields.io/badge/arXiv-2407.13012-b31b1b.svg)](https://arxiv.org/abs/2407.13012)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12750207.svg)](https://doi.org/10.5281/zenodo.12750207)

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Conda Environment Requirement](#conda-environment-requirement)
   - [Installing from Source](#installing-from-source)
   - [Security Note](#security-note)
   - [Troubleshooting](#troubleshooting)
3. [Usage](#usage)
4. [License and Compliance](#license-and-compliance)
5. [Disclaimer](#disclaimer)
6. [Citation](#citation)

## Overview

CUAOA is a GPU accelerated QAOA simulation framework utilizing the [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit). This framework offers a complete interface for QAOA simulations, enabling the calculation of (exact) expectation values, direct access to the statevector, fast sampling, and high-performance optimization methods using an advanced state-of-the-art gradient calculation technique. The framework is designed for use in Python and Rust, providing flexibility for integration into a wide range of applications, including those requiring fast algorithm implementations leveraging QAOA at its core.

## Installation

To streamline the installation of our project, utilize the [`install.sh`](./install.sh) script. This script automates the process by cloning the repository, building the project, and installing it on your system.
We plan to make CUAOA installable via pip in the upcoming future.

### Prerequisites

Before proceeding with the installation, ensure the following tools are installed on your system:

- [Rust and Cargo](https://www.rust-lang.org/tools/install): Required to compile the Rust libraries.
- [g++](https://gcc.gnu.org/): Required to compile the C++ library.
- [CUDA and nvcc](https://developer.nvidia.com/cuda-downloads): Required for CUDA-accelerated computations and the compilation.
- [git](https://git-scm.com/): Required for cloning the repository.
- [pip](https://pip.pypa.io/en/stable/installing/): Necessary for Python package installations.
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html): A crucial tool for environment and package management.
- [Python >= 3.11](https://www.python.org/downloads/): Required for running the Python code. Other versions may work but have not been tested.

### Conda Environment Requirement

> **IMPORTANT:** The installation script must be run within an active `conda` environment tailored for this project. If such an environment is not yet set up, follow these instructions to create and activate one:

```sh
# Create a new conda environment
conda create -n your-env-name python=3.11

# Activate the conda environment
conda activate your-env-name
```

Replace `your-env-name` with a desired name for your conda environment.

Within the activated `conda` environment, the script will install the following essential packages:

1. [`custatevec`](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html): A library for state vector manipulation in CUDA,
2. [`lbfgs`](https://github.com/chokkan/liblbfgs): An implementation of the L-BFGS optimization algorithm in C.

Make sure to read and understand the license of `cuStateVec` prior to running the installation script.

### Installing from Source

To initiate the installation, execute the following command in your terminal:

```sh
./install.sh
```

For a more detailed output during the installation process, include the `--verbose` option:

```sh
./install.sh --verbose
```

### Security Note

> **CAUTION:** Running scripts directly from the internet poses a risk. It is recommended to first download and review the script before execution:

```sh
# Review the script's contents
less install.sh

# Execute the script after review
bash install.sh
```

### Troubleshooting

> **TIP:** If you encounter any issues during installation:
>
> - Double-check the [prerequisites](#prerequisites) to ensure all necessary tools are installed.
> - Verify that the `conda` environment is activated before running the installation script.
> - Review permissions if encountering errors related to script execution. Adjusting script permissions with `chmod +x install.sh` may be required.

For further assistance, please visit our [Issues page](https://github.com/jflxb/cuaoa/issues) and describe the problem you're facing. We are committed to providing support and resolving installation queries.

## Usage

With CUAOA installed, you can start simulating QAOA. The essential steps for a simulation are:

1. Define the objective and convert it to one of the expected formats.
2. Create the handle required for interactions with the simulator.
3. Create the CUAOA class to access the simulator's functionality.

First, we define a simple MaxCut problem and use it throughout this usage guide. We will focus on creating the problem from the graph's adjacency matrix.

```python
import numpy as np
import pycuaoa

W = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

# Initialize the CUAOA class with the optimization problem.
# For further initialization options, please refer to the CUAOA interface.
sim = pycuaoa.CUAOA(W, depth=2)

# Create the handle for interactions with the simulator. We must specify the minimum 
# number of qubits required during the simulations using the given handle. 
# As the current MaxCut problem consists of 3 nodes, we set this value to 3. 
# Setting it to a higher value than needed will not affect the correctness of the 
# simulations but will consume more memory.
num_qubits = 3
handle = pycuaoa.create_handle(num_qubits)

expectation_value = sim.expectation_value(handle)  # Calculate the expectation value.
print(expectation_value)
# -1.3724677

sv = sim.statevector(handle)  # Retrieve the state vector.
print(sv)

sampleset = sim.sample(handle, num_shots=1)  # Obtain samples.
# The sampling process can also be seeded using the `seed` parameter:
# sampleset = sim.sample(handle, num_shots=1, seed=...)

# To access the samples in the sample set, use:
samples = sampleset.samples()
print(samples)
# [[False True True]]
# The objective value associated with each sample set can be accessed with:
costs = sampleset.energies()
print(costs)
# [-1]

gradients, ev = sim.gradients(handle)  # Calculate the gradients.
# Calculating the gradients will also compute the respective expectation value.
# You can also calculate the gradients for a given set of parameters with:
# gradients, ev = sim.gradients(
#     handle, 
#     betas=np.array([0.0, 1.0]), 
#     gammas=np.array([0.4, 1.3])
# )
# To access the gradients, you can use:
print(gradients.betas)
# [ 0.52877299 -0.86224096]
print(gradients.gammas)
# [-0.35863235 -0.50215632]

optimize_result = sim.optimize(
    handle
)  # Optimize the parameters using the built-in optimizer.
# This runs the optimization with default parameters. You can also control the 
# optimization by passing a `pycuaoa.LBFGSParameters` object, for example:
# optimize_result = sim.optimize(
#     handle, 
#     lbfgs_parameters=pycuaoa.LBFGSParameters(max_iterations=10)
# )
# The optimized parameters are automatically set in the `sim` object but can be 
# accessed with: 
# `optimize_result.parameters.betas` and 
# `optimize_result.parameters.gammas`.

# We can now recalculate the expectation value to see the effect:
expval_after_optimization = sim.expectation_value(handle)
print(expval_after_optimization)
# -1.99999999

# Sampling after optimization now also gives us the expected results:
sampleset_after_optimization = sim.sample(handle, num_shots=1)
print(sampleset_after_optimization.samples())
# array([[False, True, False]])
```

To use `CUAOA` with an arbitrary optimization problem, you can use the `from_map` function. For example, for the MaxCut problem used in the previous example:

```python
terms = {
    (0, 1): 2.0, (1, 2): 2.0,
    (0,): -1.0, (1,): -2.0, (2,): -1.0
}
sim = pycuaoa.CUAOA.from_map(num_qubits, terms, depth=2)
```

The remaining interactions with the `sim` object remain the same.

To release the allocated memory you can use the `.destory()` method on the handle:

```python
handle.destroy()
```


## License and Compliance

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

By using this software, you agree to comply with the licenses of all dependencies used in this project. Notably, the `cuStateVec` library has its own licensing terms which must be adhered to.

- [`cuStateVec License`](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/license.html)

## Citation

If you use this software, please cite it as follows:

```bibtex
@misc{stein2024cuaoanovelcudaacceleratedsimulation,
      title={CUAOA: A Novel CUDA-Accelerated Simulation Framework for the QAOA}, 
      author={Jonas Stein and Jonas Blenninger and David Bucher and Josef Peter Eder and Elif Çetiner and Maximilian Zorn and Claudia Linnhoff-Popien},
      year={2024},
      eprint={2407.13012},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2407.13012}, 
}
```

```bibtex
@software{blen_CUAOA_2024,
  author={Blenninger, Jonas and
                  Stein, Jonas and
                  Bucher, David and
                  Eder, Peter J. and
                  Çetiner, Elif and
                  Zorn, Maximilian and
                  Linnhoff-Popien, Claudia},
  title={{CUAOA: A Novel CUDA-Accelerated Simulation Framework for the QAOA}},
  month=jul,
  year=2024,
  publisher={Zenodo},
  version={0.1.0},
  doi={10.5281/zenodo.12750207},
  url={https://doi.org/10.5281/zenodo.12750207}
}
```
