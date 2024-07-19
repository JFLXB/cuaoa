# CUAOA: A Novel CUDA-Accelerated Simulation Framework for the QAOA

[![arXiv](https://img.shields.io/badge/arXiv-2407.13012-b31b1b.svg)](https://arxiv.org/abs/2407.13012)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12750207.svg)](https://doi.org/10.5281/zenodo.12750207)

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Using with Cargo](#using-with-cargo)
   - [Security Note](#security-note)
   - [Troubleshooting](#troubleshooting)
3. [License and Compliance](#license-and-compliance)
4. [Disclaimer](#disclaimer)
5. [Citation](#citation)

## Overview

CUAOA is a GPU accelerated QAOA simulation framework utilizing the [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit). This framework offers a complete interface for QAOA simulations, enabling the calculation of (exact) expectation values, direct access to the statevector, fast sampling, and high-performance optimization methods using an advanced state-of-the-art gradient calculation technique. The framework is designed for use in Python and Rust, providing flexibility for integration into a wide range of applications, including those requiring fast algorithm implementations leveraging QAOA at its core.

## Installation

To include this project in your Rust project, follow the instructions below.

### Prerequisites

Before proceeding with the installation, ensure the following tools are installed on your system:

- [Rust and Cargo](https://www.rust-lang.org/tools/install): Required to compile the Rust libraries.
- [g++](https://gcc.gnu.org/): Required to compile the C++ library.
- [CUDA and nvcc](https://developer.nvidia.com/cuda-downloads): Required for CUDA-accelerated computations and the compilation.
- [git](https://git-scm.com/): Required for cloning the repository.
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html): A crucial tool for environment and package management.

Additionally, the `cuStateVec` and `libLBFGS` libraries need to be installed. The installation script handles these dependencies, but ensure you read and understand their licensing terms, as well as the licensing terms of the other dependencies:

- [`cuStateVec License`](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/license.html)

> **NOTE:** The build process currently requires using conda for the installation, and a conda environment is required. This dependency on conda is subject to change in a future release of the library.

### Using with Cargo

To use the CUAOA Rust subpackage in your project, include it as a dependency in your `Cargo.toml` file:

```toml
[dependencies]
cuaoa = { git = "https://github.com/jflxb/cuaoa", branch = "main" }
```

This will pull the CUAOA package from the specified GitHub repository.

### Security Note

> **CAUTION:** Including external dependencies from GitHub or other sources poses a risk. It is recommended to review the source code of the dependencies to ensure they meet your security and functionality requirements.

### Troubleshooting

> **TIP:** If you encounter any issues during installation:
>
> - Double-check the [prerequisites](#prerequisites) to ensure all necessary tools are installed.
> - Verify that the `cargo` command is available and properly configured in your development environment.
> - Review permissions if encountering errors related to script execution or repository access.

For further assistance, please visit our [Issues page](https://github.com/jflxb/cuaoa/issues) and describe the problem you're facing. We are committed to providing support and resolving installation queries.

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
