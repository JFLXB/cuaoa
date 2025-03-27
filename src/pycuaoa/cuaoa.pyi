# Copyright 2024 Jonas Blenninger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numpy import float64
from numpy.typing import NDArray

from . import PyHandle
from . import LBFGSParameters, Polynomial, OptimizeResult, RXMethod

# expectation value cuaoa

def expectation_value(
    handle: PyHandle,
    /,
    num_qubits: int,
    depth: int,
    polynomial: Polynomial,
    betas: NDArray[float64],
    gammas: NDArray[float64],
    block_size: int | None = ...,
    rxmethod: RXMethod | None = ...,
) -> tuple: ...

# optimize cuaoa

def optimize(
    handle: PyHandle,
    /,
    num_qubits: int,
    depth: int,
    polynomial: Polynomial,
    betas: NDArray[float64],
    gammas: NDArray[float64],
    lbfgs_parameters: LBFGSParameters | None = ...,
    block_size: int | None = ...,
    rxmethod: RXMethod | None = ...,
) -> OptimizeResult: ...
