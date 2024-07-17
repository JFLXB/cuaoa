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

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/lib.rs
from typing import Dict, Tuple
from numpy import float64, complex64, uint64
from numpy.typing import NDArray

from .pyhandle import PyHandle
from .core import (
    ParameterizationMethod,
    Parameters,
    LBFGSParameters,
    Gradients,
    Polynomial,
    RXMethod,
    SampleSet,
    OptimizeResult,
)

# CUAOA

class CUAOA:
    def __init__(
        self,
        adjacency_matrix: NDArray[float64, float64],
        /,
        depth: int | None,
        parameterization_method: ParameterizationMethod | None = ...,
        parameters: Parameters | None = ...,
        *,
        time: float | None = ...,
        seed: int | None = ...,
        block_size: int | None = ...,
        rxmethod: RXMethod | None = ...,
    ): ...
    @staticmethod
    def from_map(
        num_qubits: int,
        dictionary: Dict[Tuple[int, ...], float],
        /,
        depth: int | None,
        parameterization_method: ParameterizationMethod | None = ...,
        parameters: Parameters | None = ...,
        *,
        time: float | None = ...,
        seed: int | None = ...,
        block_size: int | None = ...,
        rxmethod: RXMethod | None = ...,
    ) -> CUAOA: ...
    def optimize(
        self, handle: PyHandle, /, lbfgs_parameters: LBFGSParameters | None = ...
    ) -> OptimizeResult: ...
    def sample(
        self, handle: PyHandle, /, num_shots: int, *, seed: int | None = ...
    ) -> SampleSet: ...
    def statevector(self, handle: PyHandle, /) -> NDArray[complex64]: ...
    def expectation_value(self, handle: PyHandle, /) -> float: ...
    # Sampling based expectation value. -> implemented directly on the sampleset.
    def gradients(
        self,
        handle: PyHandle,
        /,
        *,
        betas: NDArray[float64] | None = ...,
        gammas: NDArray[float64] | None = ...,
    ) -> tuple[Gradients, float64]: ...
    def get_depth(self) -> int: ...
    def get_parameters(self) -> Parameters: ...
    def get_betas(self) -> NDArray[float64]: ...
    def set_betas(self, betas: NDArray[float64]): ...
    def get_gammas(self) -> NDArray[float64]: ...
    def set_gammas(self, gammas: NDArray[float64]): ...
    def get_polynomial(self) -> Polynomial: ...
    def get_num_nodes(self) -> uint64: ...

# BruteFroce

class BruteFroce:
    def __init__(
        self,
        adjacency_matrix: NDArray[float64],
        /,
        *,
        block_size: int | None = ...,
    ): ...
    @staticmethod
    def from_map(
        num_qubits: int,
        dictionary: Dict[Tuple[int, ...], float],
        /,
        *,
        block_size: int | None = ...,
    ) -> CUAOA: ...
    def solve(self, handle: PyHandle, /) -> SampleSet: ...
    def sample(self, handle: PyHandle, /) -> SampleSet: ...
    def get_polynomial(self) -> Polynomial: ...
    def get_num_nodes(self) -> uint64: ...

# make Polynomial

def make_polynomial(adjacency_matrix: NDArray[float64]) -> Polynomial: ...
def make_polynomial_from_dict(
    dictionary: Dict[Tuple[int, ...], float]
) -> Polynomial: ...
