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
from typing import Dict, Tuple
from numpy import float64, complex64, uint64
from numpy.typing import NDArray

from . import cuaoa as cuaoa
from . import utils as utils


class PyHandle:
    def destroy(self): ...

def create_handle(
    max_nodes: int, device: int | None = ..., exact: bool | None = ...
) -> PyHandle: ...

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

# LBFGSParameters

class LBFGSParameters:
    def __init__(
        self,
        num_corrections: int | None = ...,
        epsilon: float | None = ...,
        past: int | None = ...,
        delta: float | None = ...,
        max_iterations: int | None = ...,
        linesearch: LBFGSLinesearchAlgorithm | None = ...,
        max_linesearch: int | None = ...,
        min_step: float | None = ...,
        max_step: float | None = ...,
        ftol: float | None = ...,
        wolfe: float | None = ...,
        gtol: float | None = ...,
        xtol: float | None = ...,
        orthantwise_c: float | None = ...,
        orthantwise_start: int | None = ...,
        orthantwise_end: int | None = ...,
        log: bool | None = ...,
    ) -> None: ...
    def num_corrections(self) -> int | None: ...
    def epsilon(self) -> float | None: ...
    def past(self) -> int | None: ...
    def delta(self) -> float | None: ...
    def max_iterations(self) -> int | None: ...
    def linesearch(self) -> LBFGSLinesearchAlgorithm | None: ...
    def max_linesearch(self) -> int | None: ...
    def min_step(self) -> float | None: ...
    def max_step(self) -> float | None: ...
    def ftol(self) -> float | None: ...
    def wolfe(self) -> float | None: ...
    def gtol(self) -> float | None: ...
    def xtol(self) -> float | None: ...
    def orthantwise_c(self) -> float | None: ...
    def orthantwise_start(self) -> int | None: ...
    def orthantwise_end(self) -> int | None: ...
    def log(self) -> bool | None: ...

# LBFGSLinesearchAlgorithm

class LBFGSLinesearchAlgorithm(Enum):
    Default: Any
    MoreThente: Any
    Armijo: Any
    Backtracking: Any
    BacktrackingWolfe: Any
    BacktrackingStrongWolfe: Any

# ParameterizationMethod

class ParameterizationMethod(Enum):
    StandardLinearRamp: Any
    Random: Any

# OptimizeResult

class OptimizeResult:
    @property
    def iteration(self) -> int64: ...
    @property
    def n_evals(self) -> int64: ...
    @property
    def parameters(self) -> Parameters: ...
    @property
    def fx_hist(self) -> NDArray[float64]: ...
    @property
    def betas_hist(self) -> NDArray[float64, float64]: ...
    @property
    def gammas_hist(self) -> NDArray[float64, float64]: ...

# Parameters

class Parameters:
    betas: NDArray[float64]
    gammas: NDArray[float64]

    def __init__(self, betas: NDArray[float64], gammas: NDArray[float64]): ...

# Polynomial

class Polynomial:
    keys: NDArray[uint64]
    values: NDArray[float64]

# SampleSet

class SampleSet:
    def samples(self) -> NDArray[bool_]: ...
    def energies(self) -> NDArray[float64]: ...
    def expectation_values(self) -> float: ...

# RXMethod

class RXMethod(Enum):
    Custatevec: Any
    QOKit: Any

# Gradients

class Gradients(Parameters): ...

# Sample

Sample = NewType("Sample", int64)
