// Copyright 2024 Jonas Blenninger
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;

use cuaoa::prelude::{BFFunctions, BFGetters, BFInit, CUBF};
use numpy::PyReadonlyArray2;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{
    core::{make_adjacency_matrix, Polynomial as PyPolynomial, SampleSet},
    handle::PyHandle,
};

#[pyclass(mapping, module = "pycuaoa", subclass)]
pub struct BruteFroce {
    inner: CUBF,
}

#[pymethods]
impl BruteFroce {
    #[new]
    #[pyo3(signature = (adjacency_matrix, block_size=None), text_signature="(adjacency_matrix, /, *, block_size=None)")]
    fn new(adjacency_matrix: PyReadonlyArray2<f64>, block_size: Option<usize>) -> Self {
        let adj = make_adjacency_matrix(adjacency_matrix);
        BruteFroce {
            inner: CUBF::new(&adj, block_size),
        }
    }

    #[staticmethod]
    #[pyo3(name = "from_map", signature=(
        num_qubits,
        dictionary,
        block_size=None
    ), text_signature="(num_qubits, dictionary, /, *, block_size=None)")]
    fn new_from_map(
        num_qubits: usize,
        dictionary: HashMap<Vec<usize>, f64>,
        block_size: Option<usize>,
    ) -> Self {
        BruteFroce {
            inner: CUBF::new_from_map(num_qubits, &dictionary, block_size),
        }
    }

    #[pyo3(name = "solve", text_signature = "(self, handle, /)")]
    pub fn solve(&self, handle: &PyHandle) -> PyResult<SampleSet> {
        let result = self.inner.solve(handle.get());
        match result {
            Ok(bf_sampleset) => Ok(SampleSet::from_bf(self.inner.num_qubits(), bf_sampleset)),
            Err(err) => Err(PyRuntimeError::new_err(err)),
        }
    }

    #[pyo3(name = "sample", text_signature = "(self, handle, /)")]
    pub fn sample(&self, handle: &PyHandle) -> PyResult<SampleSet> {
        self.solve(handle)
    }

    #[pyo3(name = "get_polynomial", text_signature = "(self)")]
    pub fn get_polynomial(&self, py: Python) -> PyPolynomial {
        PyPolynomial::from_cuaoa(py, self.inner.polynomial())
    }

    #[pyo3(name = "get_num_nodes", text_signature = "(self)")]
    pub fn get_num_nodes(&self) -> usize {
        self.inner.num_qubits()
    }
}
