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

use cuaoa::{
    algorithms::aoa::{AOAAssociatedFunctions, AOAFunctions, AOAGetters, AOAInit},
    core::{LBFGSParameters, RXMethod},
    prelude::{make_randnums, AOASetters, ParameterizationMethod, CUAOA as CUAOAInner},
};
use numpy::{Complex64, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::{
    core::{
        build_parameterization, make_adjacency_matrix, OptimizeResult, Parameters,
        Polynomial as PyPolynomial, SampleSet,
    },
    handle::PyHandle,
};

#[pyclass(mapping, module = "pycuaoa", subclass)]
pub struct CUAOA {
    inner: CUAOAInner,
}

#[pymethods]
impl CUAOA {
    #[new]
    #[pyo3(signature=(
        adjacency_matrix,
        depth,
        parameterization_method=None,
        parameters=None,
        time=None,
        seed=None,
        block_size=None,
        rxmethod=None,
    ), text_signature="(adjacency_matrix, /, depth, parameterization_method=None, parameters=None, *, time=None, seed=None, block_size=None, rxmethod=None)")]
    fn new(
        adjacency_matrix: PyReadonlyArray2<f64>,
        depth: Option<usize>,
        parameterization_method: Option<ParameterizationMethod>,
        parameters: Option<Parameters>,
        time: Option<f64>,
        seed: Option<u64>,
        block_size: Option<usize>,
        rxmethod: Option<RXMethod>,
    ) -> PyResult<Self> {
        let param = build_parameterization(depth, parameterization_method, parameters, time, seed)?;
        let adj = make_adjacency_matrix(adjacency_matrix);
        let inner = CUAOAInner::new(&adj, param, block_size, rxmethod);
        Ok(CUAOA { inner })
    }

    #[staticmethod]
    #[pyo3(name = "from_map", signature=(
        num_qubits,
        dictionary,
        depth,
        parameterization_method=None,
        parameters=None,
        time=None,
        seed=None,
        block_size=None,
        rxmethod=None,
    ), text_signature="(num_qubits, dictionary, /, depth, parameterization_method=None, parameters=None, *, time=None, seed=None, block_size=None, rxmethod=None)")]
    fn new_from_map(
        num_qubits: usize,
        dictionary: HashMap<Vec<usize>, f64>,
        depth: Option<usize>,
        parameterization_method: Option<ParameterizationMethod>,
        parameters: Option<Parameters>,
        time: Option<f64>,
        seed: Option<u64>,
        block_size: Option<usize>,
        rxmethod: Option<RXMethod>,
    ) -> PyResult<Self> {
        let param = build_parameterization(depth, parameterization_method, parameters, time, seed)?;
        let inner = CUAOAInner::new_from_map(num_qubits, &dictionary, param, block_size, rxmethod);
        Ok(CUAOA { inner })
    }

    #[pyo3(name = "optimize", signature = (handle, lbfgs_parameters=None), text_signature = "(self, handle, /, lbfgs_parameters=None)")]
    pub fn optimize(
        &mut self,
        handle: &PyHandle,
        lbfgs_parameters: Option<&LBFGSParameters>,
    ) -> PyResult<OptimizeResult> {
        let result = self.inner.optimize(handle.get(), lbfgs_parameters);
        match result {
            Ok(res) => {
                let it = res.iteration;
                let n_evals = res.n_evals;
                let params = Parameters::from_raw(res.betas, res.gammas);
                Ok(OptimizeResult::new(
                    it,
                    n_evals,
                    params,
                    res.fx_log,
                    res.beta_log,
                    res.gamma_log,
                ))
            }
            Err(err) => Err(PyRuntimeError::new_err(err)),
        }
    }

    #[pyo3(name = "sample", signature = (handle, num_shots, seed=None), text_signature = "(self, handle, /, num_shots, *, seed)")]
    pub fn sample(
        &self,
        handle: &PyHandle,
        num_shots: u32,
        seed: Option<u64>,
    ) -> PyResult<SampleSet> {
        let randnums = make_randnums(num_shots, seed);
        let result = self
            .inner
            .sample(handle.get(), num_shots, Some(randnums.as_slice()), seed);
        match result {
            Ok(sampleset) => Ok(SampleSet::from_aoa(self.inner.num_qubits(), sampleset)),
            Err(err) => Err(PyRuntimeError::new_err(err)),
        }
    }

    #[pyo3(name = "statevector", text_signature = "(self, handle, /)")]
    pub fn statevector(&self, py: Python, handle: &PyHandle) -> PyResult<Py<PyArray1<Complex64>>> {
        let result = self.inner.statevector(handle.get());
        match result {
            Ok(vec) => Ok(vec.to_pyarray(py).into()),
            Err(err) => Err(PyRuntimeError::new_err(err)),
        }
    }

    #[pyo3(name = "expectation_value", text_signature = "(self, handle, /)")]
    pub fn expectation_value(&mut self, handle: &PyHandle) -> PyResult<f64> {
        let expval = self.inner.expectation_value(handle.get());
        match expval {
            Ok(ev) => Ok(ev),
            Err(err) => Err(PyRuntimeError::new_err(err)),
        }
    }

    #[pyo3(name = "get_depth", text_signature = "(self)")]
    pub fn get_depth(&self) -> usize {
        self.inner.depth()
    }

    #[pyo3(name = "get_parameters", text_signature = "(self)")]
    pub fn get_parameters(&self) -> Parameters {
        Parameters {
            betas: self.inner.betas().to_vec(),
            gammas: self.inner.gammas().to_vec(),
        }
    }

    #[pyo3(name = "get_betas", text_signature = "(self)")]
    pub fn get_betas(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.betas().to_pyarray(py).into()
    }

    #[pyo3(name = "set_betas", text_signature = "(self, betas)")]
    pub fn set_betas(&mut self, betas: PyReadonlyArray1<f64>) {
        self.inner.set_betas(betas.to_vec().unwrap())
    }

    #[pyo3(name = "get_gammas", text_signature = "(self)")]
    pub fn get_gammas(&self, py: Python) -> Py<PyArray1<f64>> {
        self.inner.gammas().to_pyarray(py).into()
    }

    #[pyo3(name = "set_gammas", text_signature = "(self, gammas)")]
    pub fn set_gammas(&mut self, gammas: PyReadonlyArray1<f64>) {
        self.inner.set_gammas(gammas.to_vec().unwrap())
    }

    #[pyo3(name = "get_num_nodes", text_signature = "(self)")]
    pub fn get_num_nodes(&self) -> usize {
        self.inner.num_qubits()
    }

    #[pyo3(name = "get_polynomial", text_signature = "(self)")]
    pub fn get_polynomial(&self) -> PyPolynomial {
        PyPolynomial::from_cuaoa(self.inner.polynomial())
    }

    #[pyo3(name = "gradients", signature = (handle, betas = None, gammas = None), text_signature = "(self, handle, /, *, betas, gammas)")]
    pub fn grads(
        &self,
        handle: &PyHandle,
        betas: Option<PyReadonlyArray1<f64>>,
        gammas: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<(Parameters, f64)> {
        let result = self.inner.gradients(
            handle.get(),
            self.unwrap_checked(betas)?.as_deref(),
            self.unwrap_checked(gammas)?.as_deref(),
        );
        match result {
            Ok(gradients) => {
                let expval = gradients.expectation_value;
                Ok((Parameters::from_gradients(gradients), expval))
            }
            Err(err) => Err(PyRuntimeError::new_err(err)),
        }
    }
}

impl CUAOA {
    fn unwrap_checked(&self, a: Option<PyReadonlyArray1<f64>>) -> PyResult<Option<Vec<f64>>> {
        match a {
            Some(v) => {
                let o = v.to_vec().unwrap();
                if o.len() != self.inner.depth() {
                    Err(PyValueError::new_err(
                        "length of parameters must be equal to depth",
                    ))
                } else {
                    Ok(Some(o))
                }
            }
            None => Ok(None),
        }
    }
}

#[pyfunction]
#[pyo3(name = "expectation_value", signature = (handle, num_qubits, depth, polynomial, betas, gammas, block_size=None, rxmethod=None), text_signature = "(handle, /, num_qubits, depth, polynomial, betas, gammas, block_size=None, rxmethod=None)")]
pub fn expectation_value(
    handle: &PyHandle,
    num_qubits: usize,
    depth: usize,
    polynomial: &PyPolynomial,
    betas: PyReadonlyArray1<f64>,
    gammas: PyReadonlyArray1<f64>,
    block_size: Option<usize>,
    rxmethod: Option<RXMethod>,
) -> f64 {
    <CUAOAInner as AOAAssociatedFunctions>::expectation_value(
        handle.get(),
        num_qubits,
        depth,
        &polynomial.into(),
        betas.as_slice().unwrap(),
        gammas.as_slice().unwrap(),
        block_size,
        rxmethod,
    )
    .unwrap()
}

#[pyfunction]
#[pyo3(name = "optimize", signature = (handle, num_qubits, depth, polynomial, betas, gammas, lbfgs_parameters=None, block_size=None, rxmethod=None), text_signature = "(handle, /, num_qubits, depth, polynomial, betas, gammas, lbfgs_parameters=None, block_size=None, rxmethod=None)")]
pub fn optimize(
    handle: &PyHandle,
    num_qubits: usize,
    depth: usize,
    polynomial: &PyPolynomial,
    betas: PyReadonlyArray1<f64>,
    gammas: PyReadonlyArray1<f64>,
    lbfgs_parameters: Option<&LBFGSParameters>,
    block_size: Option<usize>,
    rxmethod: Option<RXMethod>,
) -> OptimizeResult {
    let optresult = <CUAOAInner as AOAAssociatedFunctions>::optimize(
        handle.get(),
        num_qubits,
        depth,
        &polynomial.into(),
        betas.as_slice().unwrap(),
        gammas.as_slice().unwrap(),
        lbfgs_parameters,
        block_size,
        rxmethod,
    )
    .unwrap();
    let it = optresult.iteration;
    let n_evals = optresult.n_evals;
    let params = Parameters::from_raw(optresult.betas, optresult.gammas);
    OptimizeResult::new(
        it,
        n_evals,
        params,
        optresult.fx_log,
        optresult.beta_log,
        optresult.gamma_log,
    )
}
