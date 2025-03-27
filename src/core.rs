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

use bitvec::vec::BitVec;
use cuaoa::prelude::AOASampleSet;
use cuaoa::prelude::BFSampleSet;
use cuaoa::prelude::Gradients;
use cuaoa::prelude::Parameterization;
use cuaoa::prelude::ParameterizationMethod;
use cuaoa::prelude::Polynomial as CuaoaPolynomial;
use numpy::IntoPyArray;
use numpy::PyArray2;
use numpy::PyArrayMethods;
use numpy::PyReadonlyArray1;
use numpy::{PyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(mapping, module = "pycuaoa", subclass)]
pub struct Polynomial {
    pub keys: Vec<usize>,
    pub values: Vec<f64>,
}

impl Polynomial {
    pub fn from_cuaoa(polynomial: &CuaoaPolynomial) -> Polynomial {
        Polynomial {
            keys: polynomial.keys.clone(),
            values: polynomial.values.clone(),
        }
    }

    pub fn into(&self) -> CuaoaPolynomial {
        CuaoaPolynomial {
            keys: self.keys.clone(),
            values: self.values.clone(),
        }
    }
}

#[pyclass(mapping, module = "pycuaoa", subclass)]
#[derive(Clone)]
pub struct Parameters {
    #[pyo3(get)]
    pub betas: Vec<f64>,
    #[pyo3(get)]
    pub gammas: Vec<f64>,
}

#[pymethods]
impl Parameters {
    #[new]
    #[pyo3(signature = (betas, gammas), text_signature="(betas, gammas)")]
    fn new(betas: PyReadonlyArray1<f64>, gammas: PyReadonlyArray1<f64>) -> PyResult<Self> {
        Ok(Parameters {
            betas: betas.to_vec()?,
            gammas: gammas.to_vec()?,
        })
    }
}

impl Parameters {
    pub fn from_gradients(gradients: Gradients) -> Parameters {
        Parameters {
            betas: gradients.beta,
            gammas: gradients.gamma,
        }
    }

    pub fn from_raw(betas: Vec<f64>, gammas: Vec<f64>) -> Self {
        Self {
            betas,  // betas.to_pyarray(py).into(),
            gammas, // : gammas.to_pyarray(py).into(),
        }
    }
}

#[pyclass(module = "pycuaoa", subclass)]
#[derive(Clone)]
pub struct OptimizeResult {
    #[pyo3(get)]
    pub iteration: i64,
    #[pyo3(get)]
    pub n_evals: i64,
    #[pyo3(get)]
    pub parameters: Parameters,
    // #[pyo3(get)]
    pub fx_hist: Option<Vec<f64>>,
    // #[pyo3(get)]
    pub betas_hist: Option<Vec<Vec<f64>>>,
    // #[pyo3(get)]
    pub gammas_hist: Option<Vec<Vec<f64>>>,
}

impl OptimizeResult {
    pub fn new(
        iter: i64,
        n_evals: i64,
        parameters: Parameters,
        fx_hist: Option<Vec<f64>>,
        betas_hist: Option<Vec<Vec<f64>>>,
        gammas_hist: Option<Vec<Vec<f64>>>,
    ) -> Self {
        Self {
            iteration: iter,
            n_evals,
            parameters,
            fx_hist,
            betas_hist,
            gammas_hist,
        }
    }
}

#[pyclass(module = "pycuaoa", subclass)]
#[derive(Clone)]
pub struct SampleSet {
    num_qubits: usize,
    sample_vecs: Vec<BitVec>,
    energies: Vec<f64>,
    pub samples: Vec<i64>,
}

#[pymethods]
impl SampleSet {
    #[pyo3(name = "samples", text_signature = "(self)")]
    pub fn samples<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray2<bool>> {
        self.sample_vecs
            .iter()
            .map(|sample| sample.iter().map(|bit| *bit).collect())
            .flat_map(|v: Vec<bool>| v.into_iter())
            .collect::<Vec<bool>>()
            .into_pyarray(py)
            .reshape([self.samples.len(), self.num_qubits])
            .unwrap()
            .to_owned()
    }
    #[pyo3(name = "energies", text_signature = "(self)")]
    pub fn energies(&self, py: Python) -> Py<PyArray1<f64>> {
        self.energies.to_pyarray(py).into()
    }

    #[pyo3(name = "expectation_value", text_signature = "(self)")]
    pub fn expectation_value(&self) -> f64 {
        self.energies.iter().sum::<f64>() / self.energies.len() as f64
    }
}

impl SampleSet {
    pub fn from_bf(num_qubits: usize, sampleset: BFSampleSet) -> SampleSet {
        SampleSet {
            num_qubits,
            energies: vec![sampleset.energy; sampleset.samples.len()],
            sample_vecs: sampleset.samples_as_bitvecs(num_qubits),
            samples: sampleset.samples,
        }
    }

    pub fn from_aoa(num_qubits: usize, sampleset: AOASampleSet) -> SampleSet {
        SampleSet {
            num_qubits,
            sample_vecs: sampleset.samples_as_bitvecs(num_qubits),
            samples: sampleset.samples,
            energies: sampleset.energies,
        }
    }
}

pub fn make_adjacency_matrix(adjacency_matrix: PyReadonlyArray2<f64>) -> Vec<Vec<f64>> {
    let adj: Vec<Vec<f64>> = adjacency_matrix
        .as_array()
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    adj
}

pub fn build_parameterization(
    depth: Option<usize>,
    parameterization_method: Option<ParameterizationMethod>,
    parameters: Option<Parameters>,
    time: Option<f64>,
    seed: Option<u64>,
) -> PyResult<Parameterization> {
    match (depth, parameterization_method, parameters, seed) {
        (Some(d), Some(m), None, _) => Ok(Parameterization::new_from_method(m, d, time, seed)),
        (Some(d), None, None, None) => Ok(Parameterization::default(d, time)),
        (Some(d), None, None, _) => Ok(Parameterization::new_random(d, seed)),
        (_, None, Some(p), _) => {
            let ba = p.betas; // .as_ref(py).readonly().to_vec()?;
            let ga = p.gammas; //.as_ref(py).readonly().to_vec()?;

            match depth {
                Some(d) => {
                    let ba_match_d = d == ba.len();
                    let ga_match_d = d == ga.len();
                    if !ba_match_d {
                        return Err(PyValueError::new_err("length of betas doesn't match depth"));
                    }
                    if !ga_match_d {
                        return Err(PyValueError::new_err(
                            "length of gammas doesn't match depth",
                        ));
                    }
                    if !ba_match_d && !ga_match_d {
                        return Err(PyValueError::new_err(
                            "length of betas and gammas don't match depth",
                        ));
                    }
                }
                None => (),
            }
            let result = Parameterization::new(ba, ga);
            match result {
                Ok(p) => Ok(p),
                Err(err) => return Err(PyRuntimeError::new_err(err)),
            }
        }
        _ => Err(PyValueError::new_err(
            "can not build parameterization from the given parameters",
        )),
    }
}

#[pyfunction]
#[pyo3(name = "make_polynomial")]
pub fn make_polynomial(adjacency_matrix: PyReadonlyArray2<f64>) -> Polynomial {
    Polynomial::from_cuaoa(&cuaoa::make_polynomial(&make_adjacency_matrix(
        adjacency_matrix,
    )))
}

#[pyfunction]
#[pyo3(name = "make_polynomial_from_dict")]
pub fn make_polynomial_from_map(dictionary: HashMap<Vec<usize>, f64>) -> Polynomial {
    Polynomial::from_cuaoa(&cuaoa::make_polynomial_from_map(&dictionary))
}
