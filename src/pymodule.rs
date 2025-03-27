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

use cuaoa::core::{LBFGSLinesearchAlgorithm, LBFGSParameters, RXMethod};
use cuaoa::parameters::ParameterizationMethod;
use pyo3::prelude::*;

use crate::brute_force::BruteFroce;
use crate::core::{
    make_polynomial, make_polynomial_from_map, OptimizeResult, Parameters, Polynomial, SampleSet,
};
use crate::cuaoa::{expectation_value, optimize, CUAOA};
use crate::handle::{create_handle, PyHandle};
use crate::utils::{get_cuda_devices_info, PyCudaDevice};

fn register_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let utils_module = PyModule::new(m.py(), "utils")?;
    utils_module.add_function(wrap_pyfunction!(get_cuda_devices_info, &utils_module)?)?;
    utils_module.add_class::<PyCudaDevice>()?;

    m.add_submodule(&utils_module)?;
    Ok(())
}

fn register_cuaoa(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let cuaoa_module = PyModule::new(m.py(), "cuaoa")?;
    cuaoa_module.add_function(wrap_pyfunction!(expectation_value, &cuaoa_module)?)?;
    cuaoa_module.add_function(wrap_pyfunction!(optimize, &cuaoa_module)?)?;
    m.add_submodule(&cuaoa_module)?;
    Ok(())
}

pub fn register_pycuaoa(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LBFGSParameters>()?;
    m.add_class::<LBFGSLinesearchAlgorithm>()?;
    m.add_class::<ParameterizationMethod>()?;
    m.add_class::<OptimizeResult>()?;
    m.add_class::<Parameters>()?;
    m.add_class::<Polynomial>()?;
    m.add_class::<SampleSet>()?;
    m.add_class::<CUAOA>()?;
    m.add_class::<BruteFroce>()?;
    m.add_class::<RXMethod>()?;

    m.add_function(wrap_pyfunction!(make_polynomial, m)?)?;
    m.add_function(wrap_pyfunction!(make_polynomial_from_map, m)?)?;

    m.add_class::<PyHandle>()?;
    m.add_function(wrap_pyfunction!(create_handle, m)?)?;

    register_cuaoa(m)?;
    register_utils(m)?;
    Ok(())
}
