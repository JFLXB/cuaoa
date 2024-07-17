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

use crate::core::RXMethod;
use libc::c_void;
use std::ffi::CString;
use std::slice;

use crate::{
    bindings::core::{
        freeAOAGradientsResult, getBetaGradientsData, getGammaGradientsData, AOAGradientsResult,
    },
    core::Gradients,
};

use super::{constants::DEFAULT_CUDA_BLOCK_SIZE, Handle};

extern "C" {
    fn gradientsCuaoaWrapper(
        handle: *mut c_void,
        size: usize,
        depth: usize,
        polynomialKeys: *const usize,
        polynomialValues: *const f64,
        polynomialSize: usize,
        betas: *const f64,
        gammas: *const f64,
        blockSize: usize,
        rxmethod: *const i8,
    ) -> AOAGradientsResult;
}

pub fn gradients(
    handle: &Handle,
    num_qubits: usize,
    depth: usize,
    polykeys: &[usize],
    polyvalues: &[f64],
    betas: &[f64],
    gammas: &[f64],
    block_size: Option<usize>,
    rxmethod: Option<RXMethod>,
) -> Result<Gradients, String> {
    let lock_guard = handle.ptr().lock().unwrap();
    let c_rxmethod = CString::new(rxmethod.unwrap_or(RXMethod::Custatevec).value()).unwrap();
    let result = unsafe {
        gradientsCuaoaWrapper(
            lock_guard.as_ptr(),
            num_qubits,
            depth,
            polykeys.as_ptr(),
            polyvalues.as_ptr(),
            polykeys.len(),
            betas.as_ptr(),
            gammas.as_ptr(),
            block_size.unwrap_or(DEFAULT_CUDA_BLOCK_SIZE),
            c_rxmethod.as_ptr(),
        )
    };

    let gamma_data_ptr = unsafe { getGammaGradientsData(&result) };
    let beta_data_ptr = unsafe { getBetaGradientsData(&result) };
    let gamma_gradients = unsafe { slice::from_raw_parts(gamma_data_ptr, result.size) }.to_vec();
    let beta_gradients = unsafe { slice::from_raw_parts(beta_data_ptr, result.size) }.to_vec();

    let expval = result.expectation_value;

    unsafe {
        freeAOAGradientsResult(&result as *const _ as *mut AOAGradientsResult);
    }

    Ok(Gradients {
        gamma: gamma_gradients,
        beta: beta_gradients,
        expectation_value: expval,
    })
}
