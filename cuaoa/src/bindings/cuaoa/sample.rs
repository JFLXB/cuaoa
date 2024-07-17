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

use crate::bindings::core::{freeAOASampleSet, getAOAEnergies, getAOASamples, AOASampleSetResult};
use crate::core::AOASampleSet;

use super::{constants::DEFAULT_CUDA_BLOCK_SIZE, Handle};

extern "C" {
    fn sampleCuaoaWrapper(
        handle: *mut c_void,
        size: usize,
        depth: usize,
        polynomialKeys: *const usize,
        polynomialValues: *const f64,
        polynomialSize: usize,
        betas: *const f64,
        gammas: *const f64,
        maxShots: u32,
        numShots: u32,
        randnums: *const f64,
        blockSize: usize,
        rxmethod: *const i8,
    ) -> AOASampleSetResult;
}

pub fn sample(
    handle: &Handle,
    num_qubits: usize,
    depth: usize,
    polykeys: &[usize],
    polyvalues: &[f64],
    betas: &[f64],
    gammas: &[f64],
    max_shots: u32,
    num_shots: u32,
    randnums: &[f64],
    block_size: Option<usize>,
    rxmethod: Option<RXMethod>,
) -> Result<AOASampleSet, String> {
    let result = unsafe {
        let lock_guard = handle.ptr().lock().unwrap();
        let c_rxmethod = CString::new(rxmethod.unwrap_or(RXMethod::Custatevec).value()).unwrap();
        sampleCuaoaWrapper(
            lock_guard.as_ptr(),
            num_qubits,
            depth,
            polykeys.as_ptr(),
            polyvalues.as_ptr(),
            polykeys.len(),
            betas.as_ptr(),
            gammas.as_ptr(),
            max_shots,
            num_shots,
            randnums.as_ptr(),
            block_size.unwrap_or(DEFAULT_CUDA_BLOCK_SIZE),
            c_rxmethod.as_ptr(),
        )
    };

    let samples =
        unsafe { slice::from_raw_parts(getAOASamples(&result), num_shots as usize) }.to_vec();
    let energies =
        unsafe { slice::from_raw_parts(getAOAEnergies(&result), num_shots as usize) }.to_vec();

    unsafe {
        freeAOASampleSet(&result as *const _ as *mut AOASampleSetResult);
    }

    Ok(AOASampleSet { samples, energies })
}
