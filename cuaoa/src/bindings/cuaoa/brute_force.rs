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

use core::slice;
use libc::c_void;

use crate::{
    bindings::core::{freeBFSampleSetResult, getBFSamples, BFSampleSetResult},
    core::BFSampleSet,
};

use super::{constants::DEFAULT_CUDA_BLOCK_SIZE, Handle};

extern "C" {
    fn bruteFroceCuaoaWrapper(
        handle: *mut c_void,
        numberOfNodes: usize,
        polynomialKeys: *const usize,
        polynomialValues: *const f64,
        polynomialSize: usize,
        blockSize: usize,
    ) -> BFSampleSetResult;
}

pub fn brute_force(
    handle: &Handle,
    num_qubits: usize,
    polykeys: &[usize],
    polyvalues: &[f64],
    block_size: Option<usize>,
) -> Result<BFSampleSet, String> {
    let lock_guard = handle.ptr().lock().unwrap();
    let result = unsafe {
        bruteFroceCuaoaWrapper(
            lock_guard.as_ptr(),
            num_qubits,
            polykeys.as_ptr(),
            polyvalues.as_ptr(),
            polykeys.len(),
            block_size.unwrap_or(DEFAULT_CUDA_BLOCK_SIZE),
        )
    };
    let cost = result.energy;
    let samples = unsafe { slice::from_raw_parts(getBFSamples(&result), result.size) }.to_vec();

    unsafe {
        freeBFSampleSetResult(&result as *const _ as *mut BFSampleSetResult);
    }

    Ok(BFSampleSet {
        samples,
        energy: cost,
    })
}
