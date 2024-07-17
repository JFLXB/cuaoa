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

use crate::core::Polynomial;

#[repr(C)]
struct PolynomialResult {
    keys: *const usize,
    keys_size: usize,
    values: *const f64,
    values_size: usize,
}

extern "C" {
    fn makePolynomialsWrapper(flat: *const f64, dimension: usize) -> PolynomialResult;
    fn freePolynomialResult(result: PolynomialResult);
}

pub fn make_polynomial(adjacency_matrix: &Vec<Vec<f64>>) -> Polynomial {
    let dimension = adjacency_matrix.len();
    let mut flat_array = Vec::with_capacity(dimension * dimension);
    for row in adjacency_matrix {
        flat_array.extend(row);
    }

    let result = unsafe { makePolynomialsWrapper(flat_array.as_ptr(), dimension) };

    let keys = unsafe { slice::from_raw_parts(result.keys, result.keys_size) }.to_vec();
    let values = unsafe { slice::from_raw_parts(result.values, result.values_size) }.to_vec();

    unsafe {
        freePolynomialResult(result);
    }

    Polynomial { keys, values }
}
