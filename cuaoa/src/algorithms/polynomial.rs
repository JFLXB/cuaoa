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

use core::f64;
use std::collections::HashMap;

use crate::{bindings, core::Polynomial};

pub fn make_polynomial(adjacency_matrix: &Vec<Vec<f64>>) -> Polynomial {
    bindings::make_polynomial(adjacency_matrix)
}

pub fn make_polynomial_from_map(data: &HashMap<Vec<usize>, f64>) -> Polynomial {
    let mut keys: Vec<usize> = vec![];
    let mut values: Vec<f64> = vec![];
    data.iter().for_each(|(ks, v)| {
        let k: usize = ks.iter().map(|e| 1 << e).sum();
        keys.push(k);
        values.push(*v);
    });
    Polynomial { keys, values }
}
