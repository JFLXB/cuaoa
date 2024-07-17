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

use crate::{
    bindings::cuaoa::{Handle, DEFAULT_CUDA_BLOCK_SIZE},
    core::{BFSampleSet, Polynomial},
};
use std::collections::HashMap;

use super::polynomial::{make_polynomial, make_polynomial_from_map};

pub struct BF {
    pub num_qubits: usize,
    pub polynomial: Polynomial,
    pub block_size: usize,
}

pub trait BFInterface {
    fn create(bf: BF) -> Self;
    fn bf(&self) -> &BF;
    fn mutable_bf(&mut self) -> &mut BF;
    fn set_bf(&mut self, bf: BF);
}

pub trait BFSetters {
    fn set_polynomial(&mut self, num_qubits: usize, polynomial: Polynomial);
    fn set_block_size(&mut self, block_size: usize);
}

impl<T: BFInterface> BFSetters for T {
    fn set_polynomial(&mut self, num_qubits: usize, polynomial: Polynomial) {
        self.mutable_bf().num_qubits = num_qubits;
        self.mutable_bf().polynomial = polynomial;
    }

    fn set_block_size(&mut self, block_size: usize) {
        self.mutable_bf().block_size = block_size;
    }
}

pub trait BFGetters {
    fn num_qubits(&self) -> usize;
    fn polynomial(&self) -> &Polynomial;
    fn block_size(&self) -> usize;
}

impl<T: BFInterface> BFGetters for T {
    fn num_qubits(&self) -> usize {
        self.bf().num_qubits
    }

    fn polynomial(&self) -> &Polynomial {
        &self.bf().polynomial
    }

    fn block_size(&self) -> usize {
        self.bf().block_size
    }
}

pub trait BFInit {
    fn new(adjacency_matrix: &Vec<Vec<f64>>, block_size: Option<usize>) -> Self
    where
        Self: BFInit;
    fn new_from_map(
        num_qubits: usize,
        data: &HashMap<Vec<usize>, f64>,
        block_size: Option<usize>,
    ) -> Self
    where
        Self: BFInit;
}

impl<T: BFInterface> BFInit for T {
    fn new(adjacency_matrix: &Vec<Vec<f64>>, block_size: Option<usize>) -> Self
    where
        Self: BFInterface,
    {
        let num_qubits = adjacency_matrix.len();
        let polynomial = make_polynomial(adjacency_matrix);
        Self::create(BF {
            num_qubits,
            polynomial,
            block_size: block_size.unwrap_or(DEFAULT_CUDA_BLOCK_SIZE),
        })
    }

    fn new_from_map(
        num_qubits: usize,
        data: &HashMap<Vec<usize>, f64>,
        block_size: Option<usize>,
    ) -> Self
    where
        Self: BFInterface,
    {
        let polynomial = make_polynomial_from_map(data);
        Self::create(BF {
            num_qubits,
            polynomial,
            block_size: block_size.unwrap_or(DEFAULT_CUDA_BLOCK_SIZE),
        })
    }
}

pub trait BFAssociatedFunctions {
    fn solve(
        handle: &Handle,
        num_qubits: usize,
        polynomial: &Polynomial,
        block_size: Option<usize>,
    ) -> Result<BFSampleSet, String>;
}

pub trait BFFunctions {
    fn solve(&self, handle: &Handle) -> Result<BFSampleSet, String>;
}

impl<T: BFAssociatedFunctions> BFFunctions for T
where
    T: BFInterface,
{
    fn solve(&self, handle: &Handle) -> Result<BFSampleSet, String> {
        T::solve(
            handle,
            self.num_qubits(),
            self.polynomial(),
            Some(self.block_size()),
        )
    }
}
