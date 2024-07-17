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
    bindings::{self, cuaoa::Handle},
    core::{BFSampleSet, Polynomial},
};

use super::bf::{BFAssociatedFunctions, BFInterface, BF};

pub struct CUBF {
    bf: BF,
}

impl BFInterface for CUBF {
    fn create(bf: BF) -> Self {
        CUBF { bf }
    }

    fn bf(&self) -> &BF {
        &self.bf
    }

    fn mutable_bf(&mut self) -> &mut BF {
        &mut self.bf
    }

    fn set_bf(&mut self, bf: BF) {
        self.bf = bf;
    }
}

impl BFAssociatedFunctions for CUBF {
    fn solve(
        handle: &Handle,
        num_qubits: usize,
        polynomial: &Polynomial,
        block_size: Option<usize>,
    ) -> Result<BFSampleSet, String> {
        bindings::cuaoa::brute_force(
            handle,
            num_qubits,
            polynomial.keys(),
            polynomial.values(),
            block_size,
        )
    }
}
