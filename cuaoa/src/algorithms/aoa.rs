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

use num::complex::Complex64;

use crate::{
    bindings::cuaoa::{Handle, DEFAULT_CUDA_BLOCK_SIZE},
    core::{AOASampleSet, Gradients, LBFGSParameters, OptimizationResult, Polynomial, RXMethod},
    parameters::Parameterization,
    random::make_randnums,
};

use super::polynomial::{make_polynomial, make_polynomial_from_map};

pub struct AOA {
    pub num_qubits: usize,
    pub polynomial: Polynomial,
    pub betas: Vec<f64>,
    pub gammas: Vec<f64>,
    pub depth: usize,
    pub block_size: usize,
    pub rxmethod: Option<RXMethod>,
}

pub trait AOAInterface {
    fn create(aoa: AOA) -> Self;
    fn aoa(&self) -> &AOA;
    fn mutable_aoa(&mut self) -> &mut AOA;
    fn set_aoa(&mut self, aoa: AOA);
}

pub trait AOASetters {
    // fn set_num_qubits(&mut self, num_qubits: usize);
    fn set_polynomial(&mut self, num_qubits: usize, polynomial: Polynomial);
    fn set_betas(&mut self, betas: Vec<f64>);
    fn set_gammas(&mut self, gammas: Vec<f64>);
    fn set_parameters(&mut self, parameters: Parameterization);
    fn set_block_size(&mut self, block_size: usize);
}

impl<T: AOAInterface> AOASetters for T {
    fn set_polynomial(&mut self, num_qubits: usize, polynomial: Polynomial) {
        self.mutable_aoa().num_qubits = num_qubits;
        self.mutable_aoa().polynomial = polynomial;
    }

    fn set_betas(&mut self, betas: Vec<f64>) {
        self.mutable_aoa().betas = betas;
    }

    fn set_gammas(&mut self, gammas: Vec<f64>) {
        self.mutable_aoa().gammas = gammas;
    }

    fn set_parameters(&mut self, parameters: Parameterization) {
        self.mutable_aoa().betas = parameters.betas;
        self.mutable_aoa().gammas = parameters.gammas;
    }

    fn set_block_size(&mut self, block_size: usize) {
        self.mutable_aoa().block_size = block_size;
    }
}

pub trait AOAGetters {
    fn num_qubits(&self) -> usize;
    fn polynomial(&self) -> &Polynomial;
    fn betas(&self) -> &[f64];
    fn gammas(&self) -> &[f64];
    fn depth(&self) -> usize;
    fn block_size(&self) -> usize;
    fn rxmethod(&self) -> Option<RXMethod>;
    // fn handle(&self) -> &Handle;
}

impl<T: AOAInterface> AOAGetters for T {
    fn num_qubits(&self) -> usize {
        self.aoa().num_qubits
    }

    fn polynomial(&self) -> &Polynomial {
        &self.aoa().polynomial
    }

    fn betas(&self) -> &[f64] {
        self.aoa().betas.as_slice()
    }

    fn gammas(&self) -> &[f64] {
        self.aoa().gammas.as_slice()
    }

    fn depth(&self) -> usize {
        self.aoa().depth
    }

    fn block_size(&self) -> usize {
        self.aoa().block_size
    }

    fn rxmethod(&self) -> Option<RXMethod> {
        self.aoa().rxmethod
    }
}

pub trait AOAInit {
    fn new(
        adjacency_matrix: &Vec<Vec<f64>>,
        parameterization: Parameterization,
        block_size: Option<usize>,
        rxmethod: Option<RXMethod>,
    ) -> Self
    where
        Self: AOAInit;
    fn new_from_map(
        num_qubits: usize,
        data: &HashMap<Vec<usize>, f64>,
        parameterization: Parameterization,
        block_size: Option<usize>,
        rxmethod: Option<RXMethod>,
    ) -> Self
    where
        Self: AOAInit;
}

impl<T: AOAInterface> AOAInit for T {
    fn new(
        adjacency_matrix: &Vec<Vec<f64>>,
        parameterization: Parameterization,
        block_size: Option<usize>,
        rxmethod: Option<RXMethod>,
    ) -> Self
    where
        Self: AOAInterface,
    {
        let num_qubits = adjacency_matrix.len();
        let polynomial = make_polynomial(adjacency_matrix);
        Self::create(AOA {
            num_qubits,
            polynomial,
            betas: parameterization.betas,
            gammas: parameterization.gammas,
            depth: parameterization.depth,
            block_size: block_size.unwrap_or(DEFAULT_CUDA_BLOCK_SIZE),
            rxmethod,
        })
    }

    fn new_from_map(
        num_qubits: usize,
        data: &HashMap<Vec<usize>, f64>,
        parameterization: Parameterization,
        block_size: Option<usize>,
        rxmethod: Option<RXMethod>,
    ) -> Self
    where
        Self: AOAInterface,
    {
        let polynomial = make_polynomial_from_map(data);
        Self::create(AOA {
            num_qubits,
            polynomial,
            betas: parameterization.betas,
            gammas: parameterization.gammas,
            depth: parameterization.depth,
            block_size: block_size.unwrap_or(DEFAULT_CUDA_BLOCK_SIZE),
            rxmethod,
        })
    }
}

pub trait AOAAssociatedFunctions {
    fn statevector(
        handle: &Handle,
        num_qubits: usize,
        depth: usize,
        polynomial: &Polynomial,
        betas: &[f64],
        gammas: &[f64],
        block_size: Option<usize>,
        rxmethod: Option<RXMethod>,
    ) -> Result<Vec<Complex64>, String>;
    fn sample(
        handle: &Handle,
        num_qubits: usize,
        depth: usize,
        polynomial: &Polynomial,
        betas: &[f64],
        gammas: &[f64],
        num_shots: u32,
        randnums: &[f64],
        block_size: Option<usize>,
        rxmethod: Option<RXMethod>,
    ) -> Result<AOASampleSet, String>;
    // red: TODO: Sample from AOASampleSet.
    fn optimize(
        handle: &Handle,
        num_qubits: usize,
        depth: usize,
        polynomial: &Polynomial,
        betas: &[f64],
        gammas: &[f64],
        lbfgs_parameters: Option<&LBFGSParameters>,
        block_size: Option<usize>,
        rxmethod: Option<RXMethod>,
    ) -> Result<OptimizationResult, String>;
    fn gradients(
        handle: &Handle,
        num_qubits: usize,
        depth: usize,
        polynomial: &Polynomial,
        betas: &[f64],
        gammas: &[f64],
        block_size: Option<usize>,
        rxmethod: Option<RXMethod>,
    ) -> Result<Gradients, String>;
    fn expectation_value(
        handle: &Handle,
        num_qubits: usize,
        depth: usize,
        polynomial: &Polynomial,
        betas: &[f64],
        gammas: &[f64],
        block_size: Option<usize>,
        rxmethod: Option<RXMethod>,
    ) -> Result<f64, String>;
}

pub trait AOAFunctions {
    fn statevector(&self, handle: &Handle) -> Result<Vec<Complex64>, String>;
    fn sample(
        &self,
        handle: &Handle,
        num_shots: u32,
        randnums: Option<&[f64]>,
        seed: Option<u64>,
    ) -> Result<AOASampleSet, String>;
    // red: TODO: Sample from AOASampleSet.
    fn optimize(
        &mut self,
        handle: &Handle,
        lbfgs_parameters: Option<&LBFGSParameters>,
    ) -> Result<OptimizationResult, String>;
    fn gradients(
        &self,
        handle: &Handle,
        betas: Option<&[f64]>,
        gammas: Option<&[f64]>,
    ) -> Result<Gradients, String>;
    fn expectation_value(&self, handle: &Handle) -> Result<f64, String>;
}

impl<T: AOAAssociatedFunctions> AOAFunctions for T
where
    T: AOAInterface,
{
    fn statevector(&self, handle: &Handle) -> Result<Vec<Complex64>, String> {
        T::statevector(
            handle,
            self.num_qubits(),
            self.depth(),
            self.polynomial(),
            self.betas(),
            self.gammas(),
            Some(self.block_size()),
            self.rxmethod(),
        )
    }

    fn sample(
        &self,
        handle: &Handle,
        num_shots: u32,
        randnums: Option<&[f64]>,
        seed: Option<u64>,
    ) -> Result<AOASampleSet, String> {
        let generated: Vec<f64>;
        let rands: &[f64] = match randnums {
            Some(nums) => nums,
            None => {
                generated = make_randnums(num_shots, seed);
                generated.as_slice()
            }
        };
        T::sample(
            handle,
            self.num_qubits(),
            self.depth(),
            self.polynomial(),
            self.betas(),
            self.gammas(),
            num_shots,
            rands,
            Some(self.block_size()),
            self.rxmethod(),
        )
    }

    fn optimize(
        &mut self,
        handle: &Handle,
        lbfgs_parameters: Option<&LBFGSParameters>,
    ) -> Result<OptimizationResult, String> {
        let res = T::optimize(
            handle,
            self.num_qubits(),
            self.depth(),
            self.polynomial(),
            self.betas(),
            self.gammas(),
            lbfgs_parameters,
            Some(self.block_size()),
            self.rxmethod(),
        )?;

        self.set_betas(res.betas.clone());
        self.set_gammas(res.gammas.clone());

        Ok(res)
    }

    fn gradients(
        &self,
        handle: &Handle,
        betas: Option<&[f64]>,
        gammas: Option<&[f64]>,
    ) -> Result<Gradients, String> {
        T::gradients(
            handle,
            self.num_qubits(),
            self.depth(),
            self.polynomial(),
            betas.unwrap_or(self.betas()),
            gammas.unwrap_or(self.gammas()),
            Some(self.block_size()),
            self.rxmethod(),
        )
    }

    fn expectation_value(&self, handle: &Handle) -> Result<f64, String> {
        T::expectation_value(
            handle,
            self.num_qubits(),
            self.depth(),
            self.polynomial(),
            self.betas(),
            self.gammas(),
            Some(self.block_size()),
            self.rxmethod(),
        )
    }
}
