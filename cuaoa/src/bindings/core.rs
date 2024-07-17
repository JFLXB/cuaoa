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

use crate::core::{LBFGSLinesearchAlgorithm, RXMethod};
use num::complex::Complex64;
use std::os::raw::{c_char, c_void};

impl LBFGSLinesearchAlgorithm {
    pub fn value(&self) -> i32 {
        match self {
            LBFGSLinesearchAlgorithm::Default => 0,
            LBFGSLinesearchAlgorithm::MoreThente => 0,
            LBFGSLinesearchAlgorithm::Armijo => 1,
            LBFGSLinesearchAlgorithm::Backtracking => 2,
            LBFGSLinesearchAlgorithm::BacktrackingWolfe => 2,
            LBFGSLinesearchAlgorithm::BacktrackingStrongWolfe => 3,
        }
    }
}

impl RXMethod {
    pub fn value(&self) -> String {
        match self {
            RXMethod::Custatevec => "custatevec".to_string(),
            RXMethod::QOKit => "qokit".to_string(),
        }
    }
}

#[repr(C)]
pub struct BFSampleSetResult {
    pub size: usize,
    pub bit_strings: *const c_void,
    pub energy: f64,
}

#[repr(C)]
pub struct AOAStatevectorResult {
    pub error: *const c_char,
    pub statevector: *const c_void,
    pub size: usize,
}

#[repr(C)]
pub struct AOAGradientsResult {
    pub gamma_gradients: *const c_void,
    pub beta_gradients: *const c_void,
    pub size: usize,
    pub expectation_value: f64,
}

#[repr(C)]
pub struct AOAOptimizationResult {
    pub gammas: *const c_void,
    pub betas: *const c_void,
    pub fx_log: *const c_void,
    pub x_log: *const c_void,
    pub has_log: bool,
    pub iteration: i64,
    pub n_evals: i64,
    pub status: i32,
}

#[repr(C)]
pub struct AOASampleSetResult {
    pub size: usize,
    pub bit_strings: *const c_void,
    pub energies: *const c_void,
}

extern "C" {
    // BFSampleSetResult
    pub fn freeBFSampleSetResult(result: *mut BFSampleSetResult);
    pub fn getBFSamples(result: *const BFSampleSetResult) -> *const i64;
    // AOAStatevectorResult
    pub fn freeAOAStatevectorResult(result: *mut AOAStatevectorResult);
    pub fn getStateVectorData(result: *const AOAStatevectorResult) -> *const Complex64;
    // AOAGradientsResult
    pub fn freeAOAGradientsResult(result: *mut AOAGradientsResult);
    pub fn getGammaGradientsData(result: *const AOAGradientsResult) -> *const f64;
    pub fn getBetaGradientsData(result: *const AOAGradientsResult) -> *const f64;
    // AOAOptimizationResult
    pub fn freeAOAOptimizationResult(result: *mut AOAOptimizationResult);
    pub fn getBetas(result: *const AOAOptimizationResult) -> *const f64;
    pub fn getGammas(result: *const AOAOptimizationResult) -> *const f64;
    pub fn getFxLog(result: *const AOAOptimizationResult) -> *const f64;
    pub fn getXLog(result: *const AOAOptimizationResult) -> *const f64;
    // AOASampleSet
    pub fn freeAOASampleSet(result: *mut AOASampleSetResult);
    pub fn getAOASamples(result: *const AOASampleSetResult) -> *const i64;
    pub fn getAOAEnergies(result: *const AOASampleSetResult) -> *const f64;
}
