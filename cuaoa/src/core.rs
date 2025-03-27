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

// #[cfg(feature = "py")]
// extern crate pyo3;
use std::usize;

use bitvec::vec::BitVec;
#[cfg(feature = "py")]
use pyo3::prelude::*;

pub fn sample_to_bitvec(sample: &i64, size: usize) -> BitVec {
    let mut bitvector = BitVec::with_capacity(size);
    for i in 0..size {
        bitvector.push((sample & (1 << i)) != 0);
    }
    bitvector
}

pub trait ToBitVec {
    fn to_bitvec(&self, size: usize) -> BitVec;
}

impl ToBitVec for i64 {
    fn to_bitvec(&self, size: usize) -> BitVec {
        sample_to_bitvec(self, size)
    }
}

pub fn samples_to_bitvec(samples: &Vec<i64>, size: usize) -> Vec<BitVec> {
    samples.iter().map(|s| s.to_bitvec(size)).collect()
}

#[derive(Debug)]
pub struct BFSampleSet {
    pub samples: Vec<i64>,
    pub energy: f64,
}

impl BFSampleSet {
    pub fn samples_as_bitvecs(&self, size: usize) -> Vec<BitVec> {
        samples_to_bitvec(&self.samples, size)
    }
}

#[derive(Debug)]
pub struct AOASampleSet {
    pub samples: Vec<i64>,
    pub energies: Vec<f64>,
}

impl AOASampleSet {
    pub fn samples_as_bitvecs(&self, size: usize) -> Vec<BitVec> {
        samples_to_bitvec(&self.samples, size)
    }
}

pub struct Gradients {
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub expectation_value: f64,
}

pub struct OptimizationResult {
    pub gammas: Vec<f64>,
    pub betas: Vec<f64>,
    pub iteration: i64,
    pub n_evals: i64,
    pub fx_log: Option<Vec<f64>>,
    pub beta_log: Option<Vec<Vec<f64>>>,
    pub gamma_log: Option<Vec<Vec<f64>>>,
}

pub struct Polynomial {
    pub keys: Vec<usize>,
    pub values: Vec<f64>,
}

impl Polynomial {
    pub fn keys(&self) -> &[usize] {
        self.keys.as_slice()
    }
    pub fn values(&self) -> &[f64] {
        self.values.as_slice()
    }
}

#[cfg_attr(feature = "py", pyclass(mapping, module = "pycuaoa", subclass))]
#[derive(Clone, Debug)]
pub struct LBFGSParameters {
    pub num_corrections: Option<i32>,
    pub epsilon: Option<f64>,
    pub past: Option<i32>,
    pub delta: Option<f64>,
    pub max_iterations: Option<i32>,
    pub linesearch: Option<LBFGSLinesearchAlgorithm>,
    pub max_linesearch: Option<i32>,
    pub min_step: Option<f64>,
    pub max_step: Option<f64>,
    pub ftol: Option<f64>,
    pub wolfe: Option<f64>,
    pub gtol: Option<f64>,
    pub xtol: Option<f64>,
    pub orthantwise_c: Option<f64>,
    pub orthantwise_start: Option<i32>,
    pub orthantwise_end: Option<i32>,
    pub log: Option<bool>,
}

impl LBFGSParameters {
    pub fn new(
        num_corrections: Option<i32>,
        epsilon: Option<f64>,
        past: Option<i32>,
        delta: Option<f64>,
        max_iterations: Option<i32>,
        linesearch: Option<LBFGSLinesearchAlgorithm>,
        max_linesearch: Option<i32>,
        min_step: Option<f64>,
        max_step: Option<f64>,
        ftol: Option<f64>,
        wolfe: Option<f64>,
        gtol: Option<f64>,
        xtol: Option<f64>,
        orthantwise_c: Option<f64>,
        orthantwise_start: Option<i32>,
        orthantwise_end: Option<i32>,
        log: Option<bool>,
    ) -> Self {
        Self {
            num_corrections,
            epsilon,
            past,
            delta,
            max_iterations,
            linesearch,
            max_linesearch,
            min_step,
            max_step,
            ftol,
            wolfe,
            gtol,
            xtol,
            orthantwise_c,
            orthantwise_start,
            orthantwise_end,
            log,
        }
    }

    pub fn default() -> Self {
        Self {
            num_corrections: None,
            epsilon: None,
            past: None,
            delta: None,
            max_iterations: None,
            linesearch: None,
            max_linesearch: None,
            min_step: None,
            max_step: None,
            ftol: None,
            wolfe: None,
            gtol: None,
            xtol: None,
            orthantwise_c: None,
            orthantwise_start: None,
            orthantwise_end: None,
            log: None,
        }
    }

    pub fn delta(&self) -> f64 {
        self.delta.unwrap_or(1e-5)
    }
    pub fn epsilon(&self) -> f64 {
        self.epsilon.unwrap_or(1e-5)
    }
    pub fn ftol(&self) -> f64 {
        self.ftol.unwrap_or(1e-4)
    }
    pub fn gtol(&self) -> f64 {
        self.gtol.unwrap_or(0.9)
    }
    pub fn linesearch(&self) -> LBFGSLinesearchAlgorithm {
        self.linesearch.unwrap_or(LBFGSLinesearchAlgorithm::Default)
    }
    pub fn num_corrections(&self) -> i32 {
        self.num_corrections.unwrap_or(6)
    }
    pub fn max_iterations(&self) -> i32 {
        self.max_iterations.unwrap_or(0)
    }
    pub fn max_linesearch(&self) -> i32 {
        self.max_linesearch.unwrap_or(40)
    }
    pub fn max_step(&self) -> f64 {
        self.max_step.unwrap_or(1e+20)
    }
    pub fn min_step(&self) -> f64 {
        self.min_step.unwrap_or(1e-20)
    }
    pub fn orthantwise_c(&self) -> f64 {
        self.orthantwise_c.unwrap_or(0.0)
    }
    pub fn orthantwise_end(&self) -> i32 {
        self.orthantwise_end.unwrap_or(-1)
    }
    pub fn orthantwise_start(&self) -> i32 {
        self.orthantwise_start.unwrap_or(0)
    }
    pub fn past(&self) -> i32 {
        self.past.unwrap_or(0)
    }
    pub fn wolfe(&self) -> f64 {
        self.wolfe.unwrap_or(0.9)
    }
    pub fn xtol(&self) -> f64 {
        self.xtol.unwrap_or(1e-16)
    }
    pub fn log(&self) -> bool {
        self.log.unwrap_or(false)
    }
}

#[cfg_attr(feature = "py", pyclass(eq, eq_int, mapping, module = "pycuaoa"))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RXMethod {
    Custatevec,
    QOKit,
}

#[cfg_attr(feature = "py", pyclass(eq, eq_int, mapping, module = "pycuaoa"))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LBFGSLinesearchAlgorithm {
    Default,                 // = 0,
    MoreThente,              // = 0,
    Armijo,                  // = 1,
    Backtracking,            // = 2,
    BacktrackingWolfe,       // = 2,
    BacktrackingStrongWolfe, // = 3,
}

#[derive(Debug)]
pub enum LBFGSStatus {
    Success = 0,
    AlreadyMinimized = 2,
    ErrUnknownerror = -1024,
    ErrLogicerror = -1023,
    ErrOutofmemory = -1022,
    ErrCanceled = -1021,
    ErrInvalidN = -1020,
    ErrInvalidNSse = -1019,
    ErrInvalidXSse = -1018,
    ErrInvalidEpsilon = -1017,
    ErrInvalidTestperiod = -1016,
    ErrInvalidDelta = -1015,
    ErrInvalidLinesearch = -1014,
    ErrInvalidMinstep = -1013,
    ErrInvalidMaxstep = -1012,
    ErrInvalidFtol = -1011,
    ErrInvalidWolfe = -1010,
    ErrInvalidGtol = -1009,
    ErrInvalidXtol = -1008,
    ErrInvalidMaxlinesearch = -1007,
    ErrInvalidOrthantwise = -1006,
    ErrInvalidOrthantwiseStart = -1005,
    ErrInvalidOrthantwiseEnd = -1004,
    ErrOutofinterval = -1003,
    ErrIncorrectTminmax = -1002,
    ErrRoundingError = -1001,
    ErrMinimumstep = -1000,
    ErrMaximumstep = -999,
    ErrMaximumlinesearch = -998,
    ErrMaximumiteration = -997,
    ErrWidthtoosmall = -996,
    ErrInvalidparameters = -995,
    ErrIncreasegradient = -994,
}

impl From<i32> for LBFGSStatus {
    fn from(value: i32) -> Self {
        match value {
            0 => return LBFGSStatus::Success,
            2 => return LBFGSStatus::AlreadyMinimized,
            -1024 => return LBFGSStatus::ErrUnknownerror,
            -1023 => return LBFGSStatus::ErrLogicerror,
            -1022 => return LBFGSStatus::ErrOutofmemory,
            -1021 => return LBFGSStatus::ErrCanceled,
            -1020 => return LBFGSStatus::ErrInvalidN,
            -1019 => return LBFGSStatus::ErrInvalidNSse,
            -1018 => return LBFGSStatus::ErrInvalidXSse,
            -1017 => return LBFGSStatus::ErrInvalidEpsilon,
            -1016 => return LBFGSStatus::ErrInvalidTestperiod,
            -1015 => return LBFGSStatus::ErrInvalidDelta,
            -1014 => return LBFGSStatus::ErrInvalidLinesearch,
            -1013 => return LBFGSStatus::ErrMinimumstep,
            -1012 => return LBFGSStatus::ErrMaximumstep,
            -1011 => return LBFGSStatus::ErrInvalidFtol,
            -1010 => return LBFGSStatus::ErrInvalidWolfe,
            -1009 => return LBFGSStatus::ErrInvalidGtol,
            -1008 => return LBFGSStatus::ErrInvalidXtol,
            -1007 => return LBFGSStatus::ErrInvalidMaxlinesearch,
            -1006 => return LBFGSStatus::ErrInvalidOrthantwise,
            -1005 => return LBFGSStatus::ErrInvalidOrthantwiseStart,
            -1004 => return LBFGSStatus::ErrInvalidOrthantwiseEnd,
            -1003 => return LBFGSStatus::ErrOutofinterval,
            -1002 => return LBFGSStatus::ErrIncorrectTminmax,
            -1001 => return LBFGSStatus::ErrRoundingError,
            -1000 => return LBFGSStatus::ErrMinimumstep,
            -999 => return LBFGSStatus::ErrMaximumstep,
            -998 => return LBFGSStatus::ErrMaximumlinesearch,
            -997 => return LBFGSStatus::ErrMaximumiteration,
            -996 => return LBFGSStatus::ErrWidthtoosmall,
            -995 => return LBFGSStatus::ErrInvalidparameters,
            -994 => return LBFGSStatus::ErrIncreasegradient,
            _ => return LBFGSStatus::ErrUnknownerror,
        }
    }
}
