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

use rand::{
    distributions::{Distribution, Uniform},
    rngs::StdRng,
    SeedableRng,
};
use std::f64::consts::PI;
#[cfg(feature = "py")]
extern crate pyo3;
#[cfg(feature = "py")]
use pyo3::prelude::*;

#[cfg_attr(feature = "py", pyclass(eq, eq_int))]
#[derive(Clone, PartialEq)]
pub enum ParameterizationMethod {
    StandardLinearRamp,
    Random,
}

#[derive(Debug)]
pub struct Parameterization {
    pub betas: Vec<f64>,
    pub gammas: Vec<f64>,
    pub depth: usize,
}

const DEFAULT_TIME: f64 = 0.7;

impl Parameterization {
    pub fn new(betas: Vec<f64>, gammas: Vec<f64>) -> Result<Self, String> {
        check_parameters(&betas, &gammas)?;
        Ok(Self {
            depth: betas.len(),
            betas,
            gammas,
        })
    }

    pub fn default(depth: usize, time: Option<f64>) -> Self {
        Self::new_standard_linear_ramp(depth, time)
    }

    pub fn new_from_method(
        method: ParameterizationMethod,
        depth: usize,
        time: Option<f64>,
        seed: Option<u64>,
    ) -> Self {
        match method {
            ParameterizationMethod::Random => Self::new_random(depth, seed),
            ParameterizationMethod::StandardLinearRamp => {
                Self::new_standard_linear_ramp(depth, time)
            }
        }
    }

    pub fn new_standard_linear_ramp(depth: usize, time: Option<f64>) -> Self {
        let t = time.unwrap_or_else(|| DEFAULT_TIME * depth as f64);
        let dt = t / depth as f64;
        let start = (dt / t) * (t * (1.0 - 0.5 / depth as f64));
        let end = (dt / t) * (t * 0.5 / depth as f64);
        let betas = linspace(start, end, depth);
        let gammas: Vec<f64> = betas.iter().rev().cloned().collect();
        Self {
            betas,
            gammas,
            depth,
        }
    }

    pub fn new_random(depth: usize, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(seed_value) => StdRng::seed_from_u64(seed_value),
            None => StdRng::from_entropy(),
        };
        let uniform = Uniform::new(0.0, PI);
        let betas: Vec<f64> = (0..depth).map(|_| uniform.sample(&mut rng)).collect();
        let gammas: Vec<f64> = (0..depth).map(|_| uniform.sample(&mut rng)).collect();
        Self {
            betas,
            gammas,
            depth,
        }
    }
}

pub fn check_parameters(a: &[f64], b: &[f64]) -> Result<(), String> {
    if a.len() != b.len() {
        Err("betas and gammas must have the same dimensions.".to_string())
    } else {
        Ok(())
    }
}

fn linspace(start: f64, end: f64, num_steps: usize) -> Vec<f64> {
    if num_steps == 1 {
        return vec![start];
    }

    (0..num_steps)
        .map(|i| start + (end - start) * i as f64 / (num_steps - 1) as f64)
        .collect()
}
