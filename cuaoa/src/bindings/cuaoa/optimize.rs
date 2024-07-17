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
    bindings::core::{getFxLog, getXLog},
    core::RXMethod,
};
use core::slice;
use libc::c_void;
use std::ffi::CString;

use crate::{
    bindings::core::{freeAOAOptimizationResult, getBetas, getGammas, AOAOptimizationResult},
    core::{LBFGSParameters, OptimizationResult},
};

use super::{constants::DEFAULT_CUDA_BLOCK_SIZE, Handle};

extern "C" {
    fn optimizeCuaoaParametersWrapper(
        handle: *mut c_void,
        size: usize,
        depth: usize,
        polynomialKeys: *const usize,
        polynomialValues: *const f64,
        polynomialSize: usize,
        betas: *const f64,
        gammas: *const f64,
        m: i32,
        epsilon: f64,
        past: i32,
        delta: f64,
        max_iterations: i32,
        linesearch: i32,
        max_linesearch: i32,
        min_step: f64,
        max_step: f64,
        ftol: f64,
        wolfe: f64,
        gtol: f64,
        xtol: f64,
        orthantwise_c: f64,
        orthantwise_start: i32,
        orthantwise_end: i32,
        log: bool,
        blockSize: usize,
        rxmethod: *const i8,
    ) -> AOAOptimizationResult;
}

pub fn optimize(
    handle: &Handle,
    num_qubits: usize,
    depth: usize,
    polykeys: &[usize],
    polyvalues: &[f64],
    betas: &[f64],
    gammas: &[f64],
    lbfgs_parameters: Option<&LBFGSParameters>,
    block_size: Option<usize>,
    rxmethod: Option<RXMethod>,
) -> Result<OptimizationResult, String> {
    let lock_guard = handle.ptr().lock().unwrap();
    let c_rxmethod = CString::new(rxmethod.unwrap_or(RXMethod::Custatevec).value()).unwrap();

    let default_params = LBFGSParameters::default();
    let params = lbfgs_parameters.unwrap_or(&default_params);

    let res = unsafe {
        optimizeCuaoaParametersWrapper(
            lock_guard.as_ptr(),
            num_qubits,
            depth,
            polykeys.as_ptr(),
            polyvalues.as_ptr(),
            polykeys.len(),
            betas.as_ptr(),
            gammas.as_ptr(),
            params.num_corrections(),
            params.epsilon(),
            params.past(),
            params.delta(),
            params.max_iterations(),
            params.linesearch().value(),
            params.max_linesearch(),
            params.min_step(),
            params.max_step(),
            params.ftol(),
            params.wolfe(),
            params.gtol(),
            params.xtol(),
            params.orthantwise_c(),
            params.orthantwise_start(),
            params.orthantwise_end(),
            params.log(),
            block_size.unwrap_or(DEFAULT_CUDA_BLOCK_SIZE),
            c_rxmethod.as_ptr(),
        )
    };

    let betas_ptr = unsafe { getBetas(&res) };
    let opt_betas = unsafe { slice::from_raw_parts(betas_ptr, depth) }.to_vec();
    let gammas_ptr = unsafe { getGammas(&res) };
    let opt_gammas = unsafe { slice::from_raw_parts(gammas_ptr, depth) }.to_vec();

    let fx_log: Option<Vec<f64>>;
    let beta_log: Option<Vec<Vec<f64>>>;
    let gamma_log: Option<Vec<Vec<f64>>>;
    if res.has_log {
        fx_log =
            Some(unsafe { slice::from_raw_parts(getFxLog(&res), res.iteration as usize) }.to_vec());
        let x_log_long: Vec<f64> =
            unsafe { slice::from_raw_parts(getXLog(&res), 2 * depth * res.iteration as usize) }
                .to_vec();
        let x_log: Vec<Vec<f64>> = x_log_long.chunks(depth).map(|s| s.to_vec()).collect();

        let mut beta_log_: Vec<Vec<f64>> = Vec::new();
        let mut gamma_log_: Vec<Vec<f64>> = Vec::new();

        x_log.into_iter().enumerate().for_each(|(idx, data)| {
            if idx % 2 == 0 {
                beta_log_.push(data);
            } else {
                gamma_log_.push(data);
            }
        });
        beta_log = Some(beta_log_);
        gamma_log = Some(gamma_log_);
    } else {
        fx_log = None;
        beta_log = None;
        gamma_log = None;
    }

    let out = OptimizationResult {
        betas: opt_betas,
        gammas: opt_gammas,
        iteration: res.iteration,
        n_evals: res.n_evals,
        fx_log,
        beta_log,
        gamma_log,
    };

    unsafe {
        freeAOAOptimizationResult(&res as *const _ as *mut AOAOptimizationResult);
    }

    Ok(out)
}
