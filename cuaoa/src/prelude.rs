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

//! AOA's prelude.
//!
//! The purpose of this module is to alleviate imports of many commonly used items of the CUAOA crate
//! by adding glob import to the top of aoa heavy modules:
//!
//! ```
//! # #![allow(unused_imports)]
//! use aoa::prelude::*;
//! ````

#![allow(unused_imports)]

pub use crate::algorithms::aoa::{
    AOAAssociatedFunctions, AOAFunctions, AOAGetters, AOAInit, AOASetters,
};
pub use crate::algorithms::cuaoa::CUAOA;

pub use crate::algorithms::bf::{BFAssociatedFunctions, BFFunctions, BFGetters, BFInit, BFSetters};
pub use crate::algorithms::cubf::CUBF;

pub use crate::core::{
    AOASampleSet, BFSampleSet, Gradients, OptimizationResult, Polynomial, ToBitVec,
};

pub use crate::parameters::{Parameterization, ParameterizationMethod};

pub use crate::algorithms::polynomial::{make_polynomial, make_polynomial_from_map};
pub use crate::bindings::cuaoa::get_cuda_devices_info;
pub use crate::bindings::cuaoa::CudaDevice;
pub use crate::bindings::cuaoa::Handle;
pub use crate::random::make_randnums;
