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

use cuaoa::prelude::{get_cuda_devices_info as gcdi, CudaDevice};
use pyo3::prelude::*;

#[pyclass(mapping, module = "utils", subclass)]
pub struct PyCudaDevice {
    #[pyo3(get)]
    pub id: i32,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub memory: usize,
}

trait ToPy {
    fn to_py(self) -> PyCudaDevice;
}

impl ToPy for CudaDevice {
    fn to_py(self) -> PyCudaDevice {
        PyCudaDevice {
            id: self.id,
            name: self.name,
            memory: self.memory,
        }
    }
}

#[pyfunction]
pub fn get_cuda_devices_info() -> Vec<PyCudaDevice> {
    return gcdi().into_iter().map(|cd| cd.to_py()).collect();
}
