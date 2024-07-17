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

use cuaoa::prelude::Handle;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pyclass(mapping, module = "pycuaoa")]
pub struct PyHandle {
    is_destroyed: bool,
    handle: Handle,
}

unsafe impl Send for PyHandle {}

impl PyHandle {
    pub fn get(&self) -> &Handle {
        &self.handle
    }
}

#[pymethods]
impl PyHandle {
    pub fn destroy(&mut self) -> PyResult<()> {
        if self.is_destroyed {
            return Err(PyRuntimeError::new_err("handle is already destroyed"));
        }
        self.handle.destroy();
        self.is_destroyed = true;
        Ok(())
    }
}

#[pyfunction]
#[pyo3[name = "create_handle", signature = (max_nodes, device=None, exact=None), text_signature="(max_nodes, device=None, exact=None)"]]
pub fn create_handle(max_nodes: usize, device: Option<i32>, exact: Option<bool>) -> PyHandle {
    PyHandle {
        is_destroyed: false,
        handle: Handle::new(max_nodes, device, exact),
    }
}
