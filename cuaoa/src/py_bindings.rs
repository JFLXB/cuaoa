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

#[cfg(feature = "py")]
mod py_bindings {
    use pyo3::prelude::*;

    use crate::core::{LBFGSLinesearchAlgorithm, LBFGSParameters};

    #[pymethods]
    impl LBFGSParameters {
        #[new]
        #[pyo3(signature = (
            num_corrections=None,
            epsilon=None,
            past=None,
            delta=None,
            max_iterations=None,
            linesearch=None,
            max_linesearch=None,
            min_step=None,
            max_step=None,
            ftol=None,
            wolfe=None,
            gtol=None,
            xtol=None,
            orthantwise_c=None,
            orthantwise_start=None,
            orthantwise_end=None,
            log=None,
        ), text_signature="(
            num_corrections=None,
            epsilon=None,
            past=None,
            delta=None,
            max_iterations=None,
            linesearch=None,
            max_linesearch=None,
            min_step=None,
            max_step=None,
            ftol=None,
            wolfe=None,
            gtol=None,
            xtol=None,
            orthantwise_c=None,
            orthantwise_start=None,
            orthantwise_end=None,
            log=None,
        )")]
        fn new_py(
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
            Self::new(
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
            )
        }
        #[pyo3(name = "num_corrections", text_signature = "(self)")]
        pub fn py_num_corrections(&self) -> Option<i32> {
            self.num_corrections
        }

        #[pyo3(name = "epsilon", text_signature = "(self)")]
        pub fn py_epsilon(&self) -> Option<f64> {
            self.epsilon
        }
        #[pyo3(name = "past", text_signature = "(self)")]
        pub fn py_past(&self) -> Option<i32> {
            self.past
        }
        #[pyo3(name = "delta", text_signature = "(self)")]
        pub fn py_delta(&self) -> Option<f64> {
            self.delta
        }
        #[pyo3(name = "max_iterations", text_signature = "(self)")]
        pub fn py_max_iterations(&self) -> Option<i32> {
            self.max_iterations
        }
        #[pyo3(name = "linesearch", text_signature = "(self)")]
        pub fn py_linesearch(&self) -> Option<LBFGSLinesearchAlgorithm> {
            self.linesearch
        }
        #[pyo3(name = "max_linesearch", text_signature = "(self)")]
        pub fn py_max_linesearch(&self) -> Option<i32> {
            self.max_linesearch
        }
        #[pyo3(name = "min_step", text_signature = "(self)")]
        pub fn py_min_step(&self) -> Option<f64> {
            self.min_step
        }
        #[pyo3(name = "max_step", text_signature = "(self)")]
        pub fn py_max_step(&self) -> Option<f64> {
            self.max_step
        }
        #[pyo3(name = "ftol", text_signature = "(self)")]
        pub fn py_ftol(&self) -> Option<f64> {
            self.ftol
        }
        #[pyo3(name = "wolfe", text_signature = "(self)")]
        pub fn py_wolfe(&self) -> Option<f64> {
            self.wolfe
        }
        #[pyo3(name = "gtol", text_signature = "(self)")]
        pub fn py_gtol(&self) -> Option<f64> {
            self.gtol
        }
        #[pyo3(name = "xtol", text_signature = "(self)")]
        pub fn py_xtol(&self) -> Option<f64> {
            self.xtol
        }
        #[pyo3(name = "orthantwise_c", text_signature = "(self)")]
        pub fn py_orthantwise_c(&self) -> Option<f64> {
            self.orthantwise_c
        }
        #[pyo3(name = "orthantwise_start", text_signature = "(self)")]
        pub fn py_orthantwise_start(&self) -> Option<i32> {
            self.orthantwise_start
        }
        #[pyo3(name = "orthantwise_end", text_signature = "(self)")]
        pub fn py_orthantwise_end(&self) -> Option<i32> {
            self.orthantwise_end
        }
        #[pyo3(name = "log", text_signature = "(self)")]
        pub fn py_log(&self) -> Option<bool> {
            self.log
        }
    }
}
