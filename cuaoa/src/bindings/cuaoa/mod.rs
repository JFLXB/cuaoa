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

mod brute_force;
mod constants;
mod device_info;
mod expval;
mod gradients;
mod handle;
mod optimize;
mod sample;
mod statevector;

pub use brute_force::brute_force;
pub use constants::DEFAULT_CUDA_BLOCK_SIZE;
pub use device_info::{get_cuda_devices_info, CudaDevice};
pub use expval::expval;
pub use gradients::gradients;
pub use handle::Handle;
pub use optimize::optimize;
pub use sample::sample;
pub use statevector::statevector;
