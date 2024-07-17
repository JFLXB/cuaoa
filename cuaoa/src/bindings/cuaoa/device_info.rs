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

use core::slice;
use std::{ffi::CStr, os::raw::c_void};

pub struct CudaDevice {
    pub id: i32,
    pub name: String,
    pub memory: usize,
}

#[repr(C)]
struct CudaDeviceInfo {
    id: i32,
    name: *const i8,
    memory: usize,
}

#[repr(C)]
struct CudaDeviceInfoResult {
    num_devices: usize,
    devices: *const c_void,
}

extern "C" {
    fn getCudaDevicesInfo() -> CudaDeviceInfoResult;
    fn getCudaDevicesInfoData(result: *const CudaDeviceInfoResult) -> *const CudaDeviceInfo;
    fn freeCudaDevicesInfoResult(result: *mut CudaDeviceInfoResult);
}

pub fn get_cuda_devices_info() -> Vec<CudaDevice> {
    let result = unsafe { getCudaDevicesInfo() };
    let c_cuda_devices =
        unsafe { slice::from_raw_parts(getCudaDevicesInfoData(&result), result.num_devices) };
    let cuda_devices = c_cuda_devices
        .iter()
        .map(|c_device| CudaDevice {
            id: c_device.id,
            name: unsafe { CStr::from_ptr(c_device.name).to_string_lossy().into_owned() },
            memory: c_device.memory,
        })
        .collect();
    unsafe {
        freeCudaDevicesInfoResult(&result as *const _ as *mut CudaDeviceInfoResult);
    }
    cuda_devices
}
