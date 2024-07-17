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

use std::{ffi::c_void, sync::Mutex};

#[link(name = "cuaoalg")]
extern "C" {
    fn createHandle(numNodes: usize, device: i32, exact: bool) -> *mut c_void;
    fn destroyHandle(handle: *mut c_void);
}

#[derive(Debug)]
pub struct SafePtr {
    inner: *mut c_void,
}

unsafe impl Send for SafePtr {}

impl SafePtr {
    fn new(ptr: *mut c_void) -> Self {
        SafePtr { inner: ptr }
    }
    pub fn as_ptr(&self) -> *mut c_void {
        self.inner
    }
}

#[derive(Debug)]
pub struct Handle {
    ptr: Mutex<SafePtr>,
    is_alive: bool,
}

impl Handle {
    pub fn new(num_bits: usize, device: Option<i32>, exact: Option<bool>) -> Self {
        let dev = device.unwrap_or(0);
        let ptr = unsafe { createHandle(num_bits, dev, exact.unwrap_or(false)) };
        Handle {
            ptr: Mutex::new(SafePtr::new(ptr)),
            is_alive: true,
        }
    }

    pub fn ptr(&self) -> &Mutex<SafePtr> {
        return &self.ptr;
    }

    pub fn destroy(&mut self) {
        if self.is_alive {
            let lock_guard = self.ptr.lock().unwrap();
            unsafe { destroyHandle(lock_guard.as_ptr()) }
            self.is_alive = false;
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if self.is_alive {
            let lock_guard = self.ptr.lock().unwrap();
            unsafe { destroyHandle(lock_guard.as_ptr()) };
        }
    }
}
