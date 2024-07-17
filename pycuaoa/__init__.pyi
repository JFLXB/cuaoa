# Copyright 2024 Jonas Blenninger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import cuaoa as cuaoa
from . import utils as utils

from .pyhandle import PyHandle as PyHandle
from .pyhandle import create_handle as create_handle

from .pycuaoa import CUAOA as CUAOA
from .pycuaoa import BruteFroce as BruteFroce
from .pycuaoa import make_polynomial as make_polynomial

from .core import LBFGSParameters as LBFGSParameters
from .core import LBFGSLinesearchAlgorithm as LBFGSLinesearchAlgorithm
from .core import ParameterizationMethod as ParameterizationMethod
from .core import OptimizeResult as OptimizeResult
from .core import Parameters as Parameters
from .core import Polynomial as Polynomial
from .core import SampleSet as SampleSet
from .core import RXMethod as RXMethod
from .core import Gradients as Gradients
from .core import Sample as Sample

from .utils import PyCudaDevice as PyCudaDevice
from .utils import get_cuda_devices_info as get_cuda_devices_info
