"""
Activations for CAREamics models.

Copied from
https://github.com/CAREamics/careamics/blob/main/src/careamics/models/activation.py

BSD 3-Clause License

Copyright (c) 2023, CAREamics contributors

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from collections.abc import Callable
from typing import Union

import torch.nn as nn

from .support import SupportedActivation


def get_activation(activation: Union[SupportedActivation, str]) -> Callable:
    """
    Get activation function.

    Parameters
    ----------
    activation : str
        Activation function name.

    Returns
    -------
    Callable
        Activation function.
    """
    if activation == SupportedActivation.RELU:
        return nn.ReLU()
    elif activation == SupportedActivation.ELU:
        return nn.ELU()
    elif activation == SupportedActivation.LEAKYRELU:
        return nn.LeakyReLU()
    elif activation == SupportedActivation.TANH:
        return nn.Tanh()
    elif activation == SupportedActivation.SIGMOID:
        return nn.Sigmoid()
    elif activation == SupportedActivation.SOFTMAX:
        return nn.Softmax(dim=1)
    elif activation == SupportedActivation.NONE:
        return nn.Identity()
    else:
        raise ValueError(f"Activation {activation} not supported.")
