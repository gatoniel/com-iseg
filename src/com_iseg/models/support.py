"""
Activations supported by CAREamics.

Copied from
https://github.com/CAREamics/careamics/blob/main/src/careamics/config/support/supported_activations.py

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

from enum import Enum, EnumMeta
from typing import Any


class _ContainerEnum(EnumMeta):
    """Metaclass for Enum with __contains__ method."""

    def __contains__(cls, item: Any) -> bool:
        """Check if an item is in the Enum.

        Parameters
        ----------
        item : Any
            Item to check.

        Returns
        -------
        bool
            True if the item is in the Enum, False otherwise.
        """
        try:
            cls(item)
        except ValueError:
            return False
        return True

    @classmethod
    def has_value(cls, value: Any) -> bool:
        """Check if a value is in the Enum.

        Parameters
        ----------
        value : Any
            Value to check.

        Returns
        -------
        bool
            True if the value is in the Enum, False otherwise.
        """
        return value in cls._value2member_map_


class BaseEnum(Enum, metaclass=_ContainerEnum):
    """Base Enum class, allowing checking if a value is in the enum.

    Example
    -------
    >>> from careamics.utils.base_enum import BaseEnum
    >>> # Define a new enum
    >>> class BaseEnumExtension(BaseEnum):
    ...     VALUE = "value"
    >>> # Check if value is in the enum
    >>> "value" in BaseEnumExtension
    True
    """

    pass


class SupportedActivation(str, BaseEnum):
    """Supported activation functions.

    - None, no activation will be used.
    - Sigmoid
    - Softmax
    - Tanh
    - ReLU
    - LeakyReLU

    All activations are defined in PyTorch.

    See: https://pytorch.org/docs/stable/nn.html#loss-functions
    """

    NONE = "None"
    SIGMOID = "Sigmoid"
    SOFTMAX = "Softmax"
    TANH = "Tanh"
    RELU = "ReLU"
    LEAKYRELU = "LeakyReLU"
    ELU = "ELU"
