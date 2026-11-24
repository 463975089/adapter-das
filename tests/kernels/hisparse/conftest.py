# SPDX-License-Identifier: Apache-2.0
"""Explicit artifact selection; never compiles or changes an installation."""


def pytest_addoption(parser):
    group = parser.getgroup('hisparse-native')
    group.addoption('--hisparse-library', default=None,
                    help='Explicit isolated hisparse_ops .so; default uses the paired wheel')
    group.addoption('--hisparse-device', type=int, default=0,
                    help='Logical HIP device, only selected when a numerical test runs')
