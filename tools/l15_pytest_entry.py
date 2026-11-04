# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Pytest entry supporting Python 3.10 upstream tests which import tomllib."""

import sys

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:
    import tomli

    sys.modules["tomllib"] = tomli


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main(sys.argv[1:]))
