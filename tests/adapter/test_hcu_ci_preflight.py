# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.hcu_ci_preflight import check_devices


TABLE = """HCU Temp AvgPwr Perf PwrCap VRAM% HCU% Dec% Enc% Mode
0 25.0C 158.0W manual 800.0W 89% 0.0% 0.0% 0.0% Normal
4 30.0C 158.0W manual 800.0W 0% 0.0% 0.0% 0.0% Normal
5 30.0C 158.0W manual 800.0W 4% 0.0% 0.0% 0.0% Normal
"""


def test_only_explicit_reserved_devices_are_checked():
    assert check_devices(TABLE, "4,5") == ["4", "5"]


@pytest.mark.parametrize("devices", ["", "4,4", "4,", "-1", "04", "auto", "0", "8"])
def test_invalid_missing_or_busy_reservation_is_refused(devices):
    with pytest.raises(ValueError):
        check_devices(TABLE, devices)


@pytest.mark.parametrize("replacement", ["5% 0.0%", "0% 1.0%", "nan% 0%",
                                         "0% inf%", "0 0", "101% 0%"])
def test_busy_or_unparseable_device_is_refused(replacement):
    with pytest.raises(ValueError):
        check_devices(TABLE.replace("0% 0.0%", replacement), "4")
