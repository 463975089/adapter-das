# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Check explicitly reserved CI devices against the observed HCU SMI table."""

import argparse
import re
import subprocess


def check_devices(table, requested):
    ids = requested.split(",")
    if (not ids or any(not re.fullmatch(r"0|[1-9][0-9]*", x) for x in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("configure distinct numeric HCU_GPU_IDS")
    rows = {}
    for line in table.splitlines():
        columns = line.split()
        if columns and re.fullmatch(r"[0-9]+", columns[0]):
            if len(columns) < 7:
                raise ValueError("unrecognized HCU SMI row")
            try:
                if not columns[5].endswith("%") or not columns[6].endswith("%"):
                    raise ValueError()
                memory = float(columns[5][:-1])
                utilization = float(columns[6][:-1])
                if not (0 <= memory <= 100 and 0 <= utilization <= 100):
                    raise ValueError()
            except ValueError:
                raise ValueError("unrecognized HCU SMI utilization") from None
            if columns[0] in rows:
                raise ValueError("duplicate HCU SMI device row")
            rows[columns[0]] = (memory, utilization)
    for device in ids:
        if device not in rows:
            raise ValueError(f"device {device} absent from HCU SMI")
        memory, utilization = rows[device]
        if memory >= 5 or utilization != 0:
            raise ValueError(f"reserved device {device} is busy; defer this job")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True)
    args = parser.parse_args()
    try:
        result = subprocess.run(["rocm-smi"], capture_output=True, text=True,
                                check=True, timeout=30)
        ids = check_devices(result.stdout, args.devices)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"HCU resource preflight failed: {exc}\n")
    print("HCU reserved-device preflight: PASS (" + ",".join(ids) + ")")


if __name__ == "__main__":
    main()
