# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from cinotificate.record import operator_results_pass
from optest.config import CONFIG_PATH

from export_optest_report import REPORT, export_reports


def publish(staging, output_dir):
    daily_dir = staging / "daily"
    filename = "total_output.json"
    daily_result = daily_dir / filename
    if daily_result.is_file():
        data = json.loads(daily_result.read_text(encoding="utf-8"))
    else:
        data = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "operators": []}

    config = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    expected = {name for name, settings in config.items() if settings.get("enabled", True)}
    operators = data["operators"]
    reported = [item.get("op_name") for item in operators]
    complete = bool(expected) and len(reported) == len(expected) and set(reported) == expected
    passed = os.environ["OPTEST_OUTCOME"] == "success" and complete and operator_results_pass(data)
    status = "ci_pass" if passed else "ci_fail"
    data.update(image=os.environ["IMAGE_TAG"], test_status=status)
    daily_result.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_root = output_dir.parent
    site = staging / "site"
    site.mkdir()
    archive = site / output_dir.name
    daily_dir.rename(archive)
    export_reports(site, archive, output_root)

    report_path = output_root / REPORT
    archive.rename(output_dir)
    (site / REPORT).replace(report_path)

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as outputs:
        print("report_path={}".format(report_path), file=outputs)
        print("result_path={}".format(output_dir / filename), file=outputs)
        print("test_status={}".format(status), file=outputs)
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
        print("Optest results: `{}`".format(output_dir), file=summary)
        print("Optest report: `{}`".format(report_path), file=summary)


if __name__ == "__main__":
    publish(Path(sys.argv[1]), Path(sys.argv[2]))
