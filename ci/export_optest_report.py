# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import html
import json
import posixpath
import re
from pathlib import Path
from urllib.parse import quote

from optest import display


REPORT = display.REPORT_FILENAME
DAILY_ASSETS = Path("report-assets/daily")
OVERVIEW_ASSETS = Path("report-assets/overview")
DAILY_NAME = re.compile(r"daily-\d{4}-\d{2}-\d{2}-\d+-\d+")


def _relative_url(target, parent):
    return quote(posixpath.relpath(target.as_posix(), parent.as_posix()), safe="/")


def _archive_data(archive):
    data = display.collect_display_data(archive)
    for operator in data["operators"]:
        for version in operator["versions"]:
            stem, _, stamp = version["id"].rpartition("_")
            if not re.fullmatch(r"\d{12}", stamp):
                stem, stamp = version["id"], "000000000000"
            # Same source and commit can appear in multiple daily runs.
            version["id"] = f"{stem}_{archive.name}_{stamp}"
            version["archive"] = archive.name
            # The displayed date is the daily archive date, with the original result time.
            version["updated"] = archive.name[6:16] + version["updated"][10:]
    return data


def _write_scope(root, entry, assets, data, *, overview=False):
    operators = data["operators"]
    if not operators:
        (root / entry).write_text(
            '<!doctype html><meta charset="utf-8"><title>Optest</title><p>本次没有可展示的算子结果。</p>',
            encoding="utf-8",
        )
        return

    pages = assets / "pages"
    (root / pages).mkdir(parents=True, exist_ok=True)
    for index, operator in enumerate(operators):
        operator_id = operator["id"]
        civiz = assets / "civiz" / operator_id
        (root / civiz).mkdir(parents=True, exist_ok=True)
        links = []
        for version in operator["versions"]:
            version_id = version["id"]
            version_dir = root / civiz / version_id
            version_dir.mkdir(exist_ok=True)
            meta = display._civiz_meta(version)
            if overview:
                daily_data = Path(version["archive"]) / DAILY_ASSETS / "civiz" / operator_id / version_id
                meta["base_path"] = _relative_url(daily_data, civiz)
            else:
                meta["base_path"] = quote(version_id, safe="")
                (version_dir / "tables").mkdir(exist_ok=True)
                for name, table in display._civiz_table_entries(version):
                    (version_dir / "tables" / (name + ".json")).write_text(
                        json.dumps(table, ensure_ascii=False), encoding="utf-8",
                    )
            (version_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            links.append(f'<a href="{quote(version_id, safe="")}/">{html.escape(version_id)}/</a>')
        (root / civiz / "index.html").write_text(
            '<!doctype html><meta charset="utf-8">\n' + "\n".join(links), encoding="utf-8",
        )

        targets = [pages / (operator_id + ".html")]
        if index == 0:
            targets.append(entry)
        for target in targets:
            report = display._report_html(data, operator_id)
            report = report.replace(
                f'<base href="/{display.VIRTUAL_CIVIZ_ROOT}/{operator_id}/">',
                f'<base href="{_relative_url(civiz, target.parent)}/">',
            ).replace(
                f'new URL("/{REPORT}", window.location.origin)',
                f'new URL("{_relative_url(pages, target.parent)}/" + '
                'encodeURIComponent(this.value) + ".html", window.location.href)',
            ).replace('fetch("./?t="', 'fetch("./index.html?t="')
            (root / target).write_text(report, encoding="utf-8")


def export_reports(site, archive, published):
    """Prepare the new daily and overview in staging; historical table data stays in each daily."""
    archives = sorted(
        path for path in published.iterdir()
        if path.is_dir() and DAILY_NAME.fullmatch(path.name) and (path / "total_output.json").is_file()
    )
    operators = {}
    for day in [*archives, archive]:
        data = _archive_data(day)
        # Older archives produced before the static exporter may only contain raw results.
        if not (day / DAILY_ASSETS / ".complete").is_file():
            root = site if day == archive else published
            _write_scope(root, Path(day.name) / REPORT, Path(day.name) / DAILY_ASSETS, data)
            assets = day / DAILY_ASSETS
            assets.mkdir(parents=True, exist_ok=True)
            (assets / ".complete").touch()
        for operator in data["operators"]:
            merged = operators.setdefault(operator["id"], dict(operator, versions=[]))
            merged["versions"].extend(operator["versions"])
            merged["has_accuracy"] |= operator["has_accuracy"]
            merged["has_performance"] |= operator["has_performance"]
    for operator in operators.values():
        operator["versions"].sort(key=lambda version: version["updated"], reverse=True)

    _write_scope(
        site, Path(REPORT), Path(archive.name) / OVERVIEW_ASSETS,
        {"operators": list(operators.values())}, overview=True,
    )
