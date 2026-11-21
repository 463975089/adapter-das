# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Atomic file write shared by the text-patch adapters.

Every install() runs in up to 9 processes at once (parent + 8 TP scheduler
children within seconds of each other). Plain `write_text` truncates then
writes, so a concurrent reader can see a half-written file -- observed live
as phantom "needle missing" RuntimeErrors (the reader sees neither needle
nor replacement) and, worse, as torn last-writer-wins corruption
(paged_mqa_metadata.cuh lost 119 lines to it, 2026-10-20). tempfile +
os.replace in the SAME directory keeps every reader on the old or the new
inode, never a partial file.
"""

import os
import tempfile


def atomic_write(path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".hcu_atomic_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
