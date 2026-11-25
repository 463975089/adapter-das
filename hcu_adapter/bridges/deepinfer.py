# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

def install():
    from ._deepinfer.bootstrap import register_worker_import_hook
    register_worker_import_hook()


def install_runtime(**options):
    from ._deepinfer.bootstrap import install
    install(**options)


def install_distributed_workers():
    from ._deepinfer.distributed_workers import install

    install()
