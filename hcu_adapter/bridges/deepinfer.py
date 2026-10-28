def install():
    from ._deepinfer.bootstrap import register_worker_import_hook
    register_worker_import_hook()


def install_runtime(**options):
    from ._deepinfer.bootstrap import install
    install(**options)
