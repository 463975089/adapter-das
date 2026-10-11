import os

import sgl_kernel._adapters.hcu_env as h

V = "HCU_DEEPGEMM_BRIDGE"
print("file:", h.__file__)
print("env raw:", repr(os.environ.get(V, "<unset>")))
print("resolve:", h.resolve(V))
print("registry default:", h.REGISTRY[V]["default"], "auto?", "auto" in h.REGISTRY[V])
