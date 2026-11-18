import os
from contextlib import contextmanager
from functools import wraps


def install():
    from hcu_adapter import registry
    from sglang.srt.platforms.cuda import CudaDeviceMixin

    original = CudaDeviceMixin.reindex_device_id
    if getattr(original, "_hcu_sglang_visibility", False):
        return

    @contextmanager
    @wraps(original)
    def reindex(platform, device_id):
        before = {
            name: os.environ.get(name)
            for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
        }
        if os.environ.get("ROCR_VISIBLE_DEVICES") is not None:
            raise ValueError("SGLang single-device HIP mapping requires ROCR_VISIBLE_DEVICES unset")
        hip_mask = before["HIP_VISIBLE_DEVICES"]
        if before["CUDA_VISIBLE_DEVICES"] is None and hip_mask is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = hip_mask
        try:
            with original(platform, device_id) as logical_id:
                selected = os.environ["CUDA_VISIBLE_DEVICES"]
                if hip_mask is not None and selected not in hip_mask.split(","):
                    raise ValueError("SGLang device assignment exceeds the HIP allocation")
                os.environ["HIP_VISIBLE_DEVICES"] = selected
                yield logical_id
        finally:
            for name, value in before.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    reindex._hcu_sglang_visibility = True
    CudaDeviceMixin.reindex_device_id = reindex
    registry.register(
        "platform_detect", reindex,
        target="sglang.srt.platforms.cuda:CudaDeviceMixin.reindex_device_id",
        mechanism=registry.MECH_REBIND,
        detail="SGLang child HIP visibility follows upstream single-device assignment",
    )
