import types
import unittest
from unittest.mock import Mock, patch

from hcu_adapter.bridges._sglang import sampling


class GrammarLaunchTests(unittest.TestCase):
    def install(self, warp_size=64, hip="6.3"):
        self.calls = []

        class Kernel:
            def __getitem__(kernel, grid):
                return lambda *args, **kwargs: kernel.run(*args, grid=grid, **kwargs)

            def run(kernel, *args, **kwargs):
                if kwargs.get("num_warps", 4) * warp_size > 1024:
                    raise RuntimeError("threads: Required: 2048, Hardware limit: 1024")
                self.calls.append((args, kwargs))
                return "launched"

        self.kernel = Kernel()
        self.original = self.kernel.run
        self.target = Mock(return_value=types.SimpleNamespace(warp_size=warp_size))
        self.record = Mock()
        modules = {
            "torch": types.SimpleNamespace(
                version=types.SimpleNamespace(hip=hip),
                cuda=types.SimpleNamespace(get_device_properties=self.target)),
            "sglang.kernels.ops.grammar.bitmask_ops": types.SimpleNamespace(
                apply_token_bitmask_inplace_kernel=self.kernel),
        }
        self.context = patch.dict("sys.modules", modules)
        self.context.start()
        self.addCleanup(self.context.stop)
        self.registry = patch("hcu_adapter.registry.register", self.record)
        self.registry.start()
        self.addCleanup(self.registry.stop)
        sampling.install_grammar_bitmask()

    def test_public_grid_launch_bounds_wave64_threads(self):
        self.install()
        logits = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))
        with self.assertRaisesRegex(RuntimeError, "Required: 2048"):
            self.original(logits, num_warps=32)
        self.assertEqual(self.kernel[(3,)](logits, num_warps=32, num_stages=3), "launched")
        self.assertEqual(self.calls[0][1], {"grid": (3,), "num_warps": 16, "num_stages": 3})
        self.assertEqual(self.calls[0][0], (logits,))
        self.record.assert_called_once()

    def test_wave32_keeps_legal_launch(self):
        self.install(warp_size=32)
        self.kernel[(1,)](types.SimpleNamespace(device=types.SimpleNamespace(type="cuda")), num_warps=32)
        self.assertEqual(self.calls[0][1]["num_warps"], 32)

    def test_non_gpu_does_not_initialize_driver(self):
        self.install()
        self.kernel[(1,)](types.SimpleNamespace(device=types.SimpleNamespace(type="cpu")), num_warps=4)
        self.target.assert_not_called()

    def test_install_is_idempotent_and_keeps_small_launch(self):
        self.install()
        wrapped = self.kernel.run
        sampling.install_grammar_bitmask()
        self.assertIs(wrapped, self.kernel.run)
        self.kernel[(1,)](types.SimpleNamespace(device=types.SimpleNamespace(type="cuda")), num_warps=8)
        self.assertEqual(self.calls[0][1]["num_warps"], 8)
        self.record.assert_called_once()

    def test_non_hip_keeps_original(self):
        self.install(hip=None)
        self.assertEqual(self.original, self.kernel.run)
        self.record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
