import unittest
import torch
from sgl_kernel import minimax_h3_rotary_embedding


class H3RotaryCpuTests(unittest.TestCase):
    def reference(self, value, positions, cache, head_size=128):
        half = cache.shape[1] // 2
        result = value.clone()
        view = result.view(value.shape[0], -1, head_size)
        source = value.view(value.shape[0], -1, head_size).double()
        a, b = source[..., :half], source[..., half:2 * half]
        c = cache[positions, :half].double().unsqueeze(1)
        s = cache[positions, half:].double().unsqueeze(1)
        if value.dtype == torch.bfloat16:
            y1 = (a*c).to(value.dtype).double() - (b*s).to(value.dtype).double()
            y2 = (b*c).to(value.dtype).double() + (a*s).to(value.dtype).double()
        else:
            y1 = (a*c - (b*s).to(value.dtype).double()).float()
            y2 = (b*c + (a*s).to(value.dtype).double()).float()
        view[..., :half] = y1.to(value.dtype)
        view[..., half:2 * half] = y2.to(value.dtype)
        return result

    def test_dtype_stride_partial_tail_and_optional_key(self):
        generator = torch.Generator(device="cpu").manual_seed(91)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for with_key in (False, True):
                q_buffer = torch.randn(7, 3 * 128 + 13, generator=generator).to(dtype)
                k_buffer = torch.randn(7, 2 * 128 + 19, generator=generator).to(dtype)
                q = q_buffer[:, :3 * 128]
                k = k_buffer[:, :2 * 128] if with_key else None
                q_before, k_before = q_buffer.clone(), k_buffer.clone()
                positions = (torch.arange(14) % 11)[::2]
                cache = torch.randn(11, 96, generator=generator).to(dtype)
                expected_q = self.reference(q.clone(), positions, cache)
                expected_k = None if k is None else self.reference(k.clone(), positions, cache)
                result = minimax_h3_rotary_embedding(positions, q, k, 128, cache, True)
                self.assertIsNone(result)
                self.assertTrue(torch.equal(q, expected_q))
                self.assertTrue(torch.equal(q_buffer[:, 3 * 128:], q_before[:, 3 * 128:]))
                self.assertTrue(torch.equal(q.view(7, 3, 128)[..., 96:], q_before[:, :3 * 128].view(7, 3, 128)[..., 96:]))
                if k is not None:
                    self.assertTrue(torch.equal(k, expected_k))
                    self.assertTrue(torch.equal(k_buffer[:, 2 * 128:], k_before[:, 2 * 128:]))

    def test_same_qk_view_rotates_once(self):
        q = torch.arange(256, dtype=torch.float32).reshape(2, 128)
        cache = torch.ones(2, 96)
        positions = torch.arange(2)
        expected = self.reference(q.clone(), positions, cache)
        minimax_h3_rotary_embedding(positions, q, q, 128, cache, True)
        self.assertTrue(torch.equal(q, expected))

    def test_empty_input(self):
        q = torch.empty(0, 128)
        self.assertIsNone(minimax_h3_rotary_embedding(torch.empty(0, dtype=torch.int64), q, None, 128, torch.empty(1, 96), True))

    def test_invalid_geometry_is_rejected(self):
        q = torch.ones(2, 128)
        positions = torch.arange(2)
        with self.assertRaisesRegex(ValueError, "head_size"):
            minimax_h3_rotary_embedding(positions, q, None, 32, torch.ones(2, 96), True)
        with self.assertRaisesRegex(ValueError, "cache"):
            minimax_h3_rotary_embedding(positions, q, None, 128, torch.ones(2, 64), True)
        with self.assertRaisesRegex(ValueError, "dtype"):
            minimax_h3_rotary_embedding(positions, q, None, 128, torch.ones(2, 96, dtype=torch.bfloat16), True)
        with self.assertRaisesRegex(ValueError, "integer"):
            minimax_h3_rotary_embedding(positions.float(), q, None, 128, torch.ones(2, 96), True)


if __name__ == "__main__":
    unittest.main()
